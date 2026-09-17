"""V32 synthetic CPU tests. No checkpoint downloads, GPU work, or probe datasets."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from analysis import v6_capability_geometry as v6
from analysis import v28_new_source_prediction as v28
from analysis import v32_prune_descriptors as v32


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 2)
        self.layers = nn.ModuleList([nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)])
        self.lm_head = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.arange(8).reshape(4, 2)/4)
            self.layers[0].weight.copy_(torch.tensor([[0., -1.], [1., 2.]]))
            self.layers[1].weight.copy_(torch.tensor([[3., 0.], [-2., 1.]]))
            self.lm_head.weight.fill_(1.)

    def forward(self, input_ids, **_kwargs):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_retention_is_exact_v6_including_ties_embeddings_and_head(dtype):
    model = TinyLM().to(dtype=dtype)
    before = {n: p.clone() for n, p in model.named_parameters()}
    result = v32.weight_descriptors(model)
    assert result["N0"] == 8
    assert result["n_weights"] == 24
    assert result["n_layers"] == 4
    assert set(result["features"]) == set(v32.WEIGHT_KEYS)
    for density in v28.PRUNE_DEV:
        pruned = copy.deepcopy(model)
        threshold = v6.apply_global_magnitude_pruning(pruned, density)
        assert threshold == result["thresholds"][str(density)]
        for name, param in pruned.named_parameters():
            block = result["layers"][v32.layer_name(name)]
            assert block["retained_counts"][str(density)] == int(torch.count_nonzero(param))
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert result == v32.weight_descriptors(model)
    assert any(result["achieved_density"][str(d)] != pytest.approx(d) for d in v28.PRUNE_DEV)


def test_retention_aggregates_blocks_by_weight_count_and_does_not_mutate(monkeypatch):
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2)), nn.Linear(2, 2)])
    monkeypatch.setattr(v32, "CHUNK_SIZE", 3)
    result = v32.weight_descriptors(model)
    assert result["layers"]["layers.0"]["n_weights"] == 12
    assert result["layers"]["layers.1"]["n_weights"] == 4
    for d in v28.PRUNE_DEV:
        expected = [float((p.abs() > result["thresholds"][str(d)]).sum()) for n, p in
                    v6.language_weight_parameters(model) if n.startswith("layers.0.")]
        assert result["layers"]["layers.0"]["retention"][str(d)] == sum(expected)/12


def test_distribution_definitions_and_degenerate_cases():
    constant = v32.distribution([.8, .8, .8])
    assert constant["entropy"] == pytest.approx(np.log(3))
    assert constant["gini"] == pytest.approx(0)
    assert constant["skew"] == constant["variance"] == 0
    assert v32.distribution([.5, .5])["skew"] == 0
    assert all(v == 0 for v in v32.distribution([0, 0]).values())
    assert v32.distribution([0, 1])["gini"] == pytest.approx(.5)
    for invalid in ([], [np.nan], [-1], [np.inf]):
        with pytest.raises(ValueError):
            v32.distribution(invalid)


def test_streamed_activation_moments_cost_and_state_restoration(monkeypatch):
    model = TinyLM().train()
    model.layers[0].eval()
    flags = [m.training for m in model.modules()]
    before = copy.deepcopy(model.state_dict())
    batches = [{"input_ids": torch.tensor([[0, 1]])}, {"input_ids": torch.tensor([[2, 3, 0]])}]
    expected = {"layers.0": [], "layers.1": []}
    for batch in batches:
        value = model.embed_tokens(batch["input_ids"])
        for i, layer in enumerate(model.layers):
            value = layer(value)
            expected[f"layers.{i}"].extend(value.detach().abs().reshape(-1).tolist())
    monkeypatch.setattr(v32, "CHUNK_SIZE", 3)
    result = v32.activation_descriptors(model, batches, protocol={"synthetic": True})
    assert result["cost"]["extra_dense_forwards"] == 2
    assert result["cost"]["input_tokens"] == 5
    assert result["cost"]["seconds"] >= 0
    assert set(result["features"]) == set(v32.ACTIVATION_KEYS)
    for name, values in expected.items():
        assert result["layers"][name]["mean"] == pytest.approx(np.mean(values))
        assert result["layers"][name]["variance"] == pytest.approx(np.var(values))
    assert [m.training for m in model.modules()] == flags
    assert all(not m._forward_hooks for m in model.modules())
    assert all(torch.equal(p, before[n]) for n, p in model.state_dict().items())


def test_activation_hooks_cleaned_after_failure():
    model = TinyLM().train()
    def failing_batches():
        yield {"input_ids": torch.tensor([[0, 1]])}
        raise RuntimeError("synthetic failure")
    with pytest.raises(RuntimeError, match="synthetic failure"):
        v32.activation_descriptors(model, failing_batches(), protocol={})
    assert all(not m._forward_hooks and m.training for m in model.modules())


def test_shared_probe_budget_balanced_even_indices_without_dataset(monkeypatch):
    calls, texts = [], []
    def probes(n, seed):
        calls.append((n, seed))
        return {c: [{"prompt": f"{c}-{i}", "completion": " answer"} for i in range(128)]
                for c in v32.audit.CAPABILITIES}
    class Tokenizer:
        def __call__(self, text, **kwargs):
            texts.append((text, kwargs["add_special_tokens"]))
            return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))
    monkeypatch.setattr(v6, "build_probes", probes)
    batches, protocol = v32.shared_probe_batches(Tokenizer(), "cpu", forwards=6)
    assert len(list(batches)) == 6
    assert calls == [(128, 0)]
    assert [text for text, special in texts if special] == ["math-0", "code-0", "qa-0", "math-2", "code-2", "qa-2"]
    assert protocol["requested_forwards"] == 6
    assert len(protocol["probe_sha256"]) == 64
    with pytest.raises(ValueError, match="divisible"):
        v32.shared_probe_batches(Tokenizer(), "cpu", forwards=4)


def synthetic_rows():
    rows = []
    for i in range(7):
        amplitude = (-1 if i < 3 else 1)*(.2+i*.1)
        features = {k: float((i+1)**(1+j % 2)) for j, k in enumerate(v32.WEIGHT_KEYS)}
        features.update({k: float(np.sin(i+j)) for j, k in enumerate(v32.ACTIVATION_KEYS)})
        features[v32.WEIGHT_KEYS[-1]] = 1.  # train-only constant-column selection
        rows.append({"model": f"fixture-{i}", "family": "a" if i < 4 else "b", "N0": (i+1)*1e9,
                     "dense_loss": 1.5-.1*i, "capability": "qa", "descriptor_features": features,
                     "deltas": {str(d): amplitude*v28.shape("pruning", d, 2.) for d in v28.PRUNE_DEV}})
    return rows


def test_signed_labels_v1_helper_equivalence_and_nested_features():
    rows = synthetic_rows()
    result = v32.evaluate_lomo(rows, n_boot=40)
    for record, fold in zip(result["records"], result["folds"]):
        target = next(r for r in rows if r["model"] == record["model"])
        expected = target["deltas"]["0.7"]
        assert record["observed"] == pytest.approx(expected, rel=2e-5)
        fit = v28.fit_arm([r for r in rows if r["model"] != target["model"]], "pruning")
        assert record["predictions"]["V1"] == pytest.approx(v28.predict_mapping(fit["mapping"], target))
        assert set(fold["fits"]["V2"]["descriptor_keys"]) < set(fold["fits"]["V3"]["descriptor_keys"])
        assert fold["gamma"] == pytest.approx(2., abs=2e-5)
    assert result["records"][0]["observed"] < 0
    assert result["records"][0]["predictions"]["V2"] < 0
    assert result["metrics"]["V1"]["n_negative_models"] == 3


@pytest.mark.parametrize("version", ["V2", "V3"])
def test_augmented_helper_matches_joint_ridge_objective(version):
    rows = synthetic_rows()
    train, target = rows[:-1], rows[-1]
    labels = np.array([r["deltas"]["0.7"] for r in train])
    fit = v32.fit_mapping(train, labels, version)
    base = fit["base"]
    def design(r):
        basic = (np.array([np.log(r["N0"]/1e9), r["dense_loss"]])-base["center"])/base["scale"]
        family = np.array([float(r["family"] == f) for f in base["families"]])-base["family_frequencies"]
        extra = (np.array([r["descriptor_features"][k] for k in fit["descriptor_keys"]])
                 -fit["descriptor_center"])/fit["descriptor_scale"]
        return np.r_[1., basic, family, extra]
    x = np.array([design(r) for r in train])
    penalty = np.eye(x.shape[1]); penalty[0, 0] = 0
    expected = design(target) @ np.linalg.solve(x.T @ x + penalty, x.T @ labels)
    assert v32.predict_mapping(fit, target) == pytest.approx(expected, abs=1e-10)


def test_outer_target_outcomes_and_descriptors_cannot_change_source_fit():
    rows = synthetic_rows()
    before = v32.evaluate_lomo(rows, n_boot=20)
    changed = copy.deepcopy(rows)
    changed[0]["deltas"] = {str(d): 999999.*d for d in v28.PRUNE_DEV}
    changed[0]["descriptor_features"][v32.WEIGHT_KEYS[-1]] = 999999.
    after = v32.evaluate_lomo(changed, n_boot=20)
    assert before["folds"][0] == after["folds"][0]
    assert before["records"][0]["predictions"] == after["records"][0]["predictions"]
    assert before["records"][0]["observed"] != after["records"][0]["observed"]
    for fold in after["folds"]:
        assert fold["held_out"] not in fold["train_models"]
    target = copy.deepcopy(rows[-1])
    fit = before["folds"][-1]["fits"]["V3"]
    value = v32.predict_mapping(fit, target)
    target.update(observed=1e30, amplitude=1e30, deltas={"0.7": 1e30})
    target["descriptor_features"]["own_amplitude"] = 1e30
    assert v32.predict_mapping(fit, target) == value


def test_unknown_family_and_no_descriptor_fallback():
    rows = synthetic_rows()
    for row in rows:
        row["descriptor_features"] = {k: 1. for k in row["descriptor_features"]}
    labels = [r["deltas"]["0.7"] for r in rows[:-1]]
    fit = v32.fit_mapping(rows[:-1], labels, "V3")
    target = {**rows[-1], "family": "unseen"}
    assert fit["descriptor_keys"] == []
    assert v32.predict_mapping(fit, target) == v28.predict_mapping(fit["base"], target)
    with pytest.raises(ValueError, match="leaked"):
        v32.predict_mapping(fit, rows[0])


def test_paired_bootstrap_gains_and_exact_zero_sign():
    records = [{"model": f"m{i}", "observed": y,
                "predictions": {"V1": y+1., "V2": y, "zero": 0., "population_mean": .2}}
               for i, y in enumerate([-.2, .3, 0.])]
    metrics = v32.summarize(records, 100)
    assert metrics["V2"]["mae"] == 0
    assert metrics["V2"]["improvement"]["V1"] == {"mae_gain": 1., "ci95": [1., 1.]}
    assert metrics["V2"]["sign_agreement"] == 1
    assert metrics["zero"]["sign_agreement"] == pytest.approx(1/3)
    assert metrics["V1"]["negative_amplitude_sign_agreement"] == 0


def feature_json(row):
    return {"schema_version": 1, "model": row["model"], "hf_id": f"test/{row['model']}", "dtype": "bf16",
            "N0": row["N0"], "family": row["family"],
            "dense_L_c": {c: row["dense_loss"] for c in v32.audit.CAPABILITIES},
            "weights": {"features": {k: row["descriptor_features"][k] for k in v32.WEIGHT_KEYS},
                        "protocol": {"synthetic": True}},
            "activations": {"features": {k: row["descriptor_features"][k] for k in v32.ACTIVATION_KEYS},
                            "protocol": {"synthetic": True, "requested_forwards": 12, "max_length": 512},
                            "cost": {"extra_dense_forwards": 12, "input_tokens": 36, "seconds": .01}},
            "cost": {"weights": {"extra_dense_forwards": 0, "weight_bytes": 48}}}


def synthetic_json_panel(tmp_path):
    metadata = {"models": {}, "qwen3_8b": {"N0": 6945767424}}
    for row in synthetic_rows():
        features = feature_json(row)
        metadata["models"][row["model"]] = {k: features[k] for k in ("N0", "family", "hf_id")}
        path = tmp_path / "results/v6-capability-geometry" / row["model"] / "prune_losses.json"
        losses = {"1.0": features["dense_L_c"], **{str(d): {c: row["dense_loss"]+row["deltas"][str(d)]
                  for c in v32.audit.CAPABILITIES} for d in v28.PRUNE_DEV}}
        v32.write_json(path, losses)
        v32.write_json(tmp_path / "features" / row["model"] / "features.json", features)
    return metadata


def test_predict_from_synthetic_feature_json_is_cpu_only_and_preserves_inputs(tmp_path, monkeypatch):
    metadata = synthetic_json_panel(tmp_path)
    monkeypatch.setattr(v32, "geometry", lambda: pytest.fail("CPU prediction imported GPU extraction"))
    originals = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    panels, target, costs, _paths = v32.load_panel(metadata, tmp_path / "features", v32.VERSIONS, tmp_path)
    result = v32.evaluate_lomo(panels["qa"], n_boot=20)
    assert len(result["folds"]) == 7
    assert not target
    assert costs["fixture-0"]["V2"]["extra_dense_forwards"] == 0
    assert costs["fixture-0"]["V3"]["extra_dense_forwards"] == 12
    assert all(p.read_bytes() == data for p, data in originals.items())
    feature_path = tmp_path / "features/fixture-0/features.json"
    changed = json.loads(feature_path.read_text())
    changed["activations"] = None
    v32.write_json(feature_path, changed)
    with pytest.raises(ValueError, match="Missing activations"):
        v32.load_panel(metadata, tmp_path / "features", v32.VERSIONS, tmp_path)
    v32.load_panel(metadata, tmp_path / "features", ("V1", "V2"), tmp_path)


@pytest.mark.parametrize("change, message", [
    (lambda p: p.update(schema_version=999), "schema/model"),
    (lambda p: p.update(N0=123), "metadata"),
    (lambda p: p["dense_L_c"].update(qa=999.), "dense anchor"),
    (lambda p: p["weights"]["features"].update(own_amplitude=999.), "feature schema"),
    (lambda p: p["activations"]["protocol"].update(requested_forwards=24), "protocols differ"),
])
def test_descriptor_schema_identity_and_protocol_validation(tmp_path, change, message):
    metadata = synthetic_json_panel(tmp_path)
    path = tmp_path / "features/fixture-0/features.json"
    payload = json.loads(path.read_text()); change(payload)
    v32.write_json(path, payload)
    with pytest.raises(ValueError, match=message):
        v32.load_panel(metadata, tmp_path / "features", v32.VERSIONS, tmp_path)


def test_weight_loader_never_forwards_and_rejects_bad_checkpoint(monkeypatch):
    calls = []
    class Loader:
        @staticmethod
        def from_pretrained(hf_id, **kwargs):
            calls.append(kwargs)
            model = TinyLM()
            model.forward = lambda **_kwargs: pytest.fail("Weight loading performed a dense forward")
            return model, {"missing_keys": []}
    fake = ModuleType("transformers")
    fake.AutoConfig = SimpleNamespace(from_pretrained=lambda _: SimpleNamespace(tie_word_embeddings=False))
    fake.AutoModelForCausalLM = Loader
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert isinstance(v32.load_dense_weights("test/model", torch.float32), TinyLM)
    assert calls[0]["output_loading_info"] is True
    monkeypatch.setattr(Loader, "from_pretrained", lambda *_a, **_k: (TinyLM(), {"missing_keys": ["layers.0.weight"]}))
    with pytest.raises(RuntimeError, match="Checkpoint-integrity"):
        v32.load_dense_weights("test/model", torch.float32)


def test_extract_guard_and_idempotent_cached_features(tmp_path, monkeypatch):
    row = synthetic_rows()[0]
    row["model"] = "Qwen3-8B"
    payload = feature_json(row)
    payload["hf_id"] = "Qwen/Qwen3-8B"
    path = tmp_path / row["model"] / "features.json"
    v32.write_json(path, payload)
    original = path.read_bytes()
    args = v32.parser().parse_args(["extract", "--model", "Qwen3-8B", "--output-dir", str(tmp_path), "--with-activations"])
    monkeypatch.delenv("SDL_ALLOW_RESTRICTED", raising=False)
    with pytest.raises(RuntimeError, match="prohibits"):
        v32.extract(args)
    monkeypatch.setenv("SDL_ALLOW_RESTRICTED", "1")
    monkeypatch.setattr(v32, "load_dense_weights", lambda *_a: pytest.fail("Idempotent extract loaded a model"))
    assert v32.extract(args) == payload
    assert path.read_bytes() == original
    args.activation_forwards = 24
    with pytest.raises(ValueError, match="budget differs"):
        v32.extract(args)


def test_fresh_extract_and_activation_upgrade_with_mocked_gpu_loader(tmp_path, monkeypatch):
    """Exercise the GPU driver using only a tiny module that remains on CPU."""
    loads, forwards = [], []
    class CPUFixture(TinyLM):
        config = SimpleNamespace(_commit_hash="fixture-revision")
        def to(self, device):
            assert device == "cuda:0"
            return self  # Deliberately never touch CUDA in this fixture.
        def forward(self, input_ids, **kwargs):
            assert input_ids.device.type == "cpu"
            forwards.append(1)
            return super().forward(input_ids, **kwargs)
    def loader(hf_id, dtype):
        loads.append((hf_id, dtype))
        return CPUFixture()
    monkeypatch.setitem(v32.registry.MODEL_REGISTRY, "fixture", {"hf_id": "test/fixture", "family": "fixture"})
    monkeypatch.setattr(v32, "load_dense_weights", loader)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    prune = tmp_path / "prune_losses.json"
    # Compressed cells intentionally unusable: extraction only reads density 1.0.
    prune.write_text(json.dumps({"1.0": {"math": 1., "code": 2., "qa": 3.}, "0.9": "not an input"}))
    original_prune = prune.read_bytes()
    args = v32.parser().parse_args(["extract", "--model", "fixture", "--output-dir", str(tmp_path / "out"),
                                   "--prune-losses", str(prune)])
    first = v32.extract(args)
    weights = copy.deepcopy(first["weights"])
    costs = copy.deepcopy(first["cost"])
    assert len(loads) == 1 and not forwards
    assert first["activations"] is None
    assert first["N0"] == 8
    assert first["dense_L_c"] == {"math": 1., "code": 2., "qa": 3.}
    assert first["cost"]["weights"]["extra_dense_forwards"] == 0
    v32.extract(args)
    assert len(loads) == 1  # Cached weight-only extraction skips loading.
    fake = ModuleType("transformers")
    fake.AutoTokenizer = SimpleNamespace(from_pretrained=lambda _: object())
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(v32, "shared_probe_batches", lambda *_args: (
        [{"input_ids": torch.tensor([[0, 1]]), "use_cache": False} for _ in range(3)],
        {"requested_forwards": 3, "max_length": 512}))
    monkeypatch.setattr(v32, "weight_descriptors", lambda *_args: pytest.fail("Upgrade recomputed weights"))
    args.with_activations = True
    args.activation_forwards = 3
    upgraded = v32.extract(args)
    assert len(loads) == 2 and len(forwards) == 3
    assert upgraded["weights"] == weights and upgraded["cost"] == costs
    assert upgraded["activations"]["cost"]["extra_dense_forwards"] == 3
    assert upgraded["activations"]["cost"]["reload_seconds"] > 0
    destination = args.output_dir / "fixture/features.json"
    saved = destination.read_bytes()
    assert "not an input" not in saved.decode()
    v32.extract(args)
    assert destination.read_bytes() == saved
    assert len(loads) == 2 and len(forwards) == 3
    assert prune.read_bytes() == original_prune


def test_target_qwen_never_enters_development_folds(tmp_path):
    metadata = synthetic_json_panel(tmp_path)
    source = synthetic_rows()[0]
    source.update(model=v32.TARGET, family="qwen3", N0=metadata["qwen3_8b"]["N0"])
    payload = feature_json(source)
    payload["hf_id"] = v32.registry.MODEL_REGISTRY[v32.TARGET]["hf_id"]
    v32.write_json(tmp_path / "features" / v32.TARGET / "features.json", payload)
    losses = {"1.0": payload["dense_L_c"], **{str(d): {c: source["dense_loss"]+source["deltas"][str(d)]
              for c in v32.audit.CAPABILITIES} for d in v28.PRUNE_DEV}}
    path = tmp_path / "results/v6-capability-geometry/Qwen--Qwen3-8B/prune_losses.json"
    v32.write_json(path, losses)
    panels, targets, _costs, _paths = v32.load_panel(metadata, tmp_path / "features", v32.VERSIONS, tmp_path)
    assert all(r["model"] != v32.TARGET for rows in panels.values() for r in rows)
    record, fold = v32.evaluate_fold(panels["qa"], targets["qa"], v32.VERSIONS)
    assert record["observed"] < 0 and fold["held_out"] == v32.TARGET
    assert v32.TARGET not in fold["train_models"]
    assert len(fold["train_models"]) == len(metadata["models"])


def test_predict_driver_writes_report_and_costs_from_synthetic_json(tmp_path, monkeypatch):
    metadata = synthetic_json_panel(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    v32.write_json(metadata_path, metadata)
    # Generate the report's real 13-command manifest before substituting the test whitelist.
    commands = v32.extraction_commands()
    monkeypatch.setattr(v32, "extraction_commands", lambda: commands)
    monkeypatch.setattr(v28, "METADATA", metadata_path)
    load = v32.load_panel
    monkeypatch.setattr(v32, "load_panel", lambda m, d, v, **kwargs: load(m, d, v, tmp_path, **kwargs))
    args = v32.parser().parse_args(["predict", "--descriptor-dir", str(tmp_path / "features"),
                                   "--output-dir", str(tmp_path / "prediction"),
                                   "--report", str(tmp_path / "report.md"), "--n-boot", "20"])
    result = v32.predict(args)
    saved = json.loads((args.output_dir / "summary.json").read_text())
    assert saved == result
    assert set(saved["development"]) == set(v32.audit.CAPABILITIES)
    assert set(saved["development"]["qa"]["metrics"]) == {"V1", "V2", "V3", "zero", "population_mean"}
    assert "Computed versions: V1, V2, V3" in args.report.read_text()
    assert args.report.read_text() == (args.output_dir / "report.md").read_text()


def test_extraction_commands_cover_exact_dev_whitelist_and_rai_target():
    commands = v32.extraction_commands()
    tags = set(v32.audit.read_json(v28.METADATA)["models"]) | {"Qwen3-8B"}
    assert len(commands) == len(tags) == 13
    assert {c.split("--model ")[1].split()[0] for c in commands} == tags
    assert sum(c.startswith("SDL_ALLOW_RESTRICTED=1 ") for c in commands) == 4
    assert all("--with-activations" in c and "--activation-forwards 12" in c for c in commands)


def test_cli_help_does_not_import_torch_or_transformers():
    script = "from analysis import v32_prune_descriptors; import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True, cwd=v32.ROOT, capture_output=True, text=True)
    result = subprocess.run([sys.executable, "analysis/v32_prune_descriptors.py", "--help"],
                            check=True, cwd=v32.ROOT, capture_output=True, text=True)
    assert "extract" in result.stdout and "predict" in result.stdout
