"""Offline CPU checks for V88; all fixtures are synthetic, no model downloads."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from analysis import v88_displacement as v88
from analysis.tiny_test_model import TinyModel, TinyTokenizer, PROBES


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU test must not access CUDA")

    for name in ("_lazy_init", "init", "get_device_name", "is_available", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    yield
    torch.set_num_threads(previous)


class DisplacedModel(torch.nn.Module):
    """Inject an explicit final-logit displacement into a fake compressed model."""
    def __init__(self, z, r=None):
        super().__init__()
        self.register_buffer("z", z)
        self.register_buffer("r", torch.zeros_like(z) if r is None else r)
        self.calls = 0

    def forward(self, input_ids, use_cache=False, logits_to_keep=1):
        assert input_ids.device.type == "cpu" and not use_cache and logits_to_keep == 1
        self.calls += 1
        return SimpleNamespace(logits=(self.z + self.r).reshape(1, 1, -1))


def test_injected_displacement_closed_form_terms_to_1e9():
    # Dyadic uniform fixture: the independent real-number closed forms are
    # exactly representable in fp32, making a 1e-9 absolute assertion meaningful.
    z = torch.full((32,), 2.0)
    r = torch.tensor([i / 64 for i in range(-16, 16)])
    dense, compressed = DisplacedModel(z), DisplacedModel(z, r)
    sample = {"prompt": "ab", "completion": "c"}
    measured = v88.measure_pair(dense, compressed, TinyTokenizer(), {"math": [sample]})
    row = measured["per_sample"][0]
    y = TinyTokenizer().encode("c")[0]
    mean = sum(r.tolist()) / 32
    first = mean - float(r[y])
    second = 0.5 * (sum(value * value for value in r.tolist()) / 32 - mean * mean)
    assert row["first_order"] == pytest.approx(first, abs=1e-9, rel=0)
    assert row["second_order"] == pytest.approx(second, abs=1e-9, rel=0)
    assert row["second_order_prediction"] == pytest.approx(first + second, abs=1e-9, rel=0)
    assert dense.calls == compressed.calls == 1


def test_nonuniform_probability_closed_form_and_small_displacement_remainder():
    z = torch.tensor([-1.0, 0.2, 0.6, 1.3])
    r = torch.tensor([0.012, -0.008, 0.004, 0.003])
    row = v88.reduce_token(z, z + r, 1)
    p = z.double().softmax(-1)
    actual_r = (z + r).double() - z.double()
    first = float((p * actual_r).sum() - actual_r[1])
    second = 0.5 * float((p * actual_r.square()).sum() - (p * actual_r).sum().square())
    assert row["first_order"] == pytest.approx(first, abs=2e-9, rel=0)
    assert row["second_order"] == pytest.approx(second, abs=1e-9, rel=0)
    assert abs(row["measured_delta"] - row["second_order_prediction"]) < 4e-7


def test_pure_shrinkage_projection_and_missing_quadratic():
    z = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    eps = 0.125
    row = v88.reduce_token(z, (1 - eps) * z, 3)
    assert row["eps_hat"] == pytest.approx(eps, abs=1e-6, rel=0)
    assert row["sigma2_hat"] == pytest.approx(0.0, abs=1e-12)
    p = z.softmax(-1)
    var_z = (p * z.square()).sum() - (p * z).sum().square()
    expected = float(torch.tensor(row["shrinkage_prediction"]) + 0.5 * eps**2 * var_z)
    assert row["second_order_prediction"] == pytest.approx(expected, abs=1e-9, rel=0)
    assert row["second_order_prediction"] != pytest.approx(row["shrinkage_prediction"], abs=1e-9)
    assert row["shrinkage_prediction"] == pytest.approx(eps * row["B"], abs=1e-9, rel=0)


def test_pure_shrinkage_equality_when_quadratic_is_zero():
    # Equality requested in (b) holds in the constant-logit case only (or eps=0).
    z = torch.full((32,), 2.0)
    row = v88.reduce_token(z, 0.875 * z, 1)
    assert row["eps_hat"] == pytest.approx(0.125, abs=1e-6)
    assert row["sigma2_hat"] == 0
    assert row["shrinkage_prediction"] == pytest.approx(row["second_order_prediction"], abs=1e-9, rel=0)


def test_isotropic_noise_variance_and_zero_projection_with_sampling_error():
    generator = torch.Generator(device="cpu").manual_seed(88)
    z = torch.tensor([-0.1, 0.1]).repeat(4096)
    eps, variances = [], []
    for _ in range(64):
        noise = 0.1 * torch.randn(z.shape, generator=generator)
        row = v88.reduce_token(z, z + noise, 0)
        eps.append(row["eps_hat"])
        variances.append(row["sigma2_hat"])
    # Near uniform p and large vocab make V~1; projection removes one direction.
    assert sum(variances) / len(variances) == pytest.approx(0.01, rel=0.015)
    assert abs(sum(eps) / len(eps)) < 0.005  # >3 standard errors of the mean


def test_two_sample_pooling_matches_hand_computed_not_sample_mean():
    z = torch.tensor([-0.5, 0.0, 1.0, 1.5] * 8)
    r = torch.tensor([0.0, 0.25, -0.125, 0.5] * 8)
    probes = {"qa": [{"prompt": "a", "completion": "c"},
                     {"prompt": "b", "completion": "deee"}]}
    result = v88.measure_pair(DisplacedModel(z), DisplacedModel(z, r), TinyTokenizer(), probes)
    tokens = TinyTokenizer().encode("cdeee")
    dense_ce = [float(torch.logsumexp(z, 0) - z[y]) for y in tokens]
    compressed_ce = [float(torch.logsumexp(z + r, 0) - (z + r)[y]) for y in tokens]
    pooled = result["per_capability"]["qa"]
    assert pooled["scored_token_count"] == 5
    assert pooled["dense_ce"] == pytest.approx(sum(dense_ce) / 5, abs=1e-9)
    assert pooled["compressed_ce"] == pytest.approx(sum(compressed_ce) / 5, abs=1e-9)
    assert pooled["measured_delta"] == pytest.approx(sum(b-a for a, b in zip(dense_ce, compressed_ce)) / 5, abs=1e-9)
    assert pooled["dense_ce"] != pytest.approx(sum(row["dense_ce"] for row in result["per_sample"]) / 2)
    for key in v88.METRICS:
        expected = sum(row["sums"][key] for row in result["per_sample"]) / 5
        assert pooled[key] == expected


@pytest.mark.parametrize("gemma,no_bos,empty_prompt", [(False, False, False), (True, True, False), (False, True, True)])
@pytest.mark.parametrize("completion", ["", "c", "abcdefghijk"])
def test_mask_is_frozen_scorer_mask_including_truncation(gemma, no_bos, empty_prompt, completion):
    tok = TinyTokenizer()
    if gemma:
        tok.name_or_path = "gemma-test"
    if no_bos:
        encode = tok.encode
        tok.encode = lambda text, add_special_tokens=False: encode(text, False)
    sample = {"prompt": "" if empty_prompt else "abcdefghijk", "completion": completion}
    ids, count = v88.scored_inputs(tok, sample, max_len=8)
    if ids.shape[1] < 2:
        assert count == 0
        return
    capture = v88.descriptor_bv._CaptureFinal(TinyModel())
    loss, frozen_count = v88.v6.completion_loss(capture, tok, sample["prompt"], completion, "cpu", max_len=8)
    assert count == frozen_count and torch.equal(ids, capture.ids)
    result = v88.measure_pair(TinyModel(), TinyModel(), tok, {"qa": [sample]}, max_len=8)
    row = result["per_sample"][0]
    assert row["scored_token_count"] == count
    assert row["sums"]["dense_ce"] == pytest.approx(float(loss.detach()), abs=3e-6)
    if count == 0:
        assert all(row[key] is None for key in v88.METRICS)


class NoTokenVocabAllocation(TorchDispatchMode):
    """Observe operation outputs, including native forward allocations, not just reducers."""
    def __init__(self, vocab):
        self.vocab = vocab
        self.observed = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        inputs = [x for x in tree_flatten((args, kwargs or {}))[0] if isinstance(x, torch.Tensor)]
        storages = {x.untyped_storage().data_ptr() for x in inputs}
        if str(func) in ("aten.mm.default", "aten.bmm.default", "aten.addmm.default"):
            assert all(x.dtype == torch.bfloat16 for x in inputs if x.is_floating_point())
        outputs = func(*args, **(kwargs or {}))
        for tensor in tree_flatten(outputs)[0]:
            if isinstance(tensor, torch.Tensor):
                assert tensor.device.type == "cpu"
                # Views, including transposed output-head weights, allocate no
                # storage. Any forbidden parent allocation is caught earlier.
                if (tensor.ndim >= 2 and tensor.shape[-1] == self.vocab
                        and tensor.untyped_storage().data_ptr() not in storages):
                    # A tiny transition weight matrix has [vocab,vocab] shape;
                    # it is model storage, not an activation for scored tokens.
                    if tuple(tensor.shape) != (self.vocab, self.vocab):
                        assert math.prod(tensor.shape[:-1]) <= 1, (str(func), tensor.shape)
                    self.observed += 1
        return outputs


def tiny_loader(model_id, dtype, revision):
    return TinyModel(dtype), TinyTokenizer()


def test_full_run_never_allocates_token_by_vocab_and_reduces_float32(tmp_path, monkeypatch):
    seen = []
    original = torch.softmax

    def checked_softmax(x, *args, **kwargs):
        assert x.dtype == torch.float32 and kwargs["dtype"] == torch.float32
        assert x.numel() == 17
        seen.append(x.dtype)
        return original(x, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", checked_softmax)
    guard = NoTokenVocabAllocation(17)
    with guard:
        paths = v88.run_state("tiny@88", ["prune_d0.8", "b3_g4"], PROBES,
                              model_loader=tiny_loader, output_dir=tmp_path)
    assert guard.observed > 0 and seen and len(paths) == 2


def test_schema_sample_lengths_and_reused_bv(tmp_path, monkeypatch):
    original = v88.descriptor_bv.reduce_final_logits
    calls = []

    def checked(logits, targets, chunk_tokens):
        calls.append((tuple(logits.shape), chunk_tokens))
        return original(logits, targets, chunk_tokens)

    monkeypatch.setattr(v88.descriptor_bv, "reduce_final_logits", checked)
    paths = v88.run_state("pythia-410m@step16000", ["b4_g0"], PROBES,
                          model_loader=tiny_loader, output_dir=tmp_path)
    payload = json.loads(paths[0].read_text())
    assert paths[0].relative_to(tmp_path) == Path("pythia-410m--step16000/b4_g0.json")
    assert payload["model_id"] == "EleutherAI/pythia-410m" and payload["revision"] == "step16000"
    assert payload["schema_version"] == 1 and payload["experiment"] == "v88-displacement"
    assert payload["dtype"] == "bfloat16" and payload["device"] == "cpu" and payload["device_name"]
    assert payload["probe_sha256"] == v88.digest(PROBES)
    assert len(payload["weight_sha256"]) == 64
    assert payload["scored_token_counts"] == {"math": 8, "code": 2, "qa": 3}
    assert len(payload["per_sample"]) == 4
    assert len(calls) == 13 and all(shape == (1, 17) and chunk == 1 for shape, chunk in calls)
    assert payload["provenance"]["code_sha256"]["v6_capability_geometry.py"]
    for cap, samples in PROBES.items():
        rows = [row for row in payload["per_sample"] if row["capability"] == cap]
        assert len(rows) == len(samples)
        assert all(set(v88.METRICS) <= row.keys() and set(v88.METRICS) == row["sums"].keys() for row in rows)
        assert [row["sample_index"] for row in rows] == list(range(len(samples)))
    before = paths[0].read_bytes()
    with pytest.raises(FileExistsError):
        v88.run_state("pythia-410m@step16000", ["b4_g0"], PROBES,
                      model_loader=lambda *a: pytest.fail("must reject before loading"), output_dir=tmp_path)
    assert paths[0].read_bytes() == before


@pytest.mark.parametrize("config", ["prune_d0.8", "prune_d1", "b3_g4", "b4_g0"])
def test_compression_matches_existing_functions_and_preserves_dense(config, monkeypatch):
    dense = TinyModel(torch.bfloat16)
    original = dense.weight.detach().clone()
    compressed, saved = v88.compress_copy(dense, config)
    expected = copy.deepcopy(dense)
    if saved["kind"] == "prune":
        threshold = v88.v6.apply_global_magnitude_pruning(expected, saved["density"], seed=0)
        assert saved["threshold"] == (threshold if math.isfinite(threshold) else None)
        monkeypatch.setattr(v88.v6, "_sample_abs_weights", lambda *a, **kw: pytest.fail("must reuse stored threshold"))
        replay, replay_config = v88.compress_copy(dense, saved)
        assert torch.equal(replay.weight, compressed.weight)
        assert replay_config == saved
    else:
        with torch.no_grad():
            expected.weight.copy_(v88.fake_quantize_grouped(original, saved["bits"], saved["group_size"]))
    assert torch.equal(expected.weight, compressed.weight)
    assert torch.equal(dense.weight, original)
    assert dense.weight.data_ptr() != compressed.weight.data_ptr()
    assert compressed.weight.dtype == dense.weight.dtype == torch.bfloat16


class HeadModel(torch.nn.Module):
    """Context-dependent toy LM exercises the output-head hook and final softcap."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 4)
        self.lm_head = torch.nn.Linear(4, 17, bias=False)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(68).reshape(17, 4) / 100)
            self.lm_head.weight.copy_(torch.arange(68).reshape(17, 4) / 200 - 0.2)

    def forward(self, input_ids, use_cache=False):
        assert not use_cache
        hidden = self.embedding(input_ids).cumsum(dim=1)
        return SimpleNamespace(logits=1.7 * torch.tanh(self.lm_head(hidden) / 1.7))


def test_head_hook_preserves_final_softcap_context_and_is_removed():
    dense = HeadModel().to(dtype=torch.bfloat16).eval()
    sample = PROBES["math"][1]
    ids, count = v88.scored_inputs(TinyTokenizer(), sample)
    with torch.no_grad():
        reference = dense(ids).logits[0, -count-1:-1].float()
        hook = v88.TokenForward(dense)
        try:
            with NoTokenVocabAllocation(17):
                for i, pos in enumerate(range(ids.shape[1] - count, ids.shape[1])):
                    z = hook(ids[:, :pos])
                    assert torch.equal(z.float(), reference[i])
                    assert float(z.abs().max()) <= 1.7
        finally:
            hook.close()
    assert not dense.lm_head._forward_pre_hooks


def test_native_transformer_one_token_api_matches_full_cpu_forward():
    # Random config only: never from_pretrained or network access.
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(88)
        model = GPTNeoXForCausalLM(GPTNeoXConfig(
            vocab_size=31, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, max_position_embeddings=32, rotary_pct=0.5,
            attention_dropout=0, hidden_dropout=0, _attn_implementation="eager",
        )).to(dtype=torch.bfloat16).eval()
    ids = torch.tensor([[0, 2, 3, 4, 5]])
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        forward = v88.TokenForward(model)
        try:
            with NoTokenVocabAllocation(31):
                for pos in range(1, ids.shape[1]+1):
                    assert torch.allclose(forward(ids[:, :pos]).float(), full[0, pos-1].float(), atol=0.001, rtol=0.01)
        finally:
            forward.close()


def test_unsupported_head_fails_before_forward():
    class Unsupported(torch.nn.Module):
        def forward(self, input_ids, use_cache=False):
            pytest.fail("unsupported model must be rejected before allocating logits")
    with pytest.raises(ValueError, match="output head"):
        v88.TokenForward(Unsupported())


def test_rotary_outer_product_is_elementwise_and_other_fp32_matmul_rejected():
    a = torch.tensor([[[1.0], [0.1]]])
    b = torch.tensor([[[0.0, 1.0, 2.0]]])
    with v88.FP32MatmulGuard():
        assert torch.equal(torch.bmm(a, b), a * b)
        with pytest.raises(RuntimeError, match="forbids fp32 matrix"):
            torch.mm(torch.ones(2, 2), torch.ones(2, 3))


def test_allocation_guard_detects_full_native_logit_buffer():
    with pytest.raises(AssertionError):
        with NoTokenVocabAllocation(17):
            torch.zeros(1, 7, 17)


def test_zero_displacement_zero_logits_and_top16_definition():
    z = torch.zeros(32)
    zero = v88.reduce_token(z, z, 0)
    assert all(zero[key] == 0 for key in ("measured_delta", "eps_hat", "sigma2_hat", "top16_r2_share"))
    r = torch.arange(32, dtype=torch.float32) / 32
    row = v88.reduce_token(z, r, 0)
    expected = float(r[-16:].square().sum() / r.square().sum())
    assert row["top16_r2_share"] == pytest.approx(expected, abs=2e-7)
    # Rank by |r|, not p*r²: make the largest |r| coordinate very unlikely.
    z[-1] = -20
    r[-1] = 5
    row = v88.reduce_token(z, z+r, 0)
    p = z.softmax(-1)
    expected = float((p[-16:] * r[-16:].square()).sum() / (p * r.square()).sum())
    assert row["top16_r2_share"] == pytest.approx(expected, abs=2e-7)


def test_generated_probes_use_frozen_seed_and_odd_half(tmp_path, monkeypatch):
    calls = []
    probes = {cap: [dict(row, sample_id=f"{i}") for i, row in enumerate(rows * 2)] for cap, rows in PROBES.items()}

    def build(n, seed):
        calls.append((n, seed))
        return probes

    monkeypatch.setattr(v88.v6, "build_probes", build)
    paths = v88.run_state("tiny", ["b3_g4"], n_probe=6, output_dir=tmp_path, model_loader=tiny_loader)
    payload = json.loads(paths[0].read_text())
    assert calls == [(6, v88.PROBE_SEED)] and v88.PROBE_SEED == 0
    assert payload["probe_sha256"] == v88.digest({cap: rows[1::2] for cap, rows in probes.items()})


def summary_fixture(delta, exact, shrink, config="b3_g4", state="pythia-410m@step16000"):
    return {
        "experiment": "v88-displacement", "schema_version": 1,
        "state": state, "state_tag": "pythia-410m--step16000",
        "model_id": "EleutherAI/pythia-410m", "revision": "step16000",
        "config": v88.normalize_config(config), "dtype": "bfloat16", "device": "cpu",
        "probe_sha256": "probe-fixture",
        "per_capability": {"math": {"scored_token_count": 2, "measured_delta": delta,
            "second_order_prediction": exact, "shrinkage_prediction": shrink,
            "top16_r2_share": abs(delta), "top16_r2_share_nonzero_mean": abs(delta),
            "top16_r2_mass_share": abs(delta)}},
    }


def test_summary_regimes_errors_signs_zero_damage_and_frozen_read_only(tmp_path):
    output = tmp_path / "v88"
    fixtures = [(0, 0, 0.01), (-0.05, 0.05, -0.1), (0.1, 0.11, -0.2),
                (0.5, 0.4, 0.7), (-0.6, -0.5, 0.4)]
    for i, (delta, exact, shrink) in enumerate(fixtures):
        v88.write_json(output / f"state{i}" / "b3_g4.json", summary_fixture(delta, exact, shrink))
    frozen_root = tmp_path / "frozen"
    frozen_path = frozen_root / "v54-quant-group/pythia-410m--step16000/quant_group_losses.json"
    v88.write_json(frozen_path, {"_meta": {"hf_id": "EleutherAI/pythia-410m", "revision": "step16000",
        "probe_sha256": "different-probes", "model_dtype": "bf16"}, "dense": {"math": 2}, "b3_g4": {"math": 2.1}})
    before = frozen_path.read_bytes()
    summary = v88.summarize(output, frozen_root=frozen_root)
    assert [summary["regimes"][name]["count"] for name in ("mild", "moderate", "severe")] == [2, 2, 1]
    mild = summary["regimes"]["mild"]
    assert mild["second_order_prediction"]["mae"] == pytest.approx(0.05)
    assert mild["second_order_prediction"]["median_relative_error"] == pytest.approx(2)
    assert mild["second_order_prediction"]["relative_error_count"] == 1
    assert mild["second_order_prediction"]["zero_damage_count"] == 1
    assert mild["second_order_prediction"]["sign_agreement"] == 0.5
    assert mild["shrinkage_prediction"]["sign_agreement"] == 0.5
    assert mild["isotropy"]["top16_r2_share"]["mean"] == pytest.approx(0.025)
    assert len(summary["reproduction_checks"]) == 5
    for row in summary["reproduction_checks"]:
        assert row["frozen_delta"] == pytest.approx(0.1)
        assert row["difference"] == row["recomputed_delta"] - row["frozen_delta"]
        assert row["probe_sha256_match"] is False
    assert frozen_path.read_bytes() == before
    markdown = (output / "summary.md").read_text()
    assert "Median relative error" in markdown and "Difference" in markdown
    assert json.loads((output / "summary.json").read_text())["regimes"] == summary["regimes"]


def test_pruning_frozen_match_and_revision_mismatch(tmp_path):
    payload = summary_fixture(0.1, 0.11, 0.12, "prune_d0.9")
    path = tmp_path / "v6-capability-geometry/pythia-410m--step16000/prune_losses.json"
    v88.write_json(path, {"1.0": {"math": 2.0}, "0.9": {"math": 2.08}})
    rows = v88.frozen_reproductions(payload, tmp_path)
    assert len(rows) == 1 and rows[0]["difference"] == pytest.approx(0.02)
    assert rows[0]["probe_sha256_match"] is None
    payload["revision"], payload["state_tag"] = "step64000", "pythia-410m--step64000"
    assert v88.frozen_reproductions(payload, tmp_path) == []


def test_empty_summary_and_selftest_exclusion(tmp_path):
    payload = summary_fixture(0.1, 0.2, 0.3)
    payload["selftest"] = True
    v88.write_json(tmp_path / "tiny" / "b3_g4.json", payload)
    summary = v88.summarize(tmp_path, frozen_root=tmp_path / "absent")
    assert not summary["measurements"]
    assert summary["regimes"]["mild"]["second_order_prediction"]["mae"] is None


@pytest.mark.parametrize("config", ["p0", "p1.2", "b1_g4", "b3_g-1", {"kind": "grouped_rtn", "bits": 3, "group_size": 4, "mode": "asymmetric"}])
def test_invalid_configs(config):
    with pytest.raises(ValueError):
        v88.normalize_config(config)


def test_cli_pilot_full_and_separate_summary(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(v88, "run_state", lambda *args, **kwargs: calls.append((args, kwargs)))
    v88.main(["--pilot", "--states", "tiny@88", "--output-dir", str(tmp_path)])
    assert len(calls) == 1 and len(calls[0][0][1]) == 2
    calls.clear()
    v88.main(["--full", "--states", "tiny@87", "tiny@88", "--configs", "p0.9,b4_g0"])
    assert len(calls) == 2
    with pytest.raises(SystemExit):
        v88.main(["--pilot", "--states", "tiny", "--configs", "p0.9"])
    monkeypatch.setattr(v88, "run_state", lambda *a, **kw: pytest.fail("summary must not load or measure"))
    v88.main(["--summarize", "--output-dir", str(tmp_path)])


def test_cli_selftest_cpu_offline(tmp_path):
    completed = subprocess.run([sys.executable, "-m", "analysis.v88_displacement", "--selftest",
                                "--output-dir", str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "CPU selftest: passed" in completed.stdout
    paths = sorted(tmp_path.glob("*/*.json"))
    assert len(paths) == 2
    assert all(json.loads(path.read_text())["selftest"] for path in paths)


def test_output_and_float32_non_cpu_guards_before_loading(tmp_path):
    with pytest.raises(ValueError, match="results output"):
        v88.run_state("tiny", ["b3_g4"], PROBES, output_dir=v88.ROOT / "results/v6-capability-geometry")
    with pytest.raises(ValueError, match="no fp32 model matmul"):
        v88.run_state("tiny", ["b3_g4"], PROBES, device="cuda:0", dtype="float32", output_dir=tmp_path)
    with pytest.raises(ValueError, match="Conflicting"):
        v88.run_state("tiny@87", ["b3_g4"], PROBES, revision="88", output_dir=tmp_path)
    target = tmp_path / "original.json"
    target.write_text("frozen sentinel")
    link = tmp_path / "linked.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        v88.write_json(link, {"new": True}, replace=True)
    assert target.read_text() == "frozen sentinel"
