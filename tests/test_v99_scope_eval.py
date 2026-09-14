"""Fast CPU orchestration/scorer tests; run with -B and a scoped --basetemp."""
import copy
import json
import os
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest
import torch

from analysis import v67_musique_qa as v67
from analysis import v71_qa_scope as v71
from analysis import v99_scope_eval as scope


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU-only tests must not call CUDA")
    for name in ("_lazy_init", "is_available", "empty_cache", "manual_seed", "manual_seed_all", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(scope, "OUTPUT_ROOT", tmp_path / "results/v99-scope")

    def make(name="run", seed=41, updates=(0, 2, 10, 11)):
        run = tmp_path / "inputs" / name
        shared = {
            "trajectory_run_id": f"tiny/{name}", "student": "tiny", "resolved_student": "local/tiny",
            "revision": "fixed", "teacher": "gpt-5.6-luna", "recipe": "full", "domains": ["qa"],
            "n_per_domain": 2, "data_seed": seed, "data_sampling_seed": seed, "training_seed": 0,
            "training_mode": "lora", "data_pool_sha256": f"pool-{seed}",
            "unique_data_pool_tokens": 20, "unique_data_pool_examples": 2,
        }
        for update in updates:
            checkpoint = run / "trajectory" / f"update-{update}"  # Deliberately not zero padded.
            payload = {**shared, "updates": update, "processed_tokens": update * 10,
                       "completion_tokens_seen": update * 3,
                       "snapshot_kind": "trajectory" if update else "trajectory_baseline"}
            dump(checkpoint / "eval.json", payload)
            if update:
                dump(checkpoint / "adapter/adapter_config.json",
                     {"peft_type": "LORA", "base_model_name_or_path": "local/tiny"})
                (checkpoint / "adapter/adapter_model.safetensors").write_bytes(f"adapter-{name}-{update}".encode())
        curve = [{"step": step, "epoch": (step + 1) // 2,
                  "tokens": 10, "completion_tokens": 3, "processed_tokens": step * 10,
                  "completion_tokens_seen": step * 3, "unique_data_pool_tokens": 20,
                  "unique_examples_seen": min(step, 2), "unique_data_tokens": min(step, 2) * 10}
                 for step in range(1, max(updates) + 1)]
        dump(run / "train_log.json", {**shared, "training_examples": 2, "loss_curve": curve})
        return run
    return make


@pytest.fixture
def frozen(tmp_path):
    panels = {key: [{"prompt": "Q?", "completion": " A", "answer": "A"} for _ in range(v71.N)]
              for key in scope.DISTRIBUTIONS}
    metadata = {key: {**v71.SETS[key], "indices": list(range(v71.N)), "n": v71.N,
                      "prompt_reference_sha256": v71.sha_json(panels[key])}
                for key in panels}
    path = tmp_path / "inputs/register.json"
    register = {"version": 71, "max_len": v71.MAX_LEN, "batch_size": 1, "n_per_set": v71.N,
                "sets": metadata, "dataset_sources": [], "hf_id": "local/tiny", "revision": "fixed",
                "loss": "sum completion CE / sum completion tokens (native-token nats)",
                "information_condition": v71.INFORMATION}
    dump(path, register)
    protocol = scope.load_protocol(path, list(panels), "cpu", "float32")
    return path, protocol, panels, metadata


@pytest.fixture
def scoring(tmp_path, monkeypatch, frozen):
    _, protocol, panels, metadata = frozen
    calls = {"loads": [], "adapters": [], "completion": []}
    local = tmp_path / "inputs/model"
    dump(local / "config.json", {"model_type": "tiny"})

    class Base:
        update = 0

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

    class Adapter(Base):
        @classmethod
        def from_pretrained(cls, base, path, **kwargs):
            assert kwargs == {"local_files_only": True, "is_trainable": False}
            assert base.update == 0  # Each adapter gets a fresh dense base.
            calls["adapters"].append(path)
            model = cls()
            model.update = int(Path(path).parent.name.split("-")[1])
            model.peft_config = {"default": object()}
            return model

    def load(name, dtype, revision):
        assert name == str(local) and dtype is torch.float32 and revision is None
        calls["loads"].append(name)
        return Base(), types.SimpleNamespace(truncation_side="left")

    def completion_loss(model, tokenizer, prompt, completion, device, max_len):
        assert device == "cpu" and max_len == v67.MAX_LEN
        assert tokenizer.truncation_side == "right"
        assert not torch.is_grad_enabled()
        calls["completion"].append(model.update)
        tokens = 2 if completion == " A" else 1
        return (4.0 + model.update) * tokens, tokens

    v12 = types.ModuleType("analysis.v12_distill")
    v12.load_text_causal_lm = load
    v12.completion_loss = completion_loss
    peft = types.ModuleType("peft")
    peft.PeftModel = Adapter
    monkeypatch.setitem(sys.modules, "analysis.v12_distill", v12)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setattr(scope, "local_model_path", lambda *args: (local, "fixed"))
    monkeypatch.setattr(v67, "local_data_file", Mock(return_value=tmp_path / "inputs/musique.jsonl"))
    monkeypatch.setattr(v71, "prepare_panels", Mock(return_value=(panels, metadata, [])))
    scorer = Mock(wraps=v67.measure_qa)
    monkeypatch.setattr(v67, "measure_qa", scorer)
    return calls, scorer, Adapter


def run_eval(plan, protocol):
    return scope.evaluate(plan, protocol, scope.OUTPUT_ROOT, Path("unused-cache"), None, Path("unused-traces"))


def test_selection_is_numeric_first_last_plus_named(tree):
    run = tree()
    assert [p.name for p in scope.select_checkpoints(run)] == ["update-0", "update-11"]
    assert [p.name for p in scope.select_checkpoints(run, ["update-00000002", "10", "2"])] == [
        "update-0", "update-2", "update-10", "update-11"]
    assert scope.select_checkpoints(run / "trajectory", ["all"]) == scope.select_checkpoints(run, ["2", "10"])
    for selection in (["3"], ["../adapter"], ["update-abc"]):
        with pytest.raises(ValueError):
            scope.select_checkpoints(run, selection)


def test_selection_requires_own_update_zero_and_complete_metadata(tree):
    with pytest.raises(ValueError, match="update-0"):
        scope.select_checkpoints(tree("no-zero", updates=(2, 10)))
    run = tree()
    (run / "trajectory/update-2/eval.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata"):
        scope.select_checkpoints(run)


@pytest.mark.parametrize("missing", ["adapter_config.json", "adapter_model.safetensors", "directory"])
def test_missing_adapter_refused_before_any_model_or_data_load(tree, frozen, scoring, missing):
    run = tree()
    adapter = run / "trajectory/update-11/adapter"
    if missing == "directory":
        adapter.rename(adapter.with_name("absent-adapter"))
    else:
        (adapter / missing).unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete existing adapter"):
        scope.build_plan([run], [], scope.OUTPUT_ROOT)
    assert not scoring[0]["loads"]
    v71.prepare_panels.assert_not_called()
    assert not scope.OUTPUT_ROOT.exists()


def test_pool_completion_denominator_and_checkpoint_identity(tree):
    run = tree()
    plan = scope.build_plan([run], [], scope.OUTPUT_ROOT)
    baseline, final = [item["identity"] for item in plan[0]["checkpoints"]]
    assert baseline["D_U"] == final["D_U_completion"] == 6
    assert final["unique_data_pool_tokens"] == 20
    assert final["actual_supervised_tokens"] == 33 and final["reuse"] == 5.5
    assert final["D_U_source"]["first_complete_epoch_end_update"] == 2
    assert baseline["adapter_path"] is baseline["adapter_sha256"] is None
    assert not baseline["adapter_expected"]
    assert len(final["adapter_sha256"]) == 64
    assert final["pool_size"] == 2 and final["pool_seed"] == 41


@pytest.mark.parametrize("change", ["pool", "tokens", "incomplete_epoch"])
def test_invalid_pool_or_exposure_refused(tree, change):
    run = tree()
    path = run / ("train_log.json" if change == "incomplete_epoch" else "trajectory/update-11/eval.json")
    data = v71.read_json(path)
    if change == "pool":
        data["data_pool_sha256"] = "different"
    elif change == "tokens":
        data["completion_tokens_seen"] += 1
    else:
        data["training_examples"] = 3
    dump(path, data)
    with pytest.raises(ValueError):
        scope.build_plan([run], [], scope.OUTPUT_ROOT)


def test_legacy_pool_uses_and_crosschecks_v67_register(tree, tmp_path):
    run = tree("gpt-5.6-luna_full_2_p2v2_lora_dseed41")
    register_path = tmp_path / "results/v47-p2-register/register.json"
    register = {"pools": {"U2_s41": {"pool_processed_tokens": 20, "D_U_completion": 6}}}
    dump(register_path, register)
    plan = scope.build_plan([run], [], scope.OUTPUT_ROOT, root=tmp_path)
    source = plan[0]["checkpoints"][0]["identity"]["D_U_source"]["pool_register"]
    assert source == {**scope.source_identity(register_path), "pool_key": "U2_s41"}
    register["pools"]["U2_s41"]["D_U_completion"] = 7
    dump(register_path, register)
    with pytest.raises(ValueError, match="Registered D_U differs"):
        scope.build_plan([run], [], scope.OUTPUT_ROOT, root=tmp_path)


def test_imported_panel_builder_and_scorer_write_self_contained_results(tree, frozen, scoring):
    calls, scorer, _ = scoring
    plan = scope.build_plan([tree("a"), tree("b", seed=42)], [], scope.OUTPUT_ROOT)
    result = run_eval(plan, frozen[1])
    assert result == {"scored": 4, "skipped": 0}
    v71.prepare_panels.assert_called_once()
    v67.local_data_file.assert_called_once_with(None)
    assert scope.v67 is v67 and scope.v71 is v71
    assert scorer.call_count == 12  # Real imported V67 loop, 3 sets x 4 states.
    assert len(calls["completion"]) == 12 * v71.N
    assert len(calls["loads"]) == 4 and len(calls["adapters"]) == 2
    for trajectory in plan:
        baseline_path, trained_path = [Path(item["output"]) for item in trajectory["checkpoints"]]
        baseline = v71.read_json(baseline_path)
        trained = v71.read_json(trained_path)
        assert trained["status"] == baseline["status"] == "complete"
        assert trained["adapter_loaded"] is True and baseline["adapter_loaded"] is False
        assert trained["update_0"]["adapter_loaded"] is False
        assert trained["update_0"]["trajectory"] == trained["trajectory"] == trajectory["trajectory"]
        for record in (baseline, trained, trained["update_0"]):
            for key in (*scope.IDENTITY_KEYS, "pool_size", "pool_seed", "actual_supervised_tokens", "D_U",
                        "adapter_path", "adapter_sha256", "adapter_loaded", "eval_source", "D_U_source"):
                assert key in record
            assert set(record["losses"]) == set(scope.DISTRIBUTIONS)
            for key, loss in record["losses"].items():
                assert loss["probe_identity"] == frozen[1]["probes"][key]
                assert loss["n"] == v71.N and loss["tokens"] == 2 * v71.N
        assert trained["update_0"]["actual_supervised_tokens"] == 0
        assert trained["update_0"]["D_U"] == trained["D_U"] == 6
        assert set(trained["delta_from_update_0"].values()) == {11.0}


def test_rerun_skips_all_completed_work_without_loads_or_writes(tree, frozen, scoring, monkeypatch):
    run = tree()
    plan = scope.build_plan([run], [], scope.OUTPUT_ROOT)
    run_eval(plan, frozen[1])
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in scope.OUTPUT_ROOT.rglob("*.json")}
    def forbidden(*args, **kwargs):
        pytest.fail("Completed checkpoints must be skipped")
    monkeypatch.setattr(scope, "score_checkpoint", forbidden)
    monkeypatch.setattr(scope, "prepare_registered_panels", forbidden)
    monkeypatch.setattr(scope, "write_new_json", forbidden)
    plan = scope.build_plan([run], [], scope.OUTPUT_ROOT)
    assert all(item["action"] == "skip_existing" for item in plan[0]["checkpoints"])
    assert run_eval(plan, frozen[1]) == {"scored": 0, "skipped": 2}
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_resume_uses_saved_own_baseline_and_adds_only_missing_checkpoint(tree, frozen, scoring, monkeypatch):
    run = tree()
    original = scope.score_checkpoint
    def interrupted(identity, *args):
        if identity["updates"]:
            raise RuntimeError("simulated interruption")
        return original(identity, *args)
    monkeypatch.setattr(scope, "score_checkpoint", interrupted)
    plan = scope.build_plan([run], [], scope.OUTPUT_ROOT)
    with pytest.raises(RuntimeError, match="interruption"):
        run_eval(plan, frozen[1])
    initial = Path(plan[0]["checkpoints"][0]["output"])
    before = initial.read_bytes()
    monkeypatch.setattr(scope, "score_checkpoint", original)
    assert run_eval(plan, frozen[1]) == {"scored": 1, "skipped": 1}
    assert initial.read_bytes() == before
    plan = scope.build_plan([run], ["2"], scope.OUTPUT_ROOT)
    assert run_eval(plan, frozen[1]) == {"scored": 1, "skipped": 2}
    added = v71.read_json(plan[0]["checkpoints"][1]["output"])
    assert set(added["delta_from_update_0"].values()) == {2.0}


def test_adapter_loader_failure_never_scores_dense_as_trained(tree, frozen, scoring, monkeypatch):
    run = tree()
    identity = scope.build_plan([run], [], scope.OUTPUT_ROOT)[0]["checkpoints"][-1]["identity"]
    monkeypatch.setattr(scoring[2], "from_pretrained", lambda base, *args, **kwargs: base)
    with pytest.raises(ValueError, match="not actually loaded"):
        scope.score_checkpoint(identity, frozen[2], frozen[1])
    scoring[1].assert_not_called()


def test_adapter_removed_or_changed_after_plan_refused(tree, frozen, scoring):
    identity = scope.build_plan([tree()], [], scope.OUTPUT_ROOT)[0]["checkpoints"][-1]["identity"]
    weights = Path(identity["adapter_path"]) / "adapter_model.safetensors"
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after planning"):
        scope.score_checkpoint(identity, frozen[2], frozen[1])
    weights.unlink()
    with pytest.raises(FileNotFoundError):
        scope.score_checkpoint(identity, frozen[2], frozen[1])
    assert not scoring[0]["loads"]


def test_frozen_sample_drift_rejected_before_model_load(frozen, scoring):
    metadata = copy.deepcopy(frozen[3])
    metadata["musique"]["indices"].reverse()
    v71.prepare_panels.return_value = (frozen[2], metadata, [])
    with pytest.raises(ValueError, match="frozen V71 sample"):
        scope.prepare_registered_panels(frozen[1], Path("cache"), None, Path("traces"))
    assert not scoring[0]["loads"]


def test_output_confinement_and_exclusive_publication(tmp_path, tree):
    with pytest.raises(ValueError, match="must be under"):
        scope.write_new_json(tmp_path / "outside.json", {})
    with pytest.raises(ValueError, match="must be under"):
        scope.scoped_path(scope.OUTPUT_ROOT / "../escape")
    target = scope.OUTPUT_ROOT / "one.json"
    scope.write_new_json(target, {"original": True})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        scope.write_new_json(target, {"replacement": True})
    assert target.read_bytes() == before
    (scope.OUTPUT_ROOT / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        scope.write_new_json(scope.OUTPUT_ROOT / "link/escape.json", {})
    assert not list(scope.OUTPUT_ROOT.glob(".v99-*"))


def test_existing_protocol_cannot_be_changed(tree, frozen, scoring):
    plan = scope.build_plan([tree()], [], scope.OUTPUT_ROOT)
    run_eval(plan, frozen[1])
    changed = {**frozen[1], "dtype": "bfloat16"}
    with pytest.raises(ValueError, match="protocol differs"):
        run_eval(plan, changed)


def test_invalid_completed_baseline_cannot_be_reused(tree, frozen, scoring):
    plan = scope.build_plan([tree()], [], scope.OUTPUT_ROOT)
    run_eval(plan, frozen[1])
    # Corrupt a synthetic fixture to ensure an unsafe result is never resumed.
    path = Path(plan[0]["checkpoints"][-1]["output"])
    payload = v71.read_json(path)
    payload["update_0"]["trajectory"] = "another-run"
    dump(path, payload)
    calls_before = len(scoring[0]["loads"])
    with pytest.raises(ValueError, match="invalid own update-0"):
        run_eval(plan, frozen[1])
    assert len(scoring[0]["loads"]) == calls_before


def test_distribution_subset_scores_only_requested_set(tree, frozen, scoring):
    protocol = scope.load_protocol(frozen[0], ["musique"], "cpu", "float32")
    plan = scope.build_plan([tree()], [], scope.OUTPUT_ROOT)
    assert run_eval(plan, protocol) == {"scored": 2, "skipped": 0}
    assert scoring[1].call_count == 2
    result = v71.read_json(plan[0]["checkpoints"][-1]["output"])
    assert set(result["losses"]) == set(result["update_0"]["losses"]) == {"musique"}


def test_help_and_two_trajectory_dry_run_have_no_loads_or_writes(tree, frozen, scoring, capsys):
    with pytest.raises(SystemExit) as exc:
        scope.main(["--help"])
    assert exc.value.code == 0 and "--checkpoints" in capsys.readouterr().out
    runs = [tree("a"), tree("b", seed=42)]
    before = dict(os.environ)
    scope.main(["--trajectories", *map(str, runs), "--register", str(frozen[0]),
                "--output-dir", str(scope.OUTPUT_ROOT), "--device", "cpu", "--dtype", "float32", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["trajectories"]) == 2 and plan["dry_run"]
    assert plan["device"] == "cpu" and plan["dtype"] == "float32"
    assert not scope.OUTPUT_ROOT.exists()
    assert dict(os.environ) == before
    assert not scoring[0]["loads"]
    v71.prepare_panels.assert_not_called()
