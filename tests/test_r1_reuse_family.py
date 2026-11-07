"""CPU contract tests: no model download, model construction, or GPU execution."""
import copy
import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import r1_reuse_family as r1


@pytest.fixture(scope="module")
def audit():
    before = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE")}
    result = r1.audit_gemma_pools()
    assert {k: os.environ.get(k) for k in before} == before
    return result


@pytest.fixture(scope="module")
def plan(audit):
    def resolver(model):
        # Deliberately synthetic pins: never persisted outside this test.
        return {"revision": ("a" if model == r1.STUDENTS[0] else "b") * 40,
                "model_id": model, "method": "CPU test fixture"}
    return r1.make_plan(resolver=resolver, pool_audit=audit)


def test_plan_determinism_and_matrix(plan, audit):
    rebuilt = r1.make_plan(resolver=lambda model: plan["revisions"][model], pool_audit=audit)
    assert json.dumps(plan, sort_keys=True) == json.dumps(rebuilt, sort_keys=True)
    assert len(plan["runs"]) == 12
    assert [r["index"] for r in plan["runs"]] == list(range(12))
    assert len({r["output_dir"] for r in plan["runs"]}) == 12
    r1.validate_plan(plan)
    for run in plan["runs"]:
        assert shlex.split(run["command_line"]) == run["command"]
        assert run["revision"] in run["v12_argv"][1]
    changed = copy.deepcopy(plan)
    changed["runs"][0]["U"] = 67
    with pytest.raises(ValueError, match="identity changed"):
        r1.validate_plan(changed)


def test_flags_identical_to_recorded_gemma_and_launch(plan):
    for run in plan["runs"]:
        gemma = r1.flag_values(r1.gemma_arguments(run["U"], run["pool_seed"]))
        qwen = r1.flag_values(run["v12_argv"])
        assert qwen.pop("--student") == [f"{run['student']}@{run['revision']}"]
        assert gemma.pop("--student") == ["gemma3-1b"]
        assert qwen == gemma
        saved = r1.read(ROOT / run["gemma_run"] / "eval.json")
        log = r1.read(ROOT / run["gemma_run"] / "train_log.json")
        assert int(qwen["--n-per-domain"][0]) == saved["n_per_domain"]
        assert int(qwen["--data-seed"][0]) == saved["data_seed"]
        assert int(qwen["--seed"][0]) == saved["training_seed"] == 0
        assert float(qwen["--lr"][0]) == saved["learning_rate"] == log["learning_rate"] == 1e-4
        assert list(map(int, qwen["--trajectory-tokens"])) == saved["trajectory_tokens_requested"]
        assert int(qwen["--schedule-tokens"][0]) == saved["schedule_tokens"] == 678000
        assert int(qwen["--epochs"][0]) == saved["epochs"] == 60
        assert qwen["--output-suffix"] == [saved["output_suffix"]]


def test_gemma_pool_membership_reconstructs_all_recorded_hashes(audit):
    pools, _ = audit
    assert len(pools) == 6
    for pool in pools:
        saved = r1.read(ROOT / pool["gemma_run"] / "eval.json")
        assert pool["gemma_data_pool_sha256"] == saved["data_pool_sha256"]
        assert pool["sample_hashes"] == sorted(pool["sample_hashes"])
        assert pool["sample_count"] == saved["unique_data_pool_examples"]
        assert pool["sample_hashes_sha256"] == r1.digest(pool["sample_hashes"])
        assert not pool["recorded_sample_hash_fields"]  # Only aggregate V12 hashes exist.


def test_revision_resolver_calls_hub_without_downloading(monkeypatch):
    import huggingface_hub
    calls = []

    def model_info(model, **kwargs):
        calls.append((model, kwargs))
        return SimpleNamespace(id=model, sha="c" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=model_info))
    result = r1.resolve_revision(r1.STUDENTS[0])
    assert result["revision"] == "c" * 40
    assert calls == [(r1.STUDENTS[0], {"revision": "main", "timeout": 20})]


def test_actual_frozen_trainer_parser_accepts_pinned_commands(plan, monkeypatch, capsys):
    from analysis import v12_distill as v12
    monkeypatch.setenv("SDL_ALLOW_PRC", "1")
    for run in plan["runs"]:
        monkeypatch.setattr(sys, "argv", [str(r1.TRAINER), *run["v12_argv"], "--dry-run"])
        v12.main()  # Real parser/validation; dry-run never touches data/models/CUDA.
        result = json.loads(capsys.readouterr().out)
        assert result["dry_run"] is True
        assert result["student"] == run["student"]
        assert result["revision"] == run["revision"]
        assert result["output"] == str(ROOT / run["output_dir"])
        assert result["trajectory_tokens"] == r1.MILESTONES


@pytest.mark.parametrize("host", ["localhost", "hipergator", "login1", "x-rsi", "RSI-host", "rai", "rai-worker", "sn4622122543"])
def test_qwen_guard_refuses_wrong_hosts(host, monkeypatch):
    monkeypatch.setenv("SDL_ALLOW_PRC", "1")
    monkeypatch.setattr(r1.socket, "gethostname", lambda: host)
    with pytest.raises(RuntimeError, match="rsi host"):
        r1.offline_environment("QWEN/Qwen3-0.6B-Base")


@pytest.mark.parametrize("host", ["rsi", "rsi01", "sn4622131043"])
def test_qwen_guard_allows_registered_hosts(host, monkeypatch):
    monkeypatch.setattr(r1.socket, "gethostname", lambda: host)
    assert r1.offline_environment(r1.STUDENTS[0])["SDL_ALLOW_PRC"] == "1"
    r1.host_guard("google/gemma-3-1b-pt", "unrelated-host")


def test_guard_precedes_run_io(monkeypatch):
    monkeypatch.setattr(r1.socket, "gethostname", lambda: "forbidden")
    monkeypatch.setattr(r1, "load_plan", lambda: pytest.fail("plan must not be read"))
    with pytest.raises(RuntimeError, match="refused"):
        r1.main(["--run", "--index", "0"])
    with pytest.raises(RuntimeError, match="refused"):
        r1.main(["--evaluate"])


def test_nearest_supervised_selection_is_not_processed_or_loss(tmp_path):
    for step, t, proc, loss in [(0, 0, 0, 1), (1, 40000, 50000, 1), (2, 60000, 200000, 0),
                                (3, 97000, 339000, 5), (4, 188000, 678000, 9)]:
        r1.write(tmp_path / "trajectory" / f"update-{step:08d}" / "eval.json",
                 {"updates": step, "completion_tokens_seen": t, "processed_tokens": proc, "loss": loss})
    baseline, selected = r1.select_checkpoints(tmp_path)
    assert baseline[1]["updates"] == 0
    assert [ev["updates"] for _, ev in selected.values()] == [1, 3, 4]
    assert selected[50000][1]["completion_tokens_seen"] == 40000  # Lower-T tie-break.


def synthetic_rows():
    rows = []
    for size, student in enumerate(r1.STUDENTS, 1):
        for u in r1.POOLS:
            for seed in r1.SEEDS:
                for t in r1.BUDGETS:
                    for distribution in r1.DISTRIBUTIONS:
                        # Larger student's reuse cost is twice the smaller's.
                        delta = {66: .4 * size, 198: .1 * size, 594: -.2 * size}[u]
                        delta += -.01 if seed == 41 else .01
                        if distribution == "MBPP":
                            delta *= -1  # Negative result must remain in all outputs.
                        rows.append({"student": student, "U": u, "pool_seed": seed, "target_T": t,
                                     "T": t + 100, "D_U": u * 100, "E": (t + 100) / (u * 100),
                                     "distribution": distribution, "initial_loss": 2 * size,
                                     "loss": 2 * size + delta, "delta": delta})
    return rows


def test_analysis_preregistered_signs_ranges_and_own_baseline():
    rows = synthetic_rows()
    summary = r1.analyse_rows(rows)
    assert summary["status"] == "complete" and summary["measured_cells"] == 216
    assert [h["reuse_cost"] for h in summary["H6a"]] == pytest.approx([.6, 1.2])
    assert summary["H6b"]["larger_minus_smaller_reuse_cost"] == pytest.approx(.6)
    assert all(h["specified_direction_observed"] for h in summary["H6c"])
    assert len(summary["H6c"]) == 18
    first = summary["cells"][0]
    assert first["delta"]["range"] == pytest.approx([.39, .41])
    gains = summary["descriptive_net_gains"]
    assert all(g["net_gain_at_U594"] is (g["distribution"] != "MBPP") for g in gains)
    assert all(g["crosses_from_U66_non_gain_to_U594_gain"] is (g["distribution"] != "MBPP") for g in gains)
    assert len(summary["descriptive_math_code"]) == 12
    # All signs reverse, including the size direction. No outcome suppression.
    for row in rows:
        row["delta"] *= -1
        row["loss"] = row["initial_loss"] + row["delta"]
    reversed_summary = r1.analyse_rows(rows)
    assert not any(h["specified_direction_observed"] for h in reversed_summary["H6a"])
    assert reversed_summary["H6b"]["specified_direction_observed"] is False
    assert not any(h["specified_direction_observed"] for h in reversed_summary["H6c"])


def test_analysis_zero_missing_duplicates_and_invalid_accounting():
    rows = synthetic_rows()
    partial = r1.analyse_rows([r for r in rows if r["pool_seed"] == 41])
    assert partial["status"] == "partial"
    assert all(h["reuse_cost"] is None for h in partial["H6a"])
    assert partial["H6b"]["specified_direction_observed"] is None
    assert r1.analyse_rows([])["status"] == "not_measured"
    with pytest.raises(ValueError, match="Duplicate"):
        r1.analyse_rows([rows[0], rows[0]])
    bad = dict(rows[0], E=999)
    with pytest.raises(ValueError, match="accounting"):
        r1.analyse_rows([bad])
    for row in rows:
        row.update(delta=0.0, loss=row["initial_loss"])
    zero = r1.analyse_rows(rows)
    assert all(h["direction"] == "zero" and h["specified_direction_observed"] is False for h in zero["H6a"])
    assert all(g["net_gain_at_U594"] is False for g in zero["descriptive_net_gains"])


def test_reports_contain_all_cells_and_revision_caption():
    summary = r1.analyse_rows(synthetic_rows())
    tex = r1.table_tex(summary)
    assert tex.startswith("% Generated by analysis/r1_reuse_family.py; do not edit.\n")
    assert tex.count(r"\caption{\rev{") == 2
    assert tex.count(r"\begin{tabular*}{\textwidth}") == 2
    assert tex.count("2Wiki-fresh &") == 6
    assert "H6a" in r1.markdown(summary) and "H6c" in r1.markdown(summary)
    empty = r1.table_tex(r1.analyse_rows([]))
    assert "0/216" in empty and "--" in empty


def test_completed_run_skips_and_partial_run_replays(plan, tmp_path, monkeypatch):
    run = plan["runs"][0]
    state = tmp_path / r1.OUT / "runs/0"
    out = tmp_path / run["output_dir"]
    monkeypatch.setattr(r1, "offline_environment", lambda model: {
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    monkeypatch.setattr(r1, "validate_plan", lambda *a: None)
    owner = {"plan_sha256": plan["plan_sha256"], "index": 0, "v12_argv": run["v12_argv"]}
    r1.write(state / "owner.json", owner)
    pool = {"data_pool_sha256": "pool", "sample_hashes": plan["pools"][0]["sample_hashes"]}
    r1.write(out / "eval.json", pool)
    r1.write(state / "pool.json", pool)
    r1.write(out / "train_log.json", {"status": "trained"})
    monkeypatch.setattr(r1, "verify_run_metadata", lambda *a: None)
    monkeypatch.setattr(r1.subprocess, "run", lambda *a, **kw: pytest.fail("completed run must skip"))
    r1.execute_run(plan, 0, tmp_path)
    (out / "eval.json").unlink()
    from analysis import v12_distill as v12
    monkeypatch.setattr(v12, "build_probes", lambda *a, **kw: {})
    r1.write(tmp_path / r1.PRIMARY, {})
    monkeypatch.setattr(r1, "qwen_pool", lambda *a: pool)
    calls = []

    def train(argv, **kwargs):
        calls.append((argv, kwargs))
        r1.write(out / "eval.json", pool)

    monkeypatch.setattr(r1.subprocess, "run", train)
    r1.execute_run(plan, 0, tmp_path)
    assert (state / "interrupted-0001/train_log.json").is_file()
    assert calls[0][0] == [sys.executable, str(r1.TRAINER), *run["v12_argv"]]
    assert calls[0][1]["check"] is True


def test_evaluation_own_baseline_all_six_readouts_and_cache(plan, tmp_path, monkeypatch):
    from analysis import v99_scope_eval as v99
    run = plan["runs"][0]
    state, out = tmp_path / r1.OUT / "runs/0", tmp_path / run["output_dir"]
    reference = r1.read(ROOT / run["gemma_run"] / "eval.json")
    pool = {"D_U": 10000, "data_pool_sha256": reference["data_pool_sha256"]}
    r1.write(state / "pool.json", pool)
    curve = []
    for step, t in enumerate([0, 49000, 101000, 203000]):
        ev = {**reference, "updates": step, "completion_tokens_seen": t, "processed_tokens": 2 * t,
              "revision": run["revision"], "resolved_student": run["student"],
              "post_training": {cap: 2.0 + step / 10 for cap in r1.PRIMARY_KEYS}}
        r1.write(out / "trajectory" / f"update-{step:08d}" / "eval.json", ev)
        if step:
            curve.append({"step": step, "processed_tokens": 2 * t, "completion_tokens_seen": t})
    r1.write(out / "eval.json", ev)
    r1.write(out / "train_log.json", {"status": "trained", "loss_curve": curve})
    monkeypatch.setattr(r1, "offline_environment", lambda model: {})
    monkeypatch.setattr(r1, "qwen_pool", lambda *a: pool)
    panels = r1.verify_panels()
    monkeypatch.setattr(r1, "verify_panels", lambda root: panels)
    monkeypatch.setattr(v99, "load_protocol", lambda *a: {"fixture": True})
    monkeypatch.setattr(v99, "adapter_identity", lambda path, ev: {"adapter_sha256": str(ev["updates"])})
    calls = []

    def score(identity, fresh, protocol):
        step = identity["updates"]
        calls.append(step)
        assert set(fresh) == set(r1.FRESH_KEYS)
        return {"adapter_loaded": step > 0,
                "losses": {key: {"loss": 5.0 + step / 5, "tokens": 256, "n": 256} for key in fresh}}

    monkeypatch.setattr(v99, "score_checkpoint", score)
    r1.evaluate_run(plan, 0, tmp_path)
    result = r1.read(tmp_path / r1.OUT / "evaluations/0.json")
    assert len(result["rows"]) == 18
    assert calls == [0, 1, 2, 3]
    for row in result["rows"]:
        step = r1.BUDGETS.index(row["target_T"]) + 1
        assert row["T"] == [49000, 101000, 203000][step - 1]
        assert row["E"] == row["T"] / 10000
        assert row["delta"] == pytest.approx(step / (10 if row["distribution"] in r1.PRIMARY_KEYS.values() else 5))
    r1.evaluate_run(plan, 0, tmp_path)
    assert calls == [0, 1, 2, 3]  # Exact same cache; no new scoring.
    changed = r1.read(out / "trajectory/update-00000001/eval.json")
    changed["post_training"]["qa"] += .1
    r1.write(out / "trajectory/update-00000001/eval.json", changed)
    with pytest.raises(ValueError, match="Cached score"):
        r1.evaluate_run(plan, 0, tmp_path)


def test_frozen_files_and_mirrors():
    r1.frozen_checks()
    for path in ("analysis/r1_reuse_family.py", "scripts/rsi_r1_reuse.sh", "tests/test_r1_reuse_family.py"):
        assert (ROOT / path).read_bytes() == (ROOT / "paper" / path).read_bytes()
