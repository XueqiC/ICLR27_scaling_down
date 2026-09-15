"""Real cached-tokenizer checks and failure gates; no weights, GPU, or training."""
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from analysis import a3_corner_pool_search as a3
from analysis import v12_distill as trainer


@pytest.fixture(autouse=True)
def forbid_training_and_weights(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A3 must never load model weights or train")

    monkeypatch.setattr(trainer, "load_text_causal_lm", forbidden)
    monkeypatch.setattr(trainer, "run_distillation", forbidden)
    monkeypatch.setattr(trainer, "_train", forbidden)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


@pytest.fixture(scope="module")
def counters():
    return {s: a3.PoolCounter(a3.load_tokenizer(s)[0]) for s in a3.STUDENTS}


@pytest.fixture(scope="module")
def saved_plan():
    # Integration prerequisite: run the corrected search CLI first.
    plan = a3.read_json(a3.OUT / "plan.json")
    assert plan["pool_search_status"] == "completed"
    return plan


def test_selection_and_tokenization_are_imported_from_frozen_trainer():
    assert a3.load_sft_records is trainer.load_sft_records
    assert a3.tokenize_sft_example is trainer.tokenize_sft_example
    assert a3.verify_trainer_hash() == a3.read_json(a3.ROOT / a3.A1)["code_sha256"][str(a3.TRAINER)]


@pytest.mark.parametrize("student", a3.STUDENTS)
@pytest.mark.parametrize("n,seed,expected", [(66, 41, 16962), (66, 42, 17104),
                                            (132, 51, 34503), (132, 52, 34639)])
def test_counter_reproduces_all_four_recorded_totals(counters, student, n, seed, expected):
    pool = counters[student].price(n, seed)
    assert pool["D_U_pool"] == expected
    assert pool["unique_data_pool_tokens"] > expected
    assert sum(d["completion_tokens"] for d in pool["per_domain"].values()) == expected
    assert sum(Fraction(d["token_share_exact"]) for d in pool["per_domain"].values()) == 1
    assert all(d["source_rows"] == n for d in pool["per_domain"].values())
    # Empty teacher completions are omitted by V12 even under the full recipe.
    deleted = sum(d["coverage_deleted_rows"] + d["tokenization_deleted_rows"]
                  for d in pool["per_domain"].values())
    assert pool["prepared_examples"] + deleted == 3 * n
    assert pool["prepared_examples"] == (197 if n == 66 else 392)


@pytest.mark.parametrize("student", a3.STUDENTS)
def test_four_recorded_pool_hashes_match(counters, student):
    checks = a3.verify_recorded_pools(counters[student], student)
    assert len(checks) == 4
    assert all(c["observed"] == c["expected"] for c in checks)


def test_one_token_discrepancy_aborts_before_search(counters, monkeypatch, tmp_path, capsys):
    real = counters[a3.STUDENTS[0]]

    class OffByOne:
        def price(self, n, seed):
            pool = real.price(n, seed)
            pool["D_U_pool"] += 1
            return pool

    def forbidden_search(*args, **kwargs):
        pytest.fail("Search started after a one-token mismatch")

    monkeypatch.setattr(a3, "PoolCounter", lambda tokenizer: OffByOne())
    monkeypatch.setattr(a3, "load_tokenizer", lambda student: (real.tokenizer, {}))
    monkeypatch.setattr(a3, "recorded_inventory", lambda **kwargs: {})
    monkeypatch.setattr(a3, "price_grids", forbidden_search)
    assert a3.main(["--output-dir", str(tmp_path)]) != 0
    report = a3.read_json(tmp_path / "plan.json")
    assert report["pool_search_status"] == "aborted"
    assert "expected 16962, observed 16963" in report["failures"][0]
    assert "SEARCH ABORTED" in capsys.readouterr().err
    assert "search" not in report


@pytest.mark.parametrize("values", [(50500, 98000, 98400, 32900, 64000),
                                    (50800, 102000, 101600, 34200, 68300)])
def test_four_mismatch_definitions_use_actual_boundaries_and_correct_denominators(values):
    t1, t2, tc, db, dc = values
    e1, e2, e3, e4 = Fraction(t1, db), Fraction(a3.T1, a3.D_A), Fraction(tc, dc), Fraction(t2, db)
    expected = {"budget_low": Fraction(abs(t1 - a3.T1), a3.T1),
                "budget_high": Fraction(abs(tc - t2), min(tc, t2)),
                "reuse_low": abs(e1 - e3) / min(e1, e3),
                "reuse_high": abs(e2 - e4) / min(e2, e4)}
    assert a3.mismatch_fractions(*values) == expected
    design = a3.rectangle(db, dc, t1, t2, tc)
    assert Fraction(design["worst_mismatch_exact"]) == max(expected.values())
    assert set(design["mismatches"]) == set(a3.MISMATCH_NAMES)
    for name, error in expected.items():
        assert Fraction(design["mismatches"][name]["fraction_exact"]) == error


def test_pool_size_is_derived_from_corner4_boundary_not_twice_a():
    db, dc, t1, t2, tc = 33000, 64000, 50560, 98300, 98100
    d = a3.rectangle(db, dc, t1, t2, tc)
    assert Fraction(d["boundary_derived_D_B_exact"]) == Fraction(t2 * a3.D_A, a3.T1)
    assert Fraction(d["boundary_derived_D_B_exact"]) != 2 * a3.D_A
    assert Fraction(d["boundary_derived_D_C_exact"]) == Fraction(db * tc, t1)
    assert Fraction(d["boundary_derived_D_C_exact"]) != Fraction(db * db, a3.D_A)
    assert [c["T_supervised"] for c in d["corners"]] == [t1, a3.T1, tc, t2]
    assert d["corners"][1]["already_measured"]


@pytest.mark.parametrize("name", a3.MISMATCH_NAMES)
def test_each_half_percent_limit_is_inclusive_and_never_absorbed(name):
    # A rational rectangle permits exact 0.5% even when T1_A is odd.
    db = 2 * a3.D_A
    if name == "budget_low":
        args = (db, 2 * db, Fraction(a3.T1 * 201, 200), 2 * a3.T1, 2 * a3.T1)
    elif name == "budget_high":
        args = (db, 2 * db, a3.T1, 2 * a3.T1, Fraction(2 * a3.T1 * 201, 200))
    elif name == "reuse_low":
        args = (db, Fraction(2 * db * 201, 200), a3.T1, 2 * a3.T1, 2 * a3.T1)
    else:
        args = (Fraction(db * 201, 200), 2 * db, a3.T1, 2 * a3.T1, 2 * a3.T1)
    design = a3.rectangle(*args)
    assert Fraction(design["mismatches"][name]["fraction_exact"]) == Fraction(1, 200)
    assert design["mismatches"][name]["passed"]
    args = list(args)
    # Move the coordinate responsible for this residual just beyond 0.5%.
    index = {"budget_low": 2, "budget_high": 4, "reuse_low": 1, "reuse_high": 0}[name]
    args[index] *= Fraction(100501, 100500)
    failed = a3.rectangle(*args)
    assert Fraction(failed["mismatches"][name]["fraction_exact"]) == Fraction(501, 100000)
    assert not failed["mismatches"][name]["passed"]
    failures = a3.design_failures(failed)
    assert any(f.startswith(name + ":") and "0.501000000%" in f for f in failures)
    assert "Tolerance has not been widened" in failures[0]


@pytest.mark.parametrize("seed", sorted(a3.USED_SEEDS | a3.RESERVED_SEEDS | {69, -1, 2**32}))
def test_used_reserved_and_out_of_range_seeds_rejected(seed):
    assert not a3.valid_seed(seed)


def test_newly_observed_seed_is_also_rejected():
    assert a3.valid_seed(70)
    assert not a3.valid_seed(70, observed=[70])
    assert a3.valid_seed(92)


def fake_pool(n, seed, total, middle, drift="1/1"):
    tokens = [1000, 2000, 3000, *middle, 160000, 170000, 180000]
    rows = [{"update": i + 1, "supervised_tokens": t, "processed_tokens": t * 3, "epoch": 1}
            for i, t in enumerate(tokens)]
    return {"n_per_domain": n, "data_seed": seed, "D_U_pool": total,
            "data_pool_sha256": f"{n * 1000 + seed:064x}",
            "max_absolute_share_drift_pp_exact": drift,
            "max_absolute_share_drift_pp": float(Fraction(drift)), "_rows": rows}


def brute_force(b_pools, c_pools):
    ranked = {}
    for b in b_pools:
        for low in b["_rows"][3:-3]:
            for high in b["_rows"][3:-3]:
                for c in c_pools:
                    if c["data_seed"] == b["data_seed"] or c["data_pool_sha256"] == b["data_pool_sha256"]:
                        continue
                    for cp in c["_rows"][3:-3]:
                        d = a3.rectangle(b["D_U_pool"], c["D_U_pool"], low["supervised_tokens"],
                                         high["supervised_tokens"], cp["supervised_tokens"])
                        if not d["contrast_passed"]:
                            continue
                        candidate = {"B": b, "C": c, "B_low": low, "B_high": high, "C_high": cp, "design": d}
                        key = (b["n_per_domain"], b["data_seed"])
                        if key not in ranked or a3.candidate_key(candidate) < a3.candidate_key(ranked[key]):
                            ranked[key] = candidate
    return sorted(ranked.values(), key=a3.candidate_key)


def test_nested_boundary_search_matches_exhaustive_four_mismatch_minimax():
    bs = [fake_pool(124, 70, 33000, [50560, 50630, 98300, 98600], "3/2"),
          fake_pool(125, 71, 34000, [50540, 50700, 101300, 102000], "1/1"),
          fake_pool(126, 72, 33500, [50400, 50600, 99800, 100500], "1/2")]
    cs = [fake_pool(248, 73, 64000, [98000, 98300, 98500, 99000]),
          fake_pool(249, 74, 66500, [99000, 100100, 101300, 102000]),
          fake_pool(250, 70, 65500, [98000, 99000, 100000, 102000])]
    expected = brute_force(bs, cs)
    best, ranked, conditional, certificate = a3.search_boundaries(bs, cs, best_k=3)
    assert [a3.candidate_key(c) for c in ranked] == [a3.candidate_key(c) for c in expected]
    assert a3.candidate_key(best) == a3.candidate_key(conditional[0])
    assert certificate["global_minimax_certified"]
    single, _, _, _ = a3.search_boundaries(bs, cs, best_k=1)
    assert a3.candidate_key(single) == a3.candidate_key(expected[0])
    first_b = min((a3.b_boundary_candidates(p)[0] for p in bs), key=lambda b: b["lower_bound"])
    assert first_b["pool"]["data_seed"] != single["B"]["data_seed"]
    assert all(c["B"] == best["B"] and c["B_low"] == best["B_low"] and c["B_high"] == best["B_high"] for c in conditional)


def test_boundary_search_ties_break_by_domain_drift():
    bs = [fake_pool(124, 70, 33000, [50560, 98300], "3/2"),
          fake_pool(125, 71, 33000, [50560, 98300], "1/2")]
    cs = [fake_pool(248, 72, 64000, [98100, 98300], "1/4")]
    best, _, _, _ = a3.search_boundaries(bs, cs, best_k=2)
    assert best["B"]["data_seed"] == 71


def test_domain_drift_is_a_hard_exclusion_and_full_grid_is_priced(monkeypatch, counters):
    counter = counters[a3.STUDENTS[0]]
    reference = counter.price(66, 41)
    bad = deepcopy(reference)
    bad["data_seed"] = 70
    bad["D_U_pool"] = 100
    for domain, tokens in zip(a3.DOMAINS, (100, 0, 0)):
        bad["per_domain"][domain]["completion_tokens"] = tokens
    class FakeCounter:
        def price(self, *args):
            return bad
    seen = []
    original = a3.replay_accounting
    monkeypatch.setattr(a3, "replay_accounting", lambda p: (seen.append(p) or original(p)))
    with pytest.raises(ValueError, match="within 2 pp domain drift"):
        a3.price_grids(FakeCounter(), reference, [66], [70], {"hashes": [], "observed_seeds": []}, "B")
    assert len(seen) == 1


def test_hash_collision_rejected_even_with_fresh_seed(counters):
    counter = counters[a3.STUDENTS[0]]
    reference, candidate = counter.price(66, 41), counter.price(132, 70)
    with pytest.raises(ValueError, match="No fresh pool B candidates"):
        a3.price_grids(counter, reference, [132], [70],
                       {"hashes": [candidate["data_pool_sha256"]], "observed_seeds": []}, "B")


def test_inventory_finds_seeds_without_pool_metadata_and_canonical_hash_ids(tmp_path):
    result = tmp_path / "results"
    (result / "a1-development-table").mkdir(parents=True)
    digest = "a" * 64
    (result / "a1-development-table/summary.json").write_text(json.dumps({"run_audit": [{"pool_id": "sha256:" + digest}]}))
    (result / "seed_only.json").write_text(json.dumps({"nested": {"seed": 75, "data_seed": 76}, "pool_seed": 77}))
    inv = a3.recorded_inventory(root=tmp_path, output=tmp_path / "output")
    assert set(inv["observed_seeds"]) == {75, 76, 77}
    assert digest in inv["hashes"]


@pytest.mark.parametrize("n,seed", a3.RECORDED_TOTALS)
def test_cpu_ledger_replays_recorded_updates(counters, n, seed):
    pool = counters[a3.STUDENTS[0]].price(n, seed)
    rows, schedule = a3.replay_accounting(pool)
    suffix = "matrix2" if n == 66 else "critical"
    path = a3.ROOT / f"results/v12-distill/gemma3-1b/gpt-5.6-luna_full_{n}_{suffix}_lora_dseed{seed}/train_log.json"
    recorded = a3.read_json(path)
    assert len(rows) == recorded["total_updates_planned"] == schedule["schedule_updates"]
    for row, logged in zip(rows, recorded["loss_curve"]):
        assert row["update"] == logged["step"]
        assert row["supervised_tokens"] == logged["completion_tokens_seen"]
        assert row["processed_tokens"] == logged["tokens_seen"]


def test_chosen_pools_fresh_drift_valid_and_same_for_both_students(saved_plan, counters):
    inv = a3.recorded_inventory()
    b, c = saved_plan["pools"]["B"], saved_plan["pools"]["C"]
    assert b["data_seed"] != c["data_seed"]
    assert b["data_pool_sha256"] != c["data_pool_sha256"]
    for p in (b, c):
        assert a3.valid_seed(p["data_seed"], inv["observed_seeds"])
        assert p["data_pool_sha256"] not in inv["hashes"]
        assert Fraction(p["max_absolute_share_drift_pp_exact"]) <= 2
        for counter in counters.values():
            actual = counter.price(p["n_per_domain"], p["data_seed"])
            assert actual["D_U_pool"] == p["D_U_pool"]
            assert actual["data_pool_sha256"] == p["data_pool_sha256"]
    d = saved_plan["design"]
    assert d == a3.rectangle(b["D_U_pool"], c["D_U_pool"], d["T1_B"], d["T2_B"], d["T2_C"])
    assert d["contrast_passed"]
    assert saved_plan["verification"]["passed_before_search"]
    assert len(saved_plan["verification"]["recorded_pools"]) == 8
    assert len(saved_plan["verification"]["recorded_update_replays"]) == 8
    for result in (saved_plan["search"]["B"], saved_plan["search"]["C_conditional_on_achieved_B"]):
        assert len(result["best_candidates"]) == 5
        assert all(set(x["design"]["mismatches"]) == set(a3.MISMATCH_NAMES) for x in result["best_candidates"])


def test_commands_select_exact_boundaries_and_complete_local_ladders(saved_plan, counters):
    assert a3.trajectory_contract()["trajectory_tokens_unit"] == "processed_prompt_plus_completion_tokens"
    required = {"--n-per-domain", "--data-seed", "--trajectory-tokens", "--stop-after-trajectory",
                "--schedule-tokens", "--epochs", "--lr", "--training-mode", "--save-trajectory", "--output-suffix"}
    count = 0
    for student in a3.STUDENTS:
        entry = saved_plan["students"][student]
        assert entry["already_measured_corner_2"]["retrain"] is False
        assert entry["already_measured_corner_2"]["supervised_tokens"] == a3.T1
        assert len(entry["new_trajectories"]) == 2
        assert [t["corner_ids"] for t in entry["new_trajectories"]] == [[1, 4], [3]]
        for tr in entry["new_trajectories"]:
            count += 1
            argv = shlex.split(tr["command_line"])
            assert argv == tr["argv"]
            assert required <= set(argv)
            for flag, expected in (("--recipe", "full"), ("--schedule-tokens", "678000"), ("--epochs", "60"),
                                   ("--lr", "1e-4"), ("--training-mode", "lora"), ("--teacher", a3.TEACHER), ("--seed", "0")):
                assert argv[argv.index(flag) + 1] == expected
            assert argv[argv.index("--output-suffix") + 1].startswith("a3b_corners_")
            assert tr["trajectory_tokens_unit"] == "processed_prompt_plus_completion_tokens"
            p = saved_plan["pools"][tr["pool"]]
            actual = counters[student].price(p["n_per_domain"], p["data_seed"])
            rows, _ = a3.replay_accounting(actual)
            expected_updates = {target["update"] + offset for target in tr["target_boundaries"] for offset in range(-3, 4)}
            assert {cp["update"] for cp in tr["predicted_checkpoints"]} == expected_updates
            assert len(tr["trajectory_tokens"]) == len(expected_updates)
            assert tr["trajectory_tokens"] == sorted(set(tr["trajectory_tokens"]))
            for cp, flag in zip(tr["predicted_checkpoints"], tr["trajectory_tokens"]):
                crossing = next(r for r in rows if r["processed_tokens"] >= flag)
                assert flag == cp["processed_tokens"] == crossing["processed_tokens"]
                assert cp["update"] == crossing["update"]
                assert cp["supervised_tokens"] == crossing["supervised_tokens"]
            assert tr["predicted_actual_stop_supervised_tokens"] == rows[max(expected_updates) - 1]["supervised_tokens"]
    assert count == saved_plan["protocol"]["round_cap"] == 4


def test_measured_slope_contamination_is_reproducible_and_separate(saved_plan):
    result = saved_plan["residual_contamination"]
    assert result["second_difference"] == "L4 - L3 - L2 + L1"
    assert "do not sum" in result["limitation"]
    for channel in result["channels"]:
        assert set(channel["residuals"]) == set(a3.MISMATCH_NAMES)
        for residual in channel["residuals"].values():
            delta = float(Fraction(residual["signed_equivalent_supervised_displacement_exact"]))
            effects = []
            for pair in residual["adjacent_checkpoint_slopes"]:
                slope = (pair["loss_second"] - pair["loss_first"]) / (pair["T_second"] - pair["T_first"])
                assert pair["native_token_nats_per_supervised_token"] == slope
                effect = residual["second_difference_sign"] * delta * slope
                assert pair["signed_second_difference_sensitivity_nats"] == effect
                effects.append(abs(effect))
            assert residual["max_absolute_sensitivity_nats"] == max(effects)


def test_best_achievable_failure_is_retained_and_cli_returns_nonzero(monkeypatch, tmp_path):
    b = fake_pool(124, 70, 33000, [50000, 98300])  # low edge alone misses by >0.5%.
    c = fake_pool(248, 71, 64000, [98000, 98300])
    best, _, _, _ = a3.search_boundaries([b], [c], 1)
    d = best["design"]
    assert not d["tolerance_passed"]
    failures = a3.design_failures(d)
    assert failures and "No combination" in failures[0]
    plan = {"status": "failed", "launchable": False, "pool_search_status": "completed", "failures": failures,
            "best_achievable": a3.public_candidate(best)}
    monkeypatch.setattr(a3, "build_plan", lambda *args, **kwargs: plan)
    assert a3.main(["--output-dir", str(tmp_path)]) == 1
    written = a3.read_json(tmp_path / "plan.json")
    assert written["best_achievable"]["design"] == d
    assert "0.5%" in (tmp_path / "plan.md").read_text()


def test_invalid_cli_nonzero_even_under_optimized_python(tmp_path):
    result = subprocess.run([sys.executable, "-B", "-O", str(a3.ROOT / "analysis/a3_corner_pool_search.py"),
                             "--n-b", "140", "124", "--output-dir", str(tmp_path)],
                            cwd=a3.ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert a3.read_json(tmp_path / "plan.json")["status"] == "failed"


def test_saved_plan_status_and_markdown_agree_with_fixed_tolerance(saved_plan):
    failures = a3.design_failures(saved_plan["design"])
    assert saved_plan["failures"] == failures
    assert saved_plan["status"] == ("failed" if failures else "passed")
    assert saved_plan["launchable"] == (not failures)
    assert (a3.OUT / "plan.md").read_text() == a3.markdown(saved_plan)
