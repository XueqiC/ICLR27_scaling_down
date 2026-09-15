"""CPU-only fixtures and an audit of the local saved development records."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from analysis import a1_development_table as a1


@pytest.fixture(scope="module")
def built():
    return a1.build()


def test_exact_membership_sets_and_missing_members():
    core, external, archived = a1.expected_membership()
    assert (len(core), len(external), len(archived)) == (22, 41, 5)
    assert sum("_matrix2_" in p for p in core) == 18
    assert sum("_132_critical_" in p for p in core) == 4
    a1.assert_membership(core, external, archived)
    for index in range(3):
        groups = [set(g) for g in (core, external, archived)]
        groups[index].pop()
        with pytest.raises(ValueError, match="membership changed"):
            a1.assert_membership(*groups)


def test_membership_rejects_same_count_substitution_and_duplicates():
    core, external, archived = a1.expected_membership()
    changed = set(core)
    changed.remove(next(iter(changed)))
    changed.add("gemma3-1b/gpt-5.6-luna_full_999_matrix2_lora_dseed41")
    with pytest.raises(ValueError, match="membership changed"):
        a1.assert_membership(changed, external, archived)
    with pytest.raises(ValueError, match="Duplicate"):
        a1.assert_membership([*core, next(iter(core))], external, archived)


@pytest.mark.parametrize("cohort", ["core", "external"])
def test_all_five_archived_runs_are_rejected_from_either_cohort(cohort):
    original = a1.expected_membership()
    for archived_run in original[2]:
        groups = [set(g) for g in original]
        groups[0 if cohort == "core" else 1].add(archived_run)
        with pytest.raises(ValueError, match="Archived"):
            a1.assert_membership(*groups)


def test_archived_contents_cannot_be_read(tmp_path):
    # Guard must fire before opening a missing file, even when explicitly asked.
    inputs = a1.Inputs(tmp_path)
    for run in a1.expected_membership()[2]:
        with pytest.raises(ValueError, match="Archived contents"):
            inputs.read(a1.BASE_REL / run / "train_log.json")
    assert inputs.sha256 == {}


def test_assertions_remain_active_under_optimized_python():
    source = "from analysis.a1_development_table import assert_membership; assert_membership([], [], [])"
    result = subprocess.run([sys.executable, "-B", "-O", "-c", source],
                            cwd=a1.ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "membership changed" in result.stderr


def test_unlogged_domain_tokens_stay_missing_including_zero():
    payload = {"measurement_tokens": {"qa": 700}, "unique_data_tokens": 4000,
               "trace_counts": {"qa": {"source_rows": 200, "training_rows": 199}},
               "completion_tokens_seen": 1000,
               "training_examples_by_domain": {"qa": 199}, "requested_token_milestones": [3000]}
    assert a1.recorded_domain_tokens(payload, "qa", 1000) is None
    assert a1.recorded_domain_tokens(payload, "qa", 0) is None
    payload["completion_tokens_seen_by_domain"] = {"qa": 300, "math": 400, "code": 300}
    assert a1.recorded_domain_tokens(payload, "qa", 1000) == 300
    with pytest.raises(ValueError, match="disagree"):
        a1.recorded_domain_tokens(payload, "qa", 1001)
    del payload["completion_tokens_seen_by_domain"]["math"]
    with pytest.raises(ValueError, match="Incomplete"):
        a1.recorded_domain_tokens(payload, "qa", 1000)


def test_weight_revision_reconstructed_only_from_recorded_snapshot_path():
    revision = "a" * 40
    payload = dict(resolved_student="recorded/model", updates=0, model_revision=None,
                   model_local_path=f"/saved/models--recorded--model/snapshots/{revision}")
    identity = json.loads(a1.recorded_weight_identity(payload))
    assert identity["base_revision"] == revision
    assert identity["base_revision_source"] == "model_local_path"
    payload["model_revision"] = "b" * 40
    with pytest.raises(ValueError, match="revision/path disagree"):
        a1.recorded_weight_identity(payload)
    payload.update(model_revision=None, model_local_path="/saved/model/main")
    assert json.loads(a1.recorded_weight_identity(payload))["base_revision"] is None


def fixture_log():
    # Three unique examples: (processed, supervised) = (10,2), (20,3), (30,7).
    # Batch size two; step three repeats the first two examples in epoch two.
    curve = []
    processed = supervised = 0
    for step, epoch, p, t, unique_p, unique_n in (
            (1, 1, 30, 5, 30, 2), (2, 1, 30, 7, 60, 3), (3, 2, 30, 5, 60, 3)):
        processed += p
        supervised += t
        curve.append(dict(step=step, epoch=epoch, tokens=p, completion_tokens=t,
                          tokens_seen=processed, seen_tokens=processed, processed_tokens=processed,
                          completion_tokens_seen=supervised, unique_data_tokens=unique_p,
                          unique_examples_seen=unique_n, unique_data_pool_tokens=60,
                          unique_data_pool_examples=3))
    return dict(status="trained", training_examples=3, unique_data_pool_examples=3,
                unique_data_pool_tokens=60, effective_batch_size_sequences=2,
                loss_curve=curve, updates=3, optimizer_steps=3, completion_tokens_seen=17,
                processed_tokens=90, seen_tokens=90, tokens_seen=90,
                trajectory=[dict(updates=1, processed_tokens=30, completion_tokens_seen=5),
                            dict(updates=3, processed_tokens=90, completion_tokens_seen=17)])


def test_pool_and_seen_supervised_tokens_hand_accounting():
    counts, pool = a1.audit_accounting(fixture_log())
    assert pool == 12  # Never the 60 prompt-inclusive pool tokens.
    assert [counts[i]["D_U_seen"] for i in range(4)] == [0, 5, 12, 12]
    assert [counts[i]["T_actual"] for i in range(4)] == [0, 5, 12, 17]
    assert counts[3]["processed_tokens"] == 90
    changed = fixture_log()
    changed["loss_curve"][0]["completion_tokens_seen"] = 30
    with pytest.raises(ValueError, match="Supervised ledger"):
        a1.audit_accounting(changed)
    changed = fixture_log()
    changed["loss_curve"][0]["unique_examples_seen"] = 1
    with pytest.raises(ValueError, match="First pass"):
        a1.audit_accounting(changed)


def test_incomplete_pool_not_guessed_and_duplicate_examples_not_assumed():
    log = fixture_log()
    log["loss_curve"] = log["loss_curve"][:1]
    log.update(updates=1, optimizer_steps=1, completion_tokens_seen=5,
               processed_tokens=30, seen_tokens=30, tokens_seen=30,
               trajectory=log["trajectory"][:1])
    counts, du = a1.audit_accounting(log)
    assert du is None
    assert counts[1]["D_U_seen"] == 5
    log["training_examples"] = 4
    counts, du = a1.audit_accounting(log)
    assert du is None and counts[1]["D_U_seen"] is None


def test_delta_uses_own_trajectory_zero_and_matching_distribution():
    def run(name, initial, after):
        def payload(step, losses):
            return dict(updates=step, resolved_student="fixture/model", post_training=losses,
                        probe_source="fixture", probe_seed=8, probe_half="odd", n_probe_requested=2,
                        loss_definition="completion CE/token",
                        measurement_benchmarks={c: c for c in a1.CAPABILITIES},
                        measurement_samples={c: 1 for c in a1.CAPABILITIES},
                        measurement_tokens={c: 2 for c in a1.CAPABILITIES},
                        processed_tokens=step * 20, completion_tokens_seen=step * 10)
        zero, end = payload(0, initial), payload(1, after)
        return {"run_id": name, "core": True, "log_path": "fixture/train_log.json",
                "log": dict(data_sampling_seed=41, training_seed=0, student="fixture", n_per_domain=1),
                "counts": {0: dict(T_actual=0, processed_tokens=0, D_U_seen=0),
                           1: dict(T_actual=10, processed_tokens=20, D_U_seen=10)},
                "D_U_pool": 10, "pool_id": "pool", "protocol_id": "core:fixture",
                "docs": {0: ("fixture/update-00000000/eval.json", zero),
                         1: ("fixture/update-00000001/eval.json", end)},
                "baseline": zero, "baseline_path": "fixture/update-00000000/eval.json"}
    first = run("a", {"math": 4., "code": 2., "qa": 8.}, {"math": 3.5, "code": 2.25, "qa": 7.})
    second = run("b", {"math": 5., "code": 3., "qa": 6.}, {"math": 3.5, "code": 2.25, "qa": 7.})
    for fixture, expected in ((first, [-.5, .25, -1.]), (second, [-1.5, -.75, 1.])):
        rows, flags = a1.training_rows(fixture, {})
        assert [r["delta"] for r in rows[:3]] == [0., 0., 0.]
        assert [r["delta"] for r in rows[3:]] == expected
        assert all(r["T_domain_if_available"] is None for r in rows)
        assert all(f["core"] and f["reconstructed"] for f in flags)
    first["docs"][1][1]["probe_seed"] = 9
    with pytest.raises(ValueError, match="different distribution"):
        a1.training_rows(first, {})


@pytest.mark.parametrize("tolerance,within,outside", [(.005, 100500, 100501), (.01, 101000, 101001), (.05, 105000, 105001)])
def test_pair_tolerance_inclusive_boundaries_and_symmetry(tolerance, within, outside):
    assert a1.budget_match(100000, within, tolerance)
    assert a1.budget_match(within, 100000, tolerance)
    assert not a1.budget_match(100000, outside, tolerance)
    assert not a1.budget_match(outside, 100000, tolerance)
    assert not a1.budget_match(0, 0, tolerance)
    assert not a1.budget_match(0, 1, tolerance)
    assert a1.budget_match(7, 7, 0)


def point(run="a", t=100000, u=66, seed=41, student="student", protocol="core:test",
          distribution="probe", capability="qa", training_seed=0):
    return dict(run_id=run, checkpoint_id=f"update-{t:08d}", capability=capability,
                distribution=distribution, student_id=student, protocol_id=protocol, U=u,
                pool_id=f"pool:{u}:{seed}", pool_seed=seed, training_seed=training_seed,
                T_actual=t, delta=t / 100000.)


def test_pair_grouping_separates_seed_questions_and_students_protocols():
    rows = [point(), point("b", u=198), point("c", seed=42),
            point("d", training_seed=1), point("e", student="other"),
            point("f", protocol="core:other"), point("g", distribution="other"),
            point("h", capability="math")]
    pairs = a1.intervention_pairs(rows, (0.,))
    kinds = {p["kind"] for p in pairs}
    assert kinds == {"I_U", "same_U_pool_seed", "same_U_training_seed"}
    for p in pairs:
        assert p["first"][0] not in "efgh" and p["second"][0] not in "efgh"
        if p["kind"] == "I_U":
            assert p["direction"] == "66->198"
    with pytest.raises(ValueError, match="External diagnostics"):
        a1.intervention_pairs([point(), point("b", protocol="external-diagnostic:other")])
    with pytest.raises(ValueError, match="Archived run"):
        a1.intervention_pairs([point(next(iter(a1.expected_membership()[2])))])


def test_I_T_factor_two_window_and_zero_exclusion():
    rows = [point(t=t) for t in (0, 1000, 1699, 1700, 2000, 2300, 2301)]
    pairs = a1.intervention_pairs(rows)
    first_1000 = [p for p in pairs if p["T_first"] == 1000]
    assert {p["T_second"] for p in first_1000} == {1700, 2000, 2300}
    assert all(p["kind"] == "I_T" and p["T_first"] > 0 for p in pairs)


def test_budget_only_zero_identity_and_residual_confounding():
    f = lambda t: 3 * t + 2
    assert a1.budget_only_intervention(f, 100000, 100000) == 0
    assert a1.budget_only_intervention(f, 100000, 100500) == 1500
    verification = a1.verify_budget_only_zero([point(), point("b", u=198)])
    assert verification["verified"]
    assert verification["positive_exact_checkpoint_pairs"] == 1


def test_inventory_keeps_budget_student_direction_distribution_strata():
    pairs = a1.intervention_pairs([point(t=25000), point("b", t=25000, u=198),
                                  point(t=200000), point("b", t=200000, u=198)])
    inv = a1.inventory(pairs)
    assert {r["budget_band"] for r in inv["strata"]} == {"[0,37500)", "[150000,infinity)"}
    assert all(r["direction"] == "66->198" and r["capability"] == "qa" for r in inv["strata"])
    assert all(r["residual_mismatch_percent"]["max"] == 0 for r in inv["strata"])


def test_real_data_membership_values_and_scope(built):
    core, external, flags, report = built
    assert len(core) == 582 and len(external) == 684
    assert len(flags) == 1266
    assert len({a1.row_key(r) for r in core + external}) == 1266
    assert all(tuple(r) == a1.COLUMNS for r in core + external)
    assert {r["run_id"] for r in core} == a1.expected_membership()[0]
    assert {r["run_id"] for r in external} == a1.expected_membership()[1]
    assert all(r["T_domain_if_available"] is None for r in core + external)
    assert all(r["delta"] == r["loss"] - r["initial_loss"] for r in core + external)
    assert all(r["T_actual"] <= r["processed_tokens"] for r in core + external)
    assert all(r["D_U_seen"] <= min(r["T_actual"], r["D_U_pool"]) for r in core + external)
    assert {r["distribution"].split(":")[0] for r in core} == {"training_probe", "2wiki_new", "musique", "triviaqa"}
    assert not {r["protocol_id"] for r in core} & {r["protocol_id"] for r in external}
    assert len({r["protocol_id"] for r in core}) == 1
    assert sum(f["core"] for f in flags) == len(core)
    assert all(not any(part.endswith(a1.ARCHIVE_SUFFIXES) for part in Path(p).parts)
               for p in report["input_sha256"])
    # A U=132 checkpoint still in its first pool pass is not assigned D_U_pool.
    r = next(r for r in core if r["run_id"] == "gemma3-1b/gpt-5.6-luna_full_132_critical_lora_dseed51"
             and r["optimizer_steps"] == 19 and r["capability"] == "qa")
    assert (r["D_U_pool"], r["D_U_seen"], r["T_actual"], r["processed_tokens"]) == (34503, 25541, 25541, 91890)
    assert r["delta"] == r["loss"] - 5.567791334661354


def test_real_tolerances_and_exact_pair_inventory(built):
    report = built[3]
    totals = report["inventory"]["totals"]
    assert [(r["checkpoint_pairs"], r["measurement_pairs"]) for r in totals if r["kind"] == "I_U"] == [(44, 225), (69, 327), (166, 678)]
    for p in report["pairs"]:
        if p["kind"] != "I_T":
            assert a1.budget_match(p["T_first"], p["T_second"], p["tolerance"])
    assert report["budget_only_zero"]["verified"]


def test_roundtrip_default_core_only_and_external_requires_opt_in(tmp_path, built):
    core, external, flags, original = built
    report = deepcopy(original)
    out = a1.write_outputs(core, external, flags, report, tmp_path)
    assert a1.load_development_table(out) == core
    assert a1.load_development_table(out, allow_external_diagnostic=True) == core + external
    assert a1.load_intervention_pairs(out, kind="I_U", tolerance=.005) == [
        p for p in report["pairs"] if p["kind"] == "I_U" and p["tolerance"] == .005]
    with (out / "row_metadata.csv").open() as stream:
        saved_flags = list(csv.DictReader(stream))
    assert sum(f["core"] == "true" for f in saved_flags) == 582
    assert all(f["reconstructed"] == "true" for f in saved_flags)
    assert "T_domain_if_available" not in {k for f in saved_flags for k in f["reconstructed_fields"].split(";")}
    assert set(p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*")) == {"results"}
    # Even if someone refreshes a file hash after contamination, cohort identity
    # guards still reject an appended historical row in the main table.
    a1.write_csv(out / "development_table.csv", a1.COLUMNS, core + external[:1])
    import hashlib
    report["artifact_sha256"]["development_table.csv"] = hashlib.sha256((out / "development_table.csv").read_bytes()).hexdigest()
    (out / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="cohort membership"):
        a1.load_development_table(out)


def test_output_symlinks_rejected_before_writing(tmp_path, built):
    (tmp_path / "results").mkdir()
    target = tmp_path / "other"
    target.mkdir()
    (tmp_path / a1.OUT_REL).symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        a1.write_outputs(*built, root=tmp_path)
    assert list(target.iterdir()) == []
