"""CPU-only checks of the scientific claims and sealed provenance in A20."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("identity_table_under_test", ROOT / "analysis/s3_identity_table.py")
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)
EVIDENCE = ROOT / ("data_mirror" if (ROOT / "data_mirror").is_dir() else "results")


@pytest.fixture(scope="module")
def audited():
    return gen.audit(EVIDENCE)


def test_eight_roles_six_snapshots_and_only_two_new_references(audited):
    rows = audited["rows"]
    assert len(rows) == 8
    assert len({(r["model"], r["commit"]) for r in rows}) == 6
    refs = [r for r in rows if r["role"] == "reference"]
    assert [r["new_source_state"] for r in refs] == [True, True, False, False]
    assert all(r["new_run"] for r in rows)
    assert rows[0]["commit"] == rows[3]["commit"]
    assert rows[4]["commit"] == rows[7]["commit"]


def test_exact_revisions_and_storage_counts(audited):
    expected = {
        "EleutherAI/pythia-160m": ("3df7d2c5532755f7022704e07dea87f35eeaf0e1", 84934656),
        "EleutherAI/pythia-410m": ("ade882a342507b4be34455393ba750fee7a513b8", 301989888),
        "EleutherAI/pythia-1.4b": ("4e056d657ceede13435146a8126bbf7bda0a1b0e", 1207959552),
        "google/gemma-3-270m": ("9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1", 100270080),
        "google/gemma-3-1b-pt": ("fcf18a2a879aab110ca39f8bffbccd5d49d8eb29", 697761792),
        "google/gemma-3-4b-pt": ("cc012e0a6d0787b4adcc0fa2c4da74402494554d", 3208642560),
    }
    for row in audited["rows"]:
        assert (row["commit"], row["matrix_parameters"]) == expected[row["model"]]
        assert row["release"] == "Pretrained base"


def test_snapshot_evidence_is_not_confused_with_release_outcomes(audited):
    by_model = {r["model"]: r for r in audited["rows"]}
    small = by_model["google/gemma-3-270m"]["prior"]
    assert small["initial_evidence"] == ["v32-descriptors/gemma3-270m/features.json"]
    assert small["exact_outcome_evidence"] == []
    assert len(small["release_outcome_evidence"]) == 3
    one = by_model["google/gemma-3-1b-pt"]["prior"]
    assert one["exact_outcome_evidence"] == ["v71-qa-scope/measurements.json"]
    four = by_model["google/gemma-3-4b-pt"]["prior"]
    assert "adapter_base_unrecorded" in four["outcome_limit"]


def test_locked_fit_rosters_exclude_s3_and_gemma(audited):
    fits = audited["fits"]
    assert len(fits["pruning"]["states"]) == 17
    assert {b: len(s) for b, s in fits["per_channel"]["states_by_bit"].items()} == {
        "3": 17, "4": 17, "5": 4, "6": 16, "8": 16}
    assert len(fits["grouped"]["states"]) == 6
    assert fits["grouped"]["rows"] == 162
    students = fits["distillation"]["states"]
    assert len(students) == 7
    assert "pythia-1.4b@step64000" in students
    assert not set(students) & set(fits["distillation"]["excluded_students"])
    rosters = fits["pruning"]["states"] + fits["grouped"]["states"] + students
    rosters += sum(fits["per_channel"]["states_by_bit"].values(), [])
    assert all(s.startswith("pythia-") and "step120000" not in s for s in rosters)


def test_freeze_has_one_internal_time_not_invented_times(audited):
    freeze = audited["freeze"]
    assert freeze["timestamp_utc"] == "2026-09-16T23:31:54.810499+00:00"
    assert freeze["identity_timestamp_utc"] is None
    assert freeze["anchor_timestamps_utc"] is None
    assert freeze["predictions_sha256"] == "d28986ca6b13fe833e965f185de25af01c95d0f256cd36e448e53a9c9d982acd"


def test_bad_plan_seal_fails_before_any_output(tmp_path):
    directory = tmp_path / gen.S3
    directory.mkdir()
    source = gen.Evidence(EVIDENCE)
    (directory / "plan_v2.json").write_bytes(source.raw(f"{gen.S3}/plan_v2.json") + b" ")
    (directory / "plan_v2.json.sha256").write_bytes(source.raw(f"{gen.S3}/plan_v2.json.sha256"))
    with pytest.raises(ValueError, match="Artifact seal changed"):
        gen.audit(tmp_path)
    assert not list(tmp_path.rglob("*.tex"))


def test_changed_historical_snapshot_is_rejected(monkeypatch):
    original = gen.historical_evidence

    def wrong_snapshot(evidence):
        history = original(evidence)
        history["google/gemma-3-1b-pt"]["commit"] = "0" * 40
        return history

    monkeypatch.setattr(gen, "historical_evidence", wrong_snapshot)
    with pytest.raises(ValueError, match="Historical snapshot differs"):
        gen.audit(EVIDENCE)


def test_changed_locked_predictor_is_rejected():
    evidence = gen.Evidence(EVIDENCE)
    locked = evidence.read(f"{gen.S3}/inputs/locked_models.json")
    locked = copy.deepcopy(locked)
    locked["distill"]["constant"]["math"] += 0.1
    with pytest.raises(ValueError, match="differs from V78"):
        gen.fit_provenance(evidence, locked)


def test_compressed_public_evidence(tmp_path):
    (tmp_path / "record.json.gz").write_bytes(gzip.compress(b'{"value": 7}'))
    evidence = gen.Evidence(tmp_path)
    assert evidence.read("record.json") == {"value": 7}


def test_anonymization_requires_a_matching_pair_and_path(tmp_path):
    (tmp_path / "record.json").write_text('{"value": 7}')
    evidence = gen.Evidence(tmp_path)
    evidence.read("record.json")
    pair = dict(pairable=True, frozen_sha256="original", published_sha256=evidence.hashes["record.json"],
                published_at="data_mirror/record.json")
    evidence.pairs = {"record": pair}
    assert evidence.matches("record.json", "original")
    pair["pairable"] = False
    assert not evidence.matches("record.json", "original")
    pair.update(pairable=True, published_at="data_mirror/another.json")
    assert not evidence.matches("record.json", "original")
    pair.update(published_at="data_mirror/record.json", published_sha256="wrong")
    assert not evidence.matches("record.json", "original")


def test_table_layout_and_honest_wording(audited):
    table = gen.render(audited)
    assert r"\begin{table}[tb]" in table
    assert r"\begin{tabular*}{\textwidth}" in table
    assert "Weights seen in earlier work" in table
    assert "What is new in this round" in table
    assert "no measurement on any of these weights entered the fitted selection rule" in table
    assert "new set of candidates on familiar weights" in table
    assert "outside every earlier fit" not in table
    assert "all thirty earlier revisions" not in table
    # Exact commits belong in the audit record, not in a printed table.
    for row in audited["rows"]:
        assert row["commit"] not in table
        assert gen.PRETTY[row["model"]] in table
    # Public tests remain useful when the separate manuscript checkout is absent.
    table_path = ROOT / ("paper/tables/s3_identity.tex" if (ROOT / "data_mirror").is_dir()
                         else "paper/paper/tables/s3_identity.tex")
    if table_path.exists():
        assert table_path.read_text() == table


def test_audit_matches_documented_snapshot(audited):
    assert json.loads((ROOT / "docs/s3_identity_audit.json").read_text()) == audited
