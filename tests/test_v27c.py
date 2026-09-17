"""Saved-measurement integrity and unit/refit tests; no model runs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from analysis import v27c_measurement_followups as followup
from public_inputs import require_public_inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def saved():
    return json.loads((ROOT / "results/v27c/summary.json").read_text())


def synthetic_arm(arm):
    rows = []
    coordinates = followup.prediction.PRUNE_DEV if arm == "pruning" else (*followup.prediction.QUANT_DEV, 5)
    for i in range(5):
        rows.append({"model": f"model-{i}", "family": "a" if i < 3 else "b", "N0": (i + 1) * 1e9,
                     "capability": "code", "dense_loss": 1 + i * .2,
                     "deltas": {str(c): (.1 + i * .2) * followup.prediction.shape(arm, c, 2.) for c in coordinates}})
    coordinates = followup.prediction.PRUNE_TEST if arm == "pruning" else followup.prediction.QUANT_TEST
    manifest = {"folds": [{"held_out": r["model"], "fit": {"mapping": {
        "train_models": [t["model"] for t in rows if t["model"] != r["model"]]}}} for r in rows],
        "records": [{"model": r["model"], "coordinate": c} for r in rows for c in coordinates]}
    return rows, manifest


def factors_for(rows, values):
    return {r["model"]: {r["capability"]: {"token_to_byte_factor": v}}
            for r, v in zip(rows, values, strict=True)}


def test_exact_unicode_truncated_denominator_uses_scored_prefix_only():
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "é": 1, "aa": 2, "tail": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    ids, prefix = followup.scored_prefix(tokenizer, "é aa tail", limit=2)
    assert ids == [1, 2]
    assert prefix == "é aa"
    assert len(prefix.encode("utf-8")) == 5
    assert followup.scored_prefix(tokenizer, "é aa tail")[1] == "é aa tail"


def test_refuse_partial_unicode_character_at_token_cap():
    tokenizer = SimpleNamespace(encode=lambda *a, **k: SimpleNamespace(ids=[1, 2, 3], offsets=[(0, 1), (0, 1), (1, 2)]))
    with pytest.raises(ValueError, match="Unicode"):
        followup.scored_prefix(tokenizer, "éx", limit=1)


def test_corpus_conversion_is_not_equal_example_conversion():
    # Unequal lengths make both normalization and aggregation errors visible.
    items = [{"sum_ce": 2., "n_tokens": 1, "target_bytes": 2},
             {"sum_ce": 30., "n_tokens": 10, "target_bytes": 50}]
    units = followup.readout.recompute(items)
    assert units["example_token"] == 2.5
    assert units["corpus_token"] == pytest.approx(32 / 11)
    assert units["corpus_byte"] == pytest.approx(units["corpus_token"] * 11 / 52)
    assert units["example_byte"] != pytest.approx(units["example_token"] * 11 / 52)


def test_all_empty_reasoning_records_identical_and_samples_have_exact_spans(saved):
    audit = saved["empty_reasoning"]
    assert audit["n_items"] == 768 and audit["unequal_items"] == 0
    samples = [s for p in audit["panels"] for s in p["samples"]]
    assert len(samples) == 3
    for s in samples:
        assert s["r"] == "" and s["context_hash_verified"]
        a, b, c = (s["losses"][loss] for loss in followup.readout.LOSSES)
        assert a == b == c
        assert a["scored_text"] == s["y"]
        assert a["target_bytes"] == len(s["y"].encode("utf-8"))
        n = len(a["prompt_token_ids"])
        assert a["input_target_span_half_open"] == [n, n + a["n_tokens"]]
        assert a["logit_span_half_open"] == [n - 1, n + a["n_tokens"] - 1]
        assert a["loss_nats_per_token"] == a["sum_nll_nats"] / a["n_tokens"]
        panel = json.loads((ROOT / s["source"]).read_text())
        item = panel["benchmarks"][s["benchmark"]]["format_control"]["variants"]["canonical"]["items"][s["measurement_index"]]
        assert followup.readout.digest(a["prompt_token_ids"]) == item["context_tokens_sha256"]
    code = audit["panels"][0]["groups"][0]["losses"]["L_full"]
    assert code["example_token"] == pytest.approx(2.5405926364405333)
    assert code["corpus_token"] == pytest.approx(2.2899692694205327)


def test_empty_reasoning_difference_is_rejected():
    key = (0, "probe")
    item = {"sum_ce": 4., "n_tokens": 2, "target_bytes": 3, "target_sha256": "target"}
    maps = {loss: {key: dict(item)} for loss in followup.readout.LOSSES}
    maps["L_given"][key]["sum_ce"] = 5.
    with pytest.raises(ValueError, match="Empty reasoning"):
        followup.readout.regions(maps, {key: {"r": ""}})


@pytest.mark.parametrize("arm", ["pruning", "quantization"])
def test_byte_fits_exclude_all_held_out_outcomes_and_keep_calibration_disjoint(arm):
    rows, manifest = synthetic_arm(arm)
    factors = factors_for(rows, [.2, .3, .4, .5, .6])
    original = copy.deepcopy(rows)
    byte_rows = followup.convert_rows(rows, factors)
    assert rows == original
    assert byte_rows[0]["dense_loss"] == rows[0]["dense_loss"] * .2
    before = followup.replay_arm(byte_rows, arm, manifest)
    changed = copy.deepcopy(byte_rows)
    changed[0]["deltas"] = {key: 999. for key in changed[0]["deltas"]}
    after = followup.replay_arm(changed, arm, manifest)
    assert before["folds"][0] == after["folds"][0]
    a = [r for r in before["records"] if r["model"] == rows[0]["model"]]
    b = [r for r in after["records"] if r["model"] == rows[0]["model"]]
    assert [r["A"] for r in a] == [r["A"] for r in b]
    assert [r["B"] for r in a] != [r["B"] for r in b]
    assert all(r["coordinate"] != followup.prediction.CALIBRATION[arm] for r in a)
    wrong = copy.deepcopy(manifest)
    wrong["folds"][0]["fit"]["mapping"]["train_models"].append(rows[0]["model"])
    with pytest.raises(ValueError, match="membership"):
        followup.replay_arm(byte_rows, arm, wrong)


def test_common_scale_is_pure_rescaling_but_model_dependent_units_require_refit():
    rows, manifest = synthetic_arm("quantization")
    token = followup.replay_arm(rows, "quantization", manifest)
    uniform = followup.replay_arm(followup.convert_rows(rows, factors_for(rows, [.3] * len(rows))), "quantization", manifest)
    for t, b in zip(token["records"], uniform["records"], strict=True):
        for key in (*followup.MODES, "observed_delta"):
            assert b[key] == pytest.approx(t[key] * .3)
    varying = factors_for(rows, [.2, .25, .3, .4, .5])
    byte = followup.replay_arm(followup.convert_rows(rows, varying), "quantization", manifest)
    assert any(abs(b["A"] / varying[b["model"]]["code"]["token_to_byte_factor"] - t["A"]) > .01
               for t, b in zip(token["records"], byte["records"], strict=True))


def test_distillation_test_outcomes_do_not_enter_either_prediction():
    rows = [{"model": m, "D": d, "capability": "code", "row_id": f"{m}|{d}|code", "observed": .01 * (i + 1) * d}
            for i, m in enumerate(("m1", "m2", "m3")) for d in (75, 150, 300)]
    manifest = {"folds": [{"held_out": "m1", "fit": {"train_row_ids": [r["row_id"] for r in rows if r["model"] != "m1"]},
                           "mode_B_calibration_row_id": "m1|150|code"}],
                "records": [{"model": "m1", "coordinate": d} for d in (75, 300)]}
    before = followup.replay_distillation(rows, "code", manifest)
    changed = copy.deepcopy(rows)
    changed[0]["observed"] = -100
    changed[2]["observed"] = 100
    after = followup.replay_distillation(changed, "code", manifest)
    assert before["folds"] == after["folds"]
    assert [[r[k] for k in ("A", "B")] for r in before["records"]] == [[r[k] for k in ("A", "B")] for r in after["records"]]


def test_every_real_fold_and_mae_matches_frozen_token_result(saved):
    frozen = json.loads((ROOT / "results/v28-new-source-pred/frozen_predictions.json").read_text())
    assert saved["main_metric"]["frozen"] and saved["main_metric"]["unit"] == "nats/token"
    for arm, caps in saved["prediction_sensitivity"].items():
        for cap, result in caps.items():
            original = frozen["methods"][arm]["by_capability"][cap]["lomo"]
            assert result["token_replay_max_abs_error"] < 1e-12
            assert result["n_test_cells"] == {"pruning": 36, "quantization": 24, "distillation": 8}[arm]
            for token, byte, old in zip(result["token"]["folds"], result["byte"]["folds"], original["folds"], strict=True):
                assert token["held_out"] == byte["held_out"] == old["held_out"]
                if arm == "distillation":
                    assert token["fit"]["train_row_ids"] == byte["fit"]["train_row_ids"] == old["fit"]["train_row_ids"]
                else:
                    members = byte["fit"]["mapping"]["train_models"]
                    assert members == old["fit"]["mapping"]["train_models"]
                    assert byte["held_out"] not in members
            for mode, metrics in result["metrics"].items():
                assert metrics["token_mae"] == pytest.approx(original["metrics"][mode]["mae"])
                records = result["byte"]["records"]
                assert metrics["byte_mae"] == pytest.approx(sum(abs(r[mode] - r["observed_delta"]) for r in records) / len(records))
    code = saved["prediction_sensitivity"]["pruning"]["code"]["metrics"]["A"]
    assert code["material_refit_change"]
    assert code["refit_relative_mae_change_in_token_units"] == pytest.approx(.1448676768)
    for cap in followup.CAPS:
        for mode in ("A", "B"):
            r = saved["prediction_sensitivity"]["distillation"][cap]["metrics"][mode]
            assert r["byte_refit_mae_in_token_units"] == pytest.approx(r["token_mae"])


def test_mismatched_prediction_cells_or_stale_native_values_are_rejected(saved):
    r = saved["prediction_sensitivity"]["quantization"]["code"]
    original = {**r["token"], "metrics": {k: {"mae": v["token_mae"]} for k, v in r["metrics"].items()}}
    changed = copy.deepcopy(r["byte"])
    changed["records"].pop()
    with pytest.raises(ValueError, match="score cells"):
        followup.compare_predictions(r["token"], changed, original, "code", saved["denominators"])
    original["records"] = copy.deepcopy(original["records"])
    original["records"][0]["A"] += 1
    with pytest.raises(ValueError, match="replay differs"):
        followup.compare_predictions(r["token"], r["byte"], original, "code", saved["denominators"])


def test_report_and_source_provenance_are_current(saved):
    assert followup.report(saved) == (ROOT / "docs/MEASUREMENT_FOLLOWUPS.md").read_text()
    for source in saved["sources"]:
        if source["path"].startswith("HF_CACHE/"):
            continue
        assert hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest() == source["sha256"]
    assert saved["provenance"]["reference"] == {"pruning": "SOURCE own dense", "quantization": "SOURCE own dense", "distillation": "STUDENT own dense S0"}
    for factors in saved["denominators"].values():
        for cap in followup.CAPS:
            d = factors[cap]
            assert len(d["items"]) == 64
            assert d["sum_tokens"] == sum(i["n_tokens"] for i in d["items"])
            assert d["sum_bytes"] == sum(i["target_bytes"] for i in d["items"])


def test_full_rebuild_is_deterministic_cpu_only_and_read_only():
    saved = json.loads((ROOT / "results/v27c/summary.json").read_text())
    require_public_inputs(
        *(s["path"].removeprefix("HF_CACHE/") for s in saved["sources"]
          if s["path"].startswith("HF_CACHE/")),
        root=Path.home() / ".cache/huggingface",
        reason="external Hugging Face dataset/tokenizer cache is not redistributed; "
               "the 2Wiki Arrow input is 54,510,136 bytes, exceeding the 50 MB file limit")
    # Run once behind hard guards: importing model/dataset loaders or opening a
    # network connection is a failure, even if a local cached fallback exists.
    program = r'''
import importlib.abc, json, socket, sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'datasets', 'huggingface_hub'}:
            raise AssertionError('Forbidden model/network loader: ' + fullname)
sys.meta_path.insert(0, BlockModels())
def no_network(*args, **kwargs):
    raise AssertionError('Network is forbidden')
socket.socket.connect = no_network
from analysis import v27c_measurement_followups as module
expected = json.loads((module.ROOT / 'results/v27c/summary.json').read_text())
actual = module.build_summary()
assert actual == expected
'''
    paths = [ROOT / "results/v27c/summary.json", ROOT / "docs/MEASUREMENT_FOLLOWUPS.md",
             ROOT / "results/v27b-readout/summary.json", ROOT / "results/v28-new-source-pred/frozen_predictions.json"]
    before = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
    completed = subprocess.run([sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths} == before
