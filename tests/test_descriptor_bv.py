import json
import subprocess
import sys

import pytest
import torch

from analysis import descriptor_bv as bv
from analysis.tiny_test_model import TinyModel, TinyTokenizer, PROBES
from analysis.descriptor_bv_selftest import derivative_checks
from analysis import v12_distill as distill
from analysis.eval_records import aggregate_records


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_central_difference_shrinkage_and_hessian_trace():
    result = derivative_checks()
    assert result["B_relative_error"] < 1e-4
    assert result["V_relative_error"] < 2e-3
    assert result["V_finite_difference_relative_error"] < 2e-3
    # Explicitly expose the contradictory minus sign in the requested test.
    assert result["B"] == pytest.approx(-result["negative_dCE_d_shrinkage"], rel=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_float32_probability_reductions_and_final_softcap(dtype, monkeypatch):
    model, tok = TinyModel(dtype), TinyTokenizer()
    seen = []
    original = torch.softmax

    def softmax(x, *args, **kwargs):
        assert x.shape[0] <= 2  # vocabulary-sized reductions stay chunk-bounded
        seen.append((x.dtype, kwargs["dtype"]))
        return original(x, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", softmax)
    sample = PROBES["math"][1]
    row = bv.score_sample(model, tok, sample, "math", 0, "j", chunk_tokens=2)
    assert seen and all(pair == (torch.float32, torch.float32) for pair in seen)
    capture = bv._CaptureFinal(model)
    _, n = bv.completion_loss(capture, tok, sample["prompt"], sample["completion"], "cpu")
    z = capture.logits[0, -n-1:-1].detach().double()
    p = original(z, -1)
    targets = capture.ids[0, -n:]
    expected_b = (z.gather(-1, targets[:, None]).squeeze(-1) - (p*z).sum(-1)).mean()
    expected_v = (1 - p.square().sum(-1)).mean()
    assert row["B"] == pytest.approx(float(expected_b), rel=2e-6)
    assert row["V"] == pytest.approx(float(expected_v), rel=2e-6)
    assert capture.logits.abs().max() <= 1.7
    assert row["forward_logits_dtype"] == str(dtype)


def test_aggregation_matches_actual_distillation_and_unequal_lengths():
    model, tok = TinyModel(), TinyTokenizer()
    payload = bv.describe(model, tok, PROBES, "j", model_id="tiny", revision="87")
    own, counts = distill.measure_capability_losses(model, tok, PROBES, "cpu")
    for cap in own:
        assert payload["aggregates"][cap]["j"]["L"] == own[cap]
        assert payload["aggregates"][cap]["j"]["scored_token_count"] == counts[cap]
    rows = payload["per_sample"][:2]
    assert rows[0]["scored_token_count"] != rows[1]["scored_token_count"]
    assert aggregate_records(rows)["math"] != pytest.approx(sum(r["L"] for r in rows) / 2)
    for name in ("B", "V", "L"):
        key = {"B": "B_sum", "V": "V_sum", "L": "summed_nll"}[name]
        assert payload["aggregates"]["math"]["j"][name] == aggregate_records(rows, key)["math"]
        assert payload["aggregates"]["math"]["j"][name + "_per_byte"] == sum(r[key] for r in rows) / sum(r["scored_reference_byte_count"] for r in rows)
    assert payload["per_sample"][2]["scored_reference_byte_count"] == 3
    assert all(not isinstance(value, torch.Tensor) for row in payload["per_sample"] for value in row.values())


def test_legacy_truncation_gemma_bos_empty_reference_and_distribution():
    tok = TinyTokenizer()
    tok.name_or_path = "gemma-without-auto-bos"
    encode = tok.encode
    tok.encode = lambda text, add_special_tokens=False: encode(text, False)
    probes = {"math": [{"prompt": "abcdefghijkl", "completion": "mnopq", "distribution": "shifted"},
                       {"prompt": "a", "completion": ""}]}
    model = TinyModel()
    result = bv.describe(model, tok, probes, "default", model_id="tiny", revision="87", max_len=8)
    first, empty = result["per_sample"]
    assert first["scored_token_count"] == 4
    assert first["reference_byte_count"] == 5
    assert first["scored_reference_byte_count"] == 4
    loss, n = bv.completion_loss(model, tok, probes["math"][0]["prompt"], "mnopq", "cpu", max_len=8)
    assert first["summed_nll"] == float(loss.detach()) and n == first["scored_token_count"]
    assert empty["scored_token_count"] == 0 and empty["B"] is None
    assert set(result["aggregates"]["math"]) == {"default", "shifted"}


def test_unprovable_scored_bytes_do_not_change_cohort_or_use_full_reference():
    tok = TinyTokenizer()
    encode = tok.encode
    tok.encode = lambda text, add_special_tokens=False: encode(text, False)
    probes = {"qa": [{"prompt": "", "completion": "ab"},
                     {"prompt": "x", "completion": "cd"}]}
    result = bv.describe(TinyModel(), tok, probes, "j", model_id="tiny", revision="87")
    first = result["per_sample"][0]
    assert first["scored_token_count"] == 1  # V6 cannot score the first token without context
    assert first["reference_byte_count"] == 2
    assert first["scored_reference_byte_count"] is None and first["byte_count_error"]
    aggregate = result["aggregates"]["qa"]["j"]
    assert aggregate["n_scored_samples"] == 2
    assert aggregate["L_per_byte"] is None and aggregate["B_per_byte"] is None
    assert aggregate["B"] == aggregate_records(result["per_sample"], "B_sum")["qa"]


def test_cli_selftest_offline_and_cpu_default(tmp_path):
    completed = subprocess.run([sys.executable, "-m", "analysis.descriptor_bv", "--selftest",
                                "--output-dir", str(tmp_path)], check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / "descriptor_selftest.json").read_text())
    assert result["device"] == "cpu" and result["selftest"]["status"] == "passed"
    assert json.loads(completed.stdout)["input_hashes"]["probes_sha256"]


def test_cli_rejects_non_cpu_before_loading():
    result = subprocess.run([sys.executable, "-m", "analysis.descriptor_bv", "--device", "cuda:0"],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "CPU only" in result.stderr
