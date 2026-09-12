import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from analysis import v27_scoring_and_units as scoring


class CharTokenizer:
    name_or_path = "tiny"
    bos_token_id = 1
    truncation_side = "right"

    def __call__(self, text, add_special_tokens=False, return_tensors=None,
                 truncation=False, max_length=None, return_offsets_mapping=False):
        ids = ([1] if add_special_tokens else []) + [ord(c) + 2 for c in text]
        if truncation:
            ids = ids[:max_length]
        result = {"input_ids": torch.tensor([ids], dtype=torch.long) if return_tensors else ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class ConditionalModel(torch.nn.Module):
    def forward(self, input_ids, **kwargs):
        assert input_ids.device.type == "cpu"
        logits = torch.zeros((*input_ids.shape, 260))
        has_reasoning = (input_ids == ord("r") + 2).cumsum(dim=1) > 0
        logits[:, :, ord("y") + 2] = has_reasoning.float() * 4
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("key,sample,r,y", [
    ("math", {"completion": r" Work. \boxed{\frac{1}{2}}.", "answer": r"\frac{1}{2}"}, " Work. ", r"\boxed{\frac{1}{2}}."),
    ("math_gsm8k", {"completion": " We add 1+1.\n#### 2", "answer": "2"}, " We add 1+1.\n", "#### 2"),
    ("code", {"completion": "def f():\n    return 2\n"}, "", "def f():\n    return 2\n"),
    ("qa", {"completion": " Paris", "answer": "Paris"}, "", " Paris"),
])
def test_decomposition_preserves_information_and_marks_empty_reasoning(key, sample, r, y):
    result = scoring.decompose_probe(key, sample)
    assert result["supported"]
    assert result["r"] == r
    assert result["y"] == y
    assert result["r"] + result["y"] == sample["completion"]
    assert result["reasoning_available"] == bool(r)
    assert result["empty_reasoning_identity"] == (r == "")


@pytest.mark.parametrize("completion,answer", [("work, no final box", "2"),
                                               (r"\boxed{2}", "3"),
                                               (r"\boxed{2", "2"),
                                               (r"\boxed{2} More reasoning", "2")])
def test_unsupported_math_is_reported_instead_of_inventing_a_split(completion, answer):
    assert not scoring.decompose_probe("math", {"completion": completion, "answer": answer})["supported"]


def test_conditioning_masks_and_target_counts_chain_rule():
    tok, model = CharTokenizer(), ConditionalModel()
    ranges = scoring.prepare_ranges(tok, {"prompt": "x"}, {"r": "r", "y": "y"}, 16)
    full, direct, given = [ranges[name] for name in scoring.LOSSES]
    assert full[0] == direct[0] == [1, ord("x") + 2]
    assert given[0] == [*direct[0], ord("r") + 2]
    assert given[1] == direct[1] == full[1][-1:]
    full_nll = scoring.score_ids(model, *full[:2], "cpu")
    direct_nll = scoring.score_ids(model, *direct[:2], "cpu")
    given_nll = scoring.score_ids(model, *given[:2], "cpu")
    assert direct_nll == pytest.approx(math.log(260))
    assert given_nll == pytest.approx(math.log(259 + math.exp(4)) - 4, abs=3e-6)
    assert full_nll == pytest.approx(math.log(260) + given_nll)
    assert direct_nll > given_nll  # r is absent only in the direct condition
    with pytest.raises(ValueError, match="complete reasoning"):
        scoring.prepare_ranges(tok, {"prompt": "long x"}, {"r": "r" * 20, "y": "y"}, 10)


def test_mean_token_mean_and_utf8_byte_mean_use_example_totals():
    items = [{"sum_ce": 1.0, "n_tokens": 1, "target_bytes": 2},
             {"sum_ce": 27.0, "n_tokens": 9, "target_bytes": 10}]
    result = scoring.recompute_units(items)
    assert result["example_token_normalized"] == 2
    assert result["token_normalized"] == 2.8
    assert result["byte_normalized"] == pytest.approx(28 / 12)
    assert scoring.target_byte_count(CharTokenizer(), "éabc", scoring.encode(CharTokenizer(), "éa")) == 3
    with pytest.raises(ValueError, match="target_bytes"):
        scoring.recompute_units([{"sum_ce": 2, "n_tokens": 2}])
    with pytest.raises(ValueError, match="aggregate"):
        scoring.recompute_units([])


def test_partial_unicode_target_is_not_assigned_full_character_bytes():
    class ByteTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(text.encode()), "offset_mapping": [(0, 1), (0, 1)]}

    with pytest.raises(ValueError, match="Unicode"):
        scoring.target_byte_count(ByteTokenizer(), "é", [195])


def test_format_control_keeps_content_context_and_code_whitespace_fixed():
    probes = {"code": [{"prompt": "Task:", "completion": "def f():\n    return 2\n"}]}
    result = scoring.measure_benchmarks(ConditionalModel(), CharTokenizer(), probes, "cpu", 256)
    code = result["code"]
    metrics = code["scoring"]
    assert metrics["L_full"] == metrics["L_direct"] == metrics["L_given"]
    control = code["format_control"]
    assert control["conditioning"] == "L_full"
    contexts, content_hashes = set(), set()
    for name, variant in control["variants"].items():
        item = variant["items"][0]
        contexts.add(item["context_tokens_sha256"])
        content_hashes.add(item["information_sha256"])
        assert item["target_bytes"] == len((scoring.FORMAT_PREFIXES[name] + probes["code"][0]["completion"]).encode())
    assert len(contexts) == len(content_hashes) == 1
    assert content_hashes == {scoring._digest(probes["code"][0]["completion"])}
    assert len(control["paired_items"]) == 1
    assert code["legacy_v6"][scoring.VERSION_FIELD]["byte_units"] == "nats/UTF8-byte"


def test_all_formats_share_same_cohort_when_one_variant_exceeds_context():
    result = scoring.measure_benchmarks(ConditionalModel(), CharTokenizer(), {
        "qa": [{"prompt": "x", "completion": "y", "answer": "y"}]}, "cpu", 8)
    assert result["qa"]["scoring"]["L_full"]["status"] == "ok"
    control = result["qa"]["format_control"]
    assert all(v["status"] == "no eligible probes" for v in control["variants"].values())
    assert len(control["format_exclusions"]) == 1


def test_dry_run_and_compliance_guard_do_not_load_models_or_data(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail("dry run must not load models/data/tokenizers or touch CUDA")

    monkeypatch.setattr(scoring, "load_text_causal_lm", forbidden)
    monkeypatch.setattr(scoring, "measurement_probes", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "is_initialized", forbidden)
    scoring.main(["--model", "gemma3-270m", "--dry-run", "--prune-density", "0.9",
                  "--quant-bits", "4", "--output-base", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
    monkeypatch.delenv("SDL_ALLOW_RESTRICTED", raising=False)
    with pytest.raises(RuntimeError, match="prohibits"):
        scoring.main(["--model", "Qwen3-4B", "--dry-run"])


def test_reuse_preserves_original_units_and_reconstructs_only_scored_bytes(tmp_path, monkeypatch):
    import transformers

    sample = {"prompt": "x", "completion": " éabcdef", "answer": "2"}
    probes = {"math": [sample]}
    original = {
        "version": 23, "model": "gemma3-270m", "tokenizer": "google/gemma-3-270m",
        "protocol": {"n_probe_requested": 2, "probe_seed": 0, "max_len": 8,
                     "prompt_max_tokens": 4, "target_max_tokens": 4,
                     "probe_half": "measurement (odd indices, v[1::2])", "chat_template": False,
                     "probe_sha256": scoring._digest(probes)},
        "capabilities": {"math": {"primary": {"dataset": scoring.BENCHMARKS["math"][0],
            "L_c": 3.0, "token_weighted_L_c": 3.0,
            "items": [{"measurement_index": 0, "probe_sha256": scoring._digest(sample),
                       "sum_ce": 12, "n_tokens": 4, "loss": 3.0}]}}},
    }
    path = tmp_path / "original.json"
    path.write_text(json.dumps(original))
    before = path.read_bytes()
    monkeypatch.setattr(scoring, "measurement_probes", lambda *a: probes)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a: CharTokenizer())
    monkeypatch.setattr(scoring, "load_text_causal_lm", lambda *a: pytest.fail("must reuse stored NLL"))
    output = scoring.reuse_result(path, tmp_path / "out")
    saved = json.loads(output.read_text())
    group = saved["capabilities"]["math"]["primary"]
    units = group.pop(scoring.VERSION_FIELD)
    assert units["byte_normalized"] == pytest.approx(12 / 5)  # only ' éab', not full completion
    assert units["token_normalized"] == 3
    saved.pop("v27_recomputation")
    assert saved == original
    assert path.read_bytes() == before

    invalid = copy.deepcopy(original)
    invalid["capabilities"]["math"]["primary"]["items"][0]["n_tokens"] = 3
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="token count"):
        scoring.reuse_result(bad, tmp_path / "bad-out")


def test_reuse_stored_bytes_needs_no_tokenizer_and_rejects_aggregate_only(tmp_path, monkeypatch):
    path = tmp_path / "items.json"
    path.write_text(json.dumps({"items": [{"total_nll": 12, "n_tokens": 4, "target_bytes": 6}]}))
    monkeypatch.setattr(scoring, "measurement_probes", lambda *a: pytest.fail("stored bytes need no probes"))
    output = scoring.reuse_result(path, tmp_path / "out")
    assert json.loads(output.read_text())[scoring.VERSION_FIELD]["byte_normalized"] == 2
    path.write_text(json.dumps({"L_c": 3, "n_tokens": 4}))
    with pytest.raises(ValueError, match="aggregate"):
        scoring.reuse_result(path, tmp_path / "other")


def test_compression_and_adapter_are_applied_in_v15_order(tmp_path, monkeypatch):
    order = []
    probes = {"qa": [{"prompt": "x", "completion": "y", "answer": "y"}]}
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "google/gemma-3-270m"}))
    monkeypatch.setattr(scoring, "measurement_probes", lambda *a: probes)
    monkeypatch.setattr(scoring, "load_text_causal_lm", lambda *a: (ConditionalModel(), CharTokenizer()))

    def load(model, *a):
        order.append("adapter")
        return model

    monkeypatch.setattr(scoring, "_load_adapter", load)
    monkeypatch.setattr(scoring, "apply_global_magnitude_pruning", lambda *a, **kw: order.append("prune"))
    monkeypatch.setattr(scoring, "_apply_quantization", lambda *a: order.append("quant"))
    output = scoring.run_measurement(model_request="gemma3-270m", adapter=adapter,
                                     prune_density=0.9, quant_bits=4, device="cpu",
                                     output_base=tmp_path / "out")
    assert order == ["adapter", "prune", "quant"]
    saved = json.loads(output.read_text())
    assert saved["version"] == 27
    assert saved["metric_definitions"]["MAIN"]["name"] == "L_full"
    assert saved["benchmarks"]["qa"]["scoring"]["L_given"]["status"] == "ok"


def test_same_update_basename_in_two_runs_gets_distinct_output_paths(tmp_path):
    paths = []
    for run in ("seed0", "seed7"):
        adapter = tmp_path / run / "update-00000001" / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "google/gemma-3-270m"}))
        paths.append(scoring.run_measurement(model_request="gemma3-270m", adapter=adapter,
                                              output_base=tmp_path / "out", dry_run=True))
    assert paths[0] != paths[1]
    assert not (tmp_path / "out").exists()
