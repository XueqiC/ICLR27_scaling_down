"""Offline CPU coverage of the frozen E5 protocol; never load a checkpoint."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis import r1_accuracy_efficiency as r1


def pools(n=400):
    return {
        "math": [{"prompt": f"Problem: arithmetic item {i}\nSolution:", "completion": " #### 7", "answer": "7"} for i in range(n)],
        "code": [{"prompt": f"# Task: code item {i}\n# Write a Python function.\n",
                  "completion": "def f():\n    return 7", "test_list": ["assert f() == 7"],
                  "test_setup_code": ""} for i in range(n)],
        "qa": [{"prompt": f"Question: trivia item {i}?\nAnswer:", "completion": " Paris",
                "answer": "City of Paris", "answer_aliases": ["City of Paris", "Paris"]} for i in range(n)],
    }


def freeze(out, plan):
    r1.atomic_write(out / "plan.json", plan)
    (out / "plan.sha256").write_text(r1.digest(out / "plan.json") + "\n")


def test_plan_determinism_disjointness_and_hash_tampering():
    candidates = pools()
    exclusions = {"v6_both_halves": sorted(r1.text_hashes("Problem: arithmetic item 0\nSolution:")),
                  "v6_secondary": sorted(r1.text_hashes("arithmetic item 1")),
                  "v9_registry": sorted(r1.text_hashes("Task: code item 2")),
                  "teacher_trace": sorted(r1.text_hashes("Context:\nold\n\nQuestion: trivia item 3?\nAnswer:")),
                  "v15_fewshot": sorted(r1.text_hashes("here is your task: code item 4 Your code should pass these tests:"))}
    # Same question with changed wrapper/context must also be excluded.
    candidates["qa"].append({**candidates["qa"][3], "prompt": "Context:\nnew\n\nQuestion: trivia item 3?\nAnswer:"})
    first = r1.build_plan(candidates, exclusions)
    second = r1.build_plan({c: list(reversed(v)) for c, v in candidates.items()}, exclusions)
    assert first == second
    assert first["seed"] == 0
    forbidden = set().union(*(set(x) for x in exclusions.values()))
    for cap in r1.CAPS:
        assert len(first["items"][cap]) == 256
        assert len({row["id"] for row in first["items"][cap]}) == 256
        assert all(not set(row["text_hashes"]) & forbidden for row in first["items"][cap])
    assert first["protocol"]["shots"] == {"math": 4, "code": 3, "qa": 4}
    assert first["protocol"]["max_new_tokens"] == {"math": 512, "code": 384, "qa": 32}
    broken = deepcopy(first)
    broken["items"]["math"][0]["prompt"] = "different"
    with pytest.raises(ValueError, match="hash mismatch"):
        r1.validate_plan(broken)
    with pytest.raises(ValueError, match="only .* disjoint"):
        r1.build_plan(pools(10), exclusions)


def test_exclusion_registry_covers_secondary_both_halves_and_teacher(tmp_path, monkeypatch):
    calls = []

    def v6(n, seed, include_secondary):
        calls.append(("v6", n, seed, include_secondary))
        return {k: [{"prompt": f"{k} probe {i}"} for i in range(n)]
                for k in (*r1.CAPS, *r1.v6.SECONDARY_BENCHMARKS)}

    def v9(n, seed):
        calls.append(("v9", n, seed))
        return {s.name: [{"prompt": f"registry {s.name} {i}"} for i in range(n)] for s in r1.v9.PROBE_REGISTRY}

    def v15(n, seed):
        calls.append(("v15", n, seed))
        return pools(n)

    monkeypatch.setattr(r1.v6, "build_probes", v6)
    monkeypatch.setattr(r1.v9, "build_probe_registry", v9)
    monkeypatch.setattr(r1.v15, "build_easy_probes", v15)
    (tmp_path / "data/teacher").mkdir(parents=True)
    trace = tmp_path / "data/teacher/run.jsonl"
    trace.write_text(json.dumps({"messages": [{"role": "user", "content": "Question: traced question\nAnswer:"},
                                              {"role": "assistant", "content": "not a query"}]}) + "\n")
    sources, files = r1.exclusion_sources(tmp_path)
    assert calls == [("v6", 128, 0, True), ("v9", 64, 0), ("v15", 128, 0)]
    for key in (*r1.CAPS, *r1.v6.SECONDARY_BENCHMARKS):
        hashes = sources[f"v6_128_seed0/{key}/both_halves"]
        assert r1.text_hashes(f"{key} probe 0") <= set(hashes)
        assert r1.text_hashes(f"{key} probe 127") <= set(hashes)
    assert r1.text_hashes("traced question") <= set(sources["data/teacher/run.jsonl"])
    assert not r1.text_hashes("not a query") & set(sources["data/teacher/run.jsonl"])
    assert files["data/teacher/run.jsonl"]["sha256"] == r1.digest(trace)
    monkeypatch.setattr(r1.v9, "build_probe_registry", lambda *a, **kw: {})
    with pytest.raises(ValueError, match="Incomplete V9"):
        r1.exclusion_sources(tmp_path)


def test_plan_stage_freezes_once_without_dataset_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(r1, "exclusion_sources", lambda: ({"fixture": []}, {}))
    calls = []

    def loader(n, seed):
        calls.append((n, seed))
        return pools()

    monkeypatch.setattr(r1.v15, "build_easy_probes", loader)
    r1.plan_stage(tmp_path)
    original = (tmp_path / "plan.json").read_bytes()
    r1.plan_stage(tmp_path)
    assert len(calls) == 1 and calls[0] == (r1.sys.maxsize, 0)
    assert (tmp_path / "plan.json").read_bytes() == original
    assert (tmp_path / "plan.sha256").read_text().strip() == hashlib.sha256(original).hexdigest()
    (tmp_path / "plan.json").write_bytes(original + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        r1.load_plan(tmp_path)


class TinyTokenizer:
    pad_token_id, eos_token_id, bos_token_id = 0, 1, 2
    padding_side, truncation_side = "right", "right"

    def encode(self, text, add_special_tokens=False):
        return ([2] if add_special_tokens else []) + [ord(c)+3 for c in text]

    def batch_decode(self, ids, **kwargs):
        return ["".join(chr(int(i)-3) for i in row if int(i) > 2) for row in ids]

    def __call__(self, texts, *, return_tensors, padding, truncation, max_length, add_special_tokens):
        assert self.padding_side == self.truncation_side == "left"
        rows = [self.encode(text, add_special_tokens)[-max_length:] for text in texts]
        width = max(map(len, rows))
        return {"input_ids": torch.tensor([[0]*(width-len(r))+r for r in rows]),
                "attention_mask": torch.tensor([[0]*(width-len(r))+[1]*len(r) for r in rows])}


class TinyModel(torch.nn.Module):
    """Small matrix model with deterministic scripted greedy emissions."""
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False)])
        with torch.no_grad():
            self.model.layers[0].weight.copy_(torch.arange(1, 17).reshape(4, 4)/16)

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample, use_cache,
                 eos_token_id, pad_token_id, stopping_criteria=None):
        assert not do_sample and use_cache
        tok = TinyTokenizer()
        texts = tok.batch_decode(input_ids)
        outputs = []
        for text in texts:
            if "[BEGIN]" in text:
                answer = "def f():\n    return 7\n[DONE]"
            elif "Problem:" in text:
                answer = "#### 7"
            else:
                answer = "Paris\nQuestion:"
            outputs.append((tok.encode(answer)+[eos_token_id])[:max_new_tokens])
        seq = input_ids.clone()
        finished = torch.zeros(len(outputs), dtype=torch.bool)
        for step in range(max(map(len, outputs))):
            nxt = torch.tensor([pad_token_id if finished[i] else row[step] for i, row in enumerate(outputs)])
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            finished |= nxt == eos_token_id
            if stopping_criteria is not None:
                finished |= stopping_criteria(seq, None)
            if finished.all():
                break
        return seq


def test_tiny_generation_v15_scoring_and_exact_token_counts():
    plan = r1.build_plan(pools(8), {"none": []}, n=4)
    model, tok = TinyModel(), TinyTokenizer()
    for cap in r1.CAPS:
        rows = plan["items"][cap][:2]
        results = r1.generate_batch(model, tok, rows, cap, "cpu")
        assert all(r1.score_item(cap, row, gen)["correct"] for row, gen in zip(rows, results))
        assert all(g["generated_tokens"] == len(g["generated_token_ids"]) for g in results)
        if cap == "math":
            assert results[0]["generated_token_ids"][-1] == tok.eos_token_id
        else:
            assert results[0]["generated_token_ids"][-1] != tok.eos_token_id
    stop = tok.encode("[DONE]")
    assert r1.generated_length([10, *stop, 0, 0], tok, ("[DONE]",)) == 1+len(stop)
    assert r1.generated_length([10, 1, 0, 0], tok, ()) == 2
    audit = r1.agreement_check(model, tok, plan, "cpu", 2)
    assert audit["n"] == 8 and audit["agreement_rate"] == 1.0


def test_agreement_differences_are_recorded_not_fatal(monkeypatch, capsys):
    plan = r1.build_plan(pools(8), {"none": []}, n=4)
    original = r1.generate_batch

    def altered(model, tok, rows, cap, device):
        result = original(model, tok, rows, cap, device)
        if len(rows) == 1 and cap == "qa":
            result[0]["raw_generation_full"] += " different"
        return result

    monkeypatch.setattr(r1, "generate_batch", altered)
    audit = r1.agreement_check(TinyModel(), TinyTokenizer(), plan, "cpu", 8)
    assert audit["agreement_rate"] == 0.75
    assert len([r for r in audit["items"] if not r["agrees"]]) == 2
    assert "6/8" in capsys.readouterr().out


def test_measurement_resume_schema_pilot_and_pruning(tmp_path, monkeypatch):
    plan = r1.build_plan(pools(8), {"none": []}, n=4)
    freeze(tmp_path, plan)
    loads, models = [], []

    def loader(repo, dtype, revision):
        assert dtype == torch.bfloat16
        loads.append((repo, revision))
        model = TinyModel()
        models.append(model)
        return model, TinyTokenizer()

    monkeypatch.setattr(r1.v6, "load_text_causal_lm", loader)
    tag = next(iter(r1.STATES))
    real_generate = r1.generate_batch
    calls = []

    def interrupted(model, tok, rows, cap, device):
        calls.extend(row["id"] for row in rows)
        if len(calls) > 2:
            raise RuntimeError("simulated interruption")
        return real_generate(model, tok, rows, cap, device)

    monkeypatch.setattr(r1, "generate_batch", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        r1.measure_state(tag, out=tmp_path, device="cpu", batch_size=2, dense_only=True)
    path = r1.measurement_path(tmp_path, tag, 1.0)
    partial = r1.read(path)
    assert partial["status"] == "in_progress"
    assert len(partial["items"]["math"]) == 2
    monkeypatch.setattr(r1, "generate_batch", real_generate)
    r1.measure_state(tag, out=tmp_path, device="cpu", batch_size=3, dense_only=True)
    complete = r1.read(path)
    assert complete["items"]["math"][:2] == partial["items"]["math"]
    assert complete["accuracy"] == dict.fromkeys(r1.CAPS, 1.0)
    assert complete["batch_agreement"]["n"] == 8
    assert len(complete["load_run_ids"]) == 2
    assert complete["gpu_seconds"] == 0 and complete["seconds"] > 0
    sha = r1.digest(tmp_path / "plan.json")
    r1.validate_measurement(complete, plan, sha, tag, 1.0, complete=True)
    before = len(loads)
    r1.measure_state(tag, out=tmp_path, device="cpu", dense_only=True)
    assert len(loads) == before
    with pytest.raises(ValueError, match="Pilot requires every dense"):
        r1.measure_state(tag, out=tmp_path, device="cpu")
    assert len(loads) == before
    for other, spec in plan["states"].items():
        if other == tag or spec["role"] != "dev":
            continue
        record = deepcopy(complete)
        record.update(state=other, **{k: spec[k] for k in ("size", "repo", "revision", "role")})
        r1.atomic_write(r1.measurement_path(tmp_path, other, 1.0), record)
    applied = []
    real_prune = r1.v6.apply_global_magnitude_pruning

    def prune(model, d, **kwargs):
        applied.append(d)
        return real_prune(model, d, **kwargs)

    monkeypatch.setattr(r1.v6, "apply_global_magnitude_pruning", prune)
    r1.measure_state(tag, out=tmp_path, device="cpu", batch_size=4)
    assert applied == list(r1.FULL)
    for d in r1.FULL:
        record = r1.read(r1.measurement_path(tmp_path, tag, d))
        r1.validate_measurement(record, plan, sha, tag, d, complete=True)
        assert record["threshold"] > 0
        assert record["generated_tokens"] == sum(it["generated_tokens"] for rows in record["items"].values() for it in rows)
    before = len(loads)
    r1.measure_state(tag, out=tmp_path, device="cpu")
    assert len(loads) == before
    bad = deepcopy(complete)
    bad["items"]["code"][0]["id"] = "bad"
    with pytest.raises(ValueError, match="schema/order mismatch"):
        r1.validate_measurement(bad, plan, sha, tag, 1.0)


def synthetic_data(n=64):
    states, loss_states = {}, {}
    for i, (tag, (size, revision, role)) in enumerate(r1.STATES.items()):
        n0 = 1e9 if size == "1B" else 7e9
        d0 = r1.a18.tokens_of(revision)
        densities = r1.FULL if role == "dev" else r1.TEST
        loss = {"N0_block_matrices": n0, "D0": d0, "densities": list(densities), "measurements": {}}
        states[tag] = {}
        for d in (1.0, *densities):
            record = {"role": role, "size": size, "N0_block_matrices": n0, "D0": d0,
                      "items": {}, "gpu_seconds": 5.0, "generated_tokens": 100,
                      "batch_agreement": {"generated_tokens": 10} if d == 1 else None}
            loss["measurements"][str(d)] = {"probes": {}}
            for j, cap in enumerate(r1.CAPS):
                a0 = 0.4+0.02*i+0.02*j
                if cap == "qa" and size == "1B":
                    a0 = 0.0  # Exactly the registered capability-specific pilot rule.
                acc = max(0, a0-(0.07+0.005*i)*float(r1.a18.p53.shape(d, 2)))
                flags = np.arange(n) < round(acc*n)
                record["items"][cap] = [{"id": f"{cap}:{k}", "correct": bool(f)} for k, f in enumerate(flags)]
                loss["measurements"][str(d)]["probes"][cap] = {"mean_loss": 2.5-0.08*i+0.1*j+0.2*float(r1.a18.p53.shape(d, 2))}
            states[tag][str(d)] = record
        loss_states[tag] = loss
    return states, loss_states


@pytest.fixture(scope="module")
def analysed():
    states, loss = synthetic_data()
    return states, loss, r1.analyse_data(states, loss, n_resamples=100)


def test_analysis_predictor_shapes_grids_and_pilot(analysed):
    states, losses, result = analysed
    assert result["pilot_seven_b_only"] == {"math": False, "code": False, "qa": True}
    dev = {t for t, (_, _, role) in r1.STATES.items() if role == "dev"}
    test = set(r1.STATES)-dev
    for cap in r1.CAPS:
        nstates = 3 if cap == "qa" else 6
        for grid, densities in (("reduced", r1.REDUCED), ("full", r1.FULL)):
            fit = result["fits"][cap][grid]
            assert len(fit["training_rows"]) == nstates*len(densities)
            assert {r["d"] for r in fit["training_rows"]} == set(densities)
            assert {r["tag"] for r in fit["training_rows"]} <= dev
            assert not {r["tag"] for r in fit["training_rows"]} & test
            assert np.shape(fit["compact"]["beta"]) == (4,)
            assert fit["compact"]["lambda"] in r1.a18.LAMBDA_GRID
            assert fit["compact"]["gamma"] in r1.a18.p53.GAMMA_GRID
            assert all(len(beta) == 4 for beta in fit["per_density"]["anchors"].values())
            assert set(result["mae_accuracy"][cap][grid]) == set(r1.METHODS)
            assert {p["d"] for p in fit["loss_route"]["label_pairs"]} == {1.0, *r1.REDUCED}
            assert len(fit["loss_route"]["label_pairs"]) == nstates*3
        assert len(result["test_cells"][cap]) == (6 if cap == "qa" else 12)
        for cell in result["test_cells"][cap]:
            assert cell["tag"] in test
            for values in cell["predictions"].values():
                assert np.shape(list(values.values())) == (6,)
                assert all(np.isfinite(list(values.values())))
                assert values["zero"] == 0
            assert cell["predictions"]["full"]["loss_route"] == cell["predictions"]["reduced"]["loss_route"]
        assert result["mae_accuracy_points"][cap]["reduced"]["compact"] == pytest.approx(100*result["mae_accuracy"][cap]["reduced"]["compact"])
    assert result["primary"]["threshold_accuracy"] == 0.03
    # Altering held-out compressed losses cannot affect any prediction.
    cell = result["test_cells"]["math"][0]
    rows = r1.accuracy_rows(states, role="test", cap="math")
    row = next(r for r in rows if r["tag"] == cell["tag"] and r["d"] == cell["d"])
    changed = deepcopy(losses[row["tag"]])
    for d in r1.TEST:
        changed["measurements"][str(d)] = {"DO_NOT_READ": True}
    assert r1.predict_six(result["fits"]["math"]["reduced"], row, changed) == cell["predictions"]["reduced"]


def test_bootstrap_determinism_pairing_and_fixed_states(analysed):
    states, _, result = analysed
    first = r1.item_bootstrap(states, result["test_cells"])
    second = r1.item_bootstrap(states, result["test_cells"])
    assert first == second
    assert first["n_resamples"] == 2000 and first["seed"] == 0
    assert first["states_fixed"] and first["predictions_fixed"]
    assert first != r1.item_bootstrap(states, result["test_cells"], seed=9)
    for cap in r1.CAPS:
        difference = first["capabilities"][cap]["difference"]
        mae = result["mae_accuracy"][cap]
        assert difference["estimate"] == pytest.approx(mae["reduced"]["compact"]-mae["full"]["per_density"])
    # Identical predictions produce identically zero paired differences.
    cells = deepcopy(result["test_cells"])
    for cap in r1.CAPS:
        for cell in cells[cap]:
            cell["predictions"]["full"]["per_density"] = cell["predictions"]["reduced"]["compact"]
    same = r1.item_bootstrap(states, cells, n_resamples=20)
    assert all(same["capabilities"][c]["difference"]["ci95"] == [0.0, 0.0] for c in r1.CAPS)


def test_costs_include_dense_loading_and_table_house_style(analysed, tmp_path):
    states, _, result = analysed
    loads = {t: [{"load_seconds": 2.0, "device": "cuda:0"}, {"load_seconds": 3.0, "device": "cuda:0"}] for t in states}
    cost = r1.costs(states, loads)
    assert cost["reduced"]["compressed_configurations"] == 12
    assert cost["full"]["compressed_configurations"] == 24
    assert cost["reduced"]["dense_anchors"] == cost["full"]["dense_anchors"] == 6
    assert cost["reduced"]["gpu_seconds"] == 6*(5+3*5)
    assert cost["full"]["gpu_seconds"] == 6*(5+5*5)
    assert cost["reduced"]["generated_tokens_including_agreement"] == 6*(3*100+10)
    table = r1.render_table({**result, "costs": cost}, tmp_path / "table.tex").read_text()
    assert table.startswith("% Generated by analysis/r1_accuracy_efficiency.py; do not edit.\n")
    assert r"\caption{\rev{" in table
    assert table.index(r"\caption{") < table.index(r"\begin{tabular*}")
    assert r"\begin{table}[tb]" in table and r"\small" in table
    assert "2,000" in table and "0.03" in table
    assert "7B only" in table and "generated tokens" in table


def test_new_files_are_identical_in_paper_checkout():
    root = Path(__file__).resolve().parents[1]
    if not (root / "paper/analysis").exists():
        pytest.skip("Running inside the paper copy")
    for name in ("analysis/r1_accuracy_efficiency.py", "scripts/rsi_r1_accuracy.sh", "tests/test_r1_accuracy_efficiency.py"):
        assert (root / name).read_bytes() == (root / "paper" / name).read_bytes()
