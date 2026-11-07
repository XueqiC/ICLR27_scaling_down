"""Offline CPU regression tests; never load a Hub model or dataset."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from analysis import r1_probe_panel as r1
from analysis.tiny_test_model import PROBES, TinyModel, TinyTokenizer


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    import huggingface_hub
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", Mock(side_effect=AssertionError("No downloads")))
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", Mock(side_effect=AssertionError("No network")))
    monkeypatch.setattr(torch.cuda, "_lazy_init", Mock(side_effect=AssertionError("No CUDA")))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    import huggingface_hub
    commit = "a" * 40
    cache = tmp_path / "models--offline--tiny"
    snapshot = cache / "snapshots" / commit
    blobs = cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (snapshot / "config.json").write_text('{"model_type": "offline-tiny"}\n')
    shard_names = [f"model-{index:05d}-of-00002.safetensors" for index in (1, 2)]
    for name in shard_names:
        # Synthetic cache artifacts test hashing independently of the injected
        # model loader; no pretrained model is read, even from the real cache.
        content = name.encode()
        blob = blobs / hashlib.sha256(content).hexdigest()
        blob.write_bytes(content)
        (snapshot / name).symlink_to(blob)
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {f"weight{i}": name for i, name in enumerate(shard_names)}}))

    def cached(model_id, filename, revision):
        assert model_id == "offline/tiny" and revision == commit
        path = snapshot / filename
        return str(path) if path.exists() else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    loaded = []

    def load(model_id, dtype, revision):
        assert (model_id, dtype, revision) == ("offline/tiny", torch.bfloat16, "test-revision")
        model = TinyModel(dtype=dtype)
        model.config._commit_hash = commit
        loaded.append(model)
        return model, TinyTokenizer()

    loader = Mock(side_effect=load)
    monkeypatch.setattr(r1.v6, "load_text_causal_lm", loader)

    def samples(cap, n):
        return [{"prompt": f"{cap}{i}: " + PROBES[cap][i % len(PROBES[cap])]["prompt"],
                 "completion": PROBES[cap][i % len(PROBES[cap])]["completion"]}
                for i in range(n)]

    primary = Mock(side_effect=lambda n: {cap: samples(cap, n) for cap in r1.PRIMARY})
    benchmark = Mock(side_effect=lambda n, seed: {
        name: samples(r1.v9.BENCHMARK_CAPABILITY[name], n) for name in r1.BENCHMARKS
    } | {"c4": [{"prompt": "unscored", "completion": "control"}] * n})
    monkeypatch.setattr(r1.v6, "build_probes", primary)
    monkeypatch.setattr(r1.v9, "build_probe_registry", benchmark)
    return SimpleNamespace(out=tmp_path / "panel", snapshot=snapshot, loaded=loaded,
                           loader=loader, primary=primary, benchmark=benchmark,
                           shards=shard_names)


def run(tiny, configs=("dense", "prune_d0.7", "quant_b4"), **kwargs):
    return r1.run_panel("offline/tiny", "test-revision", "tiny", out=tiny.out,
                        configs=configs, device="cpu", reference_device="cpu", **kwargs)


def read(path):
    return json.loads(path.read_text())


def test_probe_selection_and_hashes(tiny):
    panel = r1.build_panel()
    tiny.primary.assert_called_once_with(128)
    tiny.benchmark.assert_called_once_with(128, seed=0)
    assert len(panel) == 704
    assert len({item["id"] for item in panel}) == 704
    for set_name, count, benchmarks in [("primary", 64, r1.PRIMARY.values()),
                                        ("benchmark", 64, r1.BENCHMARKS)]:
        for benchmark in benchmarks:
            selected = [item for item in panel if item["set"] == set_name and item["benchmark"] == benchmark]
            assert len(selected) == count
            assert [item["id"] for item in selected] == [f"{set_name}:{benchmark}:{i}" for i in range(1, 2 * count, 2)]
    for item in panel:
        assert item["benchmark"] != "c4"
        assert item["sha256"] == hashlib.sha256((item["prompt"] + item["completion"]).encode()).hexdigest()


def test_tiny_end_to_end_and_schema(tiny, monkeypatch):
    original_loss = r1.v6.completion_loss
    calls = []

    def loss(*args, **kwargs):
        assert not torch.is_grad_enabled()
        assert args[4] == "cpu"
        calls.append(args[2:4])
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(r1.v6, "completion_loss", loss)
    run(tiny)
    assert len(calls) == 704 * 3 + 8 * 2
    manifest = read(tiny.out / "manifest.json")
    assert manifest["revision"] == "test-revision"
    assert manifest["protocol"]["benchmark"]["n"] == 128
    assert manifest["reference_device"] == "cpu"
    assert manifest["checkpoint"]["config_sha256"] == r1.sha256_file(tiny.snapshot / "config.json")
    assert manifest["checkpoint"]["safetensors"][0]["hash_source"] == "full_file"
    assert manifest["checkpoint"]["safetensors"][1]["hash_source"] == "verified_lfs_filename"
    for field in ("hostname", "torch_version", "transformers_version", "git_commit"):
        assert manifest[field]
    assert manifest["gpu_name"] is None
    panel = r1.build_panel()
    model = TinyModel(dtype=torch.bfloat16).eval()
    parameters = r1.v6.language_weight_parameters(model)
    dense = [parameter.detach().clone() for _, parameter in parameters]
    sample = r1.v6._sample_abs_weights(parameters)
    threshold = float(np.quantile(sample, 1 - 0.7))
    assert manifest["pruning_thresholds"]["prune_d0.7"] == threshold
    for config in ("dense", "prune_d0.7", "quant_b4"):
        data = read(tiny.out / f"{config}.json")
        assert set(data) == {"config", "items", "seconds"}
        assert data["config"] == config and data["seconds"] >= 0
        assert len(data["items"]) == 704
        r1.v10._restore_dense_weights(parameters, dense)
        if config.startswith("prune"):
            r1.v6.apply_global_magnitude_pruning(model, 0.7, reference_weights=dense, threshold=threshold)
        elif config.startswith("quant"):
            r1.v10._apply_fake_quantization(parameters, dense, 4)
        for item, row in zip(panel, data["items"]):
            assert set(row) == {*r1.ITEM_FIELDS, "nll_sum", "n_tokens"}
            assert type(row["nll_sum"]) is float and type(row["n_tokens"]) is int
            with torch.no_grad():
                expected, count = original_loss(model, TinyTokenizer(), item["prompt"], item["completion"], "cpu")
            assert row["nll_sum"] == float(expected) and row["n_tokens"] == count
    assert torch.equal(tiny.loaded[0].weight, TinyModel(dtype=torch.bfloat16).weight)


def test_resume_skips_completed_files_and_all_done_needs_no_loading(tiny, monkeypatch):
    run(tiny, ("dense", "prune_d0.7"))
    original = {name: ((tiny.out / name).read_bytes(), (tiny.out / name).stat().st_mtime_ns)
                for name in ("dense.json", "prune_d0.7.json")}
    run(tiny)
    assert tiny.loader.call_count == 2
    for name, state in original.items():
        assert ((tiny.out / name).read_bytes(), (tiny.out / name).stat().st_mtime_ns) == state
    assert len(read(tiny.out / "manifest.json")["sessions"]) == 2
    for module, name in [(r1.v6, "load_text_causal_lm"), (r1, "build_panel"), (r1, "checkpoint_identity")]:
        monkeypatch.setattr(module, name, Mock(side_effect=AssertionError("Completed run must not load")))
    before = {path.name: path.read_bytes() for path in tiny.out.glob("*.json")}
    run(tiny)
    assert before == {path.name: path.read_bytes() for path in tiny.out.glob("*.json")}


def test_interrupted_config_is_not_committed_and_can_resume(tiny, monkeypatch):
    original = r1.measure
    count = 0

    def interrupted(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("Simulated reboot during compression")
        return original(*args)

    monkeypatch.setattr(r1, "measure", interrupted)
    with pytest.raises(RuntimeError, match="Simulated reboot"):
        run(tiny)
    assert (tiny.out / "dense.json").exists()
    assert not (tiny.out / "prune_d0.7.json").exists()
    assert torch.equal(tiny.loaded[0].weight, TinyModel(dtype=torch.bfloat16).weight)
    dense = (tiny.out / "dense.json").read_bytes()
    (tiny.out / ".prune_d0.7.json.stale").write_text('{"partial":')
    monkeypatch.setattr(r1, "measure", original)
    run(tiny)
    assert (tiny.out / "dense.json").read_bytes() == dense
    assert (tiny.out / "quant_b4.json").exists()


@pytest.mark.parametrize("hostname", ["hpg01", "loni", "workstation", "other-rsi", "rai", "rai01", "sn4622122543"])
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3-8B", "org/qWeN-custom"])
def test_qwen_guard_precedes_all_loading(tmp_path, monkeypatch, hostname, model_id):
    monkeypatch.setattr(r1.socket, "gethostname", lambda: hostname)
    build = Mock(side_effect=AssertionError("Must guard before probes"))
    load = Mock(side_effect=AssertionError("Must guard before model"))
    monkeypatch.setattr(r1, "build_panel", build)
    monkeypatch.setattr(r1.v6, "load_text_causal_lm", load)
    with pytest.raises(RuntimeError, match="Qwen runs only on rsi"):
        r1.run_panel(model_id, "commit", "guard", out=tmp_path / "blocked", device="cpu")
    assert not (tmp_path / "blocked").exists()
    build.assert_not_called()
    load.assert_not_called()


@pytest.mark.parametrize("hostname", ["rsi", "rsi-gpu01", "sn4622131043"])
def test_qwen_guard_permits_authorized_hosts_without_loading(monkeypatch, hostname):
    monkeypatch.setattr(r1.socket, "gethostname", lambda: hostname)
    r1.guard_model("Qwen/Qwen3-4B-Base")


def test_revision_required_and_configs_validated(tiny):
    with pytest.raises(SystemExit) as error:
        r1.main(["--model", "offline/tiny", "--tag", "test"])
    assert error.value.code == 2
    tiny.loader.assert_not_called()
    assert r1.parse_configs(["quant_b4,prune_d0.7", "dense"]) == ["dense", "quant_b4", "prune_d0.7"]
    for configs in ["prune_d1", "prune_dnan", "quant_b1", "../dense", "prune_d0.70"]:
        with pytest.raises(ValueError):
            r1.parse_configs(configs)


def test_dense_check_uses_token_weighted_primary_only(tmp_path):
    rows = [{"set": "primary", "capability": cap, "nll_sum": loss, "n_tokens": n}
            for cap in r1.PRIMARY for loss, n in [(2.0, 1), (12.0, 4)]]
    rows += [{"set": "benchmark", "capability": "math", "nll_sum": 999.0, "n_tokens": 1}]
    path = tmp_path / "expected.json"
    path.write_text(json.dumps({"1.0": {cap: 2.8 for cap in r1.PRIMARY}}))
    r1.check_expected_dense(rows, path)
    path.write_text(json.dumps({"1.0": {cap: 2.809 for cap in r1.PRIMARY}}))
    r1.check_expected_dense(rows, path)
    path.write_text(json.dumps({"1.0": {cap: 2.811 for cap in r1.PRIMARY}}))
    with pytest.raises(RuntimeError, match="Dense math mismatch"):
        r1.check_expected_dense(rows, path)


def test_expected_dense_failure_aborts_before_compression(tiny, tmp_path, monkeypatch):
    path = tmp_path / "expected.json"
    path.write_text(json.dumps({"1.0": {cap: 99.0 for cap in r1.PRIMARY}}))
    prune = Mock(side_effect=AssertionError("Must fail before compression"))
    monkeypatch.setattr(r1.v6, "apply_global_magnitude_pruning", prune)
    with pytest.raises(RuntimeError, match="Dense math mismatch"):
        run(tiny, expect_dense=path)
    assert not (tiny.out / "dense.json").exists()
    prune.assert_not_called()


def test_broken_restore_rejects_compressed_result(tiny, monkeypatch):
    monkeypatch.setattr(r1.v10, "_restore_dense_weights", lambda *args: None)
    with pytest.raises(RuntimeError, match="Dense restore mismatch"):
        run(tiny, ("dense", "prune_d0.5"))
    assert (tiny.out / "dense.json").exists()
    assert not (tiny.out / "prune_d0.5.json").exists()


def test_hashes_first_blob_fully_and_uses_rest_of_blob_names(tiny, monkeypatch):
    model, _ = tiny.loader("offline/tiny", torch.bfloat16, "test-revision")
    hasher = Mock(wraps=r1.sha256_file)
    monkeypatch.setattr(r1, "sha256_file", hasher)
    result = r1.checkpoint_identity("offline/tiny", "test-revision", model)
    assert [call.args[0] for call in hasher.call_args_list] == [
        (tiny.snapshot / tiny.shards[0]).resolve(), tiny.snapshot / "config.json"]
    for shard in result["safetensors"]:
        assert shard["sha256"] == Path(shard["blob_path"]).name
    (tiny.snapshot / tiny.shards[0]).resolve().write_bytes(b"corrupted cached shard")
    with pytest.raises(RuntimeError, match="SHA256 does not match"):
        r1.checkpoint_identity("offline/tiny", "test-revision", model)


@pytest.mark.parametrize("change", ["revision", "config", "probe"])
def test_resume_rejects_changed_identity(tiny, monkeypatch, change):
    run(tiny, ("dense",))
    if change == "revision":
        with pytest.raises(RuntimeError, match="manifest mismatch for revision"):
            r1.run_panel("offline/tiny", "another-revision", "tiny", out=tiny.out,
                         device="cpu", configs=("dense", "quant_b4"))
    elif change == "config":
        (tiny.snapshot / "config.json").write_text('{"different": true}')
        with pytest.raises(RuntimeError, match="checkpoint identity changed"):
            run(tiny)
    else:
        original = r1.build_panel

        def changed():
            panel = original()
            panel[0]["sha256"] = "f" * 64
            return panel

        monkeypatch.setattr(r1, "build_panel", changed)
        with pytest.raises(RuntimeError, match="probe hashes/selection changed"):
            run(tiny)
    assert not (tiny.out / "quant_b4.json").exists()


def test_cli_and_identical_paper_copy(tiny):
    r1.main(["--model", "offline/tiny", "--revision", "test-revision", "--tag", "tiny",
             "--out", str(tiny.out), "--device", "cpu", "--configs", "dense"])
    assert (tiny.out / "dense.json").exists()
    assert Path(r1.__file__).read_bytes() == (r1.ROOT / "paper/analysis/r1_probe_panel.py").read_bytes()


def test_model_list_has_requested_models_and_existing_dense_checks():
    rows = [line.split() for line in (r1.ROOT / "configs/r1_probe_panel_models.txt").read_text().splitlines()
            if line and not line.startswith("#")]
    assert len(rows) == 21
    assert len({row[0] for row in rows}) == 21
    pythia = {(row[1], row[2]) for row in rows if "pythia" in row[1]}
    assert pythia == {(f"EleutherAI/pythia-{size}", f"step{step}")
                      for size in ("160m", "410m", "1.4b") for step in (16000, 64000, 143000)}
    for row in rows:
        if row[1] == "Qwen/Qwen3-8B":
            assert row[2] == "b968826d9c46dd6066d109eabc6255188de91218"
        elif "pythia" not in row[1]:
            assert row[2] == "RESOLVE"
        if len(row) == 4:
            assert (r1.ROOT / row[3]).is_file()


def test_driver_pins_once_logs_and_continues_failures(tmp_path, monkeypatch):
    import huggingface_hub
    script = (r1.ROOT / "scripts/rsi_r1_probe_panel.sh").read_text()
    # Execute the driver's Python body with subprocess/Hub calls replaced;
    # this exercises persistence and failures without launching any GPU job.
    body = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    source = tmp_path / "models.txt"
    source.write_text("first org/first RESOLVE\nsecond org/second step16000 expected.json\n")
    original = source.read_bytes()
    model_info = Mock(return_value=SimpleNamespace(sha="c" * 40))
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    monkeypatch.setattr(sys, "argv", ["-", str(source), str(tmp_path)])
    for name in ("R1_RESOLVED_LIST", "R1_LOG_DIR", "R1_OUTPUT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        kwargs["stdout"].write("offline fake runner\n")
        return SimpleNamespace(returncode=1 if command[command.index("--tag") + 1] == "first" else 0)

    monkeypatch.setattr(r1.subprocess, "run", fake_run)
    for _ in range(2):
        with pytest.raises(SystemExit) as error:
            exec(compile(body, "rsi_r1_probe_panel.sh", "exec"), {"__name__": "__main__"})
        assert error.value.code == 1
    assert source.read_bytes() == original
    model_info.assert_called_once_with("org/first")
    resolved = (tmp_path / "configs/r1_probe_panel_models.resolved.txt").read_text()
    assert "RESOLVE" not in resolved and "c" * 40 in resolved
    assert len(commands) == 4
    assert commands[0][commands[0].index("--revision") + 1] == "c" * 40
    assert commands[1][-2:] == ["--expect-dense", "expected.json"]
    assert "exit=1" in (tmp_path / "logs/r1_probe_first.log").read_text()
    assert "exit=0" in (tmp_path / "logs/r1_probe_second.log").read_text()
