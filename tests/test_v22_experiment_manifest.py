from analysis import v22_experiment_manifest as manifest


def test_density_metadata_is_not_a_measured_configuration():
    assert manifest.numeric_keys({"1.0": {}, "0.625": {}, "_infill_meta": {}, "0.9": {}}) == ["1.0", "0.9", "0.625"]


def test_every_local_artifact_has_an_index_entry_including_ignored_weights():
    root = manifest.ROOT / "results"
    text = manifest.render(root)
    files = manifest.files_under(root)
    assert any(p.suffix == ".safetensors" for p in files)
    for path in files:
        # Unique index entry, even where the path also occurs in a run table.
        prefix = f"| `{manifest.label(path, root)}` | {path.stat().st_size} | "
        assert text.count(prefix) == 1, path
    assert "270m / 1b / 4b / 12b / 27b" in text
    assert "0.6b / 1.7b / 4b" in text
    assert "7b / 32b" in text
