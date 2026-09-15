"""Shared integration checks for the four artifact-only paper generators."""
from pathlib import Path
import hashlib

import pytest

from analysis.paper_artifacts import ROOT, frozen_run, output_path


def check_access(audit, access):
    reads, writes = access
    assert reads and writes
    for name in reads:
        path = Path(name).resolve()
        assert any(path.is_relative_to(ROOT / p) for p in ("results", "docs"))
        assert path.suffix not in (".safetensors", ".pt", ".bin")
    for name in writes:
        path = Path(name)
        assert any(path.is_relative_to(ROOT / "generated" / p) for p in ("figs", "tables"))
        assert not path.is_symlink()
    for relative, digest in audit.inputs.items():
        # All evidence remains byte-identical after generation.
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def check_figures(stem):
    assert (ROOT / f"generated/figs/{stem}.pdf").read_bytes().startswith(b"%PDF-")
    assert (ROOT / f"generated/figs/{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (ROOT / f"generated/figs/{stem}_sources.md").is_file()


def refuses_symlink(module, tmp_path, kind, component):
    target = tmp_path / "target"
    target.mkdir()
    sentinel=target/"sentinel";sentinel.write_text("untouched")
    link=tmp_path / ("generated" if component=="parent" else f"generated/{kind}")
    link.parent.mkdir(parents=True,exist_ok=True)
    link.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match="Symlink"):
        module.generate(root=tmp_path)
    assert sentinel.read_text()=="untouched"
    assert list(target.iterdir())==[sentinel]


def refuses_external_io(tmp_path):
    forbidden=tmp_path/"not-an-artifact.json"
    forbidden.write_text("{}")
    with frozen_run(ROOT):
        with pytest.raises(PermissionError,match="Read outside"):
            forbidden.read_text()
        with pytest.raises(PermissionError,match="Write outside"):
            forbidden.write_text("changed")
    assert forbidden.read_text()=="{}"


def refuses_output_file_symlink(tmp_path, kind, name):
    target=tmp_path/"sentinel";target.write_text("untouched")
    directory=tmp_path/"generated"/kind;directory.mkdir(parents=True)
    (directory/name).symlink_to(target)
    with pytest.raises(ValueError,match="Symlink"):
        output_path(tmp_path,kind,name)
    assert target.read_text()=="untouched"
