"""Read-only artifact access and output confinement for the restructured paper.

Application inputs are restricted to results/ and docs/. Python modules,
installed libraries and their bundled fonts are runtime resources, not evidence.
No model libraries, subprocesses, network, fits, or resampling are used here.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import site
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CAPS = ("math", "code", "qa")
COLORS = {"math": "#2563a6", "code": "#c26a24", "qa": "#25836b"}
ACTIVE = None
RUNTIME_ROOTS = tuple(Path(p).resolve() for p in
                      (sys.prefix, sys.base_prefix, *site.getsitepackages(), site.getusersitepackages()))
FONT_ROOTS = (Path("/usr/share/fonts"), Path("/usr/local/share/fonts"))


def no_symlinks(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlink refused: {path}")
    return path


def output_path(root, kind, name=None):
    if kind not in ("figs", "tables"):
        raise ValueError("Outputs must be generated/figs or generated/tables")
    directory = no_symlinks(Path(root) / "generated" / kind)
    path = no_symlinks(directory / name) if name else directory
    if name and (Path(name).name != name or path.parent != directory):
        raise ValueError("Output name must be a basename")
    return path


class Artifacts:
    def __init__(self, root=ROOT):
        self.root = Path(root).absolute()
        self.inputs = {}
        self.data = {}
        self.notes = []

    def path(self, relative):
        path = Path(relative)
        path = path if path.is_absolute() else self.root / path
        allowed = (self.root / "results", self.root / "docs")
        if not any(path.resolve().is_relative_to(p.resolve()) for p in allowed):
            raise ValueError(f"Input outside results/ and docs/: {path}")
        return path

    def read(self, relative):
        path = self.path(relative)
        key = path.relative_to(self.root).as_posix()
        raw = path.read_bytes()
        self.inputs[key] = hashlib.sha256(raw).hexdigest()
        result = json.loads(raw) if path.suffix == ".json" else raw.decode()
        self.data[key] = result
        return result

    def rule(self, message):
        self.notes.append(message)

    def omit(self, message):
        self.notes.append(message)

    def published_pairs(self):
        """Only scrub-reproducible pairs from this run's audited digest map."""
        try:
            entries = self.read("results/ANONYMIZATION_DIGESTS.json")["files"].values()
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return {}
        return {e["frozen_sha256"]: (e["published_sha256"], e.get("published_at", ""))
                for e in entries if e.get("pairable") is True
                and e.get("frozen_sha256") and e.get("published_sha256")}

    def matches_digest(self, recorded, computed):
        """Use the repository comparator with only this run's pairable map."""
        from unittest.mock import patch
        if __package__:
            from . import provenance
        else:
            import provenance
        if recorded == computed:
            return True
        with patch.object(provenance, "_pairs", self.published_pairs):
            return provenance.matches(recorded, computed)


def _audit(event, args):
    if ACTIVE is None:
        return
    root, reads, writes = ACTIVE
    if event in ("subprocess.Popen", "os.system", "socket.connect", "socket.bind"):
        raise PermissionError(f"Paper generator refuses {event}")
    if event == "open":
        name, mode, flags = args
        if isinstance(name, int):
            return
        path = Path(os.fsdecode(name)).absolute()
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing:
            no_symlinks(path)
            if not any(path.is_relative_to(root / "generated" / k) for k in ("figs", "tables")):
                raise PermissionError(f"Write outside paper outputs: {path}")
            writes.add(str(path))
        else:
            resolved = path.resolve()
            if any(resolved.is_relative_to(root / p) for p in ("results", "docs")):
                reads.add(str(path))
                return
            # Explicit runtime allowlist, never arbitrary repository files.
            if path.suffix in (".py", ".pyc") and resolved.is_relative_to(ROOT / "analysis"):
                return
            if any(resolved.is_relative_to(p) for p in RUNTIME_ROOTS):
                return
            # Installed fonts are read-only rendering resources, never evidence.
            if resolved.suffix.lower() in (".ttf", ".otf", ".ttc") and any(
                    resolved.is_relative_to(p) for p in FONT_ROOTS):
                return
            if resolved.is_relative_to(root / "generated/figs/.mplconfig"):
                return
            raise PermissionError(f"Read outside frozen artifact roots: {path}")
    elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename"):
        for name in (args[:2] if event == "os.rename" else args[:1]):
            path = no_symlinks(Path(os.fsdecode(name)).absolute())
            outputs = [root / "generated" / k for k in ("figs", "tables")]
            if not any(path.is_relative_to(p) or (event == "os.mkdir" and p.is_relative_to(path)) for p in outputs):
                raise PermissionError(f"Mutation outside paper outputs: {path}")


sys.addaudithook(_audit)


@contextmanager
def frozen_run(root=ROOT):
    global ACTIVE
    if ACTIVE is not None:
        raise RuntimeError("Nested paper generator guard")
    root = Path(root).absolute()
    for kind in ("tables", "figs"):
        output_path(root, kind)
    reads, writes = set(), set()
    ACTIVE = (root, reads, writes)
    try:
        yield reads, writes
    finally:
        ACTIVE = None


def pyplot(root=ROOT):
    cache = output_path(root, "figs") / ".mplconfig"
    no_symlinks(cache)
    os.environ["MPLCONFIGDIR"] = str(cache)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    # Legacy plot modules evaluate gettempdir() even when MPLCONFIGDIR is set.
    # Point that probe/cache at an allowed output directory before importing.
    import tempfile
    tempfile.tempdir = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "savefig.dpi": 220})
    return plt


def save_figure(fig, stem, audit):
    for suffix in ("pdf", "png"):
        path = output_path(audit.root, "figs", f"{stem}.{suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", metadata={"Creator": "frozen paper generator"}
                    if suffix == "pdf" else None)
    write_notes(stem, audit)


def write_notes(stem, audit, extra=None):
    path = output_path(audit.root, "figs", f"{stem}_sources.md")
    lines = [f"# {stem}: frozen sources", "", *audit.notes, "", "## Inputs (SHA-256)", ""]
    lines.extend(f"- `{p}`: `{h}`" for p, h in sorted(audit.inputs.items()))
    if extra is not None:
        lines += ["", "## Plotted records", "", "```json", json.dumps(extra, indent=2), "```"]
    path.write_text("\n".join(lines) + "\n")


def legacy_rows(audit):
    """Reuse V86's complete loaders, identities and arithmetic, with guarded reads.

    Its test-ranked Row.baselines/Row.delivered are deliberately never used.
    """
    from unittest.mock import patch
    if __package__:
        from . import v86_main_table as old
    else:
        import v86_main_table as old
    original = old.Comparison

    class Comparison(original):
        def __init__(self, relative):
            self.path = relative
            self.data = audit.read(relative)
            self.sha256 = audit.inputs[relative]

    # Digest acceptance follows the repository rule: exact equality, or a frozen/published pair
    # recorded as pairable in the anonymization digest map.
    with patch.object(old, "Comparison", Comparison), patch.object(old.provenance, "_pairs", audit.published_pairs):
        rows, _ = old.build_rows()
    audit.rule("Reused v86_main_table.build_rows, Comparison.number, row_scores and paired_rows; "
               "pruning inputs are v53-prune-dev register/predictions/compare and v72-prune-repeat freeze/compare. "
               "Did not use test-ranked Row.baselines or post-hoc Row.delivered.")
    return {row.key: row for row in rows}


def confirmation_identity(audit, directory, identity, observed):
    if __package__:
        from .plot_fig2_confirm import paired_rows
    else:
        from plot_fig2_confirm import paired_rows
    comp = audit.read(f"results/{directory}/compare.json")
    freeze = audit.read(f"results/{directory}/freeze.json")
    paired_rows(comp, freeze, identity, observed)
    return comp, freeze


def development_rows(audit):
    if __package__:
        from .a1_development_table import load_development_table
    else:
        from a1_development_table import load_development_table
    directory = audit.root / "results/a1-development-table"
    rows = load_development_table(directory)
    for name in ("summary.json", "development_table.csv", "row_metadata.csv"):
        audit.read(directory / name)
    return rows
