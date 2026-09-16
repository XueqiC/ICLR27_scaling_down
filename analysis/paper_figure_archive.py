"""Retire figure bundles reversibly, including their provenance and captions."""
from __future__ import annotations

if __package__:
    from .paper_artifacts import output_path, no_symlinks
else:
    from paper_artifacts import output_path, no_symlinks


def archive_panels(root, stems):
    directory = output_path(root, "figs")
    moved = []
    for stem in stems:
        sources = [output_path(root, "figs", stem + suffix) for suffix in
                   (".pdf", ".png", "_data.json", "_caption.txt", "_sources.md", "_files.json")]
        sources = [p for p in sources if p.exists()]
        if not sources:
            continue
        target = no_symlinks(directory / "_trash" / stem)
        index = 1
        while target.exists():
            target = no_symlinks(directory / "_trash" / f"{stem}-{index}")
            index += 1
        target.mkdir(parents=True)
        for source in sources:
            source.rename(target / source.name)
        moved.append(stem)
        print(f"Removed panel stem: {stem}; archived in {target}")
    return moved
