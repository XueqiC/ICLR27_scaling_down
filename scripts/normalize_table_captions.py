#!/usr/bin/env python3
"""Put every table caption above its tabular and every table note below it (author's rule, 2026-09-22).

Order inside each `table` environment becomes: preamble (placement, \centering, font size, column
separation, \ContinuedFloat, ...), \caption, \label, the tabular block, note minipages, anything
else in its original order, \end{table}. Figures are left alone: their captions already sit below
the graphics, which is the matching standard for figures. The same transformation is applied by
`analysis.paper_table_layout.caption_above`, so regenerated tables keep the order.

    python3 scripts/normalize_table_captions.py            # rewrite paper/paper/tables/*.tex and inline tables
    python3 scripts/normalize_table_captions.py --check    # report files that would change, exit 1 if any
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.paper_table_layout import caption_above  # noqa: E402

PAPER = ROOT / "paper/paper"
INLINE = ["appendix.tex", "setup.tex", "laws.tex", "general.tex", "observed.tex", "selection.tex", "intro.tex",
          "related.tex", "learned.tex", "statements.tex", "supplement_body.tex"]


def main() -> int:
    check = "--check" in sys.argv
    changed = []
    files = sorted((PAPER / "tables").glob("*.tex")) + [PAPER / n for n in INLINE if (PAPER / n).exists()]
    for path in files:
        before = path.read_text()
        after = caption_above(before)
        if after != before:
            changed.append(path.relative_to(ROOT))
            if not check:
                path.write_text(after)
    for p in changed:
        print(("would change " if check else "rewrote ") + str(p))
    print(f"{len(changed)} file(s) {'need' if check else 'got'} the caption-above order")
    return 1 if (check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
