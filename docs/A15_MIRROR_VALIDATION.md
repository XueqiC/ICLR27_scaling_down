# A15 evidence mirror and CPU regeneration

The four mirrored generators regenerate three byte-identical tables and ten
render-identical figure panels against the manuscript repository copies. All 51
selected tests pass. No source evidence or manuscript TeX was changed, and no
commit, GPU work, model loading, fitting, or training was performed.

## Inventory

[A15_MIRROR_INVENTORY.csv](A15_MIRROR_INVENTORY.csv) lists every added file, its
source, stored and expanded byte sizes, substitution count, and original and
published SHA-256 digests. Hashes refer to uncompressed bytes, as in
`data_mirror/ANONYMIZATION_DIGESTS.json`.

| Experiment | Added evidence files | Stored bytes | Expanded bytes |
|---|---:|---:|---:|
| A9 measurement efficiency | 3 | 2,995,429 | 25,465,327 |
| A11 efficiency confirmation, including runtime inputs | 29 | 2,049,447 | 2,049,447 |
| A12 data-requirement confirmation | 134 | 9,425,075 | 28,258,281 |
| A13 data-requirement accounting | 10 | 6,382,804 | 6,382,804 |
| Evidence total | 176 | 20,852,755 | 62,155,859 |

The inventory additionally includes the previously absent
`analysis/a13_dreq_accounting.py` and `tests/test_a13_dreq_accounting.py` (33,666
bytes together). V53, V55, V69 and V70 were already complete: every corresponding
mirror was checked against its anonymized source, including the V70 development
record used for the new Development measurements column. No replacements were
necessary. None of the four requested generators reads A10, so its artifacts were
not added.

A12 includes both preregistrations, both complete plans (including
`registration_0/plan.json`), task table, score, summary, parser validation, 18
completion records, 90 evaluation JSON files, and 18 training-metrics JSON files.
The serialized A12 dataset inputs are not needed by these consumers and are not
included. Adapter directories, weights, execution logs, and cached figures are
excluded. A11 retains its premeasurement archive and host-change note, with
publication anonymization applied.

The established input-mirror convention is deterministic gzip above 5 MB, with
mtime zero and no slimming:

| Mirror file | Stored bytes | Expanded bytes |
|---|---:|---:|
| `a9-measurement-efficiency/summary.json.gz` | 2,534,845 | 25,004,743 |
| `a12-data-requirement-confirmation/plan.json.gz` | 875,493 | 10,503,102 |
| `a12-data-requirement-confirmation/registration_0/plan.json.gz` | 455,431 | 9,661,028 |

Every added file is below 50,000,000 bytes both stored and expanded. The largest
stored file is A12 `score.json`, 2,962,327 bytes.

## Publication and bootstrap handling

The existing anonymizer and numeric-value guard were applied to all 178 added
files. The anonymizer now also handles cluster work-directory usernames, the
cluster SSH alias, and the hardware UUID in the A11 host note. The independent
anonymity patterns found no hits in the additions, including decompressed JSON.
All numeric leaves, full JSON contents apart from anonymized strings, and frozen
hash references remain intact. The digest map adds 92 scrub-only correspondences
marked `pairable=true`, plus two explicitly non-pairable code/test differences. The existing final-table
test correspondence was refreshed and remains non-pairable.

A11's frozen scorer authenticates its probe file and two runtime scripts at their
original relative locations. Their sources are published under
`data_mirror/a11-efficiency-confirmation/runtime/`; bootstrap restores the ignored
`data/a11/` and script aliases without overwriting existing files. The efficiency
figure loader translates only exact, pairable published digests during validation
by the unchanged frozen scorer. Its audit records the actual published input
hashes. New tests reject both non-pairable mappings and tampered runtime bytes.

Bootstrap was run in the public checkout and in a clean scratch reconstruction.
The checkout's previous materialized digest map was stale because bootstrap
preserves existing files; that generated copy was backed up outside the checkout
and reconstructed from the updated mirror. Repeating bootstrap writes zero files
and expands zero archives. Existing results, including frozen registrations, are
not overwritten by bootstrap.

C81, C82, C84 and C85 in `RESULTS_LEDGER.md` are verbatim source entries. C83 has
only the existing anonymizer's two workstation-name substitutions; this is the
explicit exception to verbatim copying needed to preserve publication anonymity.
No numerical, methodological, or outcome text was edited.

## Tests

Run from the public repository after bootstrap, with the CPU Matplotlib Agg backend:

```sh
MPLBACKEND=Agg python3 -B -m pytest -q \
  tests/test_plot_fig_efficiency.py \
  tests/test_a13_dreq_accounting.py \
  tests/test_a13_dreq_tables.py \
  tests/test_final_prediction_table.py \
  tests/paper_generator_checks.py
```

| Test file | Passed |
|---|---:|
| `test_plot_fig_efficiency.py` | 7 |
| `test_a13_dreq_accounting.py` | 14 |
| `test_a13_dreq_tables.py` | 13 |
| `test_final_prediction_table.py` | 17 |
| Total | 51 |

`paper_generator_checks.py` was explicitly included; it defines shared checks,
not standalone tests. Its access confinement and symlink checks are exercised by
the final-prediction-table tests. Both the public checkout and a clean scratch reconstruction pass all **51 tests**.
The clean run starts with no results, runtime aliases, or generated outputs and
rebuilds them with bootstrap. The final-table preview fixture now preserves
absence as well as existing canonical files, avoiding a dependency on past runs.

## Scratch regeneration and comparisons

All four generators were imported from the public checkout and invoked there with
`generate(root=scratch)`. The scratch root was a real directory, with copies of
`analysis/`, `docs/`, `data_mirror/`, and `bootstrap_results.py`; its bootstrap
rebuilt all inputs without relying on pre-existing results. No symlink or
manuscript output directory was used. Table generation from saved mirrored A13
accounting ran before the accounting generator recomputed its derived files.

The reproduction pattern, from this repository, is:

```python
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

source = Path.cwd()
scratch = Path(tempfile.mkdtemp(prefix="a15-"))
for name in ("analysis", "docs", "data_mirror"):
    shutil.copytree(source / name, scratch / name)
shutil.copy2(source / "bootstrap_results.py", scratch / "bootstrap_results.py")
subprocess.run([sys.executable, "-B", str(scratch / "bootstrap_results.py")], check=True)
from analysis import plot_fig_efficiency, a13_dreq_table
from analysis import a13_dreq_accounting, final_prediction_table
for module in (plot_fig_efficiency, a13_dreq_table,
               a13_dreq_accounting, final_prediction_table):
    module.generate(root=scratch)
print(scratch)
```

[A15_REGENERATION_COMPARISON.json](A15_REGENERATION_COMPARISON.json) records all 36
comparisons against `paper/paper/{figs,tables}` in the working checkout:

- `dreq_main.tex`, `dreq_full.tex`, and `main_prediction_v2.tex`: byte-identical.
  `scripts/caption_only_diff.py` additionally reports 3 tables, 0 changed,
  0 non-caption differences.
- Ten panels: `eff_{a,b,legend}`, `eff_qa_{a,b,legend}`, and
  `dreq_{a,b,c,legend}`. All PNG pixel arrays are identical. The ten PDFs differ
  in bytes but their `pdftoppm` renders at 216 dpi are pixel-identical,
  maximum channel difference zero.
- All 13 caption text files are byte-identical.

The 26 byte-identical files comprise the three tables, ten PNGs and 13 captions.
The ten PDFs have identical renders. All 402 source evidence files and 64 manuscript
TeX files were checked by SHA-256 before and after the work and are unchanged.

## Changed implementation and documentation files

- `bootstrap_results.py` and `.gitignore`: restore A11 runtime aliases and keep
  generated copies out of Git.
- `analysis/plot_fig_efficiency.py` and `tests/test_plot_fig_efficiency.py`:
  pairable-only publication-digest validation and two rejection regressions.
- `tests/test_final_prediction_table.py`: retain the existing checks while
  permitting absent generated files in a clean public checkout.
- `analysis/a13_dreq_accounting.py` and `tests/test_a13_dreq_accounting.py`:
  previously absent working-tree copies, published through the numeric guard.
- `data_mirror/ANONYMIZATION_DIGESTS.json`: preserve unrelated entries, refresh the
  existing final-table test correspondence, and append the 94 entries described above.
- `README.md`, `docs/RESULTS_LEDGER.md`, this report,
  `docs/A15_MIRROR_INVENTORY.csv`, and `docs/A15_REGENERATION_COMPARISON.json`:
  experiment index, evidence records, and verification inventory.
- Working-tree publication tools `tools/build_anon_release.py` and
  `tools/write_digest_map.py`: additional work-directory/host anonymization and
  support for runtime-mirror source resolution and Slurm script digests. These
  author-side tools are outside the public repository.

The pre-existing final-prediction and A13 table changes were retained; the
final-prediction test fixture received only the clean-checkout adjustment above. No generator's numerical or statistical logic was
changed for publication.
