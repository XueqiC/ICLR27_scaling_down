# Capability-conditioned scaling-down laws for LLM compression

Measurement and analysis code for studying how compressing a language model changes what
it can do. Three compression families are treated under one protocol, pruning, post-training
quantization and trace distillation, and the endpoint is the conditional loss on a fixed
evaluation distribution per capability rather than an aggregate score. The repository holds
the pipeline, the predictions that were frozen before each test, the measurements they were
scored against, and the records that tie every reported number to the file it came from.

## Installation

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # analysis, tables and figures: CPU only
pip install -r requirements-gpu.txt      # additionally, to measure models
```

Analysis needs no GPU and no network. Measurement was run with the stack pinned in
`requirements-gpu.txt` on single GPUs (CUDA 12.8).

## Quickstart

First reconstruct the input tree the analysis scripts read, then regenerate a table:

```
python3 bootstrap_results.py
python3 -m analysis.final_prediction_table
python3 -m analysis.plot_fig_generalization
python3 -m analysis.plot_fig_explanation
python3 -m analysis.plot_fig_responses_v2
```

`bootstrap_results.py` materialises `results/` from `data_mirror/`, expanding the gzipped
summaries and creating the directories the generators write into. It never overwrites an
existing file and never modifies `data_mirror/`.
It also restores A11's frozen probe and runtime-script aliases under `data/a11/`
and `scripts/`. Those generated copies are ignored by Git; their published sources
remain in `data_mirror/a11-efficiency-confirmation/runtime/`.

These CPU-only generators read frozen artifacts and write to `generated/tables/`
and `generated/figs/`. `plot_fig_generalization` also generates the cell-level
appendix; `python3 -m analysis.plot_fig_generalization_cells` runs that appendix
alone. No training, model evaluation, fitting or resampling is needed. The exact
input inventory, including the A5 corner-analysis checkpoints and A7 audit inputs,
is in [PAPER_GENERATOR_INPUTS.md](docs/PAPER_GENERATOR_INPUTS.md). The original V47
`freeze.json` is unavailable; the generators disclose this and use the recorded
V70 freeze for the confirmation rows.

Check these generators with:

```
python3 -m pytest -q tests/test_final_prediction_table.py tests/test_plot_fig_generalization.py tests/test_plot_fig_generalization_cells.py tests/test_plot_fig_explanation.py tests/test_plot_fig_responses_v2.py
```

The A5/A7 artifact analyses and their 75 tests were previously checked. A15 checks
`final_prediction_table` and the three measurement/data-requirement generators
below: all 51 selected tests pass. Their three tables are byte-identical to the
manuscript copies; ten PDF panels and ten PNG previews have zero pixel differences.
Publication digest acceptance is restricted to entries marked `pairable=true`.
See [A15_MIRROR_VALIDATION.md](docs/A15_MIRROR_VALIDATION.md) for the exact scope,
scratch regeneration procedure, sizes, and comparison results.

One artifact exists in two states, because experiments recorded it at different times: the
grouped-quantization measurements grew new configurations after an earlier audit had recorded
their digest. `data_mirror/v54-quant-group/` carries the current file, which the
capability-conditioning audits need, and `data_mirror/v54-quant-group-as-recorded-by-v55/`
carries the state the earlier audit recorded. Copy the latter over the former to rebuild that
earlier audit; the ledger says which entry used which.

The older versioned generators use the historical output layout. In a standalone
checkout, `python3 bootstrap_results.py --legacy-layout` creates their
`paper/paper/{tables,figs}` and `paper/code` compatibility paths. Do not use that
option when `paper/` contains a separate manuscript repository. Previously checked
with that layout: `v45_main_table`, `v52_prediction_tables`, `v74_quant_threeway`,
`v76_cap_conditioning`, `v84_main_table`, `v85_selection_decomp` and `v86_main_table`
regenerate their outputs. `v79_cond_audit` also regenerates, but it shells out to `ripgrep`, so install that first.
`v63_quant_identifiability` is the one that does not: it verifies the grouped-quantization
measurements against the digest recorded when it ran, and this repository publishes the
current file, which gained configurations afterwards. Copy
`data_mirror/v54-quant-group-as-recorded-by-v55/` over `data_mirror/v54-quant-group/`, rerun
`bootstrap_results.py` in a clean tree, and it regenerates; the capability-conditioning audits
then need the current file back. `data_mirror/ANONYMIZATION_DIGESTS.json` records every
published file whose bytes differ from the working-tree original, with the reason, and marks
which of those a digest check may accept.

Run the offline test suite (the complete suite also imports the packages in
`requirements-gpu.txt`, but uses only CPU fixtures and never loads model weights):

```
python3 -m pytest tests
```

Pytest automatically runs `bootstrap_results.py` before collection, so this command
works in a fresh checkout. Bare `python3 -m pytest` selects the same `tests/` tree;
`pytest.ini` excludes the manuscript, generated outputs and old `code/` copies.
The shared helper in `tests/public_inputs.py` skips only explicitly named absent
prerequisites: raw teacher traces under `results/traces-pilot/` (third-party text
deliberately not redistributed), model/adapter tensors (`results/**/*.safetensors`,
deliberately excluded), and external Hugging Face dataset/tokenizer caches when
absent (the required 2Wiki Arrow file alone is 54,510,136 bytes). The existing
five unavailable full-grid Pythia checks and one CUDA-only check also skip on CPU.
Use `python3 -m pytest tests -rs` to see each absent input and its reason.
All remaining measurement inputs, JSON adapter configuration metadata (no adapter
tensors), and read-only table-check snapshots are mirrored. Missing mirrored inputs
still fail; assertions are not converted to skips. See
[A17_PUBLIC_SUITE.md](docs/A17_PUBLIC_SUITE.md) for validation counts, the complete
change inventory and mirror byte totals.

`code/` is a disposable legacy copy, ignored by Git and unnecessary for the public
suite and current generators. It should stay out of the public repository and can
be removed locally. Bootstrap now creates it only with `--legacy-layout`, for older
generators that explicitly verify their historical code copies.

Each generated file names its inputs and their SHA-256 digests in a header comment, so a
regenerated table can be compared line by line with the one that was published.
API tests use mocked credentials and HTTP responses. `docs/RESULTS_LEDGER.md` names
the artifact behind every reported number.

## Experiment index: measurement efficiency, data requirements and selection

| Experiment | Published evidence | CPU consumer |
|---|---|---|
| A9 / C81 | Complete plan, summary and replicate records; `summary.json.gz` | `analysis.plot_fig_efficiency` |
| A11 / C83 | Preregistration, frozen predictions, four states' losses and metadata, premeasurement archive, frozen runtime inputs | `analysis.plot_fig_efficiency` |
| A12 / C84 | Preregistration and amendment, current and `registration_0` plans, task table, scores, 90 eval files and 18 training-metrics files | `analysis.a13_dreq_accounting` |
| A13 / C85 | Saved accounting, request/recommendation records, CSVs and validation | `analysis.a13_dreq_table` |
| S3 / C86 | Original and amended plans, numeric freeze, identity audit, eight anchors, 48 candidate measurements, four student evals and training-metrics files | `analysis.s3_paper_table` |
| V53 / V55 / V69 / V70 | Existing registers, freezes, comparisons and development records | `analysis.final_prediction_table`, including Development measurements |
| P1 (retrospective) | Fresh-item plan (384 2Wiki items), per-item losses, generations and scores for 22 models, paired summary; `data_mirror/p1-qa-behaviour/` | `analysis.p1_qa_behaviour --analyse --table` |
| P2 / A16 (retrospective) | Wanda panel losses and summary (`data_mirror/p2-wanda-panel/`); byte accounting of the selection candidates (`data_mirror/a16-storage-accounting/`) | `analysis.p2_wanda_panel --analyse --table`, `analysis.a16_storage_accounting --table` |

A10 / C82 is described in the ledger; none of A15's four consumers opens its
artifacts, so A15 does not add its 90 MB summary. The A12 serialized dataset
inputs are also unnecessary for these consumers; all scored eval JSON is included.
Adapters, model weights and execution logs are excluded. Large inputs retain their
complete contents and use deterministic gzip above 5 MB. The complete file list,
stored/expanded byte sizes and digests are in
[A15_MIRROR_INVENTORY.csv](docs/A15_MIRROR_INVENTORY.csv).

```
python3 -m pytest -q tests/test_plot_fig_efficiency.py tests/test_a13_dreq_accounting.py tests/test_a13_dreq_tables.py tests/test_final_prediction_table.py tests/paper_generator_checks.py
```

S3's independent cross-method selection appendix table regenerates from
`data_mirror/s3-selection-validation/{score_v2.json,plan_v2.json}`. From this
repository root, write a preview and check its bytes with:

```
python3 bootstrap_results.py
python3 -c 'from analysis.s3_paper_table import render; from pathlib import Path; Path("generated/tables/s3_selection.tex").write_bytes(render().encode("utf-8"))'
python3 -m pytest -q tests/test_s3_paper_table.py -rs
```

The test uses an isolated copy of the mirrored inputs and checks the published
table snapshot even in a standalone checkout. A second comparison checks
`paper/tables/s3_selection.tex` when the separate manuscript checkout is present;
otherwise it skips with the named reason `S3_MANUSCRIPT_NOT_DISTRIBUTED`. Missing
mirrored evidence or the published snapshot is a failure. The original generator
is preserved byte-for-byte; its direct module entry point uses the historical
output layout, so the preview command above writes to `generated/tables/`.
See [A18_MIRROR_VALIDATION.md](docs/A18_MIRROR_VALIDATION.md), the complete
[file/size/digest inventory](docs/A18_MIRROR_INVENTORY.csv), and the
[omitted-file reasons](docs/A18_MIRROR_EXCLUSIONS.csv).

## Layout

| Path | Contents |
|---|---|
| `analysis/` | the pipeline, one module per experiment, named `v<N>_<topic>.py` |
| `tests/` | offline test suite |
| `configs/` | sweep configurations |
| `scripts/*.sh` | grid drivers for the larger measurement runs |
| `data_mirror/` | frozen registers, predictions, measurements and comparison files, one directory per experiment |
| `results/` | curated per-experiment reports and summaries |
| `docs/` | results ledger, claim-evidence matrix, measurement definitions, pre-registrations, reference verification, reviewer self-audit |

## Measuring a model

Checkpoints resolve through `analysis/model_registry.py`; all of them are public and
pinned by revision.

- Pruning: `analysis/v6_capability_geometry.py` applies global magnitude pruning at a
  deterministic sampled threshold and scores the probe sets.
- Quantization: `analysis/v10_quantization.py` for per-channel round-to-nearest,
  `analysis/v54_quant_group.py` for grouped round-to-nearest with a choice of symmetric
  or asymmetric mode.
- Distillation: `analysis/v12_distill.py` trains a LoRA student on stored teacher
  traces. Generating traces needs an API endpoint and credentials, read from a
  `.secrets.env` file that is not part of the repository.

## Evidence and provenance

`docs/RESULTS_LEDGER.md` is the spine: one entry per experiment, in order, recording what was
frozen, when, what was measured and how it came out, including the entries that record
failures and the ones that correct earlier entries. `docs/CLAIM_EVIDENCE_MATRIX.md` maps each
reported claim to the artifact behind it. `docs/prereg/` holds registrations written before
the corresponding measurement, and `docs/REFERENCE_VERIFICATION.md` records the check of every
bibliography entry against arXiv, DBLP or the publisher.

Freeze discipline: wherever a test is described as independent, the predictions were written
and hashed before the measurement, and the register and freeze files under `data_mirror/`
carry those digests and timestamps. Nothing overwrites a frozen file; corrections arrive as
new versions and are recorded in the ledger.

## Not included

Model weights, LoRA adapters, downloaded checkpoints and dataset caches, all public or
regenerable from the pipeline; teacher trace files, which are third-party model outputs;
credentials; and run logs. Absolute paths in the stored artifacts are rewritten to
`$HOME`-relative or repository-relative form, so a digest recomputed over one of those files
can differ from the digest recorded inside it. No measurement value was altered.

## License

Released for review and reuse in research. A permissive license will be attached on
publication. Third-party datasets and checkpoints keep their own terms; see
`analysis/model_registry.py` and `docs/MEASUREMENT_DEFINITIONS.md` for the sources.
