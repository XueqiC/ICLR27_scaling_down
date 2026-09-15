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

Current validation: `plot_fig_explanation` and `plot_fig_responses_v2` and their
tests pass. `final_prediction_table`, `plot_fig_generalization` and
`plot_fig_generalization_cells` reach the V72 inputs but stop at `C52 freeze hash`:
the shared helper forces raw digest equality even though the publication digest
map records the exact anonymization correspondence. Their scoring code and digest
acceptance rules are unchanged. The A5/A7 artifact analyses and their 75 tests pass.

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

Run the offline test suite:

```
python3 -m pytest tests
```

Each generated file names its inputs and their SHA-256 digests in a header comment, so a
regenerated table can be compared line by line with the one that was published. Tests that
reach for model weights, adapters, dataset caches or credentials cannot pass from a checkout
alone, since those are not part of the repository, and inputs that were never mirrored cannot
be reconstructed; `docs/RESULTS_LEDGER.md` names the artifact behind every reported number.

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
