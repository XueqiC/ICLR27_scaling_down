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
python3 analysis/v86_main_table.py
```

`bootstrap_results.py` materialises `results/` from `data_mirror/`, expanding the gzipped
summaries and creating the directories the generators write into. It never overwrites an
existing file and never modifies `data_mirror/`.

One artifact exists in two states, because experiments recorded it at different times: the
grouped-quantization measurements grew new configurations after an earlier audit had recorded
their digest. `data_mirror/v54-quant-group/` carries the current file, which the
capability-conditioning audits need, and `data_mirror/v54-quant-group-as-recorded-by-v55/`
carries the state the earlier audit recorded. Copy the latter over the former to rebuild that
earlier audit; the ledger says which entry used which.

What runs from a plain checkout, measured on a fresh clone of this repository after
`bootstrap_results.py`: the main prediction tables and the audits behind them regenerate
(`v45_main_table`, `v52_prediction_tables`, `v63_quant_identifiability`,
`v74_quant_threeway`, `v84_main_table`, `v85_selection_decomp`, `v86_main_table`). The two
capability-conditioning audits (`v76_cap_conditioning`, `v79_cond_audit`) stop on their own
integrity check, because an input they recorded at freeze time is not byte-identical to the
copy published here; `data_mirror/ANONYMIZATION_DIGESTS.json` records every such difference
and its reason, and `docs/RESULTS_LEDGER.md` reports their results. `v79_cond_audit` also
shells out to `ripgrep`.

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
