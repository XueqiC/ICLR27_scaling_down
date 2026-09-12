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

Regenerate a table or figure from the mirrored measurements:

```
PYTHONPATH=code python3 code/analysis/v86_main_table.py
```

Run the offline test suite:

```
PYTHONPATH=code python3 -m pytest code/tests
```

The generators read the frozen JSON under `data_mirror/`. Each generated file names its
inputs and their SHA-256 digests in a header comment, so a regenerated table can be compared
line by line with the one that was published. Tests that reach for model weights, adapters,
dataset caches or credentials cannot pass from a checkout alone, since those are not part of
the repository.

## Layout

| Path | Contents |
|---|---|
| `code/analysis/` | the pipeline, one module per experiment, named `v<N>_<topic>.py` |
| `code/tests/` | offline test suite |
| `code/configs/` | sweep configurations |
| `code/*.sh` | grid drivers for the larger measurement runs |
| `data_mirror/` | frozen registers, predictions, measurements and comparison files, one directory per experiment |
| `results/` | curated per-experiment reports and summaries |
| `docs/` | results ledger, claim-evidence matrix, measurement definitions, pre-registrations, reference verification, reviewer self-audit |

## Measuring a model

Checkpoints resolve through `code/analysis/model_registry.py`; all of them are public and
pinned by revision.

- Pruning: `code/analysis/v6_capability_geometry.py` applies global magnitude pruning at a
  deterministic sampled threshold and scores the probe sets.
- Quantization: `code/analysis/v10_quantization.py` for per-channel round-to-nearest,
  `code/analysis/v54_quant_group.py` for grouped round-to-nearest with a choice of symmetric
  or asymmetric mode.
- Distillation: `code/analysis/v12_distill.py` trains a LoRA student on stored teacher
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
`code/analysis/model_registry.py` and `docs/MEASUREMENT_DEFINITIONS.md` for the sources.
