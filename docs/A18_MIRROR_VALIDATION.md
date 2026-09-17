# A18: S3 independent cross-method selection mirror

The public repository now carries the inputs for C86's appendix table and the
saved records needed to inspect its registration, identity gate, anchors and
candidate outcomes. `analysis/s3_paper_table.py` was already byte-identical to the
working generator and was retained unchanged. C86 is copied verbatim into
`docs/RESULTS_LEDGER.md`; its text required no host/user substitutions.

## Regeneration and checks

From the public repository root:

```sh
python3 bootstrap_results.py
python3 -c 'from analysis.s3_paper_table import render; from pathlib import Path; Path("generated/tables/s3_selection.tex").write_bytes(render().encode("utf-8"))'
python3 -m pytest tests/test_s3_paper_table.py -rs
```

The preview writes outside the manuscript. The original generator's direct
module entry point retains its historical `paper/paper/tables/` output path;
use the command above for this public layout.

The new tests copy `score_v2.json` and `plan_v2.json` directly from `data_mirror/`
into a temporary `results/` layout, then compare the renderer's UTF-8 bytes with
`data_mirror/published-tables/s3_selection.tex`. This avoids stale existing
`results/` files, which bootstrap deliberately does not overwrite. A second test
compares those bytes with the adjacent manuscript's `paper/tables/s3_selection.tex`
(workspace path `paper/paper/tables/s3_selection.tex`). Neither test writes to it.

The manuscript comparison uses the existing `require_public_inputs` helper and
skips only when that separately maintained manuscript file is absent, with the
explicit reason `S3_MANUSCRIPT_NOT_DISTRIBUTED`. The mirrored inputs and snapshot
are required: their absence fails rather than skips. In an isolated public test
checkout containing no manuscript, the snapshot test passed and the manuscript
comparison skipped for exactly that named reason: **1 passed, 1 skipped**.
Separate negative checks confirmed that a missing mirrored score fails even
with an existing bootstrapped copy, and a one-byte snapshot change fails the
comparison. The documented preview command also regenerated the exact 1,836-byte
manuscript table, SHA-256
`8c4c6f3922e249778a9d5a0d7050b57e0977599db4391ccb12943d561e8360df`.

Full-suite results from `paper/`:

| Command | Passed | Skipped | Failed / errors | Time |
|---|---:|---:|---:|---:|
| `python3 -m pytest tests` | 1,448 | 31 | 0 / 0 | 169.52 s |
| `python3 -m pytest tests -p no:randomly` | 1,448 | 31 | 0 / 0 | 172.83 s |

Both commands run with CUDA hidden and Hugging Face/Transformers offline. The
environment is Python 3.11.7 / pytest 7.4.0. `pytest-randomly` is not installed,
so these runs check repeatability with ordinary ordering, not randomized ordering.
Each run reported one existing FontTools deprecation warning. Both new S3 tests
passed in each full run; the 31 skips are existing suite exclusions. No code or
evidence inputs changed between the full-suite runs.
Local full stdout is retained in `/tmp/a18-pytest-default.log` and
`/tmp/a18-pytest-no-randomly.log`; execution logs are not published.

## Mirrored inventory

[A18_MIRROR_INVENTORY.csv](A18_MIRROR_INVENTORY.csv) lists every mirrored file with
its exact path, stored/original sizes, original/published SHA-256 and whether it
was anonymised. Paths are relative to this public repository except the `source`
column, which is relative to the working repository.

| Evidence group | Files | Stored bytes |
|---|---:|---:|
| S3 top-level plans, registrations, identity, score, validation and checksum files | 14 | 4,014,062 |
| `anchors/`: eight role records and eight sidecars | 16 | 8,792 |
| `inputs/`: locked models and frozen probes | 2 | 693,975 |
| `measurements/`: 44 dense/pruned/quantized candidate records and sidecars | 88 | 23,486 |
| `students/`: four measurement records and sidecars, four evals, four numeric training logs | 16 | 493,948 |
| **S3 evidence subtotal** | **136** | **5,234,263** |
| `published-tables/s3_selection.tex`: comparison snapshot | 1 | 1,836 |
| **New mirrored files total** | **137** | **5,236,099** |

The largest addition is `s3-selection-validation/identity.json`, **2,231,804
bytes**. Every file is below 50,000,000 bytes. None reaches the existing mirror's
5 MB gzip threshold, so the evidence is stored uncompressed and bootstrap needs
no changes. Existing portable `--step` names are preserved; bootstrap restores
its normal `--step`/`@step` aliases.

The digest ledger grows from 262,113 to 263,142 bytes: **1,029 additional bytes**.
Including that update, net `data_mirror/` growth is **5,237,128 bytes**. The
unchanged 3,322-byte generator and the new test/docs are outside these evidence
totals.

## Anonymisation and provenance

The existing publication anonymiser was applied without new substitutions.
Only `plan_v2.json` and `identity.json` changed: cluster snapshot paths become
`$CLUSTER_PROJECT` paths. Their exact original/published digest pairs were added
to `data_mirror/ANONYMIZATION_DIGESTS.json` with `pairable=true`. All 404 previous
entries were preserved. The complete JSON structures were checked against
string-only anonymisation, including every numeric, Boolean and null leaf.
Independent anonymity patterns found no hits in the additions.

All 60 mirrored `.sha256` sidecars retain their original bytes and validate either
directly or through those two digest pairs. All 48 registered candidates have
measurement records; the four student eval hashes and post-training losses match
their normalized measurement records. The plan's numeric `predictions_sha256`
and `maps_sha256` still verify exactly. No frozen digest or score was rewritten.

`SHA256SUMS_v2` is preserved as historical evidence. Of its entries within this
S3 mirror, 30 validate; one mismatch already exists in the source evidence:

| Artifact | Digest recorded in `SHA256SUMS_v2` | Actual source and published digest |
|---|---|---|
| `score_v2.json` | `92f1a88992b950ecdbafef9a0b630d5d4e5710c44724505817bf861f2a955e70` | `11fe915389a8e2d26067ef048860531e583c68fc6cdcd420cc71b97425feb810` |

This discrepancy is not anonymisation and has no digest-pair exception. The
published score is byte-identical to the provided source score and regenerates
the manuscript table. The other 47 manifest entries concern omitted historical
or operational records, runtime code and external trace inputs; the manifest is
not a checksum list for only this mirror. A18 reproduces the saved table and
publishes audit records; it does not rerun the tensor-identity gate or GPU jobs.

## Deliberate exclusions and preserved files

[A18_MIRROR_EXCLUSIONS.csv](A18_MIRROR_EXCLUSIONS.csv) lists all **36 existing S3
source files omitted**, with their sizes and individual reasons:

- Eight files within the four adapter directories are excluded, including their
  README/configuration metadata. No adapter directory is mirrored.
- Fourteen SLURM stdout/stderr logs are excluded; sealed measurements and numeric
  training records supply the outcome evidence.
- Four cluster preparation, execution, download and cache records are excluded.
- Three retrospective opportunity/preparation reports are excluded because they
  describe earlier candidate coverage, not the completed independent S3 outcomes.
- Seven superseded score/validation or parser/test outputs are excluded; the final
  score and v2 validation record are included.

There is no separate `panels/` directory in the provided source tree. Its panel
measurements are under `measurements/` and are all included. No model weights,
LoRA tensors, optimizer state, `.safetensors` or `.bin` files were copied. None of
those binary files was present in the inspected S3 source tree. The numeric
`train_log.json` records contain training metrics and recipe metadata, not
serialized optimizer state. External teacher-trace text and checkpoint/dataset
caches are not needed for this table and remain outside the mirror.

Final verification confirmed that all **172 source evidence files** and all
**65 manuscript `.tex` files** retained their original SHA-256 digests and
modification times. No commits were made; pre-existing changes outside A18 were
retained.
