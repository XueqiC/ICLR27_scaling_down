# A17: public offline test suite

The public suite bootstraps its inputs before collection and runs on CPU. It does
not need the adjacent manuscript checkout, author deliverables, generated figures,
or the old `code/` tree. All original tests remain; no numerical, identity,
provenance, or figure assertion was removed to obtain a passing result.

The required commands ran consecutively from the public repository, with no code or
input changes between runs. `-rs` and JUnit output were added to record skip reasons
and counts.

| Run | Passed | Skipped | Failed / errors | Time |
|---|---:|---:|---:|---:|
| `python3 -m pytest tests -p no:randomly` | 1,446 | 31 | 0 / 0 | 169.27 s |
| `python3 -m pytest tests` | 1,446 | 31 | 0 / 0 | 173.69 s |
| Standalone public checkout, full suite | 1,446 | 31 | 0 / 0 | 171.02 s |

Each run emitted one existing FontTools deprecation warning. The environment was
Python 3.11.7 / pytest 7.4.0; `pytest-randomly` was not installed, so the second run
used ordinary pytest ordering. These runs demonstrate repeatability, not randomized
ordering coverage. The standalone copy had no manuscript, private deliverables,
old code copies, or preexisting generated outputs; existing Git metadata was copied
for the descriptor tests' commit-provenance stamps, without creating commits.

The 31 skips have identical grouped reasons in all three runs:

| Count | Absent prerequisite and reason |
|---:|---|
| 19 | `results/traces-pilot/gpt-5.6-luna_{math,qa,code}.jsonl`: raw teacher-trace text is deliberately not redistributed (A3). |
| 5 | `results/traces-pilot/gpt-5.6-luna_qa.jsonl` and `claude-sonnet-4-6_qa.jsonl`: raw teacher-trace text is deliberately not redistributed (V34). |
| 1 | `results/**/*.safetensors`: model weights and adapters are deliberately not redistributed. |
| 5 | Full Pythia panel unavailable: the existing V36 exclusion records collapsed upstream Pythia-2.8B step revisions; the available-panel analysis remains tested. |
| 1 | CUDA is unavailable; the suite is forced onto CPU. |

Full local stdout and JUnit evidence are retained under `/tmp/a17-public-suite/`
as `final-no-randomly.{log,xml}`, `final-default.{log,xml}`, and
`standalone-verified.{log,xml}`. They are not published because local execution logs
contain machine paths and host metadata.

`pytest.ini` restricts bare collection to `tests/` and excludes `paper`, `code`,
`data_mirror`, `results`, `generated`, `.git`, and `.venv`. Bare collection in a
standalone copy found 1,477 items, all beneath `tests/`. The initial working-copy
baseline was 47 failed, 1,313 passed, 13 skipped, and 99 setup errors.

The data repair adds 782 files (65,507,835 stored bytes), restores 27 full V91
per-sample descriptors in place of truncated aggregates, and updates the publication
digest ledger. Net `data_mirror/` growth is **67,405,171 bytes**, including the
ledger and descriptor growth. The largest new file is
`v29-prune-diagnosis/summary.json`, 2,958,122 bytes. Every added file is below
50,000,000 bytes and total growth is below 500,000,000 bytes. Large JSON uses
reproducible gzip (empty stored filename, mtime zero). All new JSON measurements
retain their numerical values. Path, author, host, institution, and credential
patterns were checked using the repository's independent anonymity rules.

[The mirror inventory](A17_MIRROR_INVENTORY.csv) lists every added or changed mirror
file, its source, stored/expanded size, original SHA-256, published SHA-256, and
stored SHA-256. [The complete change list](A17_CHANGED_FILES.csv) lists every source
file added or changed by A17, including this report and both inventories. Paths in
these inventories are relative to the public repository. Existing user changes to
the three public documents and the V54 step-64000 inputs were preserved.

The mirror contains saved evaluations, numeric training ledgers, frozen predictions,
summary audits, preregistration, adapter **configuration JSON only**, and read-only
copies of table inputs used by caption/language tests. No model tensors, adapter
tensors, raw teacher-trace JSONL, dataset Arrow caches, or checkpoints were added.
The table copies do not alter the manuscript sources. All 64 manuscript `.tex`
files retained their recorded hashes. The excluded selection-validation experiment
was not modified. No commits were made.

The executable repairs include the root A3 no-ripgrep fallback, the current panel
export helper's `center` argument, the missing corner-figure module, and the root
V17 historical model/bit allowlist. Bootstrap preserves V12's portable student
identifiers as one directory per run: making both `--step` and `@step` aliases
would duplicate observations and violate V12's exact path/metadata checks. Other
state-tagged artifacts retain both spellings. The frozen V24–V27 analysis bytes
were retained; tests point their report checks at the public `docs/` layout.

Test repairs align stale API/key-ring and descriptor mocks with their real
signatures, seed the drift export inside its own test, check both the current
nine-state V92 panel and the historical six-state panel against their respective
published membership, and use mirrored tables instead of private directories.
All original checks remain, including deliberate tampering and missing-input
failures. New bootstrap checks cover V12 identity and the other experiments'
combined directory/filename aliases. The previously skipped A1 development checks
now run against the complete mirrored records.

Publication exclusions use `tests/public_inputs.py`, which checks named
prerequisites before the affected fixture/test runs. It does not intercept arbitrary
file errors or assertion failures. Raw teacher text remains excluded; the weights
inventory integration test requires non-redistributed tensors. A V27c full rebuild
also skips when its external Hugging Face cache inputs are absent: the required
2Wiki Arrow input alone is 54,510,136 bytes. That cache was already present on this
validation machine, so the rebuild ran and its exact source hashes and complete
summary equality were checked. API tests use synthetic credentials and mocked HTTP.

`code/` had zero tracked files and was already ignored. It is unnecessary for the
public suite and current generators, so it should stay out of the public repository.
Its prior local contents were preserved in the validation backup directory and
removed from the working copy. Bootstrap creates legacy code copies only when
explicitly invoked with `--legacy-layout`; older frozen generators still reference
those copies. Test-created `results/`, `generated/`, caches and runtime aliases are
reconstructible outputs rather than publication additions.
