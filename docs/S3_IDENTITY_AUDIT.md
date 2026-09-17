**A20: identity and provenance of the selection validation.**

Only **Pythia 410 million and Pythia 1.4 billion at training step 120000** can be called new source states relative to the searched registered records. **Gemma 3 one billion and Gemma 3 four billion cannot.** Their initial releases had already been studied, including compression and distillation outcomes. S3 supplies fresh runs and new selection decisions on these Gemma references, not first use of those weights. Even “new intervention” needs qualification: several settings and the student recipe repeat earlier conditions.

A narrower statement is supported: none of the four S3 references, and none of its student bases at these exact stages, supplied response labels to **the particular V78 locked selection predictor**. That predictor was fitted on Pythia states listed below. This does not make Gemma unseen in the wider development, model-selection or confirmation history.

The identity artifact's PASS is relative to **thirty Pythia revisions in the registered exclusions**, not every state ever used in the repository. There are eight roles but only six unique snapshots. Pythia 410 million at step 120000 is both a reference and another reference's student base; Gemma 3 one billion is similarly shared. Shared training traces and probe texts, and the seventeen budgets evaluated on one candidate panel, provide no additional independent experimental replications.

**Exact identities and storage convention.** All eight roles use pretrained base releases, not instruction-tuned releases. Model names are exact registry identifiers. Pythia's requested revision is `step120000`; the identity artifact resolves it to the immutable commits below. Gemma was already pinned by commit. The count is the number of text decoder transformer-matrix parameters used in `plan_v2.json`, verified against the learned-tensor shapes. It excludes input embeddings, output heads, biases, normalization parameters, Gemma 4 billion's vision tower and multimodal projector. It is not the model-name parameter total or measured checkpoint size. Nominal storage is density for pruning, bits/16 for per-channel quantization, (bits + 16/group size)/16 for grouped quantization, and student count/reference count for a merged student.

| Role | Exact model identifier | Requested revision | Resolved commit | Matrix parameters | Initial weights seen before? |
|---|---|---|---|---:|---|
| Reference 1 | `EleutherAI/pythia-410m` | step120000 | `ade882a342507b4be34455393ba750fee7a513b8` | 301,989,888 | No earlier occurrence found at this stage. |
| Student for reference 1 | `EleutherAI/pythia-160m` | step120000 | `3df7d2c5532755f7022704e07dea87f35eeaf0e1` | 84,934,656 | No earlier occurrence found at this stage. |
| Reference 2 | `EleutherAI/pythia-1.4b` | step120000 | `4e056d657ceede13435146a8126bbf7bda0a1b0e` | 1,207,959,552 | No earlier occurrence found at this stage. |
| Student for reference 2 | `EleutherAI/pythia-410m` | step120000 | `ade882a342507b4be34455393ba750fee7a513b8` | 301,989,888 | No earlier occurrence found at this stage. |
| Reference 3 | `google/gemma-3-1b-pt` | Pinned commit | `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` | 697,761,792 | Yes; see evidence below. |
| Student for reference 3 | `google/gemma-3-270m` | Pinned commit | `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1` | 100,270,080 | Yes; see evidence below. |
| Reference 4 | `google/gemma-3-4b-pt` | Pinned commit | `cc012e0a6d0787b4adcc0fa2c4da74402494554d` | 3,208,642,560 | Yes; see evidence below. |
| Student for reference 4 | `google/gemma-3-1b-pt` | Pinned commit | `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` | 697,761,792 | Yes; see evidence below. |

Release classification is consistent with the local `analysis/model_registry.py` and the publishers' model documentation: [Pythia model card](https://huggingface.co/EleutherAI/pythia-410m), [Gemma 3 release guide](https://developers.googleblog.com/introducing-gemma3/), and [Gemma 3 270 million release announcement](https://developers.googleblog.com/pt-br/introducing-gemma-3-270m). The unsuffixed `google/gemma-3-270m` is the pretrained release; its instruction-tuned counterpart has the `-it` suffix. These web sources classify the releases; the repository's identity records, not current remote branch heads, establish the resolved commits used here.

**Earlier use and earlier outcomes, with the limits of exact identity evidence.**

| Snapshot and S3 roles | Earlier initial-state evidence | Earlier outcomes and fitting or confirmation rounds | What S3 adds |
|---|---|---|---|
| Pythia 410 million, step 120000; reference 1 and student for reference 2 | No earlier occurrence of this revision or resolved commit was found in the searched records. The identity report also distinguishes its learned values from every registered excluded Pythia revision. | No earlier compression or distillation outcome found for this stage. The same architecture at other training steps occurs in V36, V39, V53, V64, V69 and V78; those are different source states. | Fresh compression and a freshly trained student at the new stage. The shared reference/student role is not a second independent snapshot. |
| Pythia 1.4 billion, step 120000; reference 2 | No earlier occurrence of this revision or commit found; identity comparison passes against the registered exclusions. | No earlier outcome found for this stage. Earlier 1.4 billion stages occur in the same controlled Pythia development history. | New source state and fresh compression runs. |
| Pythia 160 million, step 120000; student for reference 1 | No earlier occurrence of this revision or commit found; identity comparison passes against the registered exclusions. | No earlier outcome found at this stage. Smaller or earlier-stage Pythia students do not establish reuse of these initial weights. | Fresh training from a base state not previously recorded at this stage. |
| Gemma 3 one billion; reference 3 and student for reference 4 | `results/v32-descriptors/gemma3-1b/features.json` records the exact identifier and `fcf18a2...` weight revision. `results/v47-p2-register/register.json` pins that same `student_snapshot`. `results/v71-qa-scope/register.json` and `measurements.json` pin it again for a completed confirmation. | V6 pruning, V10 per-channel quantization and V12 distillation already have outcomes for this release. V71 independently pins compression outcomes to this exact revision, including density 0.7, per-channel four bits and grouped four bits/group 128. V50/V56 used Gemma trajectories in development and tests; V70 reclassified prior observed trajectories as development and confirmed on new pools. V99 and later A12 add further readouts/confirmation. These Gemma fits are separate from the V78 selection predictor. | Fresh compression measurements and a fresh one-billion student run for reference 4. Several compression conditions and the two-epoch, first-600-row recipe were already studied. |
| Gemma 3 four billion; reference 4 | `results/v77-model-arch/model_arch.json` records `cc012e0...`. Stronger evidence of actual base loading occurs in `results/v99-scope/rerun-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41-0e02d1102f506ae9/update-00000152.json`: `model_local_path` resolves to that exact weight snapshot. | V6 pruning, V10 per-channel quantization and V12 distillation already have release-level outcomes. Four-billion trajectories occur in V56's tests and V70's later 25-trajectory development fit. V99 records an adapter outcome evaluated on this exact base. Its `adapter_base_unrecorded=true` explicitly prevents certifying the original adapter's training initialization from that record alone. A12 also studies this release; its dates are not needed to establish the earlier V-numbered reuse. | Fresh compression runs on a previously studied source; the smaller one-billion student is newly trained for this selection round. |
| Gemma 3 270 million; student for reference 3 | `results/v32-descriptors/gemma3-270m/features.json` records actual weight features at the exact `9b0cfec...` revision, and V77 records the same snapshot. | V6 pruning, V10 quantization and V12 distillation already have outcomes for the same named release. V50/V56 develop distillation forms on this release, and V70 both refits previous trajectories and confirms new pools. However, the old V6/V10/V12 outcome files do not record the weight commit, and a pinned tokenizer in a later registration is not a retrospective weight-identity proof. Exact historical outcome identity is therefore unverified for this snapshot from those files alone. | Fresh training from previously measured base weights. The first-600-row, two-epoch recipe repeats the earlier release-level recipe. |

The oldest outcomes are `results/v6-capability-geometry/gemma3-{270m,1b,4b}/prune_losses.json`, `results/v10-quantization/gemma3-{270m,1b,4b}/quant_losses.json`, and `results/v12-distill/gemma3-{270m,1b,4b}/gpt-5.6-luna_full_600/eval.json`. All contain measured outcomes, not merely planned commands. The old 270-million and one-billion `train_log.json` records show two epochs, seed zero, the full recipe and low-rank adaptation. Their incomplete historical revision/library records prevent a claim of bitwise optimization equivalence. Fresh reruns are not copies of those outcome files, but neither are they first-ever interventions on the releases.

`results/v70-distill-confirm/develop.json` explicitly pools 12 earlier development trajectories, nine earlier test trajectories and four held-out four-billion trajectories: 25 trajectories and 100 points. Its points contain 36 observations each for Gemma 270 million and one billion and 28 for four billion. This is a concrete earlier **fit** containing the Gemma references and student release. Thus “outside every earlier fit” is too broad even though the V78 selection fit is disjoint.

The direct scope search read 3,571 registered text files, including both `docs/RESULTS_LEDGER.md` and `paper/docs/RESULTS_LEDGER.md`, `notes/exp_log.md`, and V-numbered results. [s3_identity_search.json](s3_identity_search.json) records every searched file digest, search expression and matching path. A match in a summary, code hash list, tokenizer record or configuration inventory is only a search lead, not automatically evidence of using exact initial weights. The Pythia step-120000 matches outside S3 also include the concurrent `s3-selection-control` output: it declares itself a retrospective A19 analysis of the frozen S3 panel, so it is not an earlier round. No negative search proves absence from unregistered or unavailable work.

**What fitted the predictor actually used.**

`results/s3-selection-validation/inputs/locked_models.json` is object-identical to `models.locked` in `results/v78-rule-confirm/freeze-independent.json`. The construction is in `analysis/v78_rule_confirm.py:development_objects`; prediction is in `analysis/final_rule.py` and the S3 amendment's explicit Gemma override in `analysis/s3_freeze.py`. No new coefficient is fitted by S3. V78's second, comparator object `models.v64` is not the predictor copied into S3.

The following roster is extracted from the frozen fit artifacts, not from mutable directory globs:

| Branch | Frozen fit and measured input directories | Exact states and configurations |
|---|---|---|
| Pruning | `results/v53-prune-dev/register.json`; outcomes from `results/v6-capability-geometry/` | Seventeen states: Pythia 160 million, 410 million and 1.4 billion each at steps 16000, 64000, 96000 and 143000; Pythia one billion at steps 32000, 96000 and 112000; Pythia 6.9 billion at steps 32000 and 112000. All measured densities from 0.55 through below 1 enter: 84 state-density cells, 252 capability responses. The exact per-state density list follows below. |
| Per-channel quantization | Rows frozen in `results/v64-selection-feasible/summary.json`, originally from `results/v10-quantization/`; V36 fitting functions, through V64 and V78 | Three and four bits: all seventeen pruning roster states. Six and eight bits: those states except Pythia one billion at step 96000, hence sixteen each. Five bits: Pythia one billion and 6.9 billion each at steps 32000 and 112000, hence four. Counts are per capability; the roster is not a full seventeen-by-five rectangle. |
| Grouped quantization | `results/v69-quant-confirm/develop.json`; outcomes from `results/v54-quant-group/`, with the earlier `results/v55-quant-group/register.json` retained as provenance | Pythia 160 million, 410 million and 1.4 billion each at steps 16000 and 143000. Six states times three bit widths (3, 4, 5) times three group sizes (64, 128, 256): all 54 formerly unblinded state-configuration cells, 162 capability responses. This is the V69 refit, not only the earlier V55 training subset. |
| Distillation | Nine-student cohort in `results/v39-distill-controlled/summary.json`, outcomes from `results/v12-distill/`, frozen into `results/v64-selection-feasible/summary.json` and fitted/purged by V78 | Seven fitted students: Pythia 160 million at steps 16000 and 143000; Pythia 410 million at steps 16000 and 143000; Pythia 1.4 billion at steps 16000, 64000 and 143000. The 160-million and 410-million students at step 64000 are excluded because V78 offers their historical outcomes as candidates. Every input run is `gpt-5.6-luna_full_600_lora/eval.json` under its size/step directory. |

For pruning, these are the exact recorded training densities:

| State | Retained densities |
|---|---|
| `pythia-1.4b@step143000` | 0.9, 0.8, 0.7, 0.6 |
| `pythia-1.4b@step16000` | 0.9, 0.8, 0.7, 0.6 |
| `pythia-1.4b@step64000` | 0.9, 0.8, 0.7, 0.65, 0.6, 0.55 |
| `pythia-1.4b@step96000` | 0.9, 0.8, 0.7, 0.6 |
| `pythia-160m@step143000` | 0.9, 0.8, 0.7, 0.65, 0.6, 0.55 |
| `pythia-160m@step16000` | 0.9, 0.8, 0.7, 0.6 |
| `pythia-160m@step64000` | 0.9, 0.8, 0.7, 0.6 |
| `pythia-160m@step96000` | 0.9, 0.8, 0.7, 0.6 |
| `pythia-1b@step112000` | 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55 |
| `pythia-1b@step32000` | 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55 |
| `pythia-1b@step96000` | 0.65, 0.55 |
| `pythia-410m@step143000` | 0.9, 0.8, 0.7, 0.6 |
| `pythia-410m@step16000` | 0.9, 0.8, 0.7, 0.6 |
| `pythia-410m@step64000` | 0.9, 0.8, 0.7, 0.6 |
| `pythia-410m@step96000` | 0.9, 0.8, 0.7, 0.65, 0.6, 0.55 |
| `pythia-6.9b@step112000` | 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55 |
| `pythia-6.9b@step32000` | 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55 |

All S3 references use the predictor's `new_stage` dispatch: development medians for pruning and per-channel quantization, and the grouped development median grid with the already locked boundary interpolation/extrapolation rule. For Pythia distillation, mathematics uses the frozen linear function of student matrix count, pristine student mathematics loss, and same-stage pretraining tokens. Here the latter equals 120000 × 2097152 = 251,658,240,000. Code and question answering use the frozen arithmetic means. For Gemma, the amendment uses the existing arithmetic mean for mathematics as well, since its same-stage pretraining input is not registered: mathematics 0.0666998113015672, code 0.1804815104078237, question answering −0.4803817558392179, each added to the corresponding pristine **student** loss. Gemma pretraining totals from a model card cannot replace that missing registered same-stage covariate.

V78's confirmation source states were Pythia 160 million, 410 million and 1.4 billion at step 32000, plus Pythia one billion at step 64000. Their measured confirmation outcomes did not refit these locked objects. The S3 Gemma `state_status="new_stage"` is a prediction-branch choice relative to that Pythia predictor, not proof that Gemma is historically unseen. The exact fit rosters and all consumed evidence digests are also machine-readable in [s3_identity_audit.json](s3_identity_audit.json).

**Freeze and chronology.**

The original `results/s3-selection-validation/prereg.md` registered a blocked five-slot preparation. `prereg_amendment_1.md` fixed the operative four-slot roster, dropped the problematic Pythia 2.8 billion slot, restored the first-600-row/two-epoch student recipe, and specified the Gemma mathematics fallback. `plan_v2.registered.json` preserves the contract before dense-anchor finalization; `plan_v2.json` is the finalized registration with numeric predictions and maps. The amendment itself limits identity independence to the original exclusion roster.

| Artifact or canonical object | SHA256 |
|---|---|
| `results/s3-selection-validation/prereg.md` | `a7c765bc5b3a2f147477c1b269be40b8d319d276d50c8d78cced956a483abdae` |
| `results/s3-selection-validation/prereg_amendment_1.md` | `c60e1312746317d73f83a7498d537a67cd7765ec7f68910ff0983a33eec67ecf` |
| `results/s3-selection-validation/plan.json` | `181d92f3320df078845f7151894146ea9c1859e0b77379ced8bb5d4798ae7c7c` |
| `results/s3-selection-validation/plan_v2.registered.json` | `804fbfb6f795af7fd86ee859a46bf69b82cc5f58c148096557f7382dce794132` |
| `results/s3-selection-validation/plan_v2.json` | `302efd715f307d6127e1025796f8a15419820790534c5a3df44b28af7d2139d9` |
| `results/s3-selection-validation/identity.json` | `2b6a3b5d31a2a19c88a1a5cea7515345ec357d2163564d515979cb1f2ba5788a` |
| `results/s3-selection-validation/inputs/locked_models.json` | `2c8d28bd83cc74100048be07bc3d1ea7247e0774f8efe4569b9cbaeb08779fae` |
| `results/s3-selection-validation/inputs/probes.json` | `f957c63ded36c9b991671e55a9b222f4cfbed62507bf1c7ba82c5f1062af94b8` |
| `results/s3-selection-validation/SHA256SUMS` | `3032b727c0019d3a2895ca1b9eec5c76412a361b51888e0b29128e33055bc092` |
| `results/s3-selection-validation/SHA256SUMS_v2` | `189fb8b280cf9bb9e5c1452d5ed4d50d6b552d0dd9fa5e91299413aa33e60e98` |
| `registration_sha256` (canonical object, not file bytes) | `af3c1c7734fa15c835bd8463dcb313c51c1d0b31109fc606c053dee392c7d5b9` |
| `predictions_sha256` (canonical object, not file bytes) | `d28986ca6b13fe833e965f185de25af01c95d0f256cd36e448e53a9c9d982acd` |
| `maps_sha256` (canonical object, not file bytes) | `5f976cca5a6f3e995c96b1db8e5e2703f5c8f9a9c071333f77d621bebe2c4966` |
| `results/v78-rule-confirm/freeze-independent.json` | `54b53f547800957f4be011ba4c3cbc50302144576de36f6da0e0bd63be6141a5` |

The generator checks original file sidecars, canonical registration/prediction/map seals, the freeze's identity/anchor digest bindings, the locked model object, and all 48 current candidate measurement records, including the four student evaluation hashes and their pinned initialization. The original preparation printout is not an input to this generator. The public mirror has anonymized cache paths; only explicitly `pairable=true`, path-bound original/published digest pairs in `data_mirror/ANONYMIZATION_DIGESTS.json` are accepted. The original and published bytes are not claimed identical. `SHA256SUMS_v2` is a freeze-time manifest: its scoring-placeholder entry does not authenticate the later completed score, as the existing ledger already explains.

Only the final freeze has an internal timestamp: **2026-09-16T23:31:54.810499+00:00**, stored at `plan_v2.json.final_freeze.timestamp_utc`. Neither `identity.json` nor any of the eight anchor JSON bodies stores a UTC timestamp. The identity output names job 1030713, and all anchor bodies record job 1030747. The following UTC values are the local evidence files' modification times, recorded here as filesystem metadata, **not sealed event timestamps**. Copying or touching a file can change them, so they cannot establish independently authenticated start/end times.

| Artifact | Observed modification time in UTC | Internal event time |
|---|---|---|
| `identity.json` | `2026-09-16T23:18:20+00:00` | Not recorded |
| `anchors/gemma3-1b__reference.json` | `2026-09-16T23:24:38+00:00` | Not recorded |
| `anchors/gemma3-1b__student.json` | `2026-09-16T23:25:49+00:00` | Not recorded |
| `anchors/gemma3-4b__reference.json` | `2026-09-16T23:30:27+00:00` | Not recorded |
| `anchors/gemma3-4b__student.json` | `2026-09-16T23:30:27+00:00` | Not recorded |
| `anchors/pythia-1.4b--step120000__reference.json` | `2026-09-16T23:23:21+00:00` | Not recorded |
| `anchors/pythia-1.4b--step120000__student.json` | `2026-09-16T23:23:21+00:00` | Not recorded |
| `anchors/pythia-410m--step120000__reference.json` | `2026-09-16T23:21:37+00:00` | Not recorded |
| `anchors/pythia-410m--step120000__student.json` | `2026-09-16T23:21:50+00:00` | Not recorded |
| `plan_v2.json` | `2026-09-16T23:31:54+00:00` | 2026-09-16T23:31:54.810499+00:00 |

Identity and pristine anchors are bound by their hashes in the final freeze. The code requires these steps before finalization and refuses finalization once student or compression output directories exist. The recorded dependency and seals support the intended order; they are not an independent clock attestation. Preregistration and amendment bodies also lack internal creation timestamps. The preserved registered plan has a modification time at finalization because that is when its bytes were archived; that time must not be presented as its original registration time.

The eight anchor file digests are:

| Anchor | SHA256 |
|---|---|
| `gemma3-1b__reference.json` | `cf5561a55464cfe7cc6aa12d318fe4588d3779b15b83135b66056c42cb4e0374` |
| `gemma3-1b__student.json` | `97883948a6dcec80d7f10be843f5b1db51f8713c75e2704c3044434e48cd41a2` |
| `gemma3-4b__reference.json` | `941d145cc79e84b7097d54bbb8c3b9ac6ee2e3155574623278f8e25beb9102dc` |
| `gemma3-4b__student.json` | `022ea8fb7da081553aa5dd3fcefc0670e1f36193648de4e6b36471046c18abca` |
| `pythia-1.4b--step120000__reference.json` | `d13e12aaf24df6b1d65862653218ff1561bb8447b07541e0534034b816811352` |
| `pythia-1.4b--step120000__student.json` | `4c3c2170149b374a9e988ec7e9f3f2af5bc1011f891283668fdd75a488434a9c` |
| `pythia-410m--step120000__reference.json` | `cb42a12e2f6fbee15a42da583b345a3edd4fe0c80b1d010fd54a40bca98f5975` |
| `pythia-410m--step120000__student.json` | `01dbada499500a7e2aae30abbb3a67ffbaf28a086a6d342f7c225f81d55e0842` |

**Exact manuscript sentences requiring qualification.** The quotations below preserve the source text and line breaks. No manuscript source was edited by A20.

In `paper/paper/selection.tex`, starting at line 19:

```tex
Four references outside every earlier fit, two Pythia checkpoints at a new training stage and two
Gemma-3 models, each received twelve candidates at seventeen storage budgets: four pruning
densities, six quantization settings, the dense model, and one new student distilled under the
recipe on which the delivered relation was fitted.
```

```tex
The learned weights of every reference and student base were checked against all thirty earlier
revisions, and the predicted loss of every candidate and both maps were sealed before any student
was trained or any candidate compressed.
```

In `paper/paper/appendix.tex`, starting at line 726:

```tex
The confirmation of \S\ref{sec:selection} used four references that entered no earlier fit, two
Pythia checkpoints at step 120000 and two Gemma-3 models, each with four pruning densities, six
quantization settings, the dense model and one student distilled in this round under the recipe on
which the delivered relation was fitted.
```

```tex
Before any outcome, the learned values of all six
snapshots were compared with all thirty revisions used earlier, 180 comparisons in which every
target proved distinct; the dense reference and pristine student losses were measured on the frozen
probes; and 144 capability predictions, 48 largest-increase predictions and both selection maps
were sealed.
```

The first sentence in each file should distinguish two new Pythia source states from two previously studied Gemma references, and restrict exclusion from fitting to the locked selection predictor. The second sentence in each should say “the thirty Pythia revisions in the registered exclusion list,” rather than implying an exhaustive inventory of earlier states. A suitable factual replacement is:

> The round used two Pythia checkpoints at a previously untested training step and two Gemma base releases studied in earlier experiments. None of these snapshots supplied response labels to the locked selection predictor. Compression candidates were measured anew and students were trained anew, although some Gemma settings and training recipes had been studied before. The identity check compared the six distinct snapshots with the thirty Pythia revisions in the registered exclusion list. Pristine losses were then measured, and candidate predictions and both selection maps were sealed before this round's training and compression measurements.

The appendix subsection title “Independent cross-method validation” and the existing `s3_selection.tex` caption (“four references that entered no earlier fit”) should be scoped consistently if the manuscript is revised. A20 does not edit either. There is also a separate provenance precision issue in the appendix sentence “A linear form in the student state, fit on nine Pythia states, serves only as the math predictor for the distillation candidates of the selection rule in \S\ref{sec:selection}, and enters no other relation.” The original cohort has nine states, but the locked V78/S3 predictor is fitted after excluding two, so its actual fitted cohort is seven.

**Deliverables and validation.** `analysis/s3_identity_table.py` generates `paper/paper/tables/s3_identity.tex` and `docs/s3_identity_audit.json`. The generator and `tests/test_s3_identity_table.py` are mirrored byte-for-byte under `paper/`. The table uses full model identifiers, full resolved commits, exact matrix counts, eight explicit roles, distinct columns for earlier weights, earlier outcomes and this round's fresh work, a full-sentence caption, `tabular*` at the text width, and `[!htbp]`. It is supplied as a standalone table; no manuscript input command was inserted.

Reproduce from the workspace with `CUDA_VISIBLE_DEVICES='' python -B analysis/s3_identity_table.py`. In the public repository, use `CUDA_VISIBLE_DEVICES='' python -B analysis/s3_identity_table.py --output generated/tables/s3_identity.tex`; it reads `data_mirror/` directly and requires no results bootstrap. Run the focused tests in either repository with `CUDA_VISIBLE_DEVICES='' python -B -m pytest --noconftest -q tests/test_s3_identity_table.py -o cache_dir=/tmp/a20-pytest-cache`. `--noconftest` intentionally avoids the public suite's unrelated results bootstrap so this task never writes to evidence directories.

Validation completed: **12 tests passed in the workspace and 12 passed in the public mirror**. The private and public renderings are byte-identical. A standalone LaTeX build using the manuscript's 5.5-inch text width and 9-inch text height produced one page, with no overfull boxes or oversized-float warnings; the rendered page was visually inspected. All 146 consumed evidence files retained their digests after generation and testing. Hashes of the manuscript's existing top-level `.tex` files match the pre-edit audit snapshot. A20 used no GPU, performed no fitting or training, wrote nothing under `results/`, and made no commits. Other concurrent worktree changes were left intact.
