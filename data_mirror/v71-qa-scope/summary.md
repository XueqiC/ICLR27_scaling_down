# V71 QA scope

Status: 24/24 cells measured. Unmeasured values are shown as --; no empirical conclusion until evaluation.

Gemma-3-1B; eight pre-specified states; 256 fixed references per set. Delta = state loss minus dense loss on the same set; negative means improvement.

Use the supplied question and existing reference answer on every set; supply context where the set has it. 2Wiki uses all supplied paragraphs; MuSiQue uses supporting paragraphs only (V67); TriviaQA rc.nocontext supplies none (V48). No retrieved context, chat template, generated rationale, or reference in the prompt. Thus the same available-information rule is used, but these are not matched-context datasets. Context availability and support selection limit cross-set causal interpretations. V6 direct tokenization, exactly one Gemma prompt BOS, no completion special tokens, leading answer space, right truncation at 512 prompt and 512 completion tokens. The question may be truncated after a long context, exactly as in V6/V67. Batch size 1; sum conditional completion CE / sum completion tokens, native-token nats.

## Pre-specified register

| State | Nominal supervised T | Actual supervised T | Processed trigger | Checkpoint |
| --- | ---: | ---: | ---: | --- |
| Dense | -- | -- | -- | -- |
| Prune d=0.85 | -- | -- | -- | -- |
| Prune d=0.7 | -- | -- | -- | -- |
| RTN int4/channel | -- | -- | -- | -- |
| RTN b=4 g=128 | -- | -- | -- | -- |
| KD T=35k | 35000 | 36203 | 119000 | update-00000026 |
| KD T=140k | 140000 | 138524 | 475000 | update-00000100 |
| KD T=280k | 280000 | 280329 | 949000 | update-00000199 |

## Sample and overlap audit

2Wiki: seed 71; 64 old probe and 64 old measurement indices recovered; 128 source rows excluded including duplicate questions. Selected overlap: 0. V6 files contain counts, not indices. Replayed the actual V6 builder (shared math/code/QA RNG); checked all 64 measurement prompt/reference strings against V15. Exclude both halves and their duplicated question strings.

MuSiQue: seed 0; 2417 answerable rows checked against 6 full V12 teacher pool files (600 unique question strings). Overlap count: 0 items / 0 unique questions; excluded before sampling. Selected overlap: 0. V34 question extraction; NFKC + casefold + collapsed whitespace, punctuation preserved; exact question equality. All QA trace rows must parse. Non-QA rows without a Question marker have no QA question string. No semantic or pretraining overlap claim.

TriviaQA: seed 0; V48/V9 rc.nocontext validation builder.

Source hashes, exact source indices, adapter hashes, and prompt/reference hashes are in register.json.

## Conditional loss

| State | Set | Loss | Delta from dense | Tokens | n |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense | 2wiki_new | 5.795532 | +0.000000 | 957 | 256 |
| Dense | musique | 1.919070 | +0.000000 | 1205 | 256 |
| Dense | triviaqa | 2.713957 | +0.000000 | 670 | 256 |
| Prune d=0.85 | 2wiki_new | 6.203302 | +0.407769 | 957 | 256 |
| Prune d=0.85 | musique | 2.135198 | +0.216128 | 1205 | 256 |
| Prune d=0.85 | triviaqa | 2.869875 | +0.155917 | 670 | 256 |
| Prune d=0.7 | 2wiki_new | 6.510384 | +0.714852 | 957 | 256 |
| Prune d=0.7 | musique | 3.186307 | +1.267237 | 1205 | 256 |
| Prune d=0.7 | triviaqa | 5.619263 | +2.905306 | 670 | 256 |
| RTN int4/channel | 2wiki_new | 6.594370 | +0.798838 | 957 | 256 |
| RTN int4/channel | musique | 3.324274 | +1.405204 | 1205 | 256 |
| RTN int4/channel | triviaqa | 4.360080 | +1.646123 | 670 | 256 |
| RTN b=4 g=128 | 2wiki_new | 5.759348 | -0.036184 | 957 | 256 |
| RTN b=4 g=128 | musique | 2.136030 | +0.216960 | 1205 | 256 |
| RTN b=4 g=128 | triviaqa | 3.088668 | +0.374711 | 670 | 256 |
| KD T=35k | 2wiki_new | 4.247241 | -1.548291 | 957 | 256 |
| KD T=35k | musique | 1.995124 | +0.076054 | 1205 | 256 |
| KD T=35k | triviaqa | 2.959508 | +0.245550 | 670 | 256 |
| KD T=140k | 2wiki_new | 4.537413 | -1.258119 | 957 | 256 |
| KD T=140k | musique | 2.114293 | +0.195223 | 1205 | 256 |
| KD T=140k | triviaqa | 3.058004 | +0.344047 | 670 | 256 |
| KD T=280k | 2wiki_new | 4.917410 | -0.878123 | 957 | 256 |
| KD T=280k | musique | 2.123510 | +0.204440 | 1205 | 256 |
| KD T=280k | triviaqa | 2.966772 | +0.252814 | 670 | 256 |

## Three pre-stated readings

Apply the three readings separately at each of the three fixed KD budgets. Reproduction means delta from that set's dense loss < 0 (descriptive sign only, not statistical significance). Zero is no gain. MuSiQue-only is reported as an unanticipated fourth outcome, never forced into the three planned readings. Report all budgets and both compression arms; do not select a favorable budget. TriviaQA is the V48 control and does not determine the primary/MuSiQue reading. The three adapters are dependent checkpoints of one trajectory, not three seeds.

- **Both new primary and MuSiQue reproduce**: QA loss gains extend beyond the original primary sample to a second context-supplied distribution.
- **Only the new primary sample reproduces**: QA loss gains replicate within 2Wiki but remain distribution-specific.
- **Neither reproduces**: The original QA gain does not replicate on either new panel; an original-sample-specific reading is warranted.

| KD budget | Observed reading |
| --- | --- |
| 35000 | Only the new primary sample reproduces |
| 140000 | Only the new primary sample reproduces |
| 280000 | Only the new primary sample reproduces |
