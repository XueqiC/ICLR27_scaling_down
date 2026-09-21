# A21: refresh of the published-table snapshots

`data_mirror/published-tables/` holds the published text of each manuscript
table. Each entry is meant to be byte-identical to `paper/tables/<name>.tex` in
the separately maintained manuscript repository, so that a reader of this
repository sees the table as it appears in the paper. `bootstrap_results.py`
copies the directory into `results/published-tables/`, where
`tests/test_caption_wording.py` reads it; bootstrap deliberately never
overwrites an existing file, so a stale local `results/` copy has to be
replaced by hand when the mirror moves.

The mirror was last synced on 2026-09-17. The manuscript then went through the
2026-09-18 appendix-table compression and the 2026-09-21 review round without a
matching mirror commit, so 37 of the 47 entries had fallen behind. All of them
are refreshed here, and the two manuscript tables that the directory never
carried are added.

Manuscript head at the time of this refresh: `e53364f`.

## What changed

`docs/A21_PUBLISHED_TABLES_INVENTORY.csv` lists every entry with its previous
and current byte count and SHA-256, and its status: 37 refreshed, 10 already
current, 2 added (`s3_control.tex`, `s3_identity.tex`). After the refresh every
entry in the directory is byte-identical to its manuscript source.

The refresh moves published text only; no analysis was rerun and no generator
input changed. Comparing the multiset of numbers in each file before and after,
ignoring the column widths in the `tabular*` preamble, only six files differ in
their numbers at all, and each difference is one of the manuscript passes that
the mirror had not yet caught up with:

- `models_domains.tex` loses five copies of the target-calibration count `0` and
  `rule_confirm.tex` loses thirty-two copies of `68`. These are the two constant
  columns dropped in the 2026-09-18 compression, whose values moved into the
  captions. Nothing was added to either file.
- `dreq_main.tex` and `dreq_full.tex` change every number and add exactly as many
  as they remove: the same pass reduced their printed precision. Each removed
  value rounds onto the added value that replaces it, for all 497 and 132
  occurrences; two pairs collapse together, `9.105` and `9.106` both printing as
  `9.1`. No measurement changed.
- `shared_structure.tex` is the two-table split recorded in the 2026-09-17
  amendment, together with the column widths the split required. Its measured
  values all survive: the held-out errors `0.494`, `2.962`, `2.398`, `0.542`,
  `0.275`, `0.491`, `3.999` and the three exponent triples.
- `main_prediction_v2.tex` loses the earlier unseen-bit-width row, which is why
  `0.19`, `0.21`, `0.44`, its baseline `0.55`, `0.73`, `0.54` and its measurement
  count `24` leave the file. Those numbers are now reported in Appendix E.1 of
  the manuscript, and the generator records them in
  `main_prediction_v2_sources.md` through its omission note.

The remaining 31 refreshed files differ only in wording and layout: the table
house style, the shorter cell text, and the caption sentences added on
2026-09-18.

## Checks

The whole public suite passes against the refreshed snapshots: 1481 passed, 31
skipped. `tests/test_caption_wording.py` reads the new captions, so the caption
rules and length limits are enforced against current text rather than against
the 2026-09-17 copies. The two added tables are not in that test's `USED` list,
for the same reason `s3_selection.tex` is not: the three selection tables carry
deliberately long captions that define opportunity, regret and prior use in
full.

## Supersedes

`docs/A18_MIRROR_INVENTORY.csv` and the SHA-256 quoted in
`docs/A18_MIRROR_VALIDATION.md` describe `s3_selection.tex` as it stood at A18,
1,836 bytes and `8c4c6f39...`. The 2026-09-21 round restored a truncated
sentence in that caption, so the current file is 1,975 bytes with SHA-256
`31ef5b7451eb0bd62bc275053de987b7a03d4e78526c371fe4118473b3913b97`. The A17 and
A18 records are left as they were written: they state what those tasks
produced, and this file is where the current digests live.
