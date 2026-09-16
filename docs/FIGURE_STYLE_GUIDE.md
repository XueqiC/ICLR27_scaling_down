# Paper figure style

`analysis/paper_figure_style.py` is the single palette and style authority for main and appendix figures. Never select a colour by plotting order. Use the semantic dictionaries, including for fitted curves, folds, boundaries, observations and errors. Literal colour strings belong only in the palette module.

| Meaning | Colour | Marker / line / hatch |
|---|---|---|
| Math | `#2b6cb0` blue | Same hue for observations, fits and prediction errors |
| Code | `#d97706` orange | Same hue for observations, fits and prediction errors |
| QA; 2Wiki | `#2f855a` green | QA variants remain green |
| MuSiQue | `#68b36b` mid green | QA evaluation distribution |
| TriviaQA | `#a8d5a2` light green | QA evaluation distribution |
| Pruning | `#6b46c1` purple | Method hue, never a capability hue |
| Quantization; grouped quantization | `#0e7490` teal | Method hue |
| Per-channel quantization | `#9a6b2f` brown | Distinct quantization method |
| Distillation | `#b83280` magenta | Method hue |
| Dense / no compression | `#9ca3af` grey | Method hue |
| Bits 3, 4, 5 | `#155e75`, `#0e7490`, `#67c3d9` | Dark to light teal |
| Pruning knob levels | `#443075` → `#6b46c1` → `#b49add` | Ordered purple ramp |
| Quantization group sizes / storage levels | `#155e75` → `#0e7490` → `#67c3d9` | Ordered teal ramp |
| Reference, zero, additive prediction, controls | `#4b5563` dark grey | Neutral reference line / hollow prediction glyph |
| Registered interaction prediction F_int | `#000000` black | Hollow diamond |
| Oracle method differs | `#000000` black | Hollow circle **and** `xx` cell hatch |
| Student 270M / 1B / 4B | Inherit capability hue | Dotted / dashed / solid; no student colours |
| First / second registered data seed | Inherit series hue | Circle / square |
| Measured estimate | Inherit series hue | Filled glyph |
| Baseline / prediction | Inherit series hue | Hollow glyph; frozen predictions use diamonds |
| Missing measurement interval | Inherit capability hue | Cross; do not imply coverage |
| Generalization lower-of-pair cue | Capability hue if relation is lower, reference grey otherwise | Connecting segment; no green indicator marker |

## Secondary dimensions and hatches

Use hatches only when filled areas cross a second categorical dimension. When method is the fill and capability is the second dimension: Math `///`, Code `...`, QA `xx`. Prediction bars use `///`; measured bars are plain. In policy comparisons at a capability colour, the frozen rule is plain, earlier laws `///`, quantization-only `...`, cheapest `xx`; these policy keys are explicit in the legend. Hatch ink is `darker(fill)` (55% RGB intensity), preserving its hue.

Selection maps use `xx` plus a hollow black circle for oracle mismatch, `///` for no clear winner, and grey `xx` without a circle for infeasible cells. If ambiguity and mismatch coincide, the mismatch hatch takes precedence; both recorded flags remain in the sidecar. Numerical response bars coloured by capability do not gain a redundant capability hatch.

## Physical style and export

Use the existing bold Times-family serif settings without changing the size tiers: panels ≤2 in use 7.5 pt ticks / 8.5 pt labels; panels ≤3 in use 8.5 / 9.5 pt; full width uses 9 / 10 pt. Shared legend strips use 8.5 pt. Lines are 1.1 pt, markers 3.8 pt, marker edges 0.6 pt and errorbar caps 2 pt.

Appendix panels: three per row 1.8 × 1.35 in; two per row 2.7 × 1.45 in; full width 5.5 × 1.5 in. Tall maps may reach 2.4 in. Use a second row only when the panel count cannot fit one row. Every panel has its own uncropped PDF and PNG, caption text and data/provenance sidecars. A separate legend strip sits above the row. Panel titles belong in LaTeX subcaptions, not inside plots. Complete PDF/PNG sheets are review outputs.

Display-only deterministic horizontal dodging is limited to 1.5% of the axis range per marker, enforced on both the displayed scale and native coordinate range. The added spread of a cluster never exceeds 6% of the axis range (the symmetric 1.5% cap actually limits added spread to 3%). Generators start at true numeric or categorical coordinates, without preliminary jitter. If bounded dodging cannot keep every marker partly visible, the whole crowded cluster returns to its true positions and overlap is accepted. Markers retain their 3.8-pt size, thin 0.6-pt white edges (white under-strokes for crosses), and draw above lines; smaller glyph footprints draw above larger ones. Hollow predictions retain coloured outlines. Short 0.4-pt guide connectors identify any permitted displacement; source lines and uncertainty intervals stay at true coordinates.

Every marker must be partly visible OR within the 1.5% cap with coincident neighbours documented in the sidecar. A 220-dpi Agg compositing check records stable point IDs, individual visible-ink counts, true and drawn coordinates, normalized offsets, cluster fallback, and neighbour IDs. Coincident neighbours have overlapping true-position glyph bounding boxes, including edges. Hidden measurements remain represented and recorded; never average coincident measurements or mutate frozen data to resolve overlap.

## Rebuild

Use CPU/Agg and set `MPLCONFIGDIR` below the system temporary directory. `paper_artifacts.pyplot` uses `/tmp/scaling-down-law-mpl` on this system and never changes `tempfile.tempdir`. Run pytest with a system-temp `--basetemp` and `-o cache_dir=/tmp/scaling-down-law-pytest-cache`.

Run the main generators, `python -B analysis/plot_paper_appendix.py` for the ten relation/rule appendix figures, and `python -B analysis/plot_v9_blocks_appendix.py` for all ten model similarity figures included through `appendix_blocks.tex`. The old relation/rule appendix entry points delegate to the shared renderer; V64/V80 accept `--figures-only` to avoid their analysis/table paths. Only figure outputs, captions and sidecars belong in `paper/paper/figs`. Move superseded files to `_trash/`; do not delete them.
