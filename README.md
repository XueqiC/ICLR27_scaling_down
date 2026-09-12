# Scaling-Down Laws for LLM Compression

Analysis code and curated experimental results for the
ICLR 2027 submission "Scaling-Down Laws for LLM Compression".

## Layout

| Path | Contents |
|---|---|
| `*.tex`, `references.bib`, `iclr2027_conference.*` | Paper source (Overleaf-synced; ICLR 2027 style) |
| `code/analysis/` | Full measurement & analysis pipeline (see table below) |
| `code/tests/` | Offline pytest suite for the pipeline |
| `code/configs/` | Sweep configurations |
| `results/` | Curated outputs: per-experiment `report.md` + summary JSON (large binary artifacts — Fisher vectors, checkpoints, adapters — are excluded; regenerable via the pipeline) |
| `figs/` | Paper figures |

## Pipeline map

| Script | Experiment |
|---|---|
| `v6_capability_geometry.py` | Capability-conditional Fisher spectra + magnitude-pruning damage curves (11 models, 4 families) |
| `v6b_alignment.py` | Signed first-order term & full-Fisher quadratic damage prediction |
| `v9_capability_regions.py` | 9-benchmark gradient-signature block structure (capability regions) |
| `v9b_subspaces.py` | Projected-gradient SVD subspaces, principal-angle block structure |
| `v9c_ablation.py` | Causal ablation of capability-exclusive coordinates (gold standard) |
| `v10_quantization.py` | Per-channel RTN quantization damage vs bit-width |
| `v11_geometry_damage.py` | Geometric order parameter vs behavioral cliff |
| `v12_distill.py` / `v12_sweep.py` | Black-box distillation arm (LoRA SFT on teacher traces; coverage recipes) |
| `v13_recovery.py` | Recovery-training arm (post-compression continued training, recovery-law fits) |
| `teacher_api.py` | Frontier-API teacher client (trace generation) |
| `model_registry.py` | Model registry + cluster-policy compliance guard |

Results snapshot date: see git log. Experiments still in flight are marked
"in progress" in the paper's experiment section.

## Two repositories (2026-09-12)

The paper source moved to [XueqiC/Scaling_down_law_paper](https://github.com/XueqiC/Scaling_down_law_paper),
which is the repository Overleaf syncs with; its main document is `main.tex` at the root.
This repository keeps the experiments: `code/analysis/` (analysis and table/figure generators),
`code/tests/`, `code/configs/`, `data_mirror/` (frozen predictions, measurements, comparison
files), `results/` (curated reports), and `docs/` (results ledger, claim-evidence matrix,
reference verification, reviewer self-audit, round summaries).

Text edits go to the paper repository. Generated `tables/*.tex` and `figs/*.pdf` are written
there by the generators kept here, so a change to a number starts with the generator.
Locally, `paper/` is a symlink to the paper repository's clone and is not tracked.
