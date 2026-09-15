# Additional frozen figure generators

Run these commands from the public repository root with the CPU dependencies in
`requirements.txt` installed:

```bash
export CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 bootstrap_results.py
python3 analysis/plot_fig_lr_pilot.py
python3 analysis/plot_fig_pythia_responses.py
python3 analysis/plot_fig_drift.py
python3 analysis/plot_fig_heterogeneity.py
python3 analysis/plot_fig_selection_maps.py
python3 -m pytest --rootdir=. tests/test_plot_more_main_figures.py tests/test_paper_artifact_digests.py tests/test_paper_figure_style.py -q
```

The generators only read frozen results and render on the CPU. Outputs go to
`generated/figs/`: 13 PDF panels, five PNG previews, captions, plotted records,
and source digests. No manuscript checkout is required. No model evaluation,
training, fitting, or new measurement is performed.

## Complete input trace

[ADDITIONAL_FIGURE_INPUTS.json](ADDITIONAL_FIGURE_INPUTS.json) lists every opened
frozen artifact by generator, its mirror path, and its decompressed SHA-256.
There are 65 measurement/provenance artifacts plus the publication digest map.

| Generator | Frozen inputs | Records | PDF panels |
| --- | --- | ---: | --- |
| LR pilot | All 12 `eval.json` checkpoints in six Gemma-3 1B/4B LR-pilot trajectories, including update zero | 18 | `fig1_c` |
| Pythia responses | V36 summary; V53 register; nine V6 pruning and nine V10 quantization files selected by V36; V69 develop/freeze/compare; publication digest map | 633 | `fig4_a`–`fig4_c` |
| Drift | V96 and V100 summaries; endpoint references inside those summaries are not opened as trajectories | 216 | `fig5_a`, `fig5_b` |
| Heterogeneity | V51 panel and the twelve original models' V6 pruning/V10 quantization files | 72 | `fig8_a`, `fig8_b` |
| Selection maps | V78 compare, freeze, and `freeze.json.sha256` | 272 | `fig7_a`–`fig7_d`, `fig7_legend` |

`paper_figure_style.py` reads installed fonts and Matplotlib runtime resources;
it opens no additional frozen artifact. The tests check exact saved numbers,
checkpoint selection, endpoint identities, all selection cells, typography,
PDF page sizes, output confinement, and symlink refusal.

## Publication layout and digests

- The 12 LR-pilot checkpoints and V36 summary were missing and are now mirrored.
  V51, V96, V100, V78 and all other loader inputs were already present and were
  checked against the frozen originals or their exact scrubbed bytes.
- The V96 and V100 summaries exceed 5 MB and ship as deterministic gzip files.
  Bootstrap expands them and provides both `--step` and `@step` path spellings.
- Eleven older tracked V6/V10 result copies also existed outside `data_mirror/`.
  They now match the frozen mirror, so bootstrap's existing-file preservation
  cannot leave a fresh checkout using stale losses or incomplete density grids.
- The public artifact helper retains `results/` and `docs/` as evidence roots
  and `generated/` as its output root. Its font allowlist permits installed font
  files without admitting arbitrary repository files.
- Pythia and selection-map digest checks call the repository comparator through
  `Artifacts.matches_digest`. An unequal digest is accepted only through the
  audited `results/ANONYMIZATION_DIGESTS.json` with `pairable` explicitly `true`.
  Missing pairs, stale entries, tampered hashes, and ambient maps cannot pass.
- The panel-export helper is included. The V51 loader is safe to import and
  accepts audited reads, so importing it does not regenerate tables or results.
- Tests use the output-path helper for the public layout. The public-specific
  artifact helper was edited in place; the other protected public adapters and
  test helpers were retained.

Validation: all five command-line generators and all 35 figure, style, and
digest tests pass in the public checkout. A staged-file-only export is also
bootstrapped and tested to check independence from author-side files.
