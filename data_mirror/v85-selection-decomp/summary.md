# V85 selection decomposition and exact locked rule

Generated on CPU with the Python standard library; no fitting or measurement. V78 selections, oracle candidates, and outcomes are read without modification.

Regret is the selected candidate's measured objective minus the original feasible oracle. Every state has 17 equally weighted budgets (0.20 to 1.00 by 0.05). The 51-cell subset has three step-32k states (160M, 410M, 1.4B); its compressed candidates are newly measured pruning and quantization. The 17-cell subset is 1B@64k, with newly measured pruning/quantization and two reused historical distillation outcomes (160M@64k and 410M@64k). Both subsets retain the original dense candidate, feasible at budget 1.0. Thus 51/17/68 are state-budget cells **per objective**, not counts of experiments.

Policy keys: frozen rule = `locked-rule`; source-conditioned predictor = `v64-law`; quantization-only = `quant-only` (both channel and grouped RTN, ranked by locked predictions); cheapest = `cheapest`. Multi uses `max_c(L_c - source_L0c)` over math/code/QA, not their average. QA is 2Wiki. All four policies are feasible in every cell. Differences use unrounded regrets; negative favors the frozen rule.

| Subset | Objective | Cells | Frozen rule | Source-conditioned predictor | Quantization-only | Cheapest | Frozen minus quant. |
|---|---|---:|---:|---:|---:|---:|---:|
| Three step-32k states | Math | 51 | 0.000008 | 0.019778 | 0.000015 | 2.252843 | -0.000007 |
| Three step-32k states | Code | 51 | 0.004329 | 0.141263 | 0.004328 | 3.115947 | +0.000001 |
| Three step-32k states | QA | 51 | 0.170533 | 0.282961 | 0.243316 | 1.784897 | -0.072783 |
| Three step-32k states | Multi (max) | 51 | 0.002697 | 0.002467 | 0.002954 | 3.107499 | -0.000256 |
| 1B@64k | Math | 17 | 0.000000 | 0.000434 | 0.026970 | 0.327902 | -0.026970 |
| 1B@64k | Code | 17 | 0.004321 | 0.025644 | 0.027814 | 0.390606 | -0.023494 |
| 1B@64k | QA | 17 | 0.052428 | 0.005406 | 0.280362 | 0.052428 | -0.227934 |
| 1B@64k | Multi (max) | 17 | 0.000000 | 0.000000 | 0.023494 | 0.382624 | -0.023494 |
| All four states | Math | 68 | 0.000006 | 0.014942 | 0.006754 | 1.771608 | -0.006748 |
| All four states | Code | 68 | 0.004327 | 0.112358 | 0.010199 | 2.434612 | -0.005873 |
| All four states | QA | 68 | 0.141007 | 0.213573 | 0.252577 | 1.351780 | -0.111571 |
| All four states | Multi (max) | 68 | 0.002023 | 0.001850 | 0.008089 | 2.426280 | -0.006066 |

The pooled result weights the 51-cell and 17-cell subsets 3:1. The pooled math and code advantage over quantization-only comes almost entirely from 1B@64k. On the newly measured step-32k subset, the math difference is -6.59126262227e-06 nats and the code difference is 1.16519153419e-06 nats (code is slightly worse). The frozen rule selects quantization in 48 cells and dense in three for each of math/code; their tiny differences arise at the dense budget. QA selects pruning in 27 cells and quantization in 24, giving a material improvement within the newly measured subset.

This is a descriptive decomposition of the original panel, not a distillation-removal ablation or a newly precommitted confirmation criterion. 1B@64k differs in source state as well as candidate availability, so its contribution cannot be identified solely with distillation. V78's original full-panel verdicts remain math/code/QA confirmed and multi retrospective; multi's frozen-rule regret exceeds the source-conditioned predictor.

## Exact rule and delivered-predictor differences

`locked_rule.tex` lists each capability separately in all seven compression arm/status cases, plus the dense reference (24 rows). Equivalent new-status aliases share a row. Every V78 state uses `new_stage`; hence V78 uses pruning median curves, channel per-bit medians, grouped per-configuration medians, and the distillation branch. Absolute losses add the predicted response to source dense loss for pruning/RTN, and to initial student dense loss for distillation.

- Pruning: final_deliverables allows power or A2 on seen sizes; final_rule selects power for math/code and the median curve for QA. New size or stage uses the median for every capability.
- Per-channel: final_deliverables delivers per-bit source regression on new states and explicitly flags the selection-rule exception. final_rule uses per-bit development medians on new states; source regression is only the seen-state branch.
- Grouped: both use source interpolation for seen-state math/code and medians for QA/new states. The exact rule applies V69 interpolation/boundary handling to median anchors as well.
- Distillation: final_deliverables reports exposure a_c log(1+E) for math/code and a joint budget-pool form for QA on development students. final_rule instead uses V39 fixed-recipe student-state +D0 linear math and per-capability arithmetic-mean constants for code/QA, anchored to the initial student's dense loss; QA requires 2Wiki.

The distillation constants are arithmetic means on seven retained students: code `0.1804815104078237`, QA `-0.48038175583921788` nats. The stored math constant is unused; math calls the +D0 linear model. Exact coefficients, centers, scales, per-bit rosters, and source excerpts are in `rule_spec.json`.

- **P**: V53 register: 17 states, 84 density-state responses per capability; all measured 0.55 <= d < 1 rows, excluding 2.8B. Power uses ridge 0.001 and the development gamma grid; median is per density. Source: `results/v53-prune-dev/register.json`.
- **Q**: V64 frozen paired channel responses: 16/16/4/17/17 states at 8/6/5/4/3 bits (70 responses per capability across 17 states). V36 per-bit source OLS via V64 fit_quant, or per-bit arithmetic median. Source: `results/v64-selection-feasible/summary.json`.
- **G**: V69 develop: 160M, 410M, 1.4B at 16k and 143k; bits 3,4,5 x groups 64,128,256 (54 responses per capability). Per-anchor source ridge 0.001 or per-configuration median. Source: `results/v69-quant-confirm/develop.json`.
- **K**: V39 fixed-recipe Pythia panel as frozen in V64: 160M, 410M, 1.4B at 16k,64k,143k, excluding 160M@64k and 410M@64k from both fits (7 students per capability). Teacher gpt-5.6-luna, full pool 600, 2 epochs, strict LoRA, seed 0. Math uses +D0 OLS; code/QA use mean delta. Source: `results/v64-selection-feasible/summary.json#/students`.

Verbatim module docstring from `analysis/final_rule.py`:

```text
The Round v7 locked rule. Prediction only: no files, fitting, or outcomes.

Arm                 State status                     Math/code          QA
------------------  -------------------------------  -----------------  -----------------
pruning             seen size, unseen d in [0.6,0.9]  v53 power          v53 median curve
pruning             new size or stage                v53 median curve   v53 median curve
per-channel RTN     seen bits, new state             per-bit dev median per-bit dev median
per-channel RTN     seen state                       v36 regression     v36 regression
grouped RTN         seen state                       v69 interpolation  per-config median
grouped RTN         new state                        per-config median  per-config median
distillation        new source, smaller same-stage   linear (math),     constant; 2Wiki only
                                                     constant (code)

All returns are absolute deployed losses in native-token nats. Pruning/RTN
add a development response to source L0; KD adds it to STUDENT dense loss.
Group interpolation uses v69's frozen boundary rule: adjacent log-coordinate
pairs, zero floor only outside the group grid, and no bit extrapolation.
Selection is downstream: feasible nominal storage, argmin absolute loss per
capability, or max_c(L_c-source_L0c), with v64's candidate-set heuristic.

``inputs`` contains ``models`` (development-fitted objects), scalar ``L0``,
``N0``, ``D0`` and arm-specific ``d`` / ``bit`` / ``config``. KD instead uses
``student`` with N0, D0 and a per-capability ``dense`` mapping. QA KD requires
``qa_distribution='2Wiki'``. New-size/stage status takes precedence over a
previously seen size. Undefined table cells are rejected, never improvised.

```

Verbatim evaluator (the imported helper implementations and fit construction are also captured in `rule_spec.json`):

```python
def predict(arm, capability, state_status, inputs):
    """Return the absolute loss under the docstring's locked table.

    This pure function only evaluates supplied development-fitted objects;
    it does not read outcomes, refit, change inputs, or mutate module state.
    """
    if capability not in CAPS:
        raise ValueError(f"Unknown capability: {capability}")
    c, models = capability, inputs["models"]
    arm = {"prune": "pruning", "quant_channel": "per-channel",
           "quant_group": "grouped", "distill": "distillation"}.get(arm, arm)
    if state_status not in (*NEW, "seen_state", "seen_size_unseen_density"):
        raise ValueError(f"Unknown state status: {state_status}")
    if arm == "dense":
        value = float(inputs["L0"])
    elif arm == "distillation":
        if state_status not in NEW:
            raise ValueError("Locked KD rule requires a new source")
        s = inputs["student"]
        if s["N0"] >= inputs["N0"] or s["D0"] != inputs["D0"]:
            raise ValueError("KD candidate must be smaller and at the same stage")
        if c == "qa" and inputs.get("qa_distribution") != "2Wiki":
            raise ValueError("Locked KD QA is restricted to 2Wiki")
        anchor = float(s["dense"][c])
        f = models["distill"]
        delta = (float(distill._predict(f["linear"][c], [{**s, "L0": anchor}])[0])
                 if c == "math" else float(f["constant"][c]))
        value = anchor + delta
    elif arm == "pruning":
        d = float(inputs["d"])
        if not .6 <= d <= .9:
            raise ValueError("Locked pruning domain is [0.6, 0.9]")
        f = models["prune"]
        if state_status in NEW or (state_status == "seen_size_unseen_density" and c == "qa"):
            delta = prune.linear_curve(f["models"][c]["median_curve"]["anchors"], d)
        elif state_status == "seen_size_unseen_density":
            raw = prune.raw_features(inputs["N0"], inputs["D0"], inputs["L0"])
            z = prune.standardize(raw, f["standardization"])
            p = f["models"][c]["power"]
            delta = float(np.dot(p["beta"], z) * prune.shape(d, p["gamma"]))
        else:
            raise ValueError("Seen-size pruning requires an unseen density")
        value = float(inputs["L0"]) + delta
    elif arm == "per-channel":
        f = models["channel"]
        if state_status in NEW:
            try:
                delta = f["median"][c][str(inputs["bit"])]
            except KeyError as exc:
                raise ValueError("Per-channel rule requires a seen bit-width") from exc
        elif state_status == "seen_state":
            delta = channel_delta(f["models"][c], inputs)
        else:
            raise ValueError("Invalid per-channel state status")
        value = float(inputs["L0"]) + delta
    elif arm == "grouped":
        f = models["grouped"]
        if state_status in NEW or (state_status == "seen_state" and c == "qa"):
            anchors = f["models"][c]["median"]["anchors"]
        elif state_status == "seen_state":
            raw = prune.raw_features(inputs["N0"], inputs["D0"], inputs["L0"])
            z = group.phi(raw, f["standardization"])
            anchors = {k: float(np.dot(beta, z)) for k, beta in
                       f["models"][c]["same_input_interpolation"]["anchors"].items()}
        else:
            raise ValueError("Invalid grouped state status")
        value = float(inputs["L0"]) + grouped.interpolate(anchors, inputs["config"])
    else:
        raise ValueError(f"Unknown arm: {arm}")
    if not math.isfinite(value):
        raise ValueError("Nonfinite prediction")
    return float(value)
```

## Reproduction and validation

Run `python -B analysis/v85_selection_decomp.py`; verify determinism with `python -B analysis/v85_selection_decomp.py --check`. The mirrored entry point `python -B paper/analysis/v85_selection_decomp.py --check` resolves the same repository inputs.

Generation validates the full 4 x 17 grid for each objective, frozen candidate rosters, policy feasibility, actual-minus-oracle regrets, paired differences, the 51:17 weighted identity, and reproduction of V78's published all-state means. It checks implementation hashes against V78, verbatim rule expressions, V53/V69 object identity, channel medians, and distillation mean constants/exclusions without refitting. No test cells are dropped.

Outputs: `decomposition.json` (full precision, method counts and contributions), `decomposition.csv` (12 aggregate rows), `cell_regrets.csv` (272 original cells with JSON pointers), `rule_spec.json` (24 rule rows, provenance, literal code), this summary, and the two requested LaTeX tables. The new script is mirrored under `paper/analysis/`; the existing mirrored `final_rule.py` and `v78_rule_confirm.py` already match their originals.

## Input SHA256

- `analysis/final_rule.py`: `5387d516af3c13963148bce93ec77d3a18cb4efec6f131637e52f83b5770d950`
- `analysis/v36_pythia_controlled_fit.py`: `ac665407235616c5a66801b51efae0c6f6258859e53fd9ee62243ad9e64ef91d`
- `analysis/v39_distill_controlled.py`: `437dc7576814ef1697a706a7ab0d3ba70b57abd1c7a65a53a8fbc8b4a6f9f888`
- `analysis/v53_prune_dev.py`: `5a3fa776ab1ddf491fdc70bad4bf7e7ee000754a735ec6ea10431c32a661bff2`
- `analysis/v55_quant_group_fit.py`: `596015ebcc8f635e8454f914bda159654cc85b2fc463f614243ff13df3c3f926`
- `analysis/v64_selection_feasible.py`: `3819a003cc54af531b7c2ef2aa1da543b5e7e8cfe634bc0e1b2e7f432d3dea21`
- `analysis/v69_quant_confirm.py`: `1f19cf4ad81eedae36d401a2373c4f9f6de4c152ce688ee70e78efca9c063138`
- `analysis/v78_rule_confirm.py`: `7afa1d9439f0a9753df4928df0b867bcd60842ec77cb460750afbec0c1b8eb32`
- `paper/paper/tables/final_deliverables.tex`: `a60e99142ead0daf92807ff1fdb5bbd42c7f1f012517f714656303c26b4dd9c4`
- `results/v53-prune-dev/register.json`: `7498103830f139f6bea16ee440b717a05dee41e45b9835fdac6a2251b833974d`
- `results/v64-selection-feasible/summary.json`: `bff53de6379b86cfef5466c5815e1fc01a82a5e355e2fd005978504c21f51218`
- `results/v69-quant-confirm/develop.json`: `9f035cb37a94d425064f37d822490b52dcad4c669e62864f27cba4ffa5e5643d`
- `results/v78-rule-confirm/compare.json`: `601669be112c5dd7c016e859423700a98f6a916dcd12c838bb5cd9c8c09686a8`
- `results/v78-rule-confirm/freeze-independent.json`: `54b53f547800957f4be011ba4c3cbc50302144576de36f6da0e0bd63be6141a5`
