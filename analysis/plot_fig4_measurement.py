#!/usr/bin/env python3
"""CPU-only benchmark-transfer evidence, verified against the original V23 measurements."""

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
import hashlib
import sys

sys.dont_write_bytecode = True
if __package__:
    from .plot_fig1_responses import Audit, CAPS, CAP_LABEL, COLORS, ROOT, checked_close, setup_style
else:
    from plot_fig1_responses import Audit, CAPS, CAP_LABEL, COLORS, ROOT, checked_close, setup_style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

PROTOCOLS = ("leave_model_out", "leave_density_out", "leave_model_and_density_out")
PROTOCOL_LABEL = ("Model out", "Density out", "Model + density out")
COHORTS = ("all_densities", "without_strongest")
PRIMARY_COLORS = CAPABILITY_COLORS


def load_and_verify(audit, summary):
    """Check recorded inputs, benchmark identities, and saved OOF MAEs without refitting."""
    inputs = {}
    for rel, expected in sorted(summary["input_sha256"].items()):
        payload = audit.read(rel)
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f"V23 provenance hash mismatch: {rel}")
        inputs[rel] = payload
    pairs = {}
    for rr in summary["rows"]:
        original = inputs[rr["source_path"]]
        for cap in CAPS:
            measures = original["capabilities"][cap]
            pair = tuple(measures[role]["benchmark"] for role in ("primary", "secondary"))
            if cap in pairs and pair != pairs[cap]:
                raise ValueError(f"Inconsistent benchmark pair: {cap}")
            pairs[cap] = pair
            for role in ("primary", "secondary"):
                measure = measures[role]
                checked_close(rr[role][cap], measure["delta_L_c"], f"v26/v23 {rr['row_id']}/{cap}/{role}")
                checked_close(measure["delta_L_c"], measure["L_c"] - measure["baseline_L_c"],
                              f"dense subtraction {rr['row_id']}/{cap}/{role}")
    for cohort in COHORTS:
        for protocol in PROTOCOLS:
            for cap in CAPS:
                result = summary["analyses"][cohort][protocol][cap]
                for predictor, metric in result["metrics"].items():
                    errors = [abs(r["predictions"][predictor] - r["observed"]) for r in result["records"]]
                    checked_close(np.mean(errors), metric["mae"], f"OOF MAE {cohort}/{protocol}/{cap}/{predictor}")
                    checked_close(len(errors), metric["n"], "OOF metric count")
                metrics = result["metrics"]
                checked_close(metrics["cross_selected"]["mae"] - metrics[cap]["mae"],
                              result["same_over_cross_gain"], f"paired gain {cohort}/{protocol}/{cap}")
    return pairs


def main():
    if __package__:
        from .plot_paper_appendix import generate
    else:
        from plot_paper_appendix import generate
    return generate("measurement_support")


if __name__ == "__main__":
    main()
