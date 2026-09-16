#!/usr/bin/env python3
"""Frozen prediction residuals, separated by configuration, source, and scope.

Missing measurement intervals are shown as crosses and explicitly unclassified;
neither development MAE nor a baseline-gain interval is a prediction error bar.
"""
from __future__ import annotations

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp

import sys
sys.dont_write_bytecode = True

import math

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, PRINT_RC, Artifacts, frozen_run, pyplot, save_figure, write_notes, legacy_rows, confirmation_identity
    from .plot_fig_explanation import corners, A5
else:
    from paper_artifacts import ROOT, CAPS, COLORS, PRINT_RC, Artifacts, frozen_run, pyplot, save_figure, write_notes, legacy_rows, confirmation_identity
    from plot_fig_explanation import corners, A5

GRAMMAR = "Hollow diamond: frozen prediction with a stored band; cross: interval not tested. Dashed / solid whiskers: 1B / 4B development; band membership remains in the sidecar."
FIGSIZE = (5.5, 8.2)


def point(panel,group,cap,predicted,measured,source,*,interval=None,development=False,**metadata):
    within=None if interval is None else interval[0]<=predicted<=interval[1]
    # Reflect the measurement interval about the fixed prediction, so zero is
    # contained iff the measured band contains the prediction.
    residual_interval=None if interval is None else [predicted-interval[1],predicted-interval[0]]
    return {"panel":panel,"group":group,"capability":cap,"predicted":predicted,"measured":measured,
            "residual":predicted-measured,"measurement_interval":interval,"residual_interval":residual_interval,
            "within":within,"status":"development" if development else "frozen prediction",
            "source":source,**metadata}


def build(audit):
    legacy_rows(audit)
    if __package__:
        from .plot_fig3_transfer import load_confirmation
    else:
        from plot_fig3_transfer import load_confirmation
    confirmation=load_confirmation(audit)
    audit.rule("Reused plot_fig3_transfer.load_confirmation for V53 state loading and MAE arithmetic; "
               "plot_fig2_confirm.paired_rows for V69/V70 frozen prediction identity; V86 loaders for pruning repeats. "
               "No test-ranked baseline is used in this figure.")
    result=[]
    for label,comp in confirmation.items():
        path=f"results/v53-prune-dev/predictions_{comp['tag']}.json"
        predictions=audit.data[path]
        for cap in CAPS:
            for d in comp["densities"]:
                key=str(d)
                result.append(point("B","Pythia: new stages (power)",cap,predictions["predictions"][cap][key]["power"],
                                    comp["observed_delta_loss"][cap][key],f"{path}#/predictions/{cap}/{key}/power",state=comp["tag"],density=d,
                                    measured_source=f"results/v53-prune-dev/compare_{comp['tag']}.json#/observed_delta_loss/{cap}/{key}"))
    path="results/v72-prune-repeat/compare.json"
    for i,r in enumerate(audit.data[path]["rows"]):
        result.append(point("A","Pruning: density inside range",r["capability"],r["predictions"]["power"],r["observed_delta_loss"],
                            f"{path}#/rows/{i}",density=r["density"],state=r["source"]))
    path="results/v46-p1-newsource/compare.json"
    v46=audit.read(path); freeze=audit.read("results/v46-p1-newsource/predictions_frozen.json")
    for i,r in enumerate(v46["pruning"]):
        p=freeze["pruning"][f"{r['cap']}|{r['d']}"]["power"]
        if p!=r["power"]["pred"] or not math.isclose(abs(p-r["actual"]),r["power"]["abs"]):
            raise ValueError("V46 prediction mismatch")
        group="Pruning: density outside range" if r["d"]==.55 else "Pruning: density inside range"
        result.append(point("A",group,r["cap"],p,r["actual"],f"{path}#/pruning/{i}",density=r["d"],
                            state="V46 held-out 1B@96k",range_note="V46 original coarse fit: density .6--.9; .55 is outside. V53 later includes .55."))
    audit.read("docs/RESULTS_LEDGER.md")
    audit.rule("V46 .55 is outside its original coarse .6--.9 fit; .65 is inside. "
               "Do not use the later V53 .55--.9 development range to reclassify the earlier V46 test. "
               "V72 repeats two revision labels with identical weights; these are shown as repeated records, not independent source states.")
    for directory,identity,observed in (("v69-quant-confirm",("state","config","capability"),"dL"),
                                       ("v70-distill-confirm",("student","pool","T_planned","capability"),"actual")):
        comp,freeze=confirmation_identity(audit,directory,identity,observed)
        path=f"results/{directory}/compare.json"
        for i,r in enumerate(comp["rows"]):
            if directory.startswith("v69"):
                candidate=freeze["selected"][r["capability"]]["candidate"]
                panel="A" if r["test_set"]=="development_state_boundary" else "B"
                group="Quantization: new group size" if panel=="A" else "Pythia: new quantization state"
            else:
                candidate=freeze["selected"][r["capability"]]["method"]
                panel,group="A","Distillation: new-pool budgets"
            result.append(point(panel,group,r["capability"],r["predictions"][candidate],r[observed],f"{path}#/rows/{i}",candidate=candidate))
    audit.read("results/v47-p2-register/register.json")
    if audit.path("results/v47-p2-register/freeze.json").exists():
        audit.read("results/v47-p2-register/freeze.json")
    else:
        audit.rule("Original V47 freeze.json is absent; its predictions are not tested here. V70 supplies the available registered new-pool freeze.")
    fpath="results/v78-rule-confirm/freeze.json"; freeze=audit.read(fpath)
    for i,state in enumerate(freeze["states"]):
        tag=state["tag"].replace("@","--")
        for cap in CAPS:
            descriptor=audit.read(f"results/v93-confirm-inputs/{tag}/{cap}/descriptor_bv.json")
            if descriptor["model_revision"]!=f"step{state['step']}":
                raise ValueError("V93 state identity mismatch")
        for j,config in enumerate(state["configs"]):
            if config["method"] not in ("prune","quant"):
                continue
            path=f"results/v78-rule-confirm/measurements/{tag}/{config['method']}__{config['config']}.json"
            measured=audit.read(path)
            if measured["state"]!=state["tag"] or measured["config"]["id"]!=config["id"]:
                raise ValueError("V78 measurement identity mismatch")
            for cap in CAPS:
                prediction=config["predictions"]["locked-rule"][cap]
                if prediction["status"]!="FROZEN":
                    raise ValueError("Unfrozen V78 point")
                result.append(point("B","Pythia: locked rule on new states",cap,prediction["absolute_loss"],measured["losses"][cap],
                                    f"{fpath}#/states/{i}/configs/{j}/predictions/locked-rule/{cap}/absolute_loss",
                                    state=state["tag"],config=config["id"],measured_source=f"{path}#/losses/{cap}"))
    audit.rule("V78 uses each distinct frozen configuration once, not repeated selected candidates across resource budgets. "
               "Residual is frozen absolute_loss minus stored measured loss. V93 is used only to verify source identity; "
               "its B/V descriptors contain no frozen response prediction or calibrated error interval.")
    for r in corners(audit):
        dev=r["student"]=="gemma3-4b"
        result.append(point("B" if dev else "A","4B DEVELOPMENT: corner budgets" if dev else "Distillation: corner budgets (1B)",
                            r["capability"],r["F_int"],r["measured"],r["source"],interval=r["interval"],development=dev,
                            student=r["student"],candidate="F_int",interval_source=r["interval_source"]))
    a5=audit.data[A5];protocol=audit.read("results/v99-scope/protocol.json")
    for student,s in a5["students"].items():
        fresh=s["fresh_sample_check"]
        for corner,c in fresh["corners"].items():
            measured=audit.read(c["source"])
            for dist in protocol["distributions"]:
                if measured["delta_from_update_0"][dist]!=c["responses"][dist]:
                    raise ValueError("V99/A5 fresh-distribution mismatch")
        for dist in protocol["distributions"]:
            r=fresh["readouts"][dist]
            # Additivity's registered rectangular contrast is exactly zero.
            # A7 has no fresh-scope F_int predictions: don't transport fitted
            # response coefficients or use quoted disagreement as a prediction.
            additive=audit.data["results/a7-closeout-audit/summary.json"]["checks"]["predictions"]["rows"][0]["additive_rectangle_prediction"]
            result.append(point("C",dist,"qa",additive,r["I"],f"{A5}#/students/{student}/fresh_sample_check/readouts/{dist}",
                                interval=r["interval"],development=student=="gemma3-4b",student=student,candidate="registered additive contrast"))
    audit.rule("Panel C is the registered additive corner prediction (zero), with A5 fresh-scope measured I and registered +/-2-noise bands, "
               "checked against V99 delta_from_update_0. It is not a transported fitted response law: fresh-scope F_int predictions "
               "are not stored in A7. The primary QA test failed to reject; fresh secondary readouts are shown individually, without a shared verdict. All 4B records retain their development status and solid whiskers, "
               "including the independently measured configurations, because 4B was a development student.")
    audit.rule("No measurement intervals are stored for V46/V53/V69/V70/V72/V78 prediction cells. "
               "They are crosses, with no invented whisker or coverage classification. V70 paired gain intervals "
               "and A2 conditional MAE intervals cannot be repurposed as per-measurement uncertainty. "+GRAMMAR)
    return result


def generate(root=ROOT):
    if __package__:
        from .plot_paper_appendix import generate as render
    else:
        from plot_paper_appendix import generate as render
    return render("generalization_cells", root=root)


if __name__ == "__main__":
    generate()
