#!/usr/bin/env python3
"""Uniform appendix panel exports from saved results; CPU only, no refitting.

The historical modules retain their numeric loaders. This module owns the
paper's appendix layouts and routes all writes through the frozen artifact guard.
"""
from __future__ import annotations

import argparse
import importlib
import string
import numpy as np

if __package__:
    from .paper_artifacts import ROOT, CAPS, Artifacts, frozen_run, pyplot, save_figure, write_notes
    from .paper_figure_style import (PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS,
        BIT_COLORS, STUDENT_STYLES, HATCHES, darker, surface, apply_style, panel_axes,
        legend_strip, PANEL_BANDS, combine_panels, save_panel, write_caption)
    from .paper_panel_exports import export
else:
    from paper_artifacts import ROOT, CAPS, Artifacts, frozen_run, pyplot, save_figure, write_notes
    from paper_figure_style import (PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS,
        BIT_COLORS, STUDENT_STYLES, HATCHES, darker, surface, apply_style, panel_axes,
        legend_strip, PANEL_BANDS, combine_panels, save_panel, write_caption)
    from paper_panel_exports import export

STEMS = ("final_relations", "frozen_candidates", "generalization_cells", "input_form_compact",
         "input_form_gain", "measurement_support", "rule_maps_full", "rule_maps_main",
         "rule_regret", "s3_validation", "selection_feasible")
CAP_NAMES = {"math": "Math", "code": "Code", "qa": "QA", "multi": "Maximum loss change"}
METHODS = ("prune", "quant", "distill", "dense")
SIZES = PANEL_BANDS


def module(name):
    return importlib.import_module((__package__ + "." if __package__ else "") + name)


def key(label, color=None, marker="o", *, hollow=False, ls=""):
    from matplotlib.lines import Line2D
    color = color or PALETTE["reference"]
    return Line2D([], [], color=color, marker=marker, linestyle=ls,
                  markerfacecolor=PALETTE["transparent"] if hollow else color, label=label)


def capability_keys():
    return [key(CAP_NAMES[c], CAPABILITY_COLORS[c]) for c in CAPS]


def axes(fig, size=SIZES["three"], *, left=.52, bottom=.33, right=.12):
    from matplotlib.ticker import MaxNLocator
    ax = panel_axes(fig, size, left=left, bottom=bottom, right=right)
    ax.grid(alpha=.15)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    return ax


def publish(plt, audit, stem, specs, handles, caption, columns=3, legend_artist_sizes=None, legend_size=(5.5, .42)):
    """specs=(subcaption,size,draw,records); all explanations go in sidecars."""
    panels = [(f"{stem}_{letter}", size, draw, records, label+".\n"+caption)
              for letter, (label, size, draw, records) in zip(string.ascii_lowercase, specs)]
    def draw_legend(f):
        legend = legend_strip(f, handles)
        if legend_artist_sizes is not None:
            # Retain existing legend glyph sizes during coordinate-only changes.
            lw, ms, mew = legend_artist_sizes
            for handle in legend.legend_handles:
                handle.set_linewidth(lw)
                handle.set_markersize(ms)
                handle.set_markeredgewidth(mew)
        return legend
    legend = (f"{stem}_legend", legend_size, draw_legend, [], caption)
    manifest = export(plt, audit, stem, panels, caption, columns=columns, width=5.5, legend=legend)
    audit.rule("Uniform appendix exports: top legend strip, no in-panel titles; source records unchanged. "
               "Marker dodging is bounded to 1.5% of the axis range; crowded clusters stay at true coordinates with white edges.")
    return manifest


def final_relations(audit, plt):
    old = module("plot_fig1_final")
    reg = audit.read("results/v53-prune-dev/register.json")
    pred = audit.read("results/v53-prune-dev/predictions_pythia-1.4b@step112000.json")
    losses = audit.read("results/v6-capability-geometry/pythia-1.4b--step112000/prune_losses.json")
    qf, qd, qc = [audit.read(f"results/v69-quant-confirm/{name}.json") for name in ("freeze", "develop", "compare")]
    df, dc = [audit.read(f"results/v70-distill-confirm/{name}.json") for name in ("freeze", "compare")]
    for recorded, path in ((pred["provenance"]["register_sha256"], "results/v53-prune-dev/register.json"),
                            (qc["provenance"]["freeze_sha256"], "results/v69-quant-confirm/freeze.json"),
                            (qf["provenance"]["develop_sha256"], "results/v69-quant-confirm/develop.json"),
                            (dc["freeze_sha256"], "results/v70-distill-confirm/freeze.json")):
        if not audit.matches_digest(recorded, audit.inputs[path]):
            raise ValueError("Frozen relation digest mismatch")
    def prune(f):
        ax=axes(f); target=pred["target"]
        support=[d for s in reg["dev_states"] for d in s["densities"]]
        ax.axvspan(min(support), max(support), color=PALETTE["background"])
        ds=sorted(float(k) for k in losses if not k.startswith("_"))
        for c in CAPS:
            color=CAPABILITY_COLORS[c]; x=np.linspace(min(support),1,151)
            for d, methods in pred["predictions"][c].items():
                for method in ("power","median_curve"):
                    old.close(old.prune_curve(reg,target,c,float(d),method),methods[method],"pruning freeze")
            ax.plot(x,old.prune_curve(reg,target,c,x,"power"),color=color)
            anchor=sorted(map(float,reg["models"][c]["median_curve"]["anchors"]))
            ax.plot(anchor,old.prune_curve(reg,target,c,anchor,"median_curve"),color=color,ls="--")
            ax.plot(ds,[losses[str(d)][c]-losses["1.0"][c] for d in ds],ls="",marker="o",color=color)
        ax.axhline(0,color=PALETTE["reference"])
        ax.set(xlabel="Retained density",ylabel="Loss change (nats)",xticks=[.6,.8,1],xlim=(.52,1.04))
    specs=[("A: Pruning",SIZES["three"],prune,[{"predictions":pred,"losses":losses}])]
    states={s["tag"]:s for s in qf["states"]}
    dev={(r["state"],r["config"]):r for r in qd["dev_rows"] if r["capability"]=="math"}
    conf={(r["state"],r["config"]):r for r in qc["rows"] if r["capability"]=="math"}
    frozen={(r["state"],r["config"]):r for r in qf["predictions"] if r["capability"]=="math"}
    def quant(f,bit):
        ax=axes(f); g=np.geomspace(32,512,151)
        for j,tag in enumerate(old.QUANT_STATES):
            color=CAPABILITY_COLORS["math"]
            ax.plot(g,old.quant_curve(qf,states[tag],bit,g,"same_input_interpolation"),color=color,ls=("-",":")[j])
            gs=[32,64,128,256,512]
            ax.plot(gs,[(conf if k in (32,512) else dev)[tag,f"b{bit}_g{k}"]["dL"] for k in gs],ls="",marker=("o","s")[j],color=color)
            for k in (32,512):
                rr=frozen[tag,f"b{bit}_g{k}"]
                assert rr["predictions"]==conf[tag,f"b{bit}_g{k}"]["predictions"]
                for method in ("same_input_interpolation","median"):
                    old.close(old.quant_curve(qf,states[tag],bit,k,method),rr["predictions"][method],"quant freeze")
                ax.plot(k,rr["predictions"]["same_input_interpolation"],marker="D",ls="",mfc=PALETTE["transparent"],color=color)
        ax.plot(g,old.quant_curve(qf,states[old.QUANT_STATES[0]],bit,g,"median"),ls="--",color=PALETTE["reference"])
        ax.set(xscale="log",xlabel="Group size",ylabel="Loss change (nats)",xlim=(25,680))
        ax.set_xticks([32,128,512],["32","128","512"])
        ax.minorticks_off()
    for bit in (3,4,5):
        specs.append((f"B: Math, {bit} bit",SIZES["three"],lambda f,b=bit:quant(f,b),
                      [r for r in [*qd["dev_rows"],*qc["rows"]] if r["state"] in old.QUANT_STATES and r["capability"]=="math" and r["config"].startswith(f"b{bit}_")]))
    fr={(r["student"],r["pool"],r["T_planned"],r["capability"]):r for r in df["predictions"]}
    for r in dc["rows"]:
        assert r["predictions"]==fr[r["student"],r["pool"],r["T_planned"],r["capability"]]["predictions"]
        for method in (r["selected"],r["strongest_baseline"]):
            old.close(old.distill_curve(df,r["capability"],method,r["student"],r["DU"],r["E_planned"]),r["predictions"][method],"distill freeze")
    def distill(f,student):
        ax=axes(f);style=STUDENT_STYLES[student.split("-")[-1].upper()]
        for c in CAPS:
            color=CAPABILITY_COLORS[c];drawn=set()
            for seed in range(31,37):
                rr=sorted((r for r in dc["rows"] if r["student"]==student and r["capability"]==c and r["data_seed"]==seed),key=lambda r:r["T_planned"])
                assert len(rr)==3
                du=rr[0]["DU"];xx=[r["T_actual"]/du for r in rr]
                ax.plot(xx,[r["actual"] for r in rr],marker="o",color=color,ls=style,alpha=.5)
                ee=np.geomspace(min(r["E_planned"] for r in rr),max(xx),100)
                selected=df["selected"][c]["method"]
                curve_id=(selected,du if selected!="E" else None)
                if curve_id not in drawn:
                    ax.plot(ee,old.distill_curve(df,c,selected,student,du,ee),color=color,ls=style)
                    drawn.add(curve_id)
                baseline=df["strongest_baseline"][student][c]["method"]
                yy=old.distill_curve(df,c,baseline,student,du,ee)
                ax.plot(ee,yy,color=PALETTE["reference"],ls=style,alpha=.6)
        ax.set(xscale="log",xlabel="Reuse ratio",ylabel="Loss change (nats)",xlim=(.75,4.8))
        ax.set_xticks([1,2,4],["1","2","4"]);ax.minorticks_off()
    for student in old.STUDENTS:
        specs.append(("C: "+student.split("-")[-1].upper(),SIZES["three"],lambda f,s=student:distill(f,s),[r for r in dc["rows"] if r["student"]==student]))
    handles=capability_keys()+[key("Pre-specified",marker="D",hollow=True),key("Median",marker="",ls="--"),key("270M",marker="",ls=":"),key("1B",marker="",ls="--")]
    return publish(plt,audit,"final_relations",specs,handles,"Measured responses and frozen relations. A: pruning; B: Math by bit width (410M/143k circles and solid lines; 1.4B/16k squares and dotted lines); C: the twelve U=200 trajectories, separated by student. All original measurements and frozen objects are retained. Colour always denotes capability; grey denotes baseline. Frozen predictions are hollow diamonds. Student line styles: 270M dotted, 1B dashed. No data or fits changed.",legend_artist_sizes=(1.5,6.,1.))


def frozen_candidates(audit,plt):
    old=module("v86_main_table")
    # This loader checks the full frozen alternative sets; suppress all writes.
    from unittest.mock import patch
    original=old.Comparison
    class Comparison(original):
        def __init__(self,relative):
            self.path=relative; self.data=audit.read(relative); self.sha256=audit.inputs[relative]
    with patch.object(old,"Comparison",Comparison), patch.object(old.provenance,"_pairs",audit.published_pairs):
        rows,_=old.build_rows()
    records=[old.row_record(r) for r in rows[:6]]
    specs=[]
    for start,method in zip((0,2,4),("Pruning","Grouped quantization","Distillation")):
        bars=[dict(r["capabilities"][c],cap=c,row=r["row"]) for r in records[start:start+2] for c in CAPS]
        def draw(f,bars=bars):
            ax=axes(f,left=.43)
            for y,b in enumerate(bars):
                color=CAPABILITY_COLORS[b["cap"]]
                ax.barh(y,b["gain"],height=.5,color=color,edgecolor=darker(color),hatch=HATCHES["prediction"])
                if b["delivered_differs"]:
                    ax.plot(b["delivered_gain"],y,"D",mfc=PALETTE["transparent"],color=color)
            ax.axvline(0,color=PALETTE["reference"])
            ax.set(yticks=range(6),yticklabels=[CAP_NAMES[b["cap"]] for b in bars],ylim=(5.6,-.6),xlabel="MAE gain (nats)")
        specs.append((method,SIZES["three"],draw,bars))
    return publish(plt,audit,"frozen_candidates",specs,capability_keys()+[key("Delivered",marker="D",hollow=True)],"Frozen candidate gain against the same strongest frozen alternative, per capability. Each method panel retains both registered cohorts, top then bottom; exact candidate/alternative MAEs and cohort names are in the sidecars. Hatched bars are predictions; hollow diamonds show delivered-rule gains where different. Negative gains are retained.")


def gains(audit,plt,stem):
    old=module("plot_fig_inputs" if stem=="input_form_compact" else "plot_fig2_gains")
    groups=old.load_panels(audit)
    labels=("Added inputs","Same-input forms","Capability-specific shape") if stem=="input_form_compact" else ("Full versus no-D0","Source-conditioned versus source-free","Power versus alternatives")
    specs=[]
    for rows,label in zip(groups,labels):
        def draw(f,rows=rows):
            ax=axes(f)
            for i,r in enumerate(rows):
                for j,c in enumerate(CAPS):
                    color=CAPABILITY_COLORS[c];x=i+1;y=r["values"][c]
                    if c in r["intervals"]:
                        lo,hi=r["intervals"][c]
                        ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt="none",color=color)
                    frozen=r["origin"]=="P"
                    ax.plot(x,y,marker="D" if frozen else "o",ls="",color=color,mfc=PALETTE["transparent"] if frozen else color)
            ax.axhline(0,color=PALETTE["reference"])
            ax.set(xlim=(.3,len(rows)+.7),xlabel="Test index",ylabel="MAE gain (nats)")
            ticks=sorted(set([1,(len(rows)+1)//2,len(rows)]));ax.set_xticks(ticks)
        specs.append((label,SIZES["three"],draw,rows))
    caption="Paired improvements; positive favours the first-named model. Each integer test index refers to the corresponding row in the panel sidecar, in the original test order; capabilities are horizontally dodged. All original tests, values, counts and comparator identities are retained. Whiskers are stored 95% intervals, only where computed. Hollow diamonds denote frozen prospective predictions; filled circles denote leave-one-out estimates.\n"
    for label,rows in zip(labels,groups):
        caption+=label+": "+"; ".join(f"{i+1} = {r['label']}"+(f" / {r['baseline']}" if 'baseline'in r else '') for i,r in enumerate(rows))+".\n"
    return publish(plt,audit,stem,specs,capability_keys()+[key("Pre-specified",marker="D",hollow=True),key("Leave-one-out")],caption)


def measurement_support(audit,plt):
    old=module("plot_fig4_measurement")
    summary=audit.read("results/v26-loss-validity-pred/summary.json")
    old.load_and_verify(audit,summary)
    specs=[]; row_specs=[(co,pr) for co in old.COHORTS for pr in old.PROTOCOLS]
    predictors=(*CAPS,"cross_selected","zero","train_mean")
    for kind in ("mae","gain"):
        for cap in CAPS:
            records=[dict(cohort=co,protocol=pr,**summary["analyses"][co][pr][cap]) for co,pr in row_specs]
            def draw(f,kind=kind,cap=cap,records=records):
                ax=axes(f)
                for i,r in enumerate(records):
                    if kind=="mae":
                        for j,p in enumerate(predictors):
                            color=CAPABILITY_COLORS.get(p,PALETTE["reference"])
                            marker="D" if p==cap else {"cross_selected":"s","zero":"^","train_mean":"v"}.get(p,"o")
                            ax.plot(i+1,r["metrics"][p]["mae"],marker=marker,ls="",color=color,mfc=PALETTE["transparent"])
                    else:
                        y=r["same_over_cross_gain"];lo,hi=r["gain_ci95"]
                        ax.errorbar(i+1,y,yerr=[[y-lo],[hi-y]],fmt="o",color=CAPABILITY_COLORS[cap])
                ax.axhline(0,color=PALETTE["reference"])
                ax.set(xlim=(.3,6.7),xticks=[1,3,6],xlabel="Protocol index",ylabel="MAE (nats)" if kind=="mae" else "Paired gain (nats)")
            specs.append((CAP_NAMES[cap]+(" transfer MAE" if kind=="mae" else " paired gain"),SIZES["three"],draw,records))
    handles=capability_keys()+[key("Same",marker="D",hollow=True),key("Cross",marker="s",hollow=True),key("Zero",marker="^",hollow=True),key("Mean",marker="v",hollow=True)]
    return publish(plt,audit,"measurement_support",specs,handles,"Top row: every saved primary-to-secondary prediction MAE (all markers hollow). Capability colours identify the primary predictor; grey identifies controls. Diamond = same capability, square = cross-selected, triangle up = zero, triangle down = train mean. Bottom row: measured paired same-over-cross gains with stored 95% model-bootstrap intervals, colour by target capability. Protocol indices 1–3: model / density / model-and-density out, all densities; 4–6: the same protocols at d >= 0.75. Six source model clusters; fixed fits. No fitting or resampling.")


def generalization_cells(audit,plt):
    old=module("plot_fig_generalization_cells")
    rows=old.build(audit); specs=[]
    for panel in "ABC":
        part=[r for r in rows if r["panel"]==panel]
        groups=list(dict.fromkeys(r["group"] for r in part))
        labels = {
            "Pruning: density inside range": "Prune in", "Pruning: density outside range": "Prune out",
            "Quantization: new group size": "Quant g", "Distillation: new-pool budgets": "New pool",
            "Distillation: corner budgets (1B)": "1B corner", "Pythia: new stages (power)": "Stages",
            "Pythia: new quantization state": "Quant state", "Pythia: locked rule on new states": "Locked rule",
            "4B DEVELOPMENT: corner budgets": "4B corner", "2wiki_new": "2Wiki", "musique": "MuSiQue", "triviaqa": "TriviaQA"}
        def draw(f,part=part,groups=groups,panel=panel):
            ax=axes(f,bottom=.50)
            # Y shows the actual signed error; x is categorical group index.
            # Spreading within a group is display-only, never a residual change.
            for j,group in enumerate(groups):
                rr=[r for r in part if r["group"]==group]
                for i,r in enumerate(rr):
                    x=j+1
                    color=QA_COLORS.get(group,CAPABILITY_COLORS[r["capability"]])
                    if r["residual_interval"] is not None:
                        lo,hi=r["residual_interval"];y=r["residual"]
                        whisker=ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt="none",color=color)
                        for segment in whisker.lines[2]:segment.set_linestyle("-" if r["status"]=="development" else "--")
                    ax.plot(x,r["residual"],marker="x" if r["within"] is None else "D",ls="",color=color,mfc=PALETTE["transparent"])
            ax.set_yscale("symlog",linthresh=.03)
            ax.axhline(0,color=PALETTE["reference"])
            ax.set(xlim=(.2,len(groups)+.8),xticks=range(1,len(groups)+1),ylabel="Error (nats)")
            ax.set_xticklabels([labels[g] for g in groups], rotation=50, ha="right")
            ax.set_yticks([-1,0,1],["−1","0","1"]);ax.minorticks_off()
        specs.append(({"A":"New configurations","B":"New sources or students","C":"Fresh QA distributions"}[panel],SIZES["three"],draw,part))
    caption="Per-cell prediction minus measurement for every frozen relation. Colour denotes capability; fresh QA distributions retain distinct green shades. All records retained, including repeated states and development 4B records. Cross = no stored measurement interval; hollow diamond = prediction with the stored registered corner band. No coverage classification is encoded by fill. Whiskers are registered twice-noise bands, not confidence intervals.\n"
    for panel in "ABC":caption+=panel+": "+"; ".join(f"{i+1} = {g}" for i,g in enumerate(dict.fromkeys(r['group'] for r in rows if r['panel']==panel)))+".\n"
    publish(plt,audit,"generalization_cells",specs,capability_keys()+[key("Band",marker="D",hollow=True),key("No interval",marker="x"),key("1B",marker="",ls="--"),key("4B development",marker="",ls="-")],caption)
    return rows


def maps(audit,plt,stem):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle
    old=stem=="selection_feasible"; full=stem=="rule_maps_full"
    data=audit.read("results/v64-selection-feasible/summary.json" if old else "results/v78-rule-confirm/compare.json")
    if old:states=[s["tag"] for s in data["states"]]
    else:states=[s["tag"] for s in audit.read("results/v78-rule-confirm/freeze.json")["states"]]
    caps=CAPS if old else (*CAPS,"multi")
    policies=("MAP",) if old else ("locked-rule","v64-law") if full else ("locked-rule",)
    specs=[];size=(1.8,2.4) if old else (2.7,1.45)
    inks=[METHOD_COLORS[m] for m in METHODS]+[PALETTE["reference"]]
    colors=[surface(c) for c in inks[:-1]]+[PALETTE["background"]]
    for cap in caps:
        records=data["cells"][cap]
        def draw(f,records=records):
            ax=axes(f,size,left=.69 if old else 1.02 if full else .75,bottom=.32,right=.06)
            budgets=sorted({r["budget"] for r in records});index={(r["state"],r["budget"]):r for r in records}
            matrix=np.empty((len(states)*len(policies),len(budgets)),dtype=int)
            for i,s in enumerate(states):
                for j,p in enumerate(policies):
                    y=i*len(policies)+j
                    for x,b in enumerate(budgets):
                        r=index[s,b];choice=r["policies"][p];method=choice["method"]
                        feasible=method in METHODS;value=METHODS.index(method) if feasible else 4
                        matrix[y,x]=value
                        mismatch=feasible and method!=r["oracle_method"]
                        ambiguity=r["no_clear_winner_heuristic"] if old else r["candidate_sets"][p]["no_clear_winner_heuristic"]
                        hatch=HATCHES["infeasible"] if not feasible else HATCHES["oracle_differs"] if mismatch else HATCHES["ambiguous"] if ambiguity and (old or full) else None
                        if hatch:
                            ax.add_patch(Rectangle((x-.5,y-.5),1,1,facecolor=PALETTE["transparent"],edgecolor=inks[value],hatch=hatch,linewidth=0,zorder=2))
                        if mismatch:ax.plot(x,y,marker="o",ls="",color=PALETTE["black"],mfc=PALETTE["transparent"],zorder=4)
            ax.imshow(matrix,cmap=ListedColormap(colors),vmin=-.5,vmax=4.5,aspect="auto",interpolation="nearest",zorder=0)
            labels=[]
            for s in states:
                size_name,step=s.removeprefix("pythia-").split("@step")
                for p in policies:labels.append(f"{size_name.upper()}/{int(step)//1000}k"+(" F" if p=="locked-rule" else " L") if full else f"{size_name.upper()}/{int(step)//1000}k")
            ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel="Storage budget (%)",xlim=(-.5,len(budgets)-.5),ylim=(len(labels)-.5,-.5))
            ax.set_xticks([0,8,16],["20","60","100"])
            ax.set_xticks(np.arange(-.5,len(budgets)),minor=True)
            ax.set_yticks(np.arange(-.5,len(labels)),minor=True)
            ax.grid(which="minor",color=PALETTE["white"],lw=.35,alpha=.7)
            ax.tick_params(length=0,which="both")
            for spine in ax.spines.values():spine.set_visible(False)
        specs.append((CAP_NAMES[cap],size,draw,records))
    handles=[Patch(facecolor=surface(METHOD_COLORS[m]),edgecolor=METHOD_COLORS[m],label=m.title() if m!="quant" else "Quant") for m in METHODS]
    handles += [key("Oracle differs",PALETTE["black"],hollow=True)]
    # The mismatch hatch repeats the ring, and ambiguity is drawn only on the full maps.
    if old or full:handles += [Patch(facecolor=PALETTE["white"],edgecolor=PALETTE["reference"],hatch=HATCHES["ambiguous"],label="No clear winner")]
    return publish(plt,audit,stem,specs,handles,"Saved selection maps; colour denotes the predicted compression method. A hollow black circle and cross-hatching in the cell colour both mark an oracle method mismatch. Diagonal hatching marks no-clear-winner cells without mismatch; both flags remain in sidecars when they coincide. Infeasible cells use grey cross-hatching without a circle. F = frozen rule; L = earlier source-conditioned laws. Native nominal storage budgets and all states retained; no selection rule is executed.",columns=3 if old else 2)


def rule_regret(audit,plt):
    from matplotlib.patches import Patch
    data=audit.read("results/v80-rule-addenda/summary.json")["pooled_mean_regret"]
    policies=("locked-rule","v64-law","quant-only","cheapest");caps=(*CAPS,"multi")
    def draw(f):
        ax=axes(f,SIZES["full"],left=.5)
        for j,c in enumerate(caps):
            color=CAPABILITY_COLORS.get(c,PALETTE["reference"])
            for i,p in enumerate(policies):
                hatch=("",HATCHES["prediction"],HATCHES["code"],HATCHES["qa"])[i]
                ax.bar(j+(i-1.5)*.18,data[c][p],width=.16,color=color,edgecolor=darker(color),hatch=hatch)
        ax.set(yscale="log",ylabel="Mean regret (nats)",xticks=range(4),xticklabels=["Math","Code","QA","Maximum"],xlabel="Objective")
    handles=[Patch(facecolor=PALETTE["dense"],edgecolor=darker(PALETTE["dense"]),hatch=h,label=l) for h,l in zip(("","///","...","xx"),("Final rule","Earlier laws","Quant only","Cheapest"))]
    return publish(plt,audit,"rule_regret",[("Mean regret",SIZES["full"],draw,[data])],handles,"Mean regret on four fresh states; the saved pooled means are plotted without recomputation. Colour denotes objective; policy uses plain / diagonal / dotted / cross-hatched bars. Logarithmic axis, native-token nats.",columns=1)


# Numbered as in the identity table: R1 Pythia 410M, R2 Pythia 1.4B, R3 Gemma 3 1B, R4 Gemma 3 4B.
S3_REFERENCES = (("pythia-410m--step120000", "R1"), ("pythia-1.4b--step120000", "R2"),
                 ("gemma3-1b", "R3"), ("gemma3-4b", "R4"))


# Body Figure 4: no x-axis label, so the bottom margin only holds the tick labels, and the one-row
# legend strip needs .2 in, not the .42 in reserved for two-row keys.
S3_PANEL = (1.35, 1.3)


def s3_validation(audit,plt):
    """Per-reference opportunity and policy regrets of the independent selection validation."""
    from matplotlib.patches import Patch
    score=audit.read("results/s3-selection-validation/score_v2.json")
    if score["status"]!="COMPLETE":raise ValueError("S3 scoring is not complete")
    refs={r["reference"]:r for r in score["references"]}
    if set(refs)!={k for k,_ in S3_REFERENCES}:raise ValueError("S3 references differ from the registered four")
    caps=(*CAPS,"multi")
    def values(cap):
        out=[]
        for key,_ in S3_REFERENCES:
            o=refs[key]["objectives"][cap]
            row=(o["mean_opportunity"],o["policies"]["rule"]["mean_paired_regret"],o["policies"]["quant-only"]["mean_paired_regret"])
            if any(v is None for v in row):raise ValueError(f"S3 {key}/{cap} has an undefined mean")
            out.append(row)
        return out
    data={cap:values(cap) for cap in caps}
    audit.rule("Saved S3 means are plotted without recomputation: mean_opportunity and the two policies' "
               "mean_paired_regret per reference and objective, over the sixteen budgets at which both policies find a candidate.")
    specs=[]
    for cap in caps:
        def draw(f,cap=cap):
            ax=axes(f,S3_PANEL,left=.42,bottom=.23,right=.05)
            color=CAPABILITY_COLORS.get(cap,PALETTE["reference"])
            for j,row in enumerate(data[cap]):
                ax.bar(j-.26,row[0],width=.24,color=PALETTE["white"],edgecolor=color,zorder=3)
                ax.bar(j,row[1],width=.24,color=color,edgecolor=darker(color),zorder=3)
                ax.bar(j+.26,row[2],width=.24,color=color,edgecolor=darker(color),hatch=HATCHES["code"],zorder=3)
            ax.set(xticks=range(len(S3_REFERENCES)),xticklabels=[label for _,label in S3_REFERENCES],xlim=(-.6,len(S3_REFERENCES)-.4))
            ax.set_ylim(bottom=0)
            if cap==caps[0]:ax.set_ylabel("Nats per token")
            ax.grid(axis="y",color=PALETTE["grid"],lw=.4,zorder=0)
        specs.append((CAP_NAMES[cap],S3_PANEL,draw,data[cap]))
    handles=[Patch(facecolor=PALETTE["white"],edgecolor=PALETTE["reference"],label="Opportunity"),
             Patch(facecolor=PALETTE["dense"],edgecolor=darker(PALETTE["dense"]),label="Final rule"),
             Patch(facecolor=PALETTE["dense"],edgecolor=darker(PALETTE["dense"]),hatch=HATCHES["code"],label="Quantization only")]
    return publish(plt,audit,"s3_validation",specs,handles,"Independent selection validation on four references that supplied the rule no outcome, one panel per objective. Bars per reference: the opportunity that choosing across methods opens over the best feasible quantization candidate (open), the regret of the frozen rule (solid) and the regret of a quantization-only policy (dotted), each a mean in nats per native token over the sixteen budgets at which both policies find a feasible candidate; nothing is pooled across tokenizers. Colour denotes objective. References are numbered as in the identity table: R1 Pythia 410M at step 120000, R2 Pythia 1.4B at step 120000, R3 Gemma 3 1B, R4 Gemma 3 4B.",columns=4,legend_size=(5.5,.2))


def generate(stem=None,root=ROOT):
    with frozen_run(root) as access:
        audit=Artifacts(root);plt=pyplot(root);rows=None
        for name in ((stem,) if stem else STEMS):
            if name in ("input_form_compact","input_form_gain"):result=gains(audit,plt,name)
            elif name.startswith("rule_maps") or name=="selection_feasible":result=maps(audit,plt,name)
            else:result=globals()[name](audit,plt)
            if name=="generalization_cells":rows=result
        return rows,audit,access


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure",choices=STEMS)
    generate(parser.parse_args().figure)
