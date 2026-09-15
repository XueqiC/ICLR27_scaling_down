#!/usr/bin/env python3
"""Curvature, registered corner test, and post-hoc displacement account."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from collections import defaultdict
from statistics import median

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, save_figure, write_notes
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, save_figure, write_notes

A2="results/a2-curvature-interaction/summary.json"
A5="results/a5-corner-second-difference/summary.json"
A7="results/a7-closeout-audit/summary.json"
CAPTION="failed to reject"


def curvature(audit):
    data=audit.read(A2); result=[]
    for i,row in enumerate(data["parameter_stability"]):
        # Training probes provide one comparable readout per capability.
        if not row["distribution"].startswith("training_probe:"):
            continue
        j,full=next((j,r) for j,r in enumerate(data["parameter_intervals"]) if r["distribution"]==row["distribution"])
        fit=full["fits"]["F_curv"]
        result.append({"capability":row["capability"],"p":fit["fit"]["p"],"interval":fit["p_interval"],
                       "boundary":fit["fit"]["boundary_hit"],"full_source":f"{A2}#/parameter_intervals/{j}/fits/F_curv",
                       "folds":[{"p":f["p"],"boundary":f["boundary"],"split":f["split"],"held":f["held"],
                                 "descriptor":f["descriptor"],"source":f"{A2}#/parameter_stability/{i}/folds/{k}"}
                                for k,f in enumerate(row["folds"]) if f["name"]=="F_curv"]})
    return sorted(result,key=lambda r:CAPS.index(r["capability"]))


def corners(audit):
    a7=audit.read(A7); a5=audit.read(A5); result=[]
    for i,row in enumerate(a7["checks"]["predictions"]["rows"]):
        measured=a5["students"][row["student"]]["readouts"][row["capability"]]
        if row["measured_I"]!=measured["I"] or row["registered_noise"]!=measured["noise_on_I"]:
            raise ValueError("A5/A7 second-difference or noise mismatch")
        result.append({"student":row["student"],"capability":row["capability"],"readout":row["readout"],
                       "measured":row["measured_I"],"interval":measured["interval"],"noise":row["registered_noise"],
                       "additive":row["additive_rectangle_prediction"],"F_int":row["predictions"]["F_int"]["I"],
                       "source":f"{A7}#/checks/predictions/rows/{i}",
                       "interval_source":f"{A5}#/students/{row['student']}/readouts/{row['capability']}/interval"})
    audit.rule("Corner caption: failed to reject. This wording applies to the primary QA additivity test. "
               "Math is size-dependent and code unresolved; students are never averaged. "
               "A7 check 4 rows supply additive_rectangle_prediction, predictions.F_int.I, measured_I and registered_noise; "
               "A5 supplies interval = measured I +/- twice registered noise, not a confidence interval. "
               "Shading is +/- one registered noise; whiskers are the registered +/- twice-noise band.")
    return result


def displacement(audit):
    findings="results/v88-displacement/FINDINGS.md"
    audit.read(findings)
    # The only JSON summary is the superseded, two-family uncentred run. Read it
    # for provenance, but never mix its measurements into the current run.
    audit.read("results/v88-displacement-uncentred-run/summary.json")
    groups=defaultdict(list)
    for path in sorted((audit.root/"results/v88-displacement").glob("*/*.json")):
        row=audit.read(path)
        if row.get("selftest") or row.get("experiment")!="v88-displacement":
            continue
        config=row["config"]
        for cap,metric in row["per_capability"].items():
            damage=abs(metric["measured_delta"])
            if damage==0:
                continue
            groups[config["kind"],config["id"]].append({"damage":damage,
                "relative_error":abs(metric["second_order_prediction"]-metric["measured_delta"])/damage,
                "source":f"{path.relative_to(audit.root).as_posix()}#/per_capability/{cap}","device":row["device_name"]})
    result=[{"family":family,"config":config,"damage":median(r["damage"] for r in rr),
             "median_relative_error":median(r["relative_error"] for r in rr),"n":len(rr),"records":rr}
            for (family,config),rr in sorted(groups.items())]
    if len({r["family"] for r in result})!=3:
        raise ValueError("Expected pruning, grouped RTN and per-channel RTN")
    audit.rule("V88 current narrative summary: results/v88-displacement/FINDINGS.md. "
               "No current three-family summary.json exists. The only JSON summary found is "
               "results/v88-displacement-uncentred-run/summary.json (superseded two-family run, not plotted). "
               "Panel c uses current results/v88-displacement/*/*.json per_capability.measured_delta and "
               "per_capability.second_order_prediction: per-configuration median |damage| and median |prediction-measured|/|measured|, "
               "equal state/capability records; zero damage excluded. This is aggregation, not fitting. "
               "The separate v88-displacement-cluster hardware run is not pooled. "
               "Three compression families here mean pruning, grouped RTN, per-channel RTN; there is no distillation displacement measurement.")
    return result


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit=Artifacts(root);plt=pyplot(root)
        profiles=curvature(audit);corner=corners(audit);disp=displacement(audit)
        from matplotlib.lines import Line2D
        fig,axes=plt.subplots(1,3,figsize=(12,5.2),gridspec_kw={"width_ratios":[1,1.2,1.1]})
        ax=axes[0]
        for i,r in enumerate(profiles):
            color=COLORS[r["capability"]]
            for k,f in enumerate(r["folds"]):
                # Deterministic offsets expose every fold without random draws.
                x=i-.25+.38*k/max(1,len(r["folds"])-1)
                ax.plot(x,f["p"],marker="x" if f["boundary"] else "o",color=color,ms=3,alpha=.65)
            lo,hi=r["interval"]; p=r["p"]
            ax.errorbar(i+.28,p,yerr=[[p-lo],[hi-p]],fmt="D",color=color,ms=5,capsize=3)
        ax.axhline(0,color=".65",lw=.7,ls=":");ax.axhline(1,color=".65",lw=.7,ls="--")
        ax.set(xticks=range(3),xticklabels=[c.title() for c in CAPS],ylabel="Reuse curvature exponent p (unitless)",title="a  Reuse-term curvature")
        ax.legend(handles=[Line2D([],[],marker=m,ls="",color=".3",label=l) for m,l in
                           (("o","fold"),("x","boundary hit"),("D","full development + conditional interval"))],fontsize=6,loc="upper center",bbox_to_anchor=(.5,-.15))
        ax=axes[1]
        for i,r in enumerate(corner):
            color=COLORS[r["capability"]]
            ax.fill_betweenx([i-.36,i+.36],-r["noise"],r["noise"],color=".88",zorder=0)
            ax.plot(r["additive"],i-.18,"|",color=".2",ms=8)
            ax.plot(r["F_int"],i+.18,"D",color="#7c4da1",ms=4)
            lo,hi=r["interval"];m=r["measured"]
            ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt="o",color=color,ms=4,capsize=2)
        ax.set_xscale("symlog",linthresh=.02)
        ax.set(yticks=range(len(corner)),yticklabels=[r["student"].replace("gemma3-", "")+" "+r["capability"].upper() for r in corner],
               xlabel="Second difference (native-token nats; symlog)",title="b  Corner test")
        ax.invert_yaxis()
        ax.legend(handles=[Line2D([],[],marker=m,ls="",color=c,label=l) for m,c,l in
                           (("|",".2","additive"),("D","#7c4da1","frozen F_int"),("o",".3","measured + registered band"))],
                  fontsize=6,loc="upper center",bbox_to_anchor=(.5,-.20),bbox_transform=ax.transAxes,borderaxespad=0)
        ax=axes[2]
        labels={"prune":"Pruning","pruning":"Pruning","grouped_rtn":"Grouped RTN","per_channel_rtn":"Per-channel RTN"}
        for family,color,marker in zip(sorted({r["family"] for r in disp}),COLORS.values(),("o","s","^")):
            part=sorted((r for r in disp if r["family"]==family),key=lambda r:r["damage"])
            ax.plot([r["damage"] for r in part],[100*r["median_relative_error"] for r in part],marker=marker,color=color,lw=1,label=labels.get(family,family))
        ax.set(xscale="log",yscale="log",xlabel="Median |damage| (native-token nats)",ylabel="Median relative error (%)",title="c  Displacement account (post-hoc)")
        ax.legend(fontsize=7);ax.grid(alpha=.15,which="both")
        fig.text(.5,.065,CAPTION,ha="center",fontsize=9)
        fig.text(.5,.025,"Primary QA additivity test; math size-dependent, code unresolved. Curvature intervals are conditional on development.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.14,1,1))
        audit.rule("A2 panel a uses training-probe parameter_stability[].folds[name=F_curv].p/boundary and "
                   "parameter_intervals[].fits.F_curv.fit.p / p_interval / fit.boundary_hit; no profile or bootstrap is rerun.")
        data={"profiles":profiles,"corners":corner,"displacement":disp}
        save_figure(fig,"explanation",audit);write_notes("explanation",audit,data);plt.close(fig)
        return data,audit,access


if __name__=="__main__":
    generate()
