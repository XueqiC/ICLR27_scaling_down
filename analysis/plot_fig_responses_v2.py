#!/usr/bin/env python3
"""Matched-budget observed responses from the canonical, frozen A1 table."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from collections import defaultdict

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, development_rows, save_figure, write_notes
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, development_rows, save_figure, write_notes

STUDENTS = ("gemma3-270m", "gemma3-1b", "gemma3-4b")
STYLES = (":", "--", "-")
SCOPES = ("2wiki_new", "musique", "triviaqa")


def endpoints(rows):
    """One nearest-positive checkpoint per trajectory, capability and distribution.

    T_actual and D_U_pool are the canonical supervised-token accounting; never
    pool distinct seeds, interpolate checkpoints, or use nominal epoch counts.
    """
    groups = defaultdict(list)
    for row in rows:
        if row["T_actual"] > 0:
            groups[row["run_id"], row["capability"], row["distribution"]].append(row)
    return [min(part, key=lambda r: (abs(r["T_actual"] - 200000), r["checkpoint_id"]))
            for _, part in sorted(groups.items())]


def build(audit):
    rows = endpoints(development_rows(audit))
    result = []
    for row in rows:
        distribution = row["distribution"].split(":", 1)[0]
        if distribution == "training_probe":
            panel, series = "left", row["capability"]
        elif distribution in SCOPES:
            panel, series = "right", distribution
        else:
            continue
        if not row["D_U_pool"]:
            raise ValueError("Missing positive canonical pool-token count")
        result.append({"panel": panel, "series": series, "student": row["student_id"],
                       "reuse": row["T_actual"] / row["D_U_pool"], "delta": row["delta"],
                       "pool_seed": row["pool_seed"], "U": row["U"], "T_actual": row["T_actual"],
                       "D_U_pool": row["D_U_pool"], "run_id": row["run_id"],
                       "checkpoint_id": row["checkpoint_id"], "distribution": row["distribution"],
                       "status": "development"})
    audit.rule("Reused a1_development_table.load_development_table (schema, cohort, metadata and hash checks). "
               "A1 CSV fields: run_id, checkpoint_id, student_id, pool_seed, U, T_actual, D_U_pool, "
               "capability, distribution, delta. Select positive T_actual nearest 200000 separately per trajectory/readout. "
               "reuse=T_actual/D_U_pool; y=delta (own-initial loss subtracted by A1). Both pool seeds remain separate points.")
    audit.rule("The four-rung core includes pool seeds 41/42 and, for the critical rung, 51/52. "
               "Seed markers identify the first/second registered seed within each rung; no seed averaging. "
               "Scope rows in the same CSV originate in v99-scope. All points, including 4B, are development.")
    return result


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit=Artifacts(root); plt=pyplot(root); rows=build(audit)
        from matplotlib.lines import Line2D
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.8))
        for ax,panel,series,title in zip(axes,("left","right"),(CAPS,SCOPES),
                                      ("A  Capability responses", "B  Fresh QA distributions")):
            for c,color in zip(series,COLORS.values()):
                for student,style in zip(STUDENTS,STYLES):
                    subset=[r for r in rows if r["panel"]==panel and r["series"]==c and r["student"]==student]
                    for parity,marker in ((1,"o"),(0,"s")):
                        line=sorted((r for r in subset if r["pool_seed"]%2==parity),key=lambda r:r["reuse"])
                        ax.plot([r["reuse"] for r in line],[r["delta"] for r in line],linestyle=style,
                                marker=marker,ms=4,lw=1,color=color,alpha=.85)
            ax.axhline(0,color=".4",lw=.65)
            ax.set(title=title,xlabel=r"Reuse $T/D_U$ (supervised-token passes)",
                   ylabel="Loss change (native-token nats)")
            ax.text(.02,.03,"Positive = worse",transform=ax.transAxes,fontsize=8)
            ax.grid(alpha=.15)
            ax.legend(handles=[Line2D([],[],color=color,label=c) for c,color in zip(series,COLORS.values())],loc="lower right",fontsize=7)
        handles=[Line2D([],[],color=".25",ls=style,label=student.replace("gemma3-", "")) for student,style in zip(STUDENTS,STYLES)]
        handles += [Line2D([],[],ls="",marker=m,color=".25",label=label) for m,label in (("o","first pool seed"),("s","second pool seed"))]
        fig.legend(handles=handles,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.5,.055))
        fig.text(.5,.01,"Development endpoints nearest 200k supervised tokens per trajectory; every seed shown separately.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.15,1,1))
        save_figure(fig,"responses_v2",audit);write_notes("responses_v2",audit,rows);plt.close(fig)
        return rows,audit,access


if __name__=="__main__":
    generate()
