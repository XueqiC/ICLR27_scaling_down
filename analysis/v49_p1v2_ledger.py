"""Emit the markdown ledger block for a P1-v2 source pair from compare_*.json (regenerable)."""
import json, sys
from pathlib import Path
R = Path("results/v49-p1v2")
PR = ["power", "A2", "A1", "cont", "strength_only", "median_curve", "zero"]
QU = ["full", "noD0", "per_bit_mean", "per_bit_median", "zero"]
def row(name, d, keys):
    best = min(keys, key=lambda k: d[k])
    return "| " + name + " | " + " | ".join(("**%.3f**" if k == best else "%.3f") % d[k] for k in keys) + " |"
def block(tags):
    out = []
    for tag in tags:
        c = json.load(open(R / f"compare_{tag}.json"))["protocols"]
        out.append(f"\n#### {tag}\n")
        out.append("| prune MAE (nats) | " + " | ".join(PR) + " |\n|" + "---|" * (len(PR) + 1))
        for pk in "AB":
            s = c[pk]["summary"]["prune"]
            out.append(row(f"{pk} interp d0.9-0.6", s["interp_0.9-0.6"], PR))
            out.append(row(f"{pk} extrap d0.55", s["extrap_0.55"], PR))
            rows = c[pk]["rows_prune"]
            for cap in ("math", "code", "qa"):
                rr = [x for x in rows if x["cap"] == cap and 0.6 <= x["d"] <= 0.9]
                out.append(row(f"{pk} interp {cap}", {k: sum(x[k]["abs"] for x in rr) / len(rr) for k in PR}, PR))
        out.append("\n| quant MAE (nats) | " + " | ".join(QU) + " |\n|" + "---|" * (len(QU) + 1))
        for pk in "AB":
            s = c[pk]["summary"]
            for b in ("int8", "int6", "int5", "int4", "int3"):
                out.append(row(f"{pk} {b}" + (" (rule interp 4,6)" if b == "int5" else ""), s["quant"][b], QU))
            out.append(row(f"{pk} >=4-bit aggregate", s["quant_ge4"], QU))
    return "\n".join(out)
if __name__ == "__main__":
    print(block(sys.argv[1:]))
