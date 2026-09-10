#!/usr/bin/env python3
"""CPU-only transfer matrices, including confirmation and grouped quantization."""
import sys
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
if __package__:
    from .plot_fig1_responses import Audit, CAPS, CAP_LABEL, COLORS, NOTES, ROOT, checked_close, setup_style
else:
    from plot_fig1_responses import Audit, CAPS, CAP_LABEL, COLORS, NOTES, ROOT, checked_close, setup_style
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

SOURCES = (("1b", 32000), ("1b", 96000), ("1b", 112000), ("6.9b", 32000), ("6.9b", 112000))
CONFIRMATION = (("410m", 48000, "410M@48k"), ("1.4b", 112000, "1.4B@112k"),
                ("6.9b", 80000, "6.9B@80k"))
CONFIRMATION_REGIME = "Prune interp\nd=.85/.675/.575"
GROUP_TESTS = ("bit_test", "granularity_test", "joint_test")
REGIMES = ("Prune interp\nd=.9–.6", "Prune extrap\nd=.55", "Quantization\nbits ≥4", "Quantization\nint3")
BASELINES = (("strength_only", "median_curve", "zero"), ("per_bit_mean", "per_bit_median", "zero"))
SHORT = {"strength_only": "strength", "median_curve": "median", "zero": "zero",
         "per_bit_mean": "bit mean", "per_bit_median": "bit median", "mean": "mean", "median": "median"}


class AppendAudit(Audit):
    """Reuse audited reads/saves, but preserve every byte of earlier build notes."""

    def finish(self):
        section = f"## {self.title}\n\n"
        section += "Build: CPU, matplotlib Agg; shared Fig. 1 style and colours; no fitting.\n\n"
        section += "Outputs: " + ", ".join(f"`{p}`" for p in self.outputs) + ".\n\n"
        section += "Selection and calculations:\n\n" + "".join(f"- {s}\n" for s in self.rules)
        section += "\nLimitations:\n\n"
        section += "".join(f"- {s}\n" for s in self.limitations) or "- None.\n"
        section += "\nExact data files read:\n\n" + "".join(f"- `{p}`\n" for p in self.files)
        old = NOTES.read_text() if NOTES.exists() else ""
        if section in old:
            print(f"NOTES unchanged {NOTES} (identical build already recorded)")
            return
        NOTES.parent.mkdir(parents=True, exist_ok=True)
        with NOTES.open("a") as handle:
            handle.write("\n\n" + section)
        print(f"APPENDED {NOTES} ({self.title})")


def load_v49(audit):
    """Shared source-state loading for the transfer and compact input figures."""
    data = {}
    for size, step in SOURCES:
        if step == 96000:
            continue
        path = ROOT / f"results/v49-p1v2/compare_pythia-{size}@step{step}.json"
        if path.exists():
            data[size, step] = audit.read(path)
        else:
            audit.omit(f"{path.relative_to(ROOT)} is missing: both protocol rows marked not evaluated.")
    return data


def load_confirmation(audit):
    data = {}
    for size, step, label in CONFIRMATION:
        payload = audit.read(f"results/v53-prune-dev/compare_pythia-{size}@step{step}.json")
        checked_close(payload["densities"], [.85, .675, .575], f"confirmation densities/{label}")
        for name, metrics in payload["mae"].items():
            for cap in CAPS:
                errors = payload["absolute_errors"][cap]
                checked_close(metrics[cap], np.mean([errors[str(d)][name] for d in payload["densities"]]),
                              f"confirmation MAE/{label}/{name}/{cap}")
            checked_close(metrics["mean"], np.mean([metrics[c] for c in CAPS]),
                          f"confirmation capability mean/{label}/{name}")
        data[label] = payload
    return data


def score_capability_mae(mae, candidate, baseline_names):
    """Choose one baseline after equal-weight capability averaging."""
    means = {name: float(np.mean([mae[name][c] for c in CAPS]))
             for name in (candidate, *baseline_names)}
    baseline = min(baseline_names, key=means.__getitem__)
    return {"gain": means[baseline] - means[candidate], "candidate": means[candidate],
            "baseline": baseline, "baseline_mae": means[baseline]}


def load_grouped_quantization(audit):
    payload = audit.read("results/v55-quant-group/compare.json")
    cells = {}
    for test in GROUP_TESTS:
        table = {r["candidate"]: r for r in payload["test_sets"][test]["mae_table"]}
        mae = {name: entry["mae"] for name, entry in table.items()}
        for cap in CAPS:
            rows = [r for r in payload["rows"] if r["test_set"] == test and r["capability"] == cap]
            for name in ("low_order_2d", "mean", "median", "zero"):
                checked_close(len(rows), table[name]["n"][cap], f"v55 count/{test}/{name}/{cap}")
                errors = [r["absolute_errors"][name] for r in rows]
                checked_close(errors, [abs(r["predictions"][name] - r["dL"]) for r in rows],
                              f"v55 absolute errors/{test}/{name}/{cap}")
                checked_close(mae[name][cap], np.mean(errors), f"v55 MAE/{test}/{name}/{cap}")
        cells[test] = score_capability_mae(mae, "low_order_2d", ("mean", "median", "zero"))
        cell = cells[test]
        states = sorted({r["state"] for r in payload["rows"] if r["test_set"] == test})
        audit.rule(f"v55 {test}: states={states}; low_order_2d capability-averaged MAE={cell['candidate']:.9f}; "
                   f"strongest simple={cell['baseline']}, MAE={cell['baseline_mae']:.9f}; gain={cell['gain']:+.9f}.")
    audit.rule("v55 bit_test and granularity_test evaluate seen development states; joint_test evaluates "
               "held-out 1B@96k. Use test_sets.mae_table, verified against rows; choose one of mean/median/zero "
               "after averaging math/code/QA. Candidate is always low_order_2d, without test-set selection.")
    return cells


def load_multistudent_distillation(audit):
    """Use the frozen dev choice and the summary-row means from v62_p2v2_test_table.py."""
    freeze = audit.read("results/v50-p2v2/freeze.json")
    payload = audit.read("results/v50-p2v2/compare_test.json")
    selected = {c: min(freeze["fits"][c], key=lambda form: freeze["fits"][c][form]["dev_mae"])
                for c in CAPS}
    if any(form != "joint+src" for form in selected.values()):
        raise ValueError("Frozen dev-selected headline form changed")
    agg = defaultdict(lambda: defaultdict(list))
    zero, npts = defaultdict(list), Counter()
    for key, row in payload["summary"].items():
        student, role, cap, budget = key.split("|")
        for form, mae in row["mae"].items():
            agg[student, role, cap][form].append(mae)
        zero[student, role, cap].append(row["mae_zero"])
        npts[student, role, cap] += row["n_pools"]
    groups = {}
    for student, label in (("gemma3-270m", "270M, pool 375"), ("gemma3-1b", "1B, pool 375"),
                           ("gemma3-4b", "4B*, pool 375")):
        groups[label] = {}
        for cap in CAPS:
            key = student, "test_pool", cap
            values = agg[key][selected[cap]]
            candidate = sum(values) / len(values)
            baseline = sum(agg[key]["constant"]) / len(agg[key]["constant"])
            zero_mae = sum(zero[key]) / len(zero[key])
            checked_close(npts[key], 12, f"v50 test point count/{student}/{cap}")
            groups[label][cap] = {"candidate": candidate, "baseline_mae": baseline, "zero_mae": zero_mae,
                                  "gain": baseline - candidate}
            audit.rule(f"U375 {student}/{cap}: joint+src MAE={candidate:.9f}, frozen constant MAE={baseline:.9f}, zero MAE={zero_mae:.9f}, "
                       f"gain={baseline - candidate:+.9f}; {npts[key]} points.")
    audit.rule("Multi-student distillation uses only role test_pool: mean over all summary rows for each "
               "(student, role, capability), summing n_pools for point counts, as in v62_p2v2_test_table.py. "
               "Candidate minimizes fits[cap][form].dev_mae in freeze.json 33c706c: joint+src for every "
               "capability. Predictions and baselines were frozen before tests (P/F/F/A); the headline "
               "selection rule was stated after tests (R). Test results: compare_test.json c06c856. "
               "Baseline is zero; * marks the held-out Gemma-3-4B student.")
    return groups


def regime_rows(prune, quant):
    return ([r for r in prune if .6 <= r["d"] <= .9],
            [r for r in prune if np.isclose(r["d"], .55)],
            [r for r in quant if r["bit"] >= 4],
            [r for r in quant if r["bit"] == 3])


def score_cell(rows, candidate, baseline_names, stored=None):
    if not rows:
        return None
    # Every metric is evaluated on the same source/regime rows; baseline chosen once per cell.
    names = [candidate] + [name for name in baseline_names if all(name in r for r in rows)]
    mae = {name: float(np.mean([r[name]["abs"] for r in rows])) for name in names}
    for name in names:
        checked_close([r[name]["abs"] for r in rows],
                      [abs(r[name]["pred"] - r["actual"]) for r in rows], f"row absolute errors/{name}")
    if stored is not None:
        for name in names:
            checked_close(mae[name], stored[name], f"saved regime MAE/{name}")
        mae = {name: stored[name] for name in names}
    baseline = min((name for name in names if name != candidate), key=lambda name: mae[name])
    return {"gain": mae[baseline] - mae[candidate], "candidate": mae[candidate],
            "baseline": baseline, "baseline_mae": mae[baseline], "n_rows": len(rows)}


def main():
    setup_style()
    audit = AppendAudit(3, "Figure 3 extension: confirmation pruning and grouped quantization")
    data = load_v49(audit)
    v46 = audit.read("results/v46-p1-newsource/compare.json")
    v41 = audit.read("results/v41-distill-newpool/summary.json")
    confirmation = load_confirmation(audit)
    grouped_quant = load_grouped_quantization(audit)
    multistudent = load_multistudent_distillation(audit)
    cells = {p: [[None] * 4 for _ in SOURCES] for p in ("A", "B")}
    for protocol in ("A", "B"):
        for i, (size, step) in enumerate(SOURCES):
            if step == 96000:
                if protocol == "B":
                    continue
                rows = regime_rows(v46["pruning"], v46["quantization"])
                summaries = [None] * 4
            else:
                payload = data.get((size, step), {}).get("protocols", {}).get(protocol)
                if payload is None:
                    continue
                rows = regime_rows(payload["rows_prune"], payload["rows_quant"])
                s = payload["summary"]
                summaries = [s["prune"]["interp_0.9-0.6"], s["prune"]["extrap_0.55"],
                             s["quant_ge4"], s["quant"]["int3"]]
            for j, rr in enumerate(rows):
                candidate = "power" if j < 2 else ("full_N0_L0_D0" if step == 96000 else "full")
                cell = score_cell(rr, candidate, BASELINES[0 if j < 2 else 1], summaries[j])
                cells[protocol][i][j] = cell
                if cell:
                    audit.rule(f"Protocol {protocol}, {size}@{step}, {REGIMES[j].replace(chr(10), ' ')}: "
                               f"{cell['n_rows']} rows, equal weight across math/code/QA; candidate={candidate}, "
                               f"MAE={cell['candidate']:.9f}; baseline={cell['baseline']}, "
                               f"MAE={cell['baseline_mae']:.9f}; gain={cell['gain']:+.9f}.")
    # A fifth regime column preserves the distinct confirmation density ladder.
    for rr in cells["A"]:
        rr.insert(1, None)
    for label, payload in confirmation.items():
        cell = score_capability_mae(payload["mae"], "power", BASELINES[0])
        cells["A"].append([None, cell, None, None, None])
        audit.rule(f"Protocol A confirmation, {label}, prune interp d=.85/.675/.575: "
                   f"power capability-averaged MAE={cell['candidate']:.9f}; strongest simple={cell['baseline']}, "
                   f"MAE={cell['baseline_mae']:.9f}; gain={cell['gain']:+.9f}. "
                   "Other pruning regimes and both quantization columns are not evaluated.")
    audit.rule("Read v49 summary.prune.interp_0.9-0.6, summary.prune.extrap_0.55, "
               "summary.quant_ge4 and summary.quant.int3; verify each against its exact underlying rows. "
               "Strongest source-free baseline minimizes aggregate MAE once per source/protocol/regime, "
               "not separately per capability or observation. Pruning candidates: strength_only, "
               "median_curve, zero; quantization: per_bit_mean, per_bit_median, zero where present.")
    audit.rule("For v46 1B@96k, compute MAE from pruning/quantization rows; its actual interpolation "
               "coverage is only d=.65 and its ≥4-bit coverage is only int4. Place these in protocol A "
               "as requested, labelled with an asterisk. The historical v46 freeze is preserved, not refit as v49.")
    audit.omit("1B@96k protocol B was not evaluated: four cells are hatched grey. "
               "The v46 A cells do not contain the full v49 density/bit ladders; unmeasured strengths "
               "are not filled in. Protocol summaries aggregate three capabilities as specified for this matrix.")
    audit.rule("Separate distillation panel: v41.by_cap[cap].mae.constant minus mae.E; "
               "U225 unseen pool, ordered math/code/QA. This candidate is E-only, with no post-test "
               "switch to constant in math/code. No confidence intervals are available.")
    audit.omit("No intervals are stored for these transfer comparisons; none are invented. "
               "Confirmation states appear only in protocol A; no protocol-B confirmation evaluation is implied.")
    gains = [c["gain"] for matrix in cells.values() for rr in matrix for c in rr if c]
    gains.extend(c["gain"] for c in grouped_quant.values())
    limit = max(abs(v) for v in gains)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = plt.get_cmap("RdBu")
    fig = plt.figure(figsize=(10.4, 9.0))
    grid = fig.add_gridspec(3, 1, left=.12, right=.985, bottom=.12, top=.885,
                          height_ratios=[4.8, 1.6, 1.2], hspace=.65)
    matrices = grid[0].subgridspec(1, 2, width_ratios=[5.6, 4], wspace=.12)
    axes = []
    for col, protocol in enumerate(("A", "B")):
        ax = fig.add_subplot(matrices[0, col])
        axes.append(ax)
        regimes = (REGIMES[0], CONFIRMATION_REGIME, *REGIMES[1:]) if protocol == "A" else REGIMES
        ax.set_xlim(-.5, len(regimes) - .5)
        ax.set_ylim(7.5, -.5)
        ax.set_title(f"Protocol {protocol}", fontsize=11, pad=9)
        for i, rr in enumerate(cells[protocol]):
            for j, cell in enumerate(rr):
                if cell is None:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#e1e1e1",
                                           edgecolor="#bbbbbb", hatch="///", lw=.6))
                    ax.text(j, i, "not\nevaluated", ha="center", va="center", fontsize=8,
                            bbox={"facecolor": "#e1e1e1", "edgecolor": "none", "pad": 1})
                    continue
                face = cmap(norm(cell["gain"]))
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=face, edgecolor="white", lw=1))
                text_color = "white" if abs(cell["gain"]) > .63 * limit else "#171717"
                ax.text(j, i - .23, f"{cell['gain']:+.3f}", ha="center", va="center", color=text_color, fontsize=10)
                ax.text(j, i + .03, f"MAE {cell['candidate']:.3f}", ha="center", va="center", color=text_color, fontsize=8)
                ax.text(j, i + .28, SHORT[cell["baseline"]], ha="center", va="center", color=text_color, fontsize=8)
        labels = [f"{'1B' if s == '1b' else '6.9B'}@{t // 1000}k" + ("*" if t == 96000 else "") for s, t in SOURCES]
        if protocol == "A":
            labels += list(confirmation)
            ax.axhline(4.5, color="#444444", lw=1.2)
        ax.set_yticks(range(len(labels)), labels if col == 0 else [])
        ax.set_xticks(range(len(regimes)), regimes)
        for tick, label in zip(ax.get_xticklabels(), regimes):
            tick.set_color(COLORS["pruning" if label.startswith("Prune") else "quantization"])
        ax.tick_params(length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
    color_ax = fig.add_axes([.31, .955, .5, .017])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8, length=2, pad=1)
    fig.text(.56, .982, "MAE(simple) − MAE(candidate): positive = candidate better", ha="center", fontsize=9)
    group_ax = fig.add_subplot(grid[1])
    group_ax.set_title("Grouped quantization · low_order_2d (v55)", fontsize=10, loc="left", pad=20)
    group_ax.set_xlim(-.5, 3.5)
    group_ax.set_ylim(2.5, -.5)
    group_ax.set_yticks(range(3), GROUP_TESTS)
    group_ax.set_xticks(range(4), ["Gain", "Candidate MAE", "Best simple (MAE)", "Source states"])
    group_ax.xaxis.tick_top()
    group_ax.tick_params(length=0, pad=4)
    for i, test in enumerate(GROUP_TESTS):
        cell = grouped_quant[test]
        group_ax.axhspan(i - .5, i + .5, color="#f5f5f5" if i % 2 == 0 else "white", zorder=-1)
        group_ax.add_patch(Rectangle((-.5, i - .5), 1, 1, facecolor=cmap(norm(cell["gain"])),
                                    edgecolor="white"))
        labels = [f"{cell['gain']:+.3f}", f"{cell['candidate']:.3f}",
                  f"{cell['baseline']} ({cell['baseline_mae']:.3f})",
                  "held-out 1B@96k" if test == "joint_test" else "seen development states"]
        for j, label in enumerate(labels):
            group_ax.text(j, i, label, ha="center", va="center", fontsize=9)
    for spine in group_ax.spines.values():
        spine.set_visible(False)
    distill = fig.add_subplot(grid[2])
    distill.axvline(0, color="#444444", lw=.8)
    for i, cap in enumerate(CAPS):
        mae = v41["by_cap"][cap]["mae"]
        gain = mae["constant"] - mae["E"]
        distill.plot(gain, i, marker="o", mfc="white", mec=COLORS["E"], ms=6, linestyle="none")
        distill.annotate(f"{gain:+.3f}  (E MAE {mae['E']:.3f}; constant {mae['constant']:.3f})",
                        (gain, i), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
        audit.rule(f"U225 {cap}: E MAE={mae['E']:.9f}, constant MAE={mae['constant']:.9f}, gain={gain:+.9f}.")
    distill.text(-.2, -.8, "1B, pool 225 · E-only vs constant (6 runs)", fontsize=8, va="center")
    ticks = list(range(3))
    for group_index, (label, cap_metrics) in enumerate(multistudent.items(), start=1):
        offset = 4 * group_index
        distill.text(-.2, offset - .8, f"{label} · joint+src (dev-selected, R) vs frozen constant",
                     fontsize=8, va="center")
        for i, cap in enumerate(CAPS):
            cell = cap_metrics[cap]
            y = offset + i
            ticks.append(y)
            distill.plot(cell["gain"], y, marker="o", mfc="white", mec=COLORS["E"], ms=6, linestyle="none")
            distill.annotate(f"{cell['gain']:+.3f}  (joint+src MAE {cell['candidate']:.3f}; constant {cell['baseline_mae']:.3f}; zero {cell['zero_mae']:.3f})",
                            (cell["gain"], y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
    distill.set_ylim(14.6, -1.3)
    distill.set_yticks(ticks, [CAP_LABEL[c] for _ in range(1 + len(multistudent)) for c in CAPS])
    distill.set_xlim(-.2, 1.35)
    distill.set_title("Distillation · unseen pools (* = held-out student; R = selection rule stated after tests)",
                      fontsize=10, loc="left")
    distill.set_xlabel("MAE(baseline) − MAE(candidate), nats/token; positive = candidate better")
    distill.grid(axis="x", color="#eeeeee", lw=.5)
    fig.text(.5, .018, "*1B@96k: only d=.65/.55 and int4/int3; original v46 freeze. Cell lines: gain / candidate MAE / selected simple baseline.\n"
             "Pythia cells average math, code and QA. All values are nats per native token; the distillation panel is separate.",
             ha="center", fontsize=8, linespacing=1.5)
    # Extend only the distillation panel downward; preserve the other panels in physical coordinates.
    old_height, extra_height = fig.get_figheight(), 2.4
    positions = {ax: ax.get_position().frozen() for ax in fig.axes}
    new_height = old_height + extra_height
    fig.set_size_inches(fig.get_figwidth(), new_height)
    for ax, pos in positions.items():
        bottom = pos.y0 * old_height + (0 if ax is distill else extra_height)
        height = pos.height * old_height + (extra_height if ax is distill else 0)
        ax.set_position([pos.x0, bottom / new_height, pos.width, height / new_height])
    for label in fig.texts:
        x, y = label.get_position()
        label.set_position((x, (y * old_height + (extra_height if y > .5 else 0)) / new_height))
    audit.save(fig, "transfer_limits")
    audit.finish()


if __name__ == "__main__":
    main()
