#!/usr/bin/env python3
"""A9: fixed-plan, CPU-only pruning measurement-efficiency study.

python -B analysis/a9_measurement_efficiency.py plan
python -B analysis/a9_measurement_efficiency.py run

The plan is exclusive and immutable. No model, training, or GPU imports. Fitting
requires the already-written plan, unchanged measurement inputs, and exact
reconstruction of its label-independent subsets. All artifacts stay in A9.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/a9-matplotlib")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/a9-measurement-efficiency"
CAPS = ("math", "code", "qa")
METHODS = ("power", "A2", "median_curve")
AXES = ("sources", "densities")
FRACTIONS = (.25, .5, 1.)
DENSITIES = (.6, .7, .8, .9)
SIZES = ("160m", "410m", "1.4b")
STEPS = (16000, 64000, 143000)
SEEDS = tuple(91000 + i for i in range(20))
LEVELS = (.15, .25, .35)
GAMMAS = np.arange(10, 121, dtype=float) / 20  # V53: 0.5,...,6 by 0.05
PRIMARY_SPLITS = ("seen_state_unseen_density", "new_state")
SPLITS = (*PRIMARY_SPLITS, "duplicate_weight_state_audit")
REGIONS = ("in_range", "boundary", "all")
PINNED = ("pythia-1.4b@step64000", "pythia-160m@step143000")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value, exclusive=False):
    with path.open("x" if exclusive else "w") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def install_io_guard():
    """Prevent this analysis from writing paper artifacts or starting model jobs."""
    def permitted(path):
        resolved = Path(os.fsdecode(path)).resolve()
        return resolved.is_relative_to(OUT) or resolved.is_relative_to(Path("/tmp"))

    def audit(event, args):
        if event in ("subprocess.Popen", "os.system", "socket.connect", "socket.bind"):
            raise PermissionError(f"A9 CPU analysis refuses {event}")
        if event == "open":
            name, mode, flags = args
            if not isinstance(name, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                require(permitted(name), f"A9 write outside results directory: {name}")
        if event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename"):
            for name in args[:2] if event == "os.rename" else args[:1]:
                require(permitted(name), f"A9 mutation outside results directory: {name}")
    sys.addaudithook(audit)


def state(size, step):
    return f"pythia-{size}@step{step}"


def load_data():
    """Closed source allowlists; no scanning for additional measured points."""
    hashes, mirror_audit = {}, []

    def read(relative, mirror=True):
        path = ROOT / relative
        raw = path.read_bytes()
        hashes[relative] = sha(raw)
        data = json.loads(raw)
        if mirror and relative.startswith("results/"):
            mp = ROOT / "paper/data_mirror" / relative.removeprefix("results/").replace("@", "--")
            if mp.is_file():
                other = mp.read_bytes()
                hashes[str(mp.relative_to(ROOT))] = sha(other)
                # The publication mirror sanitizes this runtime-only cache path.
                left = canonical(data).replace(os.path.expanduser('~/.cache/huggingface/hub'), '$HOME/.cache/huggingface/hub')
                right = canonical(json.loads(other))
                equal = left == right
                mirror_audit.append(dict(path=relative, mirror=str(mp.relative_to(ROOT)), semantic_match=equal,
                                         normalization="HF cache path only: the user home directory -> $HOME"))
                require(equal, f"Mirror mismatch: {relative}")
        return data

    a1 = read("results/a1-development-table/summary.json")
    a1_csv = ROOT / "results/a1-development-table/development_table.csv"
    hashes[str(a1_csv.relative_to(ROOT))] = sha(a1_csv.read_bytes())
    require("density" not in a1_csv.read_text().splitlines()[0], "Revisit A1 provenance correction")
    read("paper/paper/tables/groups.json", mirror=False)
    v36 = read("results/v36-pythia-controlled/summary.json")
    v53 = read("results/v53-prune-dev/register.json")
    architectures = v53["architectures"]

    def n0(size):
        a = architectures[size]
        h, m = a["hidden_size"], a["intermediate_size"]
        return a["num_hidden_layers"] * (4*h*h + 2*h*m)

    def row(size, step, d, cap, y, dense, path, pointer, split=None):
        tag = state(size, step)
        return dict(id=f"{tag}|{d:g}|{cap}", state=tag, size=size, step=step,
                    d=d, cap=cap, y=float(y), L0=float(dense[cap]), N0=n0(size),
                    D0=step*2097152, phi_raw=[float(np.log(n0(size))), float(dense[cap]),
                                             float(np.log(step*2097152))],
                    provenance=f"{path}#{pointer}", split=split,
                    region="in_range" if .6 <= d <= .9 else "boundary")

    dev, tables = [], {}
    for size, step in itertools.product(SIZES, STEPS):
        path = f"results/v6-capability-geometry/pythia-{size}--step{step}/prune_losses.json"
        table = read(path)
        tables[size, step] = table
        require(n0(size) == v36["inputs"]["architectures"][size]["N0"], "N0 convention drift")
        for d, cap in itertools.product(DENSITIES, CAPS):
            dev.append(row(size, step, d, cap, table[str(d)][cap]-table["1.0"][cap],
                           table["1.0"], path, f"/{d}/{cap} minus /1.0/{cap}"))

    tests = []
    path = "results/v42-prune-sameinput/summary.json"
    v42 = read(path)
    for i, r in enumerate(v42["per_point"]):
        size, step = r["source"].split("@")
        step = int(step)
        if (size, step) not in tables:
            tables[size, step] = read(f"results/v6-capability-geometry/pythia-{size}--step{step}/prune_losses.json")
        split = "new_state" if r["new_source"] else "seen_state_unseen_density"
        tests.append(row(size, step, r["d"], r["cap"], r["obs"], tables[size, step]["1.0"],
                         path, f"/per_point/{i}/obs", split))

    read("results/v38-prospective/compare.json")  # published aggregate; labels in raw measurements
    for size in ("160m", "1.4b"):
        path = f"results/v6-capability-geometry/pythia-{size}--step96000/prune_losses.json"
        table = read(path)
        for d, cap in itertools.product(DENSITIES, CAPS):
            tests.append(row(size, 96000, d, cap, table[str(d)][cap]-table["1.0"][cap],
                             table["1.0"], path, f"/{d}/{cap} minus /1.0/{cap}", "new_state"))
    path = "results/v46-p1-newsource/compare.json"
    v46 = read(path)
    for i, r in enumerate(v46["pruning"]):
        tests.append(row("1b", 96000, r["d"], r["cap"], r["actual"], v46["dense_frozen"],
                         path, f"/pruning/{i}/actual", "new_state"))
    for size, step in itertools.product(("1b", "6.9b"), (32000, 112000)):
        path = f"results/v49-p1v2/compare_{state(size, step)}.json"
        v49 = read(path)
        for i, r in enumerate(v49["protocols"]["A"]["rows_prune"]):
            tests.append(row(size, step, r["d"], r["cap"], r["actual"], v49["dense_frozen"],
                             path, f"/protocols/A/rows_prune/{i}/actual", "new_state"))
    for size, step in (("410m", 48000), ("1.4b", 112000), ("6.9b", 80000)):
        path = f"results/v53-prune-dev/compare_{state(size, step)}.json"
        compare = read(path)
        frozen = read(f"results/v53-prune-dev/predictions_{state(size, step)}.json")
        for d, cap in itertools.product(compare["densities"], CAPS):
            tests.append(row(size, step, d, cap, compare["observed_delta_loss"][cap][str(d)],
                             frozen["target"]["L0"], path, f"/observed_delta_loss/{cap}/{d}", "new_state"))
    path = "results/v72-prune-repeat/compare.json"
    v72 = read(path)
    frozen72 = read("results/v72-prune-repeat/freeze.json")
    for i, r in enumerate(v72["rows"]):
        target = next(t for t in frozen72["targets"] if t["tag"] == r["source"])
        tests.append(row(target["size"], target["step"], r["density"], r["capability"],
                         r["observed_delta_loss"], target["L0"], path, f"/rows/{i}/observed_delta_loss",
                         "duplicate_weight_state_audit"))
    require(len(dev) == 108, "Expected 36 pruning measurements per capability")
    require(len({r["id"] for r in dev}) == len(dev), "Duplicate development row")
    require(len({r["id"] for r in tests}) == len(tests), "Duplicate test row")
    require(not ({r["id"] for r in dev} & {r["id"] for r in tests}), "Train/test overlap")
    return dev, tests, dict(input_sha256=hashes, mirror_audit=mirror_audit,
                            a1_summary_keys=list(a1))


def subsets(dev):
    """Nested, label-independent subsets; keep seen-test sources actually seen."""
    states = sorted({r["state"] for r in dev})
    out = []
    for axis, seed in itertools.product(AXES, SEEDS):
        rng = np.random.default_rng(seed + (100000 if axis == "densities" else 0))
        order = list(PINNED) + list(rng.permutation([s for s in states if s not in PINNED]))
        density_order = {s: rng.permutation(DENSITIES).tolist() for s in states}
        for fraction in FRACTIONS:
            ns = int(np.floor(9*fraction + .5)) if axis == "sources" else 9
            nd = int(4*fraction) if axis == "densities" else 4
            selected = [r for r in dev if r["state"] in order[:ns] and
                        (axis == "sources" or r["d"] in density_order[r["state"]][:nd])]
            ids = sorted(r["id"] for r in selected)
            require(all(s in {r["state"] for r in selected} for s in PINNED), "Seen-test source lost")
            out.append(dict(id=f"{axis}:{fraction:g}:{seed}", axis=axis, fraction=fraction,
                            seed=seed, n_sources=ns, n_densities_per_source=nd,
                            n_measurements_per_capability=len(ids)//3, n_scalar_labels=len(ids),
                            realized_fraction=len(ids)/108, selected_ids=ids,
                            subset_sha256=sha(canonical(ids).encode())))
    return out


def make_plan():
    dev, test, provenance = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = dict(schema_version=1, study="A9", created_utc=datetime.now(timezone.utc).isoformat(),
                status="fixed before any A9 fit; retrospective reuse of published confirmation outcomes",
                source_correction="A1/C73 is Gemma distillation, not Pythia pruning. Use the explicit nine-state "
                "V36/V40/V92 pruning panel, cross-checked with data_mirror. The pruning A2 is V42/V53, not A2/C74 distillation.",
                development=dict(states=sorted({r["state"] for r in dev}), densities=list(DENSITIES),
                                 n_per_capability=36, n_scalar_labels=108),
                design=dict(fractions=list(FRACTIONS), R=20, seeds=list(SEEDS), axes=list(AXES),
                            source_rounding="nearest integer, halves up: 2,5,9 of 9 states (8,20,36 measurements/capability)",
                            source_sampling="retain the two seen-test source states; randomly permute the other seven; nested budgets",
                            density_sampling="independently permute four densities within each of nine sources; nested 1,2,4 points/source",
                            replicates="subset variability, not independent experimental repetitions; full-budget subsets repeat",
                            measurement_unit="one (source,density,capability) response; per-capability x axis equals number of source-density runs; all-capability scalar labels = 3*x",
                            dense_cost="dense anchors and metadata supplied equally; not charged as pruning measurements; count observed training-source anchors separately"),
                predictors=dict(power="five nominal parameters: four source amplitude coefficients plus gamma",
                                A2="four coefficients per retained density, linear adjacent interpolation / nearest-pair boundary extension",
                                median_curve="one median response per retained density, identical interpolation/extension; ignores equally available source inputs",
                                features=["1", "z(log N0)", "z(L0c)", "z(log D0)"],
                                standardization="population mean/std of selected training rows pooled across capabilities, as V53; no held-out inputs",
                                tuning="zero validation/hyperparameter searches for every predictor; unregularized least squares for source coefficients; no ridge rescue",
                                gamma_grid=GAMMAS.tolist(), gamma_selection="minimum training SSE; increasing-grid tie break; structural coefficient fit, not held-out tuning",
                                rank_rule="relative SVD tolerance 1e-10; full coefficient rank required; power also requires local five-column Jacobian rank 5; A2 requires rank 4 at every observed anchor; interpolation needs >=2 anchors",
                                parameter_note="four-density A2 has 16 parameters here; paper's 20 refers to V53's larger five-anchor development set"),
                scoring=dict(splits=list(SPLITS), regions=list(REGIONS), primary_region="in_range",
                             in_range="0.6 <= density <= 0.9, the original nine-state development hull; other held-out points retained as boundary results",
                             statistic="unweighted mean absolute error across identical fixed cells per capability, split and region",
                             missing="keep each failed replicate with reason and null MAE; report conditional median/IQR with explicit fitted/20 counts",
                             threshold_levels=list(LEVELS), threshold_rule="smallest observed budget with all 20 fits available and median MAE <= level; no interpolation across budgets, no monotonic smoothing; null means not reached on tested grid",
                             comparisons="paired errors only on common fitted replicates; primary dominance claims require 20/20 common fits",
                             duplicate_weights="V72's two stage labels are the same learned 2.8B state (C52). Score all 18 labelled cells separately as duplicate_weight_state_audit, never pool with primary new-state cells; D0 labels are not verified training histories.",
                             provenance="V38/V42/V46/V49 from pred_full groups sidecar; V53 and V72 from Section 6; V49 protocol A labels only, not duplicate protocol B; V36 LOSO is development, not confirmation"),
                provenance=provenance, subsets=subsets(dev),
                fixed_test_cells=[{k: r[k] for k in ("id", "state", "d", "cap", "split", "region", "provenance")} for r in test],
                data_sha256=sha(canonical(dict(development=dev, heldout=test)).encode()))
    write_json(OUT / "plan.json", plan, exclusive=True)
    print(f"Wrote immutable plan: {OUT / 'plan.json'} ({len(plan['subsets'])} subsets)")


def rank_record(matrix):
    values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = float(values[0]*1e-10) if len(values) else 0.
    rank = int(np.sum(values > tolerance))
    return dict(rank=rank, n_rows=int(matrix.shape[0]), n_columns=int(matrix.shape[1]),
                singular_values=values.tolist(), absolute_tolerance=tolerance,
                condition_number=float(values[0]/values[-1])
                if rank == matrix.shape[1] and len(values) else None)


def standardizer(rows):
    raw = np.array([r["phi_raw"] for r in rows])
    center, scale = raw.mean(axis=0), raw.std(axis=0)
    return dict(center=center.tolist(), scale=np.where(scale > 0, scale, 1.).tolist())


def design(rows, stats):
    raw = np.array([r["phi_raw"] for r in rows])
    return np.column_stack((np.ones(len(rows)), (raw-stats["center"])/stats["scale"]))


def fit_predictor(rows, method, stats):
    """No labels outside the selected development rows enter this function."""
    z = design(rows, stats)
    y = np.array([r["y"] for r in rows])
    ds = np.array([r["d"] for r in rows])
    anchors = sorted(set(ds))
    n_parameters = 5 if method == "power" else len(anchors)*(4 if method == "A2" else 1)
    record = dict(method=method, status="failed", reason=None, n_observations=len(rows),
                  n_parameters=n_parameters, n_source_coefficients=4 if method == "power" else
                  4*len(anchors) if method == "A2" else 0,
                  source_design=rank_record(z), observed_densities=anchors,
                  observations_by_density={str(d): int(np.sum(ds == d)) for d in anchors})
    if method == "power":
        sh = (1-ds)/.3
        record["design_matrix"] = rank_record(z*sh[:, None])
        if record["design_matrix"]["rank"] < 4:
            record["reason"] = "source amplitude design rank < 4; gamma cannot repair source identifiability"
            record["jacobian_rank_upper_bound"] = record["design_matrix"]["rank"]+1
            return record
        profile = []
        for gamma in GAMMAS:
            x = z*(sh**gamma)[:, None]
            beta = np.linalg.lstsq(x, y, rcond=1e-10)[0]
            sse = float(np.sum((x@beta-y)**2))
            profile.append(dict(gamma=float(gamma), sse=sse))
        best = min(profile, key=lambda p: p["sse"])
        gamma = best["gamma"]
        x = z*(sh**gamma)[:, None]
        beta = np.linalg.lstsq(x, y, rcond=1e-10)[0]
        jacobian = np.column_stack((x, (x@beta)*np.log(sh)))
        record.update(gamma=gamma, beta=beta.tolist(), profile=profile, sse=best["sse"],
                      gamma_boundary=gamma in (GAMMAS[0], GAMMAS[-1]),
                      design_matrix=rank_record(x), local_jacobian=rank_record(jacobian))
        if record["local_jacobian"]["rank"] < 5:
            record["reason"] = "local power Jacobian rank < 5; gamma not identified"
            return record
    else:
        indicator = np.column_stack([ds == d for d in anchors]).astype(float)
        x = np.column_stack([z*(ds == d)[:, None] for d in anchors]) if method == "A2" else indicator
        record["design_matrix"] = rank_record(x)
        if method == "A2":
            record["anchor_designs"] = {str(d): rank_record(z[ds == d]) for d in anchors}
        if len(anchors) < 2:
            record["reason"] = "fewer than two observed anchors; interpolation/extension undefined"
            return record
        if record["design_matrix"]["rank"] < n_parameters:
            record["reason"] = "per-density design rank < parameter count; no pseudoinverse or ridge rescue"
            return record
        record["anchors"] = {str(d): np.linalg.lstsq(z[ds == d], y[ds == d], rcond=1e-10)[0].tolist()
                             if method == "A2" else float(np.median(y[ds == d])) for d in anchors}
        fitted = x@np.concatenate(list(record["anchors"].values())) if method == "A2" else indicator@np.array(list(record["anchors"].values()))
        record["sse"] = float(np.sum((fitted-y)**2))
    record["status"] = "fitted"
    return record


def interpolate(ds, values, density):
    i = int(np.clip(np.searchsorted(ds, density, side="right")-1, 0, len(ds)-2))
    lo, hi = ds[i:i+2]
    return float(values[i]+(values[i+1]-values[i])*(density-lo)/(hi-lo))


def predict(fit, rows, stats):
    require(fit["status"] == "fitted", "Cannot predict an unidentified fit")
    z = design(rows, stats)
    ds = np.array([r["d"] for r in rows])
    if fit["method"] == "power":
        return z@np.array(fit["beta"])*((1-ds)/.3)**fit["gamma"]
    anchors = sorted(float(d) for d in fit["anchors"])
    values = np.array([fit["anchors"][str(d)] for d in anchors])
    return np.array([interpolate(anchors, values@x if fit["method"] == "A2" else values, d)
                     for x, d in zip(z, ds)])


def quantiles(values):
    if not values:
        return dict(median=None, q25=None, q75=None)
    lo, med, hi = np.quantile(values, [.25, .5, .75], method="linear")
    return dict(median=float(med), q25=float(lo), q75=float(hi))


def aggregate(records, test):
    curves, comparisons, thresholds = [], [], []
    for axis, split, region, cap, method in itertools.product(AXES, SPLITS, REGIONS, CAPS, METHODS):
        key = f"{split}/{region}"
        cells = [r for r in test if r["split"] == split and r["cap"] == cap and
                 (region == "all" or r["region"] == region)]
        if not cells:
            continue
        group = []
        for fraction in FRACTIONS:
            rr = [r for r in records if (r["axis"], r["fraction"], r["cap"], r["method"]) ==
                  (axis, fraction, cap, method)]
            values = [r["scores"][key]["mae"] for r in rr if r["fit"]["status"] == "fitted"]
            point = dict(axis=axis, split=split, region=region, cap=cap, method=method, fraction=fraction,
                         n_measurements_per_capability=rr[0]["n_measurements_per_capability"],
                         n_scalar_labels=rr[0]["n_scalar_labels"], n_test_cells=len(cells),
                         n_fitted=len(values), n_failed=20-len(values), R=20,
                         fitted_seeds=[r["seed"] for r in rr if r["fit"]["status"] == "fitted"],
                         failed_seeds=[r["seed"] for r in rr if r["fit"]["status"] != "fitted"],
                         threshold_success_count={str(level): sum(v <= level for v in values) for level in LEVELS},
                         **quantiles(values))
            curves.append(point)
            group.append(point)
        for level in LEVELS:
            eligible = [p for p in group if p["n_fitted"] == 20 and p["median"] <= level]
            by_rep = []
            for seed in SEEDS:
                hit = [r["n_measurements_per_capability"] for r in records
                       if (r["axis"], r["cap"], r["method"], r["seed"]) == (axis, cap, method, seed)
                       and r["fit"]["status"] == "fitted" and r["scores"][key]["mae"] <= level]
                by_rep.append(dict(seed=seed, minimum_measured_budget=min(hit) if hit else None))
            minimum = min((p["n_measurements_per_capability"] for p in eligible), default=None)
            thresholds.append(dict(axis=axis, split=split, region=region, cap=cap, method=method, level=level,
                                   minimum_measurements=minimum, minimum_scalar_labels=3*minimum if minimum else None,
                                   status="reached" if minimum else "not reached on tested grid",
                                   per_replicate_first_crossing=by_rep))
    for axis, split, region, cap, fraction, (a, b) in itertools.product(
            AXES, SPLITS, REGIONS, CAPS, FRACTIONS, itertools.combinations(METHODS, 2)):
        key = f"{split}/{region}"
        left = {r["seed"]: r for r in records if (r["axis"], r["cap"], r["fraction"], r["method"]) == (axis, cap, fraction, a)}
        right = {r["seed"]: r for r in records if (r["axis"], r["cap"], r["fraction"], r["method"]) == (axis, cap, fraction, b)}
        common = [seed for seed in SEEDS if left[seed]["fit"]["status"] == right[seed]["fit"]["status"] == "fitted"
                  and key in left[seed]["scores"]]
        differences = [right[s]["scores"][key]["mae"]-left[s]["scores"][key]["mae"] for s in common]
        comparisons.append(dict(axis=axis, split=split, region=region, cap=cap, fraction=fraction,
                                a=a, b=b, difference="MAE(b)-MAE(a); positive favors a", n_common=len(common),
                                seeds=common, values=differences, **quantiles(differences)))
    return curves, thresholds, comparisons


def conclusions(summary):
    primary = lambda r: r["split"] in PRIMARY_SPLITS and r["region"] == "in_range"
    table = {(r["axis"], r["split"], r["cap"], r["fraction"], r["method"]): r
             for r in summary["curves"] if primary(r)}
    wins, median_wins, eligible = [], [], 0
    for axis, split, cap, fraction in itertools.product(AXES, PRIMARY_SPLITS, CAPS, FRACTIONS):
        points = {m: table[axis, split, cap, fraction, m] for m in METHODS}
        if all(p["n_fitted"] == 20 for p in points.values()):
            eligible += 1
            p, a, m = [points[k]["median"] for k in METHODS]
            label = dict(axis=axis, split=split, cap=cap,
                         measurements=points["power"]["n_measurements_per_capability"],
                         power=p, A2=a, median_curve=m)
            if p < a and p < m:
                wins.append(label)
            if m <= p and m <= a:
                median_wins.append(label)
    targets = {(r["axis"], r["split"], r["cap"], r["level"], r["method"]): r["minimum_measurements"]
               for r in summary["thresholds"] if primary(r)}
    savings, exclusive = [], []
    for axis, split, cap, level, competitor in itertools.product(AXES, PRIMARY_SPLITS, CAPS, LEVELS, ("A2", "median_curve")):
        power = targets[axis, split, cap, level, "power"]
        other = targets[axis, split, cap, level, competitor]
        if power is not None and (other is None or power < other):
            result = dict(axis=axis, split=split, cap=cap, level=level, competitor=competitor,
                          power_measurements=power, competitor_measurements=other)
            (exclusive if other is None else savings).append(result)
    return dict(n_complete_primary_comparisons=eligible, power_lowest= wins,
                median_lowest=median_wins, measured_threshold_savings=savings,
                only_power_reaches_threshold=exclusive)


def validate(summary):
    """Executable checks of identifiability, scoring, accounting and fixed cells."""
    records, plan = summary["records"], summary["plan"]
    require(len(records) == 120*3*3, "Missing fit attempts")
    for subset in plan["subsets"]:
        rr = [r for r in records if r["id"] == subset["id"]]
        for cap in CAPS:
            cr = [r for r in rr if r["cap"] == cap]
            require(len({r["training_label_sha256"] for r in cr}) == 1, "Unequal labels")
            require(len({canonical(r["standardizer"]) for r in cr}) == 1, "Unequal source information")
    fixed = {}
    for r in records:
        for key, score in r["scores"].items():
            pair = r["cap"], key
            fixed.setdefault(pair, score["fixed_cell_sha256"])
            require(fixed[pair] == score["fixed_cell_sha256"], "Heldout cells differ across fits")
            split, region = key.split("/")
            rows = [p for p in r["predictions"] if p["split"] == split and (region == "all" or p["region"] == region)]
            if r["fit"]["status"] == "fitted":
                require(len(rows) == score["n_test"], "Incomplete score")
                require(np.isclose(np.mean([p["absolute_error"] for p in rows]), score["mae"], atol=1e-12), "MAE mismatch")
            else:
                require(score["mae"] is None and not rows, "Failed fit has scores")
    for cap, method in itertools.product(CAPS, METHODS):
        full = [r for r in records if (r["fraction"], r["cap"], r["method"]) == (1., cap, method)]
        require(len({canonical(r["predictions"]) for r in full}) == 1, "Full-budget predictions differ")
    for axis, seed in itertools.product(AXES, SEEDS):
        subs = sorted([s for s in plan["subsets"] if (s["axis"], s["seed"]) == (axis, seed)], key=lambda s:s["fraction"])
        require(set(subs[0]["selected_ids"]) < set(subs[1]["selected_ids"]) < set(subs[2]["selected_ids"]), "Subsets not nested")
        for s in subs:
            rr = [r for r in summary["development_rows"] if r["id"] in set(s["selected_ids"])]
            counts = Counter(r["state"] for r in rr)
            require(len(set(counts.values())) == 1, "Unequal per-state sampling")
            require((len(counts) == 9 if axis == "densities" else set(counts.values()) == {12}), "Mixed reduction axes")
    # Synthetic known laws test recovery and out-of-grid prediction, not fitted
    # training residuals alone. There is no optimizer or hyperparameter tuning.
    synthetic = []
    rng = np.random.default_rng(314159)
    beta = np.array([.8, .1, -.2, .15])
    for i, phi in enumerate(rng.normal(size=(10, 3))):
        for d in DENSITIES:
            synthetic.append(dict(phi_raw=phi.tolist(), d=d, y=float(np.r_[1., phi]@beta*((1-d)/.3)**2)))
    stats = dict(center=[0., 0., 0.], scale=[1., 1., 1.])
    power = fit_predictor(synthetic, "power", stats)
    queries = [dict(phi_raw=[.2, -.1, .3], d=.65)]
    expected = np.r_[1., queries[0]["phi_raw"]]@beta*((1-.65)/.3)**2
    require(power["status"] == "fitted" and abs(predict(power, queries, stats)[0]-expected) < 1e-10, "Power synthetic recovery failed")
    one_d = [r for r in synthetic if r["d"] == .6]
    require(fit_predictor(one_d, "power", stats)["status"] == "failed", "One-density exponent incorrectly identified")
    require(fit_predictor(synthetic[:8], "A2", stats)["status"] == "failed", "Rank-deficient A2 incorrectly fitted")
    linear = [dict(r, y=float(np.r_[1., r["phi_raw"]]@beta*(1-r["d"]))) for r in synthetic]
    a2 = fit_predictor(linear, "A2", stats)
    expected = np.r_[1., queries[0]["phi_raw"]]@beta*(1-.65)
    require(abs(predict(a2, queries, stats)[0]-expected) < 1e-10, "A2 interpolation failed")
    require(abs(interpolate([.6,.8], [2.,1.], .5)-2.5) < 1e-12, "Boundary extension failed")
    med = fit_predictor(linear, "median_curve", stats)
    require(all(med["anchors"][str(d)] == float(np.median([r["y"] for r in linear if r["d"] == d])) for d in DENSITIES), "Median anchors failed")
    # Independent historical implementation of the same full-budget OLS A2.
    historical = json.loads((ROOT/"results/v42-prune-sameinput/summary.json").read_text())["per_point"]
    discrepancies = []
    for cap in CAPS:
        r = next(r for r in records if (r["axis"],r["fraction"],r["cap"],r["method"],r["seed"]) == ("sources",1.,cap,"A2",SEEDS[0]))
        for p in r["predictions"]:
            if p["split"] == PRIMARY_SPLITS[0] and p["region"] == "in_range":
                source, density, _ = p["id"].split("|")
                source = source.replace("pythia-", "").replace("step", "")
                old = next(x for x in historical if (x["source"],x["d"],x["cap"]) == (source,float(density),cap))
                discrepancies.append(abs(p["predicted"]-old["A2"]))
    require(max(discrepancies) < 1e-10, "Historical A2 reproduction failed")
    return dict(passed=True, fit_attempts=len(records), failed_fit_attempts=len(summary["failure_inventory"]),
                historical_A2_max_absolute_discrepancy=max(discrepancies),
                fixed_cell_hashes={"/".join(k): v for k,v in fixed.items()},
                checks=["identical training labels and standardized source inputs", "fixed test cells, no missing fitted predictions",
                        "MAEs independently recomputed", "full-budget replicates identical", "nested single-axis subsets",
                        "synthetic power recovery on unseen density", "unidentified exponent and source fit rejected",
                        "synthetic A2 interpolation and boundary extension", "exact median anchor calculation",
                        "historical V42 full-budget A2 independently reproduced"])


def figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    try:
        from . import paper_figure_style as style
    except ImportError:
        import paper_figure_style as style
    figdir = OUT / "figs"
    figdir.mkdir(exist_ok=True)
    style.apply_style("double")
    colors = [style.PALETTE[k] for k in ("pruning", "reference", "black")]
    patterns, markers = ("-", "--", ":"), ("o", "s", "^")
    labels = ("Power", "A2", "Median")
    handles = [Line2D([], [], color=c, linestyle=l, marker=m, label=n)
               for c,l,m,n in zip(colors, patterns, markers, labels)]
    panels = []
    all_rows = summary["curves"]
    pdf = OUT / "a9_efficiency.pdf"
    with PdfPages(pdf, metadata={"Title": "A9 measurement efficiency", "CreationDate": None, "ModDate": None}) as pages:
        for axis, split in itertools.product(AXES, PRIMARY_SPLITS):
            page = plt.figure(figsize=(8.1, 2.75))
            page.suptitle(f"Fewer {axis} · {'seen states, unseen densities' if split == PRIMARY_SPLITS[0] else 'new states'}", y=.99)
            page.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.91), ncol=3)
            for cap_index, cap in enumerate(CAPS):
                rows = [r for r in all_rows if (r["axis"], r["split"], r["region"], r["cap"]) == (axis,split,"in_range",cap)]
                stem = f"a9_{chr(97+len(panels))}"
                fig = plt.figure(figsize=(2.7,2.2))
                ax = style.panel_axes(fig, (2.7,2.2), left=.50, bottom=.39, right=.08, top=.35)
                pax = page.add_axes(((cap_index*2.7+.5)/8.1, .39/2.75, 2.12/8.1, 1.46/2.75))
                ymax = max(r["q75"] for r in rows if r["q75"] is not None)*1.1
                ymax = max(ymax, .4)
                for target in (ax, pax):
                    for method,color,pattern,marker in zip(METHODS,colors,patterns,markers):
                        rr = sorted([r for r in rows if r["method"] == method], key=lambda r:r["n_measurements_per_capability"])
                        x = [r["n_measurements_per_capability"] for r in rr]
                        y, lo, hi = [np.array([r[k] if r[k] is not None else np.nan for r in rr]) for k in ("median","q25","q75")]
                        target.plot(x,y,color=color,linestyle=pattern,marker=marker)
                        target.fill_between(x,lo,hi,color=color,alpha=.10,linewidth=0)
                        target.errorbar(x,y,yerr=[y-lo,hi-y],color=color,fmt="none",capsize=2,alpha=.6)
                    for level in LEVELS:
                        target.axhline(level,color=style.PALETTE["grid"],linestyle="--",linewidth=.5,zorder=0)
                    target.set(ylim=(0,ymax), xlim=(6,38), xticks=sorted({r["n_measurements_per_capability"] for r in rows}),
                               xlabel="Measurements / capability", ylabel=f"{cap.title()} MAE (nats)")
                    target.set_yticks([v for v in target.get_yticks() if 0 <= v <= ymax])
                    style.row_axis_style(target)
                fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,1.0), ncol=3, fontsize=8.5)
                # Save directly to A9: paper_figure_style.save_panel routes to
                # paper/, which is explicitly outside this task's output scope.
                fig.savefig(figdir/f"{stem}.pdf", metadata={"CreationDate":None,"ModDate":None})
                fig.savefig(figdir/f"{stem}.png", dpi=220)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                for text in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label]:
                    box = text.get_window_extent(renderer)
                    require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1,
                            f"Figure text outside canvas: {stem}: {text.get_text()}")
                entry = dict(panel=stem, axis=axis, split=split, region="in_range", cap=cap,
                             size_inches=[2.7,2.2], kind="double", records=rows,
                             lines="conditional median over fitted replicates", bands="25th–75th percentile, not confidence interval",
                             failures="absent fits produce gaps; fitted/20 counts are in records and summary.md",
                             source_sha256=sha((ROOT/"analysis/paper_figure_style.py").read_bytes()),
                             font=dict(family=style.SERIF, ticks_pt=8.5, labels_pt=9.5, legend_pt=8.5),
                             artist_sizes=style.artist_sizes(fig))
                write_json(figdir/f"{stem}_data.json",entry)
                panels.append(entry)
                plt.close(fig)
            pages.savefig(page)
            plt.close(page)
    return dict(combined_pdf=str(pdf.relative_to(ROOT)), panels=panels)


def markdown(summary):
    plan, conclusion = summary["plan"], summary["conclusions"]
    def primary(r):
        return r["region"] == "in_range" and r["split"] in PRIMARY_SPLITS
    def short(split):
        return {"seen_state_unseen_density":"seen", "new_state":"new", "duplicate_weight_state_audit":"2.8B audit"}[split]
    def interval(p):
        if not p["n_fitted"]:
            return "unfittable (0/20)"
        return f"{p['median']:.3f} [{p['q25']:.3f}, {p['q75']:.3f}] ({p['n_fitted']}/20)"
    lines = ["# A9 measurement-efficiency study", "",
             "CPU only; no training or GPU use. Retrospective subsampling of existing development labels, evaluated on fixed historical confirmation cells. The A9 plan was written before any A9 fit; these are not newly blinded confirmation results.", "",
             "## Result", ""]
    savings = conclusion["measured_threshold_savings"]
    exclusive = conclusion["only_power_reaches_threshold"]
    lines.append(f"The compact power form has a demonstrated smaller tested measurement budget in {len(savings)} predictor-pair/target comparisons where both predictors reach the target. It alone reaches the target in {len(exclusive)} additional comparisons; those establish no finite savings estimate for the comparator. Targets are 0.15, 0.25 and 0.35 nats, with all 20 fits required at a qualifying budget.")
    lines += ["", "The advantage is restricted to density reduction: five finite comparisons against A2 and one against the median curve (new-state code at 0.35 nats: 9 versus 18 measurements). The five A2 comparisons partly reflect its inability to fit all replicates at sparse budgets. Source reduction shows no finite budget savings where both predictors reach the target. New-state QA favors the median curve in every complete three-predictor comparison; the compact form reaches none of the three QA error targets there. These statements apply only to this fixed panel, estimator and sampling plan."]
    lines.append("")
    if savings or exclusive:
        lines += ["| Reduction | Cells | Capability | Target | Power measurements | Comparator | Comparator measurements |",
                  "|---|---|---|---:|---:|---|---:|"]
        for r in savings+exclusive:
            other = str(r["competitor_measurements"]) if r["competitor_measurements"] is not None else "not reached"
            lines.append(f"| {r['axis']} | {short(r['split'])} | {r['cap']} | {r['level']:.2f} | {r['power_measurements']} | {r['competitor']} | {other} |")
    else:
        lines.append("No measurement-efficiency advantage for the compact form is established at these targets and tested budgets.")
    n, mw, pw = conclusion["n_complete_primary_comparisons"], conclusion["median_lowest"], conclusion["power_lowest"]
    lines += ["", f"Among {n} primary capability × axis × budget × test-panel comparisons with 20/20 fits for all three predictors, the median curve has the lowest (or tied-lowest) median MAE in {len(mw)} and power is strictly lowest in {len(pw)}. " +
              ("The median curve dominates all these complete comparisons." if len(mw)==n else "The median curve does not dominate all these complete comparisons."), ""]
    for label, entries in (("Power is lowest",pw),("Median is lowest",mw)):
        lines.append(label + ": " + ("; ".join(f"{r['axis']}/{short(r['split'])}/{r['cap']} at {r['measurements']}" for r in entries) or "none") + ".")
    lines += ["", "These are descriptive medians and empirical threshold crossings on a three-budget grid, not significance tests or continuous sample-complexity estimates. An unavailable A2 fit is an identifiability limit under the fixed unregularized estimator, not evidence that its prediction error is larger.", "",
              "## Provenance and fixed design", "", plan["source_correction"], "",
              "The development table is exactly 9 states (160M/410M/1.4B × 16k/64k/143k) × densities 0.6/0.7/0.8/0.9 × math/code/QA: 36 measurements per capability, 108 scalar labels. Input files, mirror checks, row-level provenance and all labels are in summary.json. No confirmation cell enters fitting.", "",
              f"Plan: `{summary['plan_sha256']}`; written {summary['plan_created_utc']}; fitting began {summary['fit_started_utc']}.", "",
              "R = 20, seeds 91000–91019, nested subsets. Source reduction retains the two seen-test states (1.4B@64k, 160M@143k), then samples the other seven: 2/5/9 states yield 8/20/36 measurements per capability (realized 22.2%/55.6%/100%; nominal 25%/50%/100%). This conditional source design keeps the fixed seen-state panel actually seen. At 25% the source subset is identical across replicates. Density reduction independently samples 1/2/4 strengths per source: 9/18/36 measurements (exact 25%/50%/100%). Axes are never mixed. At full budget all replicates repeat the same fit.", "",
              "The x axis counts pruning responses per capability (also the number of source-density evaluations). Multiply by three for all-capability scalar labels. Dense anchors and N0/D0 metadata are equally available and excluded from this response-measurement count; training dense-anchor counts are 2/5/9 per capability for source reduction and 9 throughout density reduction. Test dense anchors are fixed inputs.", "",
              "All predictors receive identical selected labels and source inputs. Power and A2 use the same training-only standardizer and unregularized least squares; no validation or hyperparameter search is allowed for any predictor. Power fits its structural exponent on the fixed V53 grid 0.5–6.0 by 0.05 using training SSE only. The source-free median deliberately ignores source covariates. A2 and median interpolate adjacent observed density anchors and linearly extend the nearest pair outside their sampled hull, without clipping. The compact form has five parameters; A2 has four per observed density (16 at full budget, not the paper's 20 from a different five-anchor panel); median has one estimated anchor per observed density.", "",
              "Full column rank is required at relative SVD tolerance 1e-10; power additionally needs local Jacobian rank five. No ridge or pseudoinverse rescue is used. The table reports conditional medians/IQRs with fitted counts, and every failed attempt remains in summary.json with rank, nominal parameter count and reason. Threshold crossings require 20/20 successful fits and median MAE at or below the target; missing crossings are not extrapolated.", "",
              "## Held-out cells", "", "| Panel | In-range cells/capability | Boundary cells/capability | Source states or labels |", "|---|---:|---:|---:|"]
    for split in SPLITS:
        rr = [r for r in summary["heldout_rows"] if r["split"]==split and r["cap"]=="math"]
        lines.append(f"| {short(split)} | {sum(r['region']=='in_range' for r in rr)} | {sum(r['region']=='boundary' for r in rr)} | {len({r['state'] for r in rr})} |")
    lines += ["", "Primary curves and thresholds use the original density range [0.6, 0.9]. Boundary points (including 0.55 and V53's 0.575, outside this nine-state panel) are retained and scored separately and in all-cell summaries. New-state cells are the union of V38/V42/V46/V49 and V53; V49 protocol B duplicates the same observations and is not counted twice. V36 leave-stage-out rows are development results and are not treated as confirmation. All predictors and replicates use the exact same fixed cells within each panel.", "",
              "V72's two 2.8B stage labels have identical learned weights (C52), so they are one weight state measured twice, not two independent training stages. All 18 labelled capability cells are scored in a separate audit using their recorded dense inputs and D0 labels. Their actual training histories are unverified; they are excluded from the primary new-state aggregate and all efficiency claims. The full audit, boundary and all-cell curves/thresholds are in summary.json.", "",
              "## Primary error curves", "", "MAE in nats: median [25th, 75th percentile] over fitted replicates (fitted/20). IQRs measure subset variation, not uncertainty across independent experiments. `seen` means unseen densities of the two retained development states.", "",
              "| Reduction | Cells | Capability | Measurements | Power | A2 | Median curve |", "|---|---|---|---:|---|---|---|"]
    table = {(r["axis"],r["split"],r["cap"],r["fraction"],r["method"]):r for r in summary["curves"] if primary(r)}
    for axis, split, cap, fraction in itertools.product(AXES,PRIMARY_SPLITS,CAPS,FRACTIONS):
        rr = [table[axis,split,cap,fraction,m] for m in METHODS]
        lines.append(f"| {axis} | {short(split)} | {cap} | {rr[0]['n_measurements_per_capability']} | " + " | ".join(interval(r) for r in rr) + " |")
    lines += ["", "## Measurements needed for the pre-set error levels", "", "Entries are minimum tested measurements per capability for 0.15 / 0.25 / 0.35 nats; NR = not reached with 20/20 fits. No interpolation between budgets and no monotonic smoothing.", "",
              "| Reduction | Cells | Capability | Power | A2 | Median curve |", "|---|---|---|---|---|---|"]
    targets = {(r['axis'],r['split'],r['cap'],r['method'],r['level']):r['minimum_measurements'] for r in summary['thresholds'] if primary(r)}
    for axis,split,cap in itertools.product(AXES,PRIMARY_SPLITS,CAPS):
        cols = [" / ".join(str(targets[axis,split,cap,m,t]) if targets[axis,split,cap,m,t] is not None else "NR" for t in LEVELS) for m in METHODS]
        lines.append(f"| {axis} | {short(split)} | {cap} | " + " | ".join(cols) + " |")
    lines += ["", "## Fit availability and identifiability", "", "Counts below aggregate the three capabilities: 60 attempts per row. Every individual rank, singular spectrum, parameter count, fitted coefficient, gamma profile and prediction is recorded in summary.json.", "",
              "| Reduction | Measurements/capability | Predictor | Fitted/60 | Parameters | Design ranks |", "|---|---:|---|---:|---|---|"]
    for axis,fraction,method in itertools.product(AXES,FRACTIONS,METHODS):
        rr = [r for r in summary['records'] if (r['axis'],r['fraction'],r['method']) == (axis,fraction,method)]
        params = sorted({r['fit']['n_parameters'] for r in rr})
        ranks = sorted({r['fit']['design_matrix']['rank'] for r in rr})
        lines.append(f"| {axis} | {rr[0]['n_measurements_per_capability']} | {method} | {sum(r['fit']['status']=='fitted' for r in rr)}/60 | {', '.join(map(str,params))} | {', '.join(map(str,ranks))} |")
    lines += ["", "## Boundary and duplicated-state audit", "", "Full-budget median MAEs; all 20 repeats are identical. These rows do not enter the primary efficiency claims.", "",
              "| Panel | Region | Capability | Power | A2 | Median curve |", "|---|---|---|---:|---:|---:|"]
    for split,region,cap in itertools.product(SPLITS,REGIONS,CAPS):
        if region == 'all' or (split in PRIMARY_SPLITS and region == 'in_range'):
            continue
        rr = [r for r in summary['curves'] if (r['axis'],r['fraction'],r['split'],r['region'],r['cap']) == ('sources',1.,split,region,cap)]
        if rr:
            bymethod = {r['method']:r for r in rr}
            lines.append(f"| {short(split)} | {region} | {cap} | " + " | ".join(f"{bymethod[m]['median']:.3f}" for m in METHODS) + " |")
    lines += ["", "## Artifacts and checks", "", "`summary.json` contains every selected subset, development and held-out row, fit/failure, prediction, MAE, median/IQR, paired difference and threshold crossing (including per-replicate crossings). `a9_efficiency.pdf` contains four pages of primary curves. Per-panel PDFs/PNGs and numeric/style sidecars are under `figs/`, following analysis/paper_figure_style.py typography, palette and full-canvas conventions.", "",
              "Panel order: a–c source reduction/seen states (math, code, QA); d–f source reduction/new states; g–i density reduction/seen states; j–l density reduction/new states. Lines show conditional medians, bands/error bars show IQRs, horizontal guides show the pre-set targets. Missing fits are gaps; availability is explicit above and in each sidecar.", "",
              f"Validation passed: {summary['validation']['fit_attempts']} fit attempts, {summary['validation']['failed_fit_attempts']} explicitly reported failures; same inputs and fixed test-cell hashes; recomputed MAEs; identical full-budget fits; nested single-axis subsets; synthetic law recovery, rank rejection, interpolation and median checks. Full-budget A2 reproduces historical V42 predictions to {summary['validation']['historical_A2_max_absolute_discrepancy']:.2g} nats. A runtime guard confines analysis writes to A9 outputs and temporary caches, and blocks subprocesses/network operations. This analysis wrote no paper TeX, figures or tables. No commit was made.", ""]
    return "\n".join(lines)


def run():
    plan_raw = (OUT / "plan.json").read_bytes()
    plan = json.loads(plan_raw)
    dev, test, provenance = load_data()
    require(plan["data_sha256"] == sha(canonical(dict(development=dev, heldout=test)).encode()), "Data changed after plan")
    require(plan["provenance"] == provenance, "Input artifacts changed after plan")
    require(plan["subsets"] == subsets(dev), "Subsample plan changed")
    require(plan["predictors"]["gamma_grid"] == GAMMAS.tolist(), "Gamma grid changed after plan")
    started = datetime.now(timezone.utc).isoformat()
    require(started > plan["created_utc"], "Fit must follow plan")
    records = []
    for subset in plan["subsets"]:
        selected_ids = set(subset["selected_ids"])
        selected = [r for r in dev if r["id"] in selected_ids]
        stats = standardizer(selected)
        for cap, method in itertools.product(CAPS, METHODS):
            training = [r for r in selected if r["cap"] == cap]
            heldout = [r for r in test if r["cap"] == cap]
            fit = fit_predictor(training, method, stats)
            scores, cells = {}, []
            if fit["status"] == "fitted":
                predictions = predict(fit, heldout, stats)
                require(np.all(np.isfinite(predictions)), "Nonfinite prediction")
                for r, prediction in zip(heldout, predictions):
                    cells.append(dict(id=r["id"], split=r["split"], region=r["region"],
                                      observed=r["y"], predicted=float(prediction), absolute_error=float(abs(prediction-r["y"]))))
            for split, region in itertools.product(SPLITS, REGIONS):
                fixed = [r for r in heldout if r["split"] == split and (region == "all" or r["region"] == region)]
                scored = [r for r in cells if r["split"] == split and (region == "all" or r["region"] == region)]
                if fixed:
                    require(not scored or len(scored) == len(fixed), "Test-set drift")
                    scores[f"{split}/{region}"] = dict(n_test=len(fixed),
                        mae=float(np.mean([r["absolute_error"] for r in scored])) if scored else None,
                        fixed_cell_sha256=sha(canonical(sorted(r["id"] for r in fixed)).encode()))
            records.append({k: v for k, v in subset.items() if k != "selected_ids"} | dict(
                cap=cap, method=method, standardizer=stats, fit=fit, scores=scores, predictions=cells,
                training_label_sha256=sha(canonical([(r["id"], r["y"]) for r in training]).encode()),
                n_training_dense_anchors_per_capability=subset["n_sources"],
                test_seen_sources_retained=list(PINNED)))
    curves, thresholds, comparisons = aggregate(records, test)
    summary = dict(schema_version=1, cpu_only=True, training_runs=0, gpu_calls=0,
                   plan_sha256=sha(plan_raw), plan_created_utc=plan["created_utc"], fit_started_utc=started,
                   analysis_sha256=sha(Path(__file__).read_bytes()), provenance=provenance, plan=plan,
                   development_rows=dev, heldout_rows=test, records=records, curves=curves,
                   thresholds=thresholds, paired_comparisons=comparisons,
                   failure_inventory=[dict(subset_id=r["id"], cap=r["cap"], method=r["method"],
                                           reason=r["fit"]["reason"], rank=r["fit"]["design_matrix"]["rank"],
                                           parameters=r["fit"]["n_parameters"])
                                      for r in records if r["fit"]["status"] != "fitted"])
    summary["conclusions"] = conclusions(summary)
    summary["validation"] = validate(summary)
    summary["figures"] = figures(summary)
    require((OUT / "plan.json").read_bytes() == plan_raw, "Plan was mutated")
    write_json(OUT / "summary.json", summary)
    report = markdown(summary)
    (OUT / "summary.md").write_text(report)
    print(report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "run"))
    args = p.parse_args()
    install_io_guard()
    if args.command == "plan":
        make_plan()
    else:
        run()


if __name__ == "__main__":
    main()
