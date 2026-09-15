#!/usr/bin/env python3
"""A2: CPU-only, canonical-input response fitting and frozen intervention audit.

Only the A1 CSV and summary are measurement inputs. No imports of earlier
analyses, model libraries, training, GPU calls, or directory discovery.
Run with -B; --nonembedding-counts accepts an explicitly supplied JSON mapping.
All result files are confined to results/a2-curvature-interaction/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import io
from itertools import combinations
import json
import math
import os
from pathlib import Path
import sys

# This module never uses a GPU. Bound BLAS concurrency before importing numpy.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import Delaunay, QhullError

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("results/a2-curvature-interaction")
T_REF = 100_000.0
STRUCTURES = ("F_log", "F_curv", "F_int")
RESPONSE_BASELINES = ("zero", "constant", "budget_only", "reuse_only", "surface")
DESCRIPTORS = ("log_parameters", "initial_loss")
LAMBDAS = (0.0001, 0.01, 1.0)
FIXED_LAMBDA = 0.01
SPLITS = ("data_rung", "student", "largest_budget", "pool_seed")
ROW_KEY = ("run_id", "checkpoint_id", "capability", "distribution")
P_GRID = np.linspace(-1, 3, 33)
P_BRACKETS = ((-1., 1/3), (1/3, 5/3), (5/3, 3.))
COUNTS_MISSING = ("The permitted A1 inputs contain no non-embedding parameter counts. "
                  "Supply verified counts for every student with --nonembedding-counts; "
                  "nominal model names are not parameter counts.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Point:
    key: tuple
    run: str
    student: str
    pool: str
    seed: int
    rung: int
    protocol: str
    T: float
    D: float
    initial_loss: float
    y: float
    n: int | None = None

    @property
    def E(self):
        return self.T / self.D


@dataclass(frozen=True)
class Pair:
    id: str
    kind: str
    first: Point
    second: Point
    tolerance: float | None
    band: str
    actual: float

    @property
    def runs(self):
        return tuple(sorted({self.first.run, self.second.run}))


def load_inputs(root=ROOT, counts=None):
    base = Path(root) / "results/a1-development-table"
    raw = {n: (base / n).read_bytes() for n in ("development_table.csv", "summary.json")}
    summary = json.loads(raw["summary.json"])
    require(summary["artifact_sha256"]["development_table.csv"] == sha(raw["development_table.csv"]),
            "Canonical table checksum mismatch")
    records = list(csv_reader(raw["development_table.csv"].decode()))
    points = []
    for r in records:
        n = None if counts is None else counts.get(r["student_id"])
        if counts is not None:
            require(type(n) is int and n > 0, COUNTS_MISSING)
        p = Point(tuple(r[k] for k in ROW_KEY), r["run_id"], r["student_id"],
                  r["pool_id"], int(r["pool_seed"]), int(r["U"]), r["protocol_id"],
                  int(r["T_actual"]), int(r["D_U_pool"]), float(r["initial_loss"]),
                  float(r["delta"]), n)
        require(p.D > 0 and p.T >= 0 and math.isfinite(p.y), "Invalid canonical measurement")
        require(math.isclose(float(r["loss"]) - p.initial_loss, p.y, abs_tol=1e-10),
                "Response sign/accounting mismatch")
        require(p.T != 0 or abs(p.y) < 1e-12, "Nonzero anchor response")
        points.append(p)
    lookup = {p.key: p for p in points}
    require(len(lookup) == len(points), "Duplicate canonical row key")
    require({p.run for p in points} == set(summary["membership"]["core"]), "Core membership mismatch")
    pairs = []
    for item in summary["pairs"]:
        if item["kind"] not in ("I_T", "I_U") or not item["protocol_id"].startswith("core:"):
            continue
        a, b = (lookup[tuple(item[k])] for k in ("first", "second"))
        require(a.key[2:] == b.key[2:] and a.student == b.student and a.protocol == b.protocol,
                "Incompatible pair endpoints")
        require(a.T > 0 and b.T > 0, "Zero anchor in positive intervention inventory")
        require(a.T == item["T_first"] and b.T == item["T_second"], "Pair budget mismatch")
        require(math.isclose(b.y - a.y, item["effect"], abs_tol=1e-12), "Stored pair effect mismatch")
        pid = sha(canonical([item["kind"], item["first"], item["second"], item["tolerance"]]).encode())[:20]
        pairs.append(Pair(pid, item["kind"], a, b, item["tolerance"], item["budget_band"], item["effect"]))
    require(len({p.id for p in pairs}) == len(pairs), "Duplicate stored pair")
    return points, pairs, {k: sha(v) for k, v in raw.items()}, summary


def csv_reader(text):
    import csv
    return csv.DictReader(io.StringIO(text))


def h_p(E, p):
    E = np.asarray(E, dtype=float)
    v = np.log1p(E)
    if p == 0:
        return v
    if abs(p) < 1e-7:
        # expm1(p*v)/p = v*(1 + p*v/2 + (p*v)^2/6 + ...).
        x = p * v
        return v * (1 + x/2 + x*x/6 + x*x*x/24)
    return np.expm1(p * v) / p


def numerical_invariants():
    e = np.array([0, 1e-10, .1, 1., 10., 100.])
    require(np.array_equal(h_p(e, 0), np.log1p(e)), "h_p(0) identity failed")
    require(np.allclose(h_p(e, 1), e, rtol=1e-13, atol=1e-13), "h_p(1) linear reuse identity failed")


def response_weights(points, multipliers=None):
    """Equal trajectories, then equal distinct positive budgets within each."""
    groups = Counter(p.run for p in points)
    w = np.array([(1 if multipliers is None else multipliers[p.run]) / groups[p.run] for p in points])
    return w / w.sum()


@dataclass(frozen=True)
class Scaler:
    descriptor: str
    mean: float
    scale: float
    students: tuple

    @classmethod
    def fit(cls, points, descriptor):
        # Equal students, then equal trajectories. Some scope baselines differ
        # slightly across runs; retain and audit these recorded measurements.
        values = defaultdict(dict)
        for p in points:
            value = cls.value(p, descriptor)
            require(p.run not in values[p.student] or values[p.student][p.run] == value,
                    "Descriptor changes within a trajectory")
            values[p.student][p.run] = value
        mean = float(np.mean([np.mean(list(v.values())) for v in values.values()]))
        var = float(np.mean([np.mean((np.array(list(v.values()))-mean)**2) for v in values.values()]))
        sd = np.sqrt(var)
        return cls(descriptor, mean, float(sd) if sd > 1e-12 else 1., tuple(sorted(values)))

    @staticmethod
    def value(point, descriptor):
        if descriptor == "log_parameters":
            require(point.n is not None, COUNTS_MISSING)
            return math.log(point.n)
        require(descriptor == "initial_loss", "Unknown descriptor")
        return point.initial_loss

    def transform(self, points):
        return np.array([(self.value(p, self.descriptor)-self.mean)/self.scale for p in points])


def design(points, name, scaler=None, p=0.):
    u = np.log1p(np.array([r.T for r in points]) / T_REF)
    e = np.array([r.E for r in points])
    v = np.log1p(e)
    if name == "zero":
        return np.zeros((len(points), 0))
    if name == "constant":
        return (u > 0).astype(float)[:, None]
    if name == "budget_only":
        return u[:, None]
    if name == "reuse_only":
        return v[:, None]
    require(scaler is not None, "Missing training scaler")
    z = scaler.transform(points)
    if name == "surface":
        return np.column_stack((u, v, u*u, v*v, u*v, z*u, z*v))
    require(name in STRUCTURES, "Unknown structure")
    reuse = h_p(e, p) if name == "F_curv" else v
    columns = [u, z*u, reuse, z*reuse]
    if name == "F_int":
        columns.append(u*v)
    return np.column_stack(columns)


def ridge(X, y, w, lam):
    if X.shape[1] == 0:
        return np.zeros(0), 0., float(w @ (y*y)), np.ones(0)
    rms = np.sqrt(w @ (X*X))
    rms[rms < 1e-12] = 1.
    A = X / rms
    gram = A.T @ (w[:, None]*A)
    inv = np.linalg.pinv(gram + lam*np.eye(X.shape[1]), rcond=1e-12)
    theta = inv @ (A.T @ (w*y))
    residual = y - A @ theta
    return theta/rms, float(np.trace(inv@gram)), float(w@(residual*residual)+lam*(theta@theta)), rms


@dataclass
class Fit:
    name: str
    coefficients: np.ndarray
    scaler: Scaler | None
    p: float
    lam: float
    edf: float
    objective: float
    profile: list
    train_keys: tuple
    fit_id: str

    def predict(self, points):
        return design(points, self.name, self.scaler, self.p) @ self.coefficients

    def intervention(self, pairs, fixed_T=False):
        # Exactly this fit object predicts BOTH endpoints; there is no pair fit.
        first = [p.first for p in pairs]
        second = [replace(p.second, T=p.first.T if fixed_T else p.second.T,
                          initial_loss=p.first.initial_loss) for p in pairs]
        return self.predict(second) - self.predict(first)

    @property
    def boundary(self):
        return self.name == "F_curv" and (self.p <= -1+1e-4 or self.p >= 3-1e-4)

    def record(self):
        out = dict(name=self.name, coefficients=self.coefficients.tolist(), p=self.p if self.name == "F_curv" else None,
                   lambda_=self.lam, effective_df_conditional=self.edf,
                   nominal_parameters=len(self.coefficients)+(self.name == "F_curv"),
                   objective=self.objective, profile=self.profile, boundary_hit=self.boundary,
                   curvature_status=("not identified: boundary optimum" if self.boundary else
                                     "interior optimum; identification requires stability") if self.name == "F_curv" else None,
                   train_keys=[list(k) for k in self.train_keys], fit_id=self.fit_id)
        if self.scaler:
            out["standardizer"] = dict(descriptor=self.scaler.descriptor, mean=self.scaler.mean,
                                       scale=self.scaler.scale, students=list(self.scaler.students))
        # Curvature EDF includes one only as a local nominal adjustment, never exact.
        out["effective_df_with_p_approx"] = self.edf + (self.name == "F_curv" and not self.boundary)
        return out


def fit_response(points, name, descriptor="log_parameters", lam=FIXED_LAMBDA,
                 profile=True, multipliers=None):
    require(points and all(r.T > 0 for r in points), "No positive training responses")
    require(len({r.key[2:] for r in points}) == 1, "Fit must have one capability/distribution")
    scaler = Scaler.fit(points, descriptor) if name in (*STRUCTURES, "surface") else None
    w = response_weights(points, multipliers)
    y = np.array([r.y for r in points])
    effective_lam = 0. if name in ("zero", "constant") else lam
    cache = {}

    def solve(p):
        p = float(p)
        if p not in cache:
            cache[p] = ridge(design(points, name, scaler, p), y, w, effective_lam)
        return cache[p]

    p = 0.
    if name == "F_curv":
        # Three fixed bounded starts/regions plus both boundaries and p=0,1.
        choices = [-1., 0., 1., 3.]
        for bounds in P_BRACKETS:
            result = minimize_scalar(lambda x: solve(x)[2], bounds=bounds, method="bounded",
                                     options={"xatol": 2e-5, "maxiter": 60})
            require(result.success, "Curvature profile optimisation failed")
            choices.append(float(result.x))
        p = min(choices, key=lambda x: (solve(x)[2], abs(x)))
    coef, edf, objective, _ = solve(p)
    curve = ([dict(p=float(x), objective=solve(x)[2], effective_df=solve(x)[1]) for x in P_GRID]
             if name == "F_curv" and profile else [])
    keys = tuple(sorted(r.key for r in points))
    fid = sha(canonical([name, descriptor, lam, p, keys, coef.tolist()]).encode())[:20]
    return Fit(name, coef, scaler, p, effective_lam, edf, objective, curve, keys, fid)


def make_folds(points, split, inner=False):
    if split == "largest_budget":
        threshold = 75_000 if inner else 150_000
        return [(f"T>={threshold}", [p for p in points if p.T < threshold],
                 [p for p in points if p.T >= threshold])]
    attr = {"data_rung": "rung", "student": "student", "pool_seed": "seed"}[split]
    return [(str(g), [p for p in points if getattr(p, attr) != g],
             [p for p in points if getattr(p, attr) == g]) for g in sorted({getattr(p, attr) for p in points})]


def pair_subset(pairs, training, testing=None, tolerance=.01):
    train = {p.key for p in training}
    test = set() if testing is None else {p.key for p in testing}
    out = []
    for p in pairs:
        if p.kind == "I_U" and p.tolerance != tolerance:
            continue
        endpoints = {p.first.key, p.second.key}
        if testing is None:
            use = endpoints <= train
        else:
            # At least one withheld endpoint. The other may be a known reference.
            use = endpoints <= (train | test) and bool(endpoints & test)
        if use:
            out.append(p)
    return out


def item_weights(items):
    if not items:
        return np.zeros(0)
    if isinstance(items[0], Point):
        return response_weights(items)
    # Student -> physical ordered pool pair -> fixed actual-token band -> pairs.
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, p in enumerate(items):
        pools = (p.first.pool, p.second.pool)
        groups[p.first.student][pools][p.band].append(i)
    w = np.zeros(len(items))
    for pools in groups.values():
        for bands in pools.values():
            for ids in bands.values():
                w[ids] = 1 / (len(groups)*len(pools)*len(bands)*len(ids))
    return w


def targets(points, pairs):
    return {"response": points, **{k: [p for p in pairs if p.kind == k] for k in ("I_T", "I_U")}}


def truth(items):
    return np.array([p.y if isinstance(p, Point) else p.actual for p in items])


def mean_effects(pairs):
    out = {}
    for kind in ("I_T", "I_U"):
        subset = [p for p in pairs if p.kind == kind]
        out[kind] = float(item_weights(subset) @ truth(subset)) if subset else None
    return out


def predict_target(fit, items):
    if not items:
        return np.zeros(0)
    return fit.predict(items) if isinstance(items[0], Point) else fit.intervention(items)


def select_inside(training, pairs, split, cache):
    folds = [(g, tr, te) for g, tr, te in make_folds(training, split, inner=True) if tr and te]
    if len(folds) < 3:
        return dict(primary="F_log", descriptor="log_parameters", lambda_=FIXED_LAMBDA,
                    baseline={"response": "constant", "I_T": "mean_effect", "I_U": "mean_effect"},
                    method="Fixed parsimonious settings: fewer than three usable inner grouped folds",
                    inner_folds=len(folds), scores=[])
    options = []
    for descriptor in DESCRIPTORS:
        for lam in LAMBDAS:
            losses = defaultdict(list)
            for g, tr, te in folds:
                trainpairs = pair_subset(pairs, tr)
                testpairs = pair_subset(pairs, tr, te)
                means = mean_effects(trainpairs)
                for name in (*STRUCTURES, *RESPONSE_BASELINES):
                    fit = cached_fit(cache, tr, name, descriptor, lam)
                    for target, items in targets(te, testpairs).items():
                        if items:
                            score = float(item_weights(items) @ np.abs(truth(items)-predict_target(fit, items)))
                            losses[name, target].append(score)
                for kind in ("I_T", "I_U"):
                    items = [p for p in testpairs if p.kind == kind]
                    if items and means[kind] is not None:
                        losses["mean_effect", kind].append(float(item_weights(items) @ np.abs(truth(items)-means[kind])))
            scores = {f"{name}:{target}": float(np.mean(x)) for (name, target), x in losses.items()}
            # Choose a single F/descriptor/lambda from response validation only.
            for name in STRUCTURES:
                options.append(dict(primary=name, descriptor=descriptor, lambda_=lam,
                                    score=scores[name+":response"], scores=scores))
    best = min(options, key=lambda x: (x["score"], STRUCTURES.index(x["primary"]),
                                       DESCRIPTORS.index(x["descriptor"]), -x["lambda_"]))
    # Baselines receive the same chosen descriptor and lambda as the candidate.
    # Target-specific inner MAE chooses baseline identity, never outer errors.
    baseline = {}
    for target in ("response", "I_T", "I_U"):
        names = list(RESPONSE_BASELINES) + ([] if target == "response" else ["mean_effect"])
        valid = [n for n in names if n+":"+target in best["scores"]]
        baseline[target] = min(valid, key=lambda n: best["scores"][n+":"+target]) if valid else "zero"
    return dict(primary=best["primary"], descriptor=best["descriptor"], lambda_=best["lambda_"],
                baseline=baseline, method="Inner grouped response MAE; same F for all three targets", inner_folds=len(folds),
                scores=[{k: v for k, v in x.items() if k != "scores"} for x in options],
                baseline_inner_scores=best["scores"])


def cached_fit(cache, points, name, descriptor, lam):
    key = (tuple(sorted(p.key for p in points)), name, descriptor, lam)
    if key not in cache:
        cache[key] = fit_response(points, name, descriptor, lam, profile=False)
    return cache[key]


def interpolate(training, testing):
    """Diagnostic only: same student, closed convex hull in fixed (u,v), no extrapolation."""
    out = np.full(len(testing), np.nan)
    for student in sorted({p.student for p in testing}):
        tr = [p for p in training if p.student == student]
        idx = [i for i, p in enumerate(testing) if p.student == student]
        if len(tr) < 3:
            continue
        xy = np.array([[np.log1p(p.T/T_REF), np.log1p(p.E)] for p in tr])
        xy, unique = np.unique(xy, axis=0, return_index=True)
        if len(xy) < 3 or np.linalg.matrix_rank(xy-xy.mean(0)) < 2:
            continue
        try:
            tri = Delaunay(xy)
        except QhullError:
            continue
        values = np.array([tr[i].y for i in unique])
        for i in idx:
            p = testing[i]
            q = np.array([np.log1p(p.T/T_REF), np.log1p(p.E)])
            simplex = tri.find_simplex(q, tol=1e-12)
            if simplex < 0:
                continue
            b = tri.transform[simplex, :2] @ (q-tri.transform[simplex, 2])
            bary = np.r_[b, 1-b.sum()]
            if bary.min() >= -1e-12:
                out[i] = bary @ values[tri.simplices[simplex]]
    return out


def observation_record(item, fold, fits, selection, means):
    is_response = isinstance(item, Point)
    actual = item.y if is_response else item.actual
    target = "response" if is_response else item.kind
    predictions = {n: float(predict_target(fit, [item])[0]) for n, fit in fits.items()}
    if not is_response and means[target] is not None:
        predictions["mean_effect"] = means[target]
    baseline = selection["baseline"][target]
    fallback = None
    if baseline not in predictions:
        fallback, baseline = "No training pairs for the preselected mean; fixed zero fallback", "zero"
    return dict(id=canonical(item.key) if is_response else item.id, fold=fold, target=target, actual=actual,
                predictions=predictions, primary=selection["primary"], baseline=baseline, baseline_fallback=fallback,
                student=item.student if is_response else item.first.student,
                runs=[item.run] if is_response else list(item.runs),
                pools=[item.pool] if is_response else [item.first.pool, item.second.pool],
                band=None if is_response else item.band,
                first_key=list(item.key) if is_response else list(item.first.key),
                second_key=None if is_response else list(item.second.key),
                fit_ids={n: fit.fit_id for n, fit in fits.items()},
                fixed_T_predictions=None if is_response or target != "I_U" else
                {n: float(fit.intervention([item], fixed_T=True)[0]) for n, fit in fits.items()},
                residual_T=None if is_response else item.second.T-item.first.T,
                initial_loss_difference=None if is_response else item.second.initial_loss-item.first.initial_loss,
                tolerance=None if is_response else item.tolerance)


def record_weights(records):
    """Equal response trajectories; intervention student/pool-pair/band hierarchy.

    Repeated appearances of the same pair in different held groups share its
    weight, rather than treating a duplicated endpoint contrast as new data.
    """
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for i, r in enumerate(records):
        if r["target"] == "response":
            a, b, c = r["runs"][0], "response", "response"
        else:
            a, b, c = r["student"], tuple(r["pools"]), r["band"]
        tree[a][b][c][r["id"]].append(i)
    w = np.zeros(len(records))
    for btree in tree.values():
        for ctree in btree.values():
            for ids in ctree.values():
                for positions in ids.values():
                    w[positions] = 1/(len(tree)*len(btree)*len(ctree)*len(ids)*len(positions))
    return w


def frozen_interval(records, values, draws, seed=20260914):
    if not records:
        return None
    runs = sorted({run for r in records for run in r["runs"]})
    if len(runs) < 2:
        return None
    rng = np.random.default_rng(seed)
    lookup = {r: i for i, r in enumerate(runs)}
    w = record_weights(records)
    # Dyadic Bayesian cluster bootstrap: each trajectory gets one multiplier;
    # cross-trajectory pairs get the product. Same-run I_T uses it only once.
    m = rng.exponential(size=(draws, len(runs)))
    weights = np.column_stack([np.prod(m[:, [lookup[x] for x in r["runs"]]], axis=1) for r in records]) * w
    means = (weights @ np.asarray(values)) / weights.sum(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def metric(records, draws):
    if not records:
        return dict(status="unscorable: no eligible held-out observations", n=0)
    w = record_weights(records)
    pe = np.array([abs(r["actual"]-r["predictions"][r["primary"]]) for r in records])
    be = np.array([abs(r["actual"]-r["predictions"][r["baseline"]]) for r in records])
    common = set.intersection(*(set(r["predictions"]) for r in records))
    all_mae = {n: float(w @ np.array([abs(r["actual"]-r["predictions"][n]) for r in records])) for n in sorted(common)}
    baselines = {n: v for n, v in all_mae.items() if n not in STRUCTURES}
    strongest = min(baselines, key=baselines.get)
    oracle_error = np.array([abs(r["actual"]-r["predictions"][strongest]) for r in records])
    return dict(status="scored", n=len(records), unique_observations=len({r["id"] for r in records}),
                trajectories=len({x for r in records for x in r["runs"]}),
                primary_candidates=dict(Counter(r["primary"] for r in records)),
                inner_selected_baselines=dict(Counter(r["baseline"] for r in records)),
                primary_mae=float(w@pe), baseline_mae=float(w@be),
                paired_improvement=float(w@(be-pe)),
                frozen_prediction_error_interval=frozen_interval(records, be-pe, draws),
                primary_mae_interval=frozen_interval(records, pe, draws),
                all_model_mae=all_mae, strongest_observed_baseline=strongest,
                strongest_baseline_mae=baselines[strongest],
                improvement_over_strongest=float(w@(oracle_error-pe)),
                interval_over_strongest=frozen_interval(records, oracle_error-pe, draws),
                strongest_baseline_note="Minimum OUTER baseline MAE; descriptive oracle, selection not covered by interval",
                exact_I_U_mae=None if records[0]["target"] == "I_U" else "not applicable")


def evaluate(points, pairs, draws=2000):
    positive = [p for p in points if p.T > 0]
    cache, results = {}, []
    for channel in sorted({p.key[2:] for p in positive}):
        rows = [p for p in positive if p.key[2:] == channel]
        channelpairs = [p for p in pairs if p.first.key[2:] == channel]
        for split in SPLITS:
            details, predictions, sensitivity = [], [], {str(t): [] for t in (.005, .05)}
            for held, tr, te in make_folds(rows, split):
                if not tr or not te:
                    details.append(dict(held=held, status="unscorable: no positive train or test responses",
                                        n_train=len(tr), n_test=len(te)))
                    continue
                selection = select_inside(tr, channelpairs, split, cache)
                descriptor, lam = selection["descriptor"], selection["lambda_"]
                fits = {n: fit_response(tr, n, descriptor, lam) for n in (*STRUCTURES, *RESPONSE_BASELINES)}
                trainpairs = pair_subset(channelpairs, tr)
                testpairs = pair_subset(channelpairs, tr, te)
                means = mean_effects(trainpairs)
                current = [observation_record(x, held, fits, selection, means) for x in [*te, *testpairs]]
                predictions.extend(current)
                interp = interpolate(tr, te)
                valid = np.isfinite(interp)
                interp_detail = dict(n_supported=int(valid.sum()), n_requested=len(te), nominal_parameters=len(tr),
                                     effective_df=len(tr), rule="same-student (u,v) closed convex hull, barycentric linear; no extrapolation")
                if valid.any():
                    supported = [p for p, ok in zip(te, valid) if ok]
                    weights = response_weights(supported)
                    interp_detail.update(mae=float(weights @ np.abs(truth(supported)-interp[valid])),
                                         primary_mae_same_support=float(weights @ np.abs(truth(supported)-fits[selection["primary"]].predict(supported))),
                                         supported_keys=[list(p.key) for p in supported])
                for tol in (.005, .05):
                    train_tol = pair_subset(channelpairs, tr, tolerance=tol)
                    test_tol = [p for p in pair_subset(channelpairs, tr, te, tolerance=tol) if p.kind == "I_U"]
                    sensitivity[str(tol)].extend(observation_record(x, held, fits, selection, mean_effects(train_tol)) for x in test_tol)
                details.append(dict(held=held, status="scored", n_train=len(tr), n_test=len(te),
                                    train_trajectories=len({p.run for p in tr}), test_keys=[list(p.key) for p in te],
                                    selection=selection, fits={n: f.record() for n, f in fits.items()},
                                    mean_effects=means, mean_training_pairs=dict(Counter(p.kind for p in trainpairs)),
                                    metrics={t: metric([r for r in current if r["target"] == t], draws) for t in ("response", "I_T", "I_U")},
                                    empirical_interpolation_diagnostic=interp_detail))
            result = dict(capability=channel[0], distribution=channel[1], split=split, folds=details,
                          predictions=predictions,
                          metrics={t: metric([r for r in predictions if r["target"] == t], draws) for t in ("response", "I_T", "I_U")},
                          tolerance_sensitivity={t: metric(r, draws) for t, r in sensitivity.items()})
            results.append(result)
            print(f"A2: {channel[1]} / {split} complete", file=sys.stderr, flush=True)
    return results


def stability(results):
    output = []
    for channel in sorted({(r["capability"], r["distribution"]) for r in results}):
        rows = []
        for result in results:
            if (result["capability"], result["distribution"]) != channel:
                continue
            for fold in result["folds"]:
                if fold["status"] != "scored":
                    continue
                for name in STRUCTURES:
                    f = fold["fits"][name]
                    rows.append(dict(split=result["split"], held=fold["held"], name=name,
                                     descriptor=f["standardizer"]["descriptor"], p=f["p"], boundary=f["boundary_hit"],
                                     coefficients=f["coefficients"], effective_df=f["effective_df_with_p_approx"],
                                     objective=f["objective"], profile=f["profile"]))
        ps = [r["p"] for r in rows if r["name"] == "F_curv"]
        output.append(dict(capability=channel[0], distribution=channel[1], folds=rows,
                           p_distribution=dict(n=len(ps), minimum=min(ps), median=float(np.median(ps)), maximum=max(ps),
                                               quantiles_025_975=np.quantile(ps, [.025, .975]).tolist(),
                                               boundary_hits=sum(r["boundary"] for r in rows if r["name"] == "F_curv")),
                           leave_one_student=[r for r in rows if r["name"] == "F_curv" and r["split"] == "student"],
                           warning="Across-fold ranges are sensitivity summaries, NOT parameter confidence intervals; folds overlap."))
    return output


def parameter_intervals(points, pairs, draws=200):
    output = []
    for channel in sorted({p.key[2:] for p in points}):
        rows = [p for p in points if p.T > 0 and p.key[2:] == channel]
        pp = [p for p in pairs if p.first.key[2:] == channel]
        choice = select_inside(rows, pp, "pool_seed", {})
        rng = np.random.default_rng(62104)
        runs = sorted({p.run for p in rows})
        multipliers = [dict(zip(runs, rng.exponential(size=len(runs)))) for _ in range(draws)]
        fits = {}
        for name in STRUCTURES:
            fit = fit_response(rows, name, choice["descriptor"], choice["lambda_"])
            reps = [fit_response(rows, name, choice["descriptor"], choice["lambda_"], profile=False, multipliers=m) for m in multipliers]
            fits[name] = dict(fit=fit.record(), coefficient_interval=np.quantile([r.coefficients for r in reps], [.025, .975], axis=0).tolist(),
                              p_interval=np.quantile([r.p for r in reps], [.025, .975]).tolist() if name == "F_curv" else None,
                              bootstrap_boundary_fraction=float(np.mean([r.boundary for r in reps])) if name == "F_curv" else None)
        output.append(dict(capability=channel[0], distribution=channel[1], selection=choice, fits=fits,
                           label="Parameter intervals: 95% trajectory-multiplier refit percentiles, conditional on descriptor/lambda and observed students",
                           draws=draws, warning="p reprofiled in each refit; selection and new-student uncertainty not included. Ridge intervals are descriptive."))
        print(f"A2: parameter intervals / {channel[1]} complete", file=sys.stderr, flush=True)
    return output


def four_corners(points):
    # Geometry uses only actual canonical T/D, exact rational equality, and never
    # the stored intervention matches (which do not define rectangular support).
    physical = {}
    for p in points:
        physical[p.student, p.protocol, p.run, p.key[1]] = p
    trajectories, coverage = [], []
    for run in sorted({p.run for p in physical.values()}):
        rows = sorted([p for p in physical.values() if p.run == run], key=lambda p: p.T)
        p = rows[0]
        trajectories.append(dict(run=run, student=p.student, U=p.rung, D_U=p.D, pool_seed=p.seed, pool_id=p.pool,
                                 checkpoints=[dict(checkpoint=r.key[1], T=r.T, E=r.E,
                                                   E_exact=str(Fraction(int(r.T), int(r.D)))) for r in rows]))
    for student in sorted({p.student for p in physical.values()}):
        rows = [p for p in physical.values() if p.student == student and p.T > 0]
        grid = defaultdict(set)
        for p in rows:
            grid[int(p.T)].add(Fraction(int(p.T), int(p.D)))
        rectangles = []
        for t1, t2 in combinations(sorted(grid), 2):
            for e1, e2 in combinations(sorted(grid[t1] & grid[t2]), 2):
                rectangles.append(dict(T1=t1, T2=t2, E1=str(e1), E2=str(e2)))
        coverage.append(dict(student=student, trajectories=len({p.run for p in rows}),
                             positive_checkpoints=len(rows), unique_T=len(grid), unique_E=len(set().union(*grid.values())),
                             T_with_two_or_more_E=sum(len(v)>1 for v in grid.values()),
                             exact_rectangles=len(rectangles), rectangles=rectangles,
                             zero_budget="Only (T,E)=(0,0); repeated anchors cannot form a nondegenerate rectangle"))
    return dict(coverage=coverage, trajectories=trajectories,
                identity="[delta(T2,E2)-delta(T2,E1)]-[delta(T1,E2)-delta(T1,E1)]=0 for A(T)+B*h(E) at fixed student",
                rule="Exact rational T/D matches within student/protocol, T>0; two fixed pools across budgets are not four corners",
                structural_exclusion="None: no observed four-corner interaction statistic exists; model-imputed or interpolated corners cannot exclude additivity")


def input_audit(points, pairs):
    """Outcome-free eligibility plus the recorded baseline discrepancies."""
    baselines, folds = [], []
    for channel in sorted({p.key[2:] for p in points}):
        rows = [p for p in points if p.key[2:] == channel and p.T > 0]
        pp = [p for p in pairs if p.first.key[2:] == channel]
        for student in sorted({p.student for p in rows}):
            values = sorted({p.initial_loss for p in rows if p.student == student})
            baselines.append(dict(capability=channel[0], distribution=channel[1], student=student,
                                  values=values, spread=max(values)-min(values)))
        for split in SPLITS:
            for held, train, test in make_folds(rows, split):
                foldpairs = pair_subset(pp, train, test)
                folds.append(dict(capability=channel[0], distribution=channel[1], split=split, held=held,
                                  positive_training_rows=len(train), positive_testing_rows=len(test),
                                  training_trajectories=len({p.run for p in train}),
                                  test_intervention_pairs=dict(Counter(p.kind for p in foldpairs)),
                                  training_intervention_pairs=dict(Counter(p.kind for p in pair_subset(pp,train))),
                                  fit_available=bool(train and test)))
    return dict(initial_loss=baselines, fold_eligibility=folds,
                n_data_pairs_with_different_initial_loss=sum(p.kind=="I_U" and p.first.initial_loss!=p.second.initial_loss for p in pairs),
                initial_loss_note="Stored effects remain differences of the canonical deltas. Their differing zero baselines are retained as a measurement limitation. An initial-loss-conditioned intervention holds the first endpoint's initial loss fixed.")


def corner_plan(points, intervals):
    # Prespecified design scale comes from the 1B U66/seed41 middle checkpoint;
    # new distinct-pool token totals are explicit requirements, not inferred U.
    source = next(p for p in points if p.student == "gemma3-1b" and p.rung == 66 and p.seed == 41 and p.T == 50563)
    d, t = source.D, source.T
    corners = [(t, 2*d), (t, d), (2*t, 4*d), (2*t, 2*d)]
    predicted = []
    for channel in intervals:
        cap, dist = channel["capability"], channel["distribution"]
        base = next(p for p in points if p.student == source.student and p.key[2:] == (cap, dist))
        queries = [replace(base, T=T, D=D) for T, D in corners]
        contrasts, values = {}, {}
        for name in STRUCTURES:
            f = channel["fits"][name]["fit"]
            s = f["standardizer"]
            scaler = Scaler(s["descriptor"], s["mean"], s["scale"], tuple(s["students"]))
            value = design(queries, name, scaler, f["p"] or 0) @ np.array(f["coefficients"])
            values[name] = value.tolist()
            contrasts[name] = float(value[3]-value[2]-value[1]+value[0])
        # Fixed 0.01 nat design threshold; used only for choosing a useful future
        # measurement, never for changing the headline/model-selection result.
        maximum = max(abs(a-b) for n in ("F_log", "F_curv") for a,b in zip(values[n], values["F_int"]))
        predicted.append(dict(capability=cap, distribution=dist, responses=values, second_differences=contrasts,
                              maximum_additive_interaction_disagreement=maximum, material=maximum >= .01))
    return dict(status="proposed only; no training performed", materiality_rule="maximum corner response disagreement >=0.01 native-token nat, fixed before evaluation",
                student=source.student, protocol=source.protocol, training_seed=0,
                source_checkpoint=list(source.key),
                corners=[dict(T=T, E=T/D, E_exact=str(Fraction(int(T), int(D))), D_U=D,
                              observed=any(p.student==source.student and p.T==T and p.D==D for p in points)) for T,D in corners],
                trajectories=[dict(U=66, D_U=d, pool_seed=41, run=source.run, checkpoints=[t], status="existing training-probe corner; scope QA at this checkpoint not recorded"),
                              dict(U=None, D_U=2*d, pool_seed=41, checkpoints=[t, 2*t], status="new distinct token-counted pool; U must be measured, never assumed to be 132"),
                              dict(U=None, D_U=4*d, pool_seed=41, checkpoints=[2*t], status="new distinct token-counted pool; U must be measured, never assumed to be 264")],
                exactness="Three trajectories total (one existing, two new). Exact supervised stops at 50563 and 101126; preserve the recorded fixed schedule horizon and recipe. Construct and verify distinct pools with exactly 33924/67848 supervised tokens. If these counts/stops cannot be realised without changing protocol, this design is infeasible and must be redesigned before collection; nominal U or ordinal checkpoints do not substitute.",
                measurement="Evaluate all four corners on the same distributions, including 2wiki_new; QA scope requires measurement even at the existing training checkpoint. Pool parentage must be recorded; existing seed labels do not prove nesting.",
                predictions=predicted, needed=any(r["material"] for r in predicted))


def protocol():
    return dict(T_ref=T_REF, p_range=[-1,3], p_initial_regions=P_BRACKETS, p_profile_grid=P_GRID.tolist(),
                coefficients="unconstrained signs", structures={"F_log": "(a+a_prime*z)*u+(b+b_prime*z)*v",
                "F_curv": "(a+a_prime*z)*u+(b+b_prime*z)*h_p(E)", "F_int": "(a+a_prime*z)*u+(b+b_prime*z)*v+k*u*v"},
                descriptors=DESCRIPTORS, lambda_grid=LAMBDAS, fixed_lambda=FIXED_LAMBDA,
                descriptor_calibration_cost="Log parameter count requires exact architecture metadata; initial_loss requires the target student's own recorded pretraining evaluation on that distribution. This is one baseline evaluation, not a post-training target calibration. Both use training-fold standardisation.",
                selection="Use >=3 usable inner folds of same grouping as outer split; select structure/descriptor/lambda by mean inner response MAE. Otherwise F_log/log_parameters/lambda=.01 are fixed. Each selected F predicts every target. All baseline surfaces share the selected descriptor/lambda.",
                standardisation="Equal training students then trajectories; log non-embedding count or own distribution-specific initial loss. Feature RMS learned in each fit, no column centering. RMS-normalised ridge minimises weighted MSE+lambda*sum(theta^2).",
                fit_weights="Positive responses only: equal trajectories, then equal budgets within trajectory; exact zero anchors imposed algebraically. 132 trajectories receive no extra weight from dense checkpoints.",
                intervention_weights="Equal student, physical pool pair, fixed A1 budget band, pair. Duplicate pair appearances across held groups share weight.",
                largest_budget="All T>=150000 withheld; fit only T<150000. QA scope has only final positive checkpoints and is unscorable here.",
                endpoints="Training pair calibration requires both endpoints in training. Evaluation needs at least one held endpoint, allowing a known reference in leave-rung/seed/budget folds. Both predictions always from same fitted F. The student's initial-loss descriptor is held at the first endpoint's recorded value in both predictions; small cross-run baseline discrepancies are audited, not interpreted as a data effect.",
                primary_targets=["code/training_probe:MBPP x I_T", "qa/2wiki_new x I_U; training_probe:2Wiki also reported separately"],
                secondary="Math, training-probe QA, MuSiQue and TriviaQA; all distributions from canonical DEVELOPMENT CSV only",
                I_U="Exact I_U has no observed positive matched-T pairs. Decision MAEs labelled 1% tolerance proxy use actual endpoints; fixed-T model predictions are stored but not scored against mismatched outcomes. 0.5%/5% separate sensitivity, never pooled.",
                I_T="Stored [1.7,2.3] budget ratios, approximate doubling; no exact 2x claim",
                error_interval="95% frozen-prediction trajectory multiplier intervals; same-run pair gets one weight, cross-run pair product. No refitting; shared pool seed dependence and baseline oracle selection not covered. Pointwise, not simultaneous.",
                new_configuration_prediction_intervals=dict(interval=None, status="not estimable with calibrated coverage",
                    reason="Only three students and four partly rung-confounded pool seeds; no independent calibration configurations. Frozen error and parameter intervals are not new-configuration prediction intervals."),
                external_diagnostic_file_read=False,
                previous_F_int="V96 source specifies 18 matrix2 trajectories, different selection/regularisation and no 132 rung. Those inputs/folds do not match A2; prior scores not read or recomputed. A2 is a reanalysis of overlapping measurements, not independent new evidence. Identical A2 fingerprints are reused on rerun.")


def baseline_inventory():
    shared = "Only training-fold canonical responses; no extra measured checkpoints"
    return [dict(name="zero", parameters=0, effective_df="0", fitting_unit="none", information="T=0 anchor", calibration_cost="none"),
            dict(name="constant", parameters=1, effective_df="1", fitting_unit="trajectory/budget weighted positive response",
                 information=shared, calibration_cost="one scalar from existing training responses",
                 intervention="identically zero for two positive budgets; at the zero anchor the step response is zero"),
            dict(name="budget_only", parameters=1, effective_df="trace ridge hat matrix, <=1; recorded per fold", fitting_unit="weighted response",
                 information="u only", calibration_cost=shared, intervention="fixed-T I_U exactly zero; unequal T can give mismatch-only effect"),
            dict(name="reuse_only", parameters=1, effective_df="trace ridge hat matrix, <=1; recorded per fold", fitting_unit="weighted response",
                 information="v only", calibration_cost=shared),
            dict(name="mean_effect", parameters="1 per intervention type; 2 in total, no response predictor", effective_df="1 per estimable mean, 0 when unavailable",
                 fitting_unit="training pairs, hierarchical student/pool/budget weights", information="observed nonzero signed budget-doubling or data-change effects separately",
                 calibration_cost="requires both existing endpoints of training pairs; no new measurements", unavailable="if no training pairs; fixed zero fallback explicitly marked"),
            dict(name="surface", parameters=7, effective_df="trace ridge hat matrix, <=7; recorded per fold", fitting_unit="weighted response",
                 information="fixed u,v,u^2,v^2,uv,zu,zv, exactly same inputs/scaler/lambda as F", calibration_cost=shared),
            dict(name="empirical_interpolation_diagnostic", parameters="one value per distinct training knot", effective_df="number of distinct knots",
                 fitting_unit="same-student response knots", information="training (u,v) closed convex hull, barycentric linear, no extrapolation or cross-student transfer",
                 calibration_cost="dense local responses already present; unsupported points abstain", scope="diagnostic only; MAE compared on identical supported subset; no structural exclusion")]


def decision_rows(results):
    out=[]
    for r in results:
        for target, m in r["metrics"].items():
            out.append(dict(capability=r["capability"], distribution=r["distribution"], target=target, split=r["split"],
                            primary_target=(r["capability"]=="code" and target=="I_T") or
                            (r["distribution"].startswith("2wiki_new:") and target=="I_U"),
                            estimand="1% tolerance proxy; exact I_U unavailable" if target=="I_U" else target, **m))
    return out


def fmt(x):
    return "NA" if x is None else f"{x:.5g}"


def ci(x):
    return "NA" if x is None else f"[{fmt(x[0])}, {fmt(x[1])}]"


def short_channel(s):
    return {"training_probe:MBPP": "code", "training_probe:MATH-500": "math",
            "training_probe:2WikiMultihopQA": "QA-probe", "2wiki_new": "QA-2Wiki", "musique": "QA-MuSiQue", "triviaqa": "QA-TriviaQA"}.get(s.rsplit(":",1)[0],s)


def decision_table(summary):
    lines=["| Distribution | Target | Split | Primary F | Strongest observed baseline | MAE F | MAE baseline | Paired gain [95% trajectory interval] |",
           "|---|---|---|---|---|---:|---:|---|"]
    for r in summary["decision_table"]:
        label = short_channel(r["distribution"])
        target = "I_U (1% proxy)" if r["target"]=="I_U" else r["target"]
        if r.get("primary_target"):
            label += " **primary**"
        if r["status"] != "scored":
            lines.append(f"| {label} | {target} | {r['split']} | NA | NA | NA | NA | unscorable |")
            continue
        names = ", ".join(sorted(r["primary_candidates"]))
        lines.append(f"| {label} | {target} | {r['split']} | {names} | {r['strongest_observed_baseline']} | {fmt(r['primary_mae'])} | {fmt(r['strongest_baseline_mae'])} | {fmt(r['improvement_over_strongest'])} {ci(r['interval_over_strongest'])} |")
    return "\n".join(lines)


def inventory_table(inventory):
    lines=["| Student | Trajectories | Positive checkpoints | Distinct T | T with ≥2 E | Actual four-corner rectangles |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in inventory["coverage"]:
        lines.append(f"| {r['student']} | {r['trajectories']} | {r['positive_checkpoints']} | {r['unique_T']} | {r['T_with_two_or_more_E']} | {r['exact_rectangles']} |")
    return "\n".join(lines)


def render(summary):
    lines=["# A2 curvature and interaction report", "", f"Status: **{summary['status']}**. CPU only; no training or new evaluation.", "",
           "The predeclared primary targets are code × budget intervention and new 2Wiki QA × data intervention. "
           "Exact positive-budget data interventions and T-by-E rectangles are absent. Tolerance-matched contrasts do not establish fixed-budget causality. "
           "These are reused development measurements, not an independent confirmation sample.", ""]
    if summary.get("blocker"):
        lines += [summary["blocker"], ""]
    lines += ["## Protocol", ""]
    for key, value in summary["protocol"].items():
        lines += [f"**{key}**: {value if isinstance(value,str) else canonical(value)}", ""]
    lines += ["## Canonical input limitations", "", summary["input_audit"]["initial_loss_note"], "",
              "| Student / distribution | Distinct recorded initial losses | Range |", "|---|---:|---:|"]
    for r in summary["input_audit"]["initial_loss"]:
        if r["spread"]:
            lines.append(f"| {r['student']} / {short_channel(r['distribution'])} | {len(r['values'])} | {fmt(r['spread'])} |")
    lines += ["", "Complete grouped-fold eligibility counts are saved in summary.json even when missing descriptors prevent fitting. No forbidden metadata or external diagnostic measurement file was opened.", ""]
    lines += ["## Baselines and calibration", "", "| Baseline | Parameters | Effective degrees of freedom | Fitting unit | Information | Calibration cost |", "|---|---|---|---|---|---|"]
    for b in summary["baselines"]:
        lines.append(f"| {b['name']} | {b['parameters']} | {b['effective_df']} | {b['fitting_unit']} | {b['information']} | {b['calibration_cost']} |")
    lines += ["", "The constant positive-budget response has identically zero differenced intervention. The signed mean-effect baseline is separately estimated for I_T and I_U; it is not forced to zero. "
              "F_log has four coefficients, F_curv four coefficients plus p, and F_int five coefficients. Conditional ridge EDF and the approximate additional p degree of freedom are saved per fold.", "",
              "## Decision table", "", "MAE in native-token nats. Positive gain favours the primary F. The strongest observed baseline is an outer-score oracle, shown as requested; its interval is conditional and does not include choosing the baseline. "
              "The separate inner-selected baseline comparisons below avoid that outer selection. Candidate identity always comes from inner responses, never the target outer score.", "", decision_table(summary), "",
              "## T-by-E coverage", "", inventory_table(summary["four_corners"]), "", summary["four_corners"]["structural_exclusion"], "",
              "A repeated (0,0) anchor is degenerate. Within each positive T there is only one E at fixed student; no horizontal two-E edge exists. Two pools at two approximately equal nominal budgets do not provide the required corners.", ""]
    for tr in summary["four_corners"]["trajectories"]:
        cps = "; ".join(f"{p['checkpoint']} T={p['T']} E={p['E_exact']}" for p in tr["checkpoints"])
        lines += [f"- `{tr['run']}`: U={tr['U']}, D_U={tr['D_U']}, seed={tr['pool_seed']}; {cps}"]
    if not summary.get("results"):
        lines += ["", "## Unavailable analyses", "",
                  "The absent exact non-embedding counts prevent the required inner comparison with the initial-loss alternative. No structure was fitted on invented counts or an unselected replacement descriptor. "
                  "Thus per-fold prediction errors, parameter stability, profile p, parameter intervals and model disagreement at missing corners are unavailable. "
                  "The conditional trigger for proposing new trajectories cannot yet be evaluated. The 72 decision rows explicitly remain unscored; this is not a completed A2 statistical decision.", ""]
        return "\n".join(lines)+"\n"
    lines += ["", "## Curvature stability and parameter intervals", "",
              "Fold p ranges are sensitivity summaries, not confidence intervals. Any boundary optimum is labelled curvature not identified. "
              "Parameter intervals refit coefficients and p under trajectory multipliers, conditional on descriptor/lambda and observed students. "
              "Frozen-prediction error intervals resample errors without refitting. New-configuration prediction intervals are unavailable; neither of the preceding interval types substitutes for them.", "",
              "| Distribution | p min / median / max | Boundary hits / folds | Leave-one-student p |", "|---|---|---|---|"]
    for r in summary["parameter_stability"]:
        d=r["p_distribution"]
        sens="; ".join(f"{x['held']}={fmt(x['p'])}{'*' if x['boundary'] else ''}" for x in r["leave_one_student"])
        lines.append(f"| {short_channel(r['distribution'])} | {fmt(d['minimum'])} / {fmt(d['median'])} / {fmt(d['maximum'])} | {d['boundary_hits']}/{d['n']} | {sens} |")
    for r in summary["parameter_intervals"]:
        f=r["fits"]["F_curv"]
        lines += ["", f"{short_channel(r['distribution'])}: full-development conditional parameter interval for p {ci(f['p_interval'])}; refit boundary fraction {fmt(f['bootstrap_boundary_fraction'])}. "
                   f"Full fit p={fmt(f['fit']['p'])}: {f['fit']['curvature_status']}. Coefficient intervals and complete profile objectives are in summary.json."]
    plan=summary["proposed_corner_measurements"]
    lines += ["", "## Missing-corner development design", "", plan["materiality_rule"]+".", "", plan["exactness"], "", plan["measurement"], "",
              "| Corner | T | E | D_U | Existing recorded training corner |", "|---|---:|---|---:|---|"]
    for i,c in enumerate(plan["corners"]):
        lines.append(f"| {i+1} | {c['T']} | {c['E_exact']} | {c['D_U']} | {c['observed']} |")
    lines += ["", "One 1B trajectory at D_U=16962 (existing U66, seed41), one new D_U=33924 trajectory at both budgets, and one new D_U=67848 trajectory at the higher budget cover the design. "
              "New pools' source-example counts cannot be inferred from token totals; they must be recorded after assembly. This is a concrete token-level requirement, not an assertion that existing named pools already satisfy it.", "",
              "| Distribution | Max additive/interaction corner disagreement | Material at 0.01 | F_int second difference |", "|---|---:|---|---:|"]
    for p in plan["predictions"]:
        lines.append(f"| {short_channel(p['distribution'])} | {fmt(p['maximum_additive_interaction_disagreement'])} | {p['material']} | {fmt(p['second_differences']['F_int'])} |")
    lines += ["", "Predictions in this design table are planning diagnostics, not imputed observations or structural evidence.", "",
              "## Per-fold decisions, inner-selected baselines and stability", "",
              "All coefficients, standardizers, training keys, predictor IDs, profile grids and per-pair endpoint predictions are retained in summary.json. The same predictor ID applies to both endpoints.", ""]
    for r in summary["results"]:
        lines += [f"### {short_channel(r['distribution'])} / {r['split']}", "", "| Held group | Target | Primary | Inner-selected baseline | MAE F | MAE baseline | Gain [frozen error interval] | p / boundary |", "|---|---|---|---|---:|---:|---|---|"]
        for fold in r["folds"]:
            if fold["status"] != "scored":
                lines.append(f"| {fold['held']} | all | NA | NA | NA | NA | {fold['status']} | NA |")
                continue
            c=fold["fits"]["F_curv"]
            for target,m in fold["metrics"].items():
                if m["status"] != "scored":
                    lines.append(f"| {fold['held']} | {target} | NA | NA | NA | NA | unscorable | {fmt(c['p'])}/{c['boundary_hit']} |")
                else:
                    lines.append(f"| {fold['held']} | {target} | {','.join(m['primary_candidates'])} | {','.join(m['inner_selected_baselines'])} | {fmt(m['primary_mae'])} | {fmt(m['baseline_mae'])} | {fmt(m['paired_improvement'])} {ci(m['frozen_prediction_error_interval'])} | {fmt(c['p'])}/{c['boundary_hit']} |")
        lines += ["", "Data-pair tolerance sensitivity (same frozen F, no retuning): " + "; ".join(f"{float(t)*100:g}% n={m['n']} F MAE={fmt(m.get('primary_mae'))}" for t,m in r["tolerance_sensitivity"].items())+".", ""]
        for f in r["folds"]:
            if f["status"]=="scored":
                s=f["selection"]; i=f["empirical_interpolation_diagnostic"]
                lines += [f"Held {f['held']}: {s['method']}; descriptor={s['descriptor']}, lambda={s['lambda_']}. "
                          f"Empirical interpolation support {i['n_supported']}/{i['n_requested']}, MAE={fmt(i.get('mae'))}, F MAE on identical support={fmt(i.get('primary_mae_same_support'))}.", ""]
    return "\n".join(lines)+"\n"


def write_outputs(summary, root=ROOT):
    root = Path(root)
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    report = root / "paper/docs/CURVATURE_INTERACTION_REPORT.md"
    require(report.parent.is_dir(), "Requested paper/docs directory is missing")
    markdown = render(summary)
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    (out/"summary.md").write_text(markdown)
    report.write_text(markdown)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonembedding-counts", help="JSON mapping student_id to exact non-embedding parameter count")
    parser.add_argument("--counts-provenance", default="user-supplied exact counts")
    parser.add_argument("--print-tables", action="store_true")
    parser.add_argument("--interval-draws", type=int, default=2000)
    parser.add_argument("--parameter-draws", type=int, default=200)
    args=parser.parse_args(argv)
    try:
        numerical_invariants()
        require(args.interval_draws >= 100 and args.parameter_draws >= 100, "At least 100 interval draws required")
        counts=json.loads(args.nonembedding_counts) if args.nonembedding_counts else None
        points,pairs,hashes,a1=load_inputs(counts=counts)
        geometry=four_corners(points)
        spec=protocol()
        summary=dict(schema_version="a2-curvature-interaction-v1", status="complete", device="cpu", training=False,
                     input_sha256=hashes, nonembedding_counts=counts, counts_provenance=args.counts_provenance,
                     protocol=spec, baselines=baseline_inventory(), four_corners=geometry, input_audit=input_audit(points,pairs),
                     inventory_counts=dict(rows=len(points), positive_rows=sum(p.T>0 for p in points), trajectories=len({p.run for p in points}),
                                           stored_core_pairs=len(pairs), exact_positive_I_U=sum(p.kind=="I_U" and p.first.T==p.second.T for p in pairs)),
                     results=[], decision_table=[])
        if counts is None:
            summary.update(status="blocked: required descriptor absent from permitted inputs", blocker=COUNTS_MISSING)
            for channel in sorted({p.key[2:] for p in points}):
                for split in SPLITS:
                    for target in ("response","I_T","I_U"):
                        summary["decision_table"].append(dict(capability=channel[0], distribution=channel[1], split=split, target=target,
                            primary_target=(channel[0]=="code" and target=="I_T") or (channel[1].startswith("2wiki_new:") and target=="I_U"),
                            status="blocked: counts missing", n=0, primary_candidates=None, strongest_observed_baseline=None,
                            primary_mae=None, strongest_baseline_mae=None, paired_improvement=None,
                            frozen_prediction_error_interval=None, parameter_stability=None))
            write_outputs(summary)
            print(decision_table(summary));print(inventory_table(geometry))
            raise ValueError(COUNTS_MISSING)
        fingerprint=sha(canonical([hashes, counts, spec, args.interval_draws, args.parameter_draws,
                                   sha(Path(__file__).read_bytes())]).encode())
        path=ROOT/OUT/"summary.json"
        if path.exists():
            prior=json.loads(path.read_text())
            if prior.get("fingerprint")==fingerprint and prior.get("status")=="complete":
                print("Reusing identical A2 inputs, protocol and folds, including F_int; no new evidence.")
                if args.print_tables:
                    print(decision_table(prior));print(inventory_table(prior["four_corners"]))
                return 0
        summary["fingerprint"]=fingerprint
        summary["results"]=evaluate(points,pairs,args.interval_draws)
        summary["decision_table"]=decision_rows(summary["results"])
        summary["parameter_stability"]=stability(summary["results"])
        summary["parameter_intervals"]=parameter_intervals(points,pairs,args.parameter_draws)
        summary["proposed_corner_measurements"]=corner_plan(points,summary["parameter_intervals"])
        # Detect concurrent canonical input edits before writing any scored result.
        for name, expected in hashes.items():
            require(sha((ROOT/"results/a1-development-table"/name).read_bytes())==expected, "Canonical input changed during analysis")
        write_outputs(summary)
        if args.print_tables:
            print(decision_table(summary)); print(inventory_table(geometry))
        return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"A2 failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
