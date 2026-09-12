"""The Round v7 locked rule. Prediction only: no files, fitting, or outcomes.

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
"""
from __future__ import annotations

import math
import numpy as np

from analysis import v53_prune_dev as prune
from analysis import v55_quant_group_fit as group
from analysis import v69_quant_confirm as grouped
from analysis import v36_pythia_controlled_fit as channel
from analysis import v39_distill_controlled as distill

CAPS = ("math", "code", "qa")
NEW = ("new_state", "new_size", "new_stage", "new_source")


def channel_delta(fit, inputs):
    """Evaluate v36's feature-major per-bit OLS without mutating CONFIGS.

    v64 extends the vocabulary to five bits. Selecting its coefficient column
    is algebraically identical to v36.predict's one-hot design matrix.
    """
    order = fit["config_order"]
    bit = inputs["bit"]
    if bit not in order:
        raise ValueError(f"No development anchor for bit {bit}")
    public = {k: inputs[k] for k in fit["input_fields"]}
    public["config"] = bit
    raw = channel.covariates([public], input_fields=fit["input_fields"])[0]
    z = np.r_[1., (raw - np.asarray(fit["center"])) / np.asarray(fit["scale"])]
    beta = np.asarray(fit["coefficients"]).reshape(len(z), len(order))
    return float(z @ beta[:, order.index(bit)])


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
