#!/usr/bin/env python3
"""Audit S3 identity from recorded evidence; generate a table without model loading.

Only the explicit output files are written. The same file runs in the private
workspace (results/) and the public paper repository (data_mirror/, including
compressed JSON). No training, fitting, network access, or tensor imports occur.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

try:
    from .paper_table_layout import house_style
except ImportError:  # run as a script
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from paper_table_layout import house_style

import re

ROOT = Path(__file__).resolve().parents[1]
S3 = "s3-selection-validation"
FOUR_B_OUTCOME = (
    "v99-scope/rerun-4b/gpt-5.6-luna_full_66_matrix2_lora_dseed41-"
    "0e02d1102f506ae9/update-00000152.json"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def object_sha(value):
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return hashlib.sha256(text.encode()).hexdigest()


class Evidence:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.hashes = {}
        manifest = self.directory / "ANONYMIZATION_DIGESTS.json"
        self.pairs = json.loads(manifest.read_text())["files"] if manifest.is_file() else {}

    def raw(self, name):
        path = self.directory / name
        if path.is_file():
            raw = path.read_bytes()
        else:
            raw = gzip.decompress(Path(str(path) + ".gz").read_bytes())
        self.hashes[name] = hashlib.sha256(raw).hexdigest()
        return raw

    def read(self, name, sealed=False):
        raw = self.raw(name)
        if sealed:
            expected = self.raw(name + ".sha256").decode().strip()
            require(self.matches(name, expected), f"Artifact seal changed: {name}")
        return json.loads(raw)

    def matches(self, name, expected):
        """Accept only an explicitly pairable, path-bound public anonymization."""
        actual = self.hashes[name]
        return actual == expected or any(
            p.get("pairable") is True and p.get("frozen_sha256") == expected and
            p.get("published_sha256") == actual and
            p.get("published_at") in ("data_mirror/" + name, "data_mirror/" + name + ".gz")
            for p in self.pairs.values())


def registration_sha(plan):
    fixed = copy.deepcopy(plan)
    for key in ("status", "launch_ready", "blockers", "predictions", "maps",
                "predictions_sha256", "maps_sha256", "identity_verification",
                "final_freeze", "registration_sha256"):
        fixed.pop(key, None)
    for ref in fixed["references"]:
        ref.pop("dense", None)
        ref.pop("eligibility", None)
        ref["student"].pop("dense", None)
    return object_sha(fixed)


def historical_evidence(evidence):
    """Separate pinned weight records from older outcomes lacking a commit field."""
    history = {}
    for tag in ("gemma3-270m", "gemma3-1b"):
        path = f"v32-descriptors/{tag}/features.json"
        record = evidence.read(path)
        history[record["hf_id"]] = dict(commit=record["revision"],
            initial_evidence=[path], exact_outcome_evidence=[],
            outcome_limit="Earlier V6/V10/V12 files name the release but omit the weight commit.")
    path = "v71-qa-scope/measurements.json"
    record = evidence.read(path)
    h = history[record["register"]["hf_id"]]
    require(record["register"]["revision"] == h["commit"], "V71 snapshot changed")
    require(record["status"] == "complete" and any(
        r["state"] == "rtn_b4_g128" for r in record["measurements"]), "Missing V71 outcome")
    h["exact_outcome_evidence"].append(path)
    record = evidence.read(FOUR_B_OUTCOME)
    require(record["status"] == "complete" and record["adapter_loaded"], "Missing V99 outcome")
    # This is the model path actually loaded for evaluation, not the tokenizer path.
    commit = Path(record["model_local_path"]).name
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "V99 has no resolved base snapshot")
    history[record["resolved_student"]] = dict(commit=commit,
        initial_evidence=[FOUR_B_OUTCOME], exact_outcome_evidence=[FOUR_B_OUTCOME],
        outcome_limit="V99 pins the evaluation base; adapter_base_unrecorded is true, so "
                      "it does not independently pin the original adapter training base.")
    for model, h in history.items():
        tag = {"google/gemma-3-270m": "gemma3-270m", "google/gemma-3-1b-pt": "gemma3-1b",
               "google/gemma-3-4b-pt": "gemma3-4b"}[model]
        paths = [f"v6-capability-geometry/{tag}/prune_losses.json",
                 f"v10-quantization/{tag}/quant_losses.json",
                 f"v12-distill/{tag}/gpt-5.6-luna_full_600/eval.json"]
        prune, channel, distill = [evidence.read(p) for p in paths]
        require(distill["resolved_student"] == model and distill["post_training"], "Wrong V12 release")
        h.update(release_outcome_evidence=paths,
                 earlier_densities=sorted(float(k) for k in prune if not k.startswith("_")),
                 earlier_bits=sorted(int(k) for k in channel if k.isdigit()))
    return history


def fit_provenance(evidence, locked):
    v78 = evidence.read("v78-rule-confirm/freeze-independent.json", sealed=True)
    require(v78["models"]["locked"] == locked, "S3 predictor differs from V78 models.locked")
    p = evidence.read("v53-prune-dev/register.json")
    g = evidence.read("v69-quant-confirm/develop.json")
    old = evidence.read("v64-selection-feasible/summary.json")
    for name in ("v53-prune-dev/register.json", "v69-quant-confirm/develop.json",
                 "v64-selection-feasible/summary.json"):
        expected = v78["models"]["input_sha256"]["results/" + name]
        require(evidence.matches(name, expected), f"V78 development artifact changed: {name}")
    require(locked["prune"] == {k: p[k] for k in ("standardization", "models")}, "V53 fit changed")
    require(locked["grouped"] == {k: g[k] for k in ("standardization", "models")}, "V69 fit changed")
    excluded = locked["distill"]["excluded_students"]
    return dict(
        pruning=dict(artifact="v53-prune-dev/register.json", measurements="v6-capability-geometry",
                     states=[s["tag"] for s in p["dev_states"]], rows=p["n_dev_rows"]),
        per_channel=dict(artifact="v64-selection-feasible/summary.json", measurements="v10-quantization",
            states_by_bit={str(b): [s["tag"] for s in old["states"] if any(
                q["law"] == "quant_channel" and q["bit"] == b for q in s["configs"])]
                for b in (3, 4, 5, 6, 8)}),
        grouped=dict(artifact="v69-quant-confirm/develop.json", measurements="v54-quant-group",
            states=sorted({r["state"] for r in g["dev_rows"]}),
            configurations=sorted({r["config"] for r in g["dev_rows"]}), rows=len(g["dev_rows"])),
        distillation=dict(artifact="v64-selection-feasible/summary.json", analysis="v39-distill-controlled",
            measurements="v12-distill", excluded_students=excluded,
            states=sorted(s["tag"] for s in old["students"].values() if s["tag"] not in excluded),
            files=[s["path"] for s in old["students"].values() if s["tag"] not in excluded],
            constants=locked["distill"]["constant"]),
        confirmation_states=[s["tag"] for s in v78["states"]])


def audit(directory):
    evidence = Evidence(directory)
    plan = evidence.read(f"{S3}/plan_v2.json", sealed=True)
    require(plan["status"] == "FROZEN_BEFORE_OUTCOMES", "S3 is not frozen")
    require(plan["registration_sha256"] == registration_sha(plan), "Registration changed")
    for name in ("predictions", "maps"):
        require(object_sha(plan[name]) == plan[name + "_sha256"], f"{name} seal changed")
    registered = evidence.read(f"{S3}/plan_v2.registered.json", sealed=True)
    require(registration_sha(registered) == plan["registration_sha256"], "Registered contract changed")
    identity = evidence.read(f"{S3}/identity.json", sealed=True)
    require(identity["status"] == "PASS" and len(identity["historical"]) == 30 and
            len(identity["comparisons"]) == 180, "Unexpected identity audit scope")
    require(all("pythia-" in k for k in identity["historical"]), "Reassess historical audit scope")
    for name, expected in plan["final_freeze"]["artifact_sha256"].items():
        name = name.removeprefix("results/")
        evidence.raw(name)
        require(evidence.matches(name, expected), f"Freeze artifact changed: {name}")
    for name, expected in plan["preserved_sha256"].items():
        name = name.removeprefix("results/")
        if name.endswith("PRINTED_PREPARATION.md"):
            continue  # Operational printout is outside this audit and the public mirror.
        evidence.raw(name)
        require(evidence.matches(name, expected), f"Preserved artifact changed: {name}")
    amendment = f"{S3}/prereg_amendment_1.md"
    require(hashlib.sha256(evidence.raw(amendment)).hexdigest() ==
            plan["runtime_sha256"]["results/" + amendment], "Amendment changed")
    history = historical_evidence(evidence)
    fits = fit_provenance(evidence, evidence.read(f"{S3}/inputs/locked_models.json"))
    rows = []
    for index, ref in enumerate(plan["references"], 1):
        for role in ("reference", "student"):
            spec = ref if role == "reference" else ref[role]
            key = spec["model"] + "@" + spec["revision"]
            target = identity["targets"][key]
            require(target["commit"] == plan["identity_verification"][ref["id"]][role]["commit"],
                    "Resolved commit disagrees with frozen identity")
            # Vision matrices, embeddings, output heads, biases and norms are excluded.
            count = sum(v["shape"][0] * v["shape"][1] for n, v in target["learned_tensors"].items()
                        if re.match(r"^(model|gpt_neox)\.layers\.\d+\.", n) and len(v["shape"]) == 2)
            require(count == spec["N0"], "Storage parameter count disagrees with tensor shapes")
            prior = history.get(spec["model"])
            require(prior is None or prior["commit"] == target["commit"], "Historical snapshot differs")
            anchor = evidence.read(f"{S3}/anchors/{ref['id']}__{role}.json", sealed=True)
            require(anchor["commit"] == target["commit"] and anchor["pristine"] and
                    not anchor["training_performed"] and not anchor["compression_applied"], "Wrong anchor")
            rows.append(dict(reference_number=index, reference=ref["id"], role=role,
                model=spec["model"], requested_revision=spec["revision"], commit=target["commit"],
                release="Pretrained base", matrix_parameters=count, initial_weights_seen_before=bool(prior),
                prior=prior, new_run=True, new_source_state=prior is None,
                anchor_job=anchor["slurm_job_id"]))
    require(len(rows) == 8 and len({(r["model"], r["commit"]) for r in rows}) == 6,
            "Expected eight roles and six unique snapshots")
    # A fresh run is attested separately from whether its initial weights were seen.
    for ref in plan["references"]:
        for candidate in ref["candidates"]:
            name = f"{S3}/measurements/{ref['id']}/{candidate['id'].replace(':', '__')}.json"
            if candidate["method"] == "distill":
                name = f"{S3}/students/{ref['id']}/measurement.json"
            record = evidence.read(name, sealed=True)
            require(record["fresh_s3_run"] and record["reference"] == ref["id"] and
                    record["candidate"] == candidate["id"] and
                    evidence.matches(f"{S3}/plan_v2.json", record["plan_sha256"]), "Unbound S3 outcome")
            if candidate["method"] == "distill":
                name = record["eval_path"].removeprefix("results/")
                require(name.startswith(f"{S3}/students/{ref['id']}/"), "Historical student reused")
                evaluation = evidence.read(name)
                require(evidence.matches(name, record["eval_sha256"]) and
                        evaluation["resolved_student"] == ref["student"]["model"] and
                        evaluation["revision"] == ref["student"]["revision"], "Student evidence changed")
    return dict(rows=rows, fits=fits, freeze=dict(
        timestamp_utc=plan["final_freeze"]["timestamp_utc"],
        identity_timestamp_utc=None, anchor_timestamps_utc=None,
        timestamp_limit="Identity and anchor JSON bodies contain no UTC timestamps.",
        registration_sha256=plan["registration_sha256"], predictions_sha256=plan["predictions_sha256"],
        maps_sha256=plan["maps_sha256"]), evidence_sha256=evidence.hashes)


def code(text):
    """Allow line breaks in exact identifiers without shortening or altering them."""
    return r"\texttt{" + r"\allowbreak{}".join(text[i:i+10] for i in range(0, len(text), 10)) + "}"


PRETTY = {
    "EleutherAI/pythia-410m": "Pythia 410M",
    "EleutherAI/pythia-160m": "Pythia 160M",
    "EleutherAI/pythia-1.4b": "Pythia 1.4B",
    "google/gemma-3-1b-pt": "Gemma 3, 1B",
    "google/gemma-3-270m": "Gemma 3, 270M",
    "google/gemma-3-4b-pt": "Gemma 3, 4B",
}


def parameters(count):
    """Matrix parameters in the units a reader thinks in."""
    return f"{count/1e9:.2f}B" if count >= 1e9 else f"{count/1e6:.0f}M"


def render(data):
    return house_style(_render(data))


def _render(data):
    caption = (
        "Identity of the eight roles in the selection validation. All are pretrained base releases, "
        "none instruction tuned, and the eight roles draw on six distinct snapshots because two "
        "serve twice. The two Pythia states appear nowhere in the registered records; the Gemma "
        "releases were measured in earlier development, so for them this round is a new set of "
        "candidates on familiar weights. Every candidate in the round is a fresh run, and no "
        "measurement on any of these weights entered the fitted selection rule. Parameter counts "
        "cover the text-decoder matrices. Exact revisions and commit identifiers, the evidence for "
        "each entry, and the freeze timestamps are recorded in the repository audit.")
    lines = ["% Generated by analysis/s3_identity_table.py; do not edit.",
             r"\begin{table}[tb]", r"\centering\footnotesize",
             "\\caption{" + caption + "}", r"\label{tab:s3-identity}",
             r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.12}"]
    widths = (.16, .21, .12, .17, .30)
    columns = "".join(r">{\raggedright\arraybackslash}p{\dimexpr " +
                      f"{w:.2f}" + r"\textwidth-" + f"{8*w:.2f}" + r"\tabcolsep\relax}"
                      for w in widths)
    lines += [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + columns + "@{}}",
              r"\toprule",
              r"Role & Model and stage & Matrix parameters & Weights seen in earlier work & "
              r"What is new in this round \\", r"\midrule"]
    for i, row in enumerate(data["rows"]):
        role = ("Reference " if row["role"] == "reference" else "Student for reference ") + str(row["reference_number"])
        stage = PRETTY.get(row["model"], row["model"])
        if row["requested_revision"] == "step120000":
            stage += ", training step 120000"
        if row["initial_weights_seen_before"]:
            seen = "Yes, in earlier development"
            new = ("A new set of candidates on familiar weights" if row["role"] == "reference"
                   else "A student trained again from familiar weights")
        else:
            seen = "No"
            new = ("A model state with no earlier recorded use, compressed here for the first time"
                   if row["role"] == "reference" else "A base with no earlier recorded use, trained here for the first time")
        lines.append(" & ".join((role, stage, parameters(row["matrix_parameters"]), seen, new)) + r" \\")
        if i % 2 == 1 and i != len(data["rows"]) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path,
                        default=ROOT / ("data_mirror" if (ROOT / "data_mirror").is_dir() else "results"))
    paper = ROOT / "paper" if (ROOT / "data_mirror").is_dir() else ROOT / "paper/paper"
    parser.add_argument("--output", type=Path, default=paper / "tables/s3_identity.tex")
    parser.add_argument("--audit-output", type=Path, default=ROOT / "docs/s3_identity_audit.json")
    args = parser.parse_args()
    # Never allow output to be redirected into the evidence tree.
    for path in (args.output, args.audit_output):
        require(not path.resolve().is_relative_to(args.evidence_root.resolve()), "Evidence is read-only")
        require(not path.resolve().is_relative_to((ROOT / "results").resolve()), "Results are read-only")
    data = audit(args.evidence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data))
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output} and {args.audit_output}")


if __name__ == "__main__":
    main()
