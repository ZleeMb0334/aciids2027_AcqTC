"""Deterministic analyses added during the final ACIIDS review cycle.

This script uses preserved aggregate results. It performs no model inference.
It reports shortcut-cutoff sensitivity and the existing fusion-form ablation.
"""
import argparse
import glob
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.dirname(HERE)
PROJECT = os.path.dirname(ARTIFACT)


def load(path):
    with open(path) as f:
        return json.load(f)


def collect_runs(results):
    """Keep one row per method and seed; targeted reruns replace older rows."""
    runs = {}
    patterns = ("fs__s*.json", "baselines__s*.json", "gatefull__s*.json",
                "semtc__s*.json")
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(results, pattern))):
            for row in load(path):
                runs[(row["method"], int(row.get("seed", 0)))] = row
    return runs


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "n": int(len(values)),
    }


def cutoff_analysis(results, runs):
    audit = load(os.path.join(results, "benchmark_audit.json"))
    fs = {
        row["subset"].split("/", 1)[1]: float(row["shortcut_auroc"])
        for row in audit if row["subset"].startswith("ForenSynths/")
    }
    out = []
    for cutoff in (0.60, 0.70, 0.80):
        retained = sorted(name for name, auc in fs.items() if auc < cutoff)
        unseen = [name for name in retained if name != "progan"]
        methods = {}
        for method in ("semantic", "semantic_tc"):
            seed_values = []
            for (name, seed), row in sorted(runs.items()):
                if name != method:
                    continue
                per = row.get("per_generator", {})
                if unseen and all(g in per for g in unseen):
                    seed_values.append(float(np.mean([per[g]["auroc"] for g in unseen])))
            methods[method] = mean_sd(seed_values) if seed_values else None
        delta = None
        if methods["semantic"] and methods["semantic_tc"]:
            delta = methods["semantic_tc"]["mean"] - methods["semantic"]["mean"]
        out.append({
            "cutoff": cutoff,
            "retained": retained,
            "retained_count": len(retained),
            "unseen": unseen,
            "unseen_count": len(unseen),
            "methods": methods,
            "semantic_tc_minus_semantic": delta,
        })
    return out


def fusion_analysis(results, runs):
    ablations = {}
    for path in sorted(glob.glob(os.path.join(results, "abl__s*.json"))):
        for row in load(path):
            ablations[(row["method"], int(row.get("seed", 0)))] = row
    for (method, seed), row in runs.items():
        if method == "gate":
            ablations[(method, seed)] = row
    rows = []
    for method in ("mean", "static", "concat", "gate"):
        selected = [row for (name, _), row in sorted(ablations.items()) if name == method]
        if not selected:
            continue
        rows.append({
            "method": method,
            "unseen_auroc": mean_sd([row["auroc_unseen_macro"] for row in selected]),
            "jpeg70_auroc": mean_sd([row["a4_jpeg70_auroc"] for row in selected]),
            "resize_auroc": mean_sd([row["a4_resize_auroc"] for row in selected]),
        })
    return rows


def fmt_ms(stat):
    if stat["sd"] is None:
        return f'{stat["mean"]:.3f}'
    return f'{stat["mean"]:.3f}$\\pm${stat["sd"]:.3f}'


def write_tables(out, cutoff, fusion):
    os.makedirs(out, exist_ok=True)
    lines = [
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Cutoff & Retained & Unseen & Semantic & \method{} \\",
        r"\midrule",
    ]
    for row in cutoff:
        lines.append(
            f'{row["cutoff"]:.2f} & {row["retained_count"]}/11 & '
            f'{row["unseen_count"]} & {fmt_ms(row["methods"]["semantic"])} & '
            f'{fmt_ms(row["methods"]["semantic_tc"])} \\\\'
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "tab_cutoff_sensitivity.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    labels = {
        "mean": "Logit mean",
        "static": "Static mixture",
        "concat": "Concatenation",
        "gate": "Sample gate",
    }
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Fusion & Unseen & JPEG 70 & Resize \\",
        r"\midrule",
    ]
    for row in fusion:
        lines.append(
            f'{labels[row["method"]]} & {fmt_ms(row["unseen_auroc"])} & '
            f'{fmt_ms(row["jpeg70_auroc"])} & {fmt_ms(row["resize_auroc"])} \\\\'
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, "tab_fusion_forms.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=os.path.join(ARTIFACT, "results"))
    parser.add_argument("--generated", default=os.path.join(PROJECT, "manuscript", "generated"))
    args = parser.parse_args()

    runs = collect_runs(args.results)
    cutoff = cutoff_analysis(args.results, runs)
    fusion = fusion_analysis(args.results, runs)
    output = {"cutoff_sensitivity": cutoff, "fusion_forms": fusion}
    outpath = os.path.join(args.results, "review_analysis.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    write_tables(args.generated, cutoff, fusion)
    print(json.dumps(output, indent=2))
    print(f"[review-analysis] wrote {outpath}")


if __name__ == "__main__":
    main()
