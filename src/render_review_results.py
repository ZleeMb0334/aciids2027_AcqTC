"""Render the experiments requested by the final ACIIDS review.

The script fails when an expected result is absent.  This prevents a partial
GPU run from being represented as a completed experiment in the manuscript.
"""
import glob
import json
import os

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.dirname(HERE)
PROJECT = os.path.dirname(ARTIFACT)
RESULTS = os.path.join(ARTIFACT, "results")
GENERATED = os.path.join(PROJECT, "manuscript", "generated")
MACROS = {}
ROW_END = " " + chr(92) * 2


def read(name):
    path = os.path.join(RESULTS, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as stream:
        return json.load(stream)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("empty measurement")
    return float(values.mean()), (float(values.std(ddof=1)) if len(values) > 1 else None)


def fmt(values, decimals=3):
    mean, sd = mean_sd(values)
    if sd is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}" + r"\sd{" + f"{sd:.{decimals}f}" + r"}"


def macro(name, value):
    MACROS[name] = value


def existing_runs():
    runs = {}
    for pattern in ("fs__s*.json", "gatefull__s*.json"):
        for path in sorted(glob.glob(os.path.join(RESULTS, pattern))):
            with open(path) as stream:
                rows = json.load(stream)
            for row in rows:
                runs[(row["method"], int(row["seed"]))] = row
    return runs


def transform_control():
    runs = existing_runs()
    for path in sorted(glob.glob(os.path.join(RESULTS, "semtc__s*.json"))):
        with open(path) as stream:
            rows = json.load(stream)
        for row in rows:
            runs[(row["method"], int(row["seed"]))] = row
    labels = {
        "semantic": "Semantic baseline",
        "semantic_tc": r"\textbf{\method{} (ours)}",
        "gate_full": r"\fusionmethod{}",
    }
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             "Model & Unseen & JPEG 70 & Resize" + ROW_END, r"\midrule"]
    for method in ("semantic", "gate_full", "semantic_tc"):
        rows = [row for (name, _), row in sorted(runs.items()) if name == method]
        if len(rows) != 3:
            raise ValueError(f"expected three {method} rows, found {len(rows)}")
        unseen = [row["auroc_unseen_macro"] for row in rows]
        jpeg = [row["a4_jpeg70_auroc"] for row in rows]
        resize = [row["a4_resize_auroc"] for row in rows]
        lines.append(f"{labels[method]} & {fmt(unseen)} & {fmt(jpeg)} & {fmt(resize)}" + ROW_END)
        tag = {"semantic": "Semantic", "semantic_tc": "AcqTc",
               "gate_full": "GateFull"}[method]
        macro(tag + "ReviewUnseen", fmt(unseen))
        macro(tag + "ReviewJpeg", fmt(jpeg))
        macro(tag + "ReviewResize", fmt(resize))
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(GENERATED, "tab_transform_control.tex"), "w") as stream:
        stream.write("\n".join(lines) + "\n")
    for key, tag in (("auroc_unseen_macro", "Unseen"),
                     ("a4_jpeg70_auroc", "Jpeg"),
                     ("a4_resize_auroc", "Resize")):
        differences = [
            runs[("semantic_tc", seed)][key] - runs[("gate_full", seed)][key]
            for seed in (0, 1, 2)
        ]
        mean, sd = mean_sd(differences)
        macro("SemanticTcMinusGateFull" + tag, f"{mean:+.4f}")
        macro("SemanticTcMinusGateFull" + tag + "Sd", f"{sd:.4f}")
        baseline_differences = [
            runs[("semantic_tc", seed)][key] - runs[("semantic", seed)][key]
            for seed in (0, 1, 2)
        ]
        base_mean, base_sd = mean_sd(baseline_differences)
        macro("AcqTcMinusSemantic" + tag, f"{base_mean:+.4f}")
        macro("AcqTcMinusSemantic" + tag + "Sd", f"{base_sd:.4f}")


def objective_ablation():
    paths = sorted(glob.glob(os.path.join(RESULTS, "objabl__s*.json")))
    if not paths:
        return
    runs = existing_runs()
    for path in sorted(glob.glob(os.path.join(RESULTS, "semtc__s*.json"))) + paths:
        with open(path) as stream:
            for row in json.load(stream):
                runs[(row["method"], int(row["seed"]))] = row
    labels = {
        "semantic": "Clean objective",
        "semantic_aug": "Transformed-view classification",
        "semantic_align": "Logit alignment",
        "semantic_tc": r"\textbf{Both (\method{})}",
    }
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             "Objective & Unseen & JPEG 70 & Resize" + ROW_END, r"\midrule"]
    measurements = {}
    for method in ("semantic", "semantic_aug", "semantic_align", "semantic_tc"):
        rows = [row for (name, _), row in sorted(runs.items()) if name == method]
        if len(rows) != 3:
            raise ValueError(f"expected three {method} rows, found {len(rows)}")
        unseen = [row["auroc_unseen_macro"] for row in rows]
        jpeg = [row["a4_jpeg70_auroc"] for row in rows]
        resize = [row["a4_resize_auroc"] for row in rows]
        measurements[method] = {"Unseen": unseen, "Jpeg": jpeg, "Resize": resize}
        lines.append(f"{labels[method]} & {fmt(unseen)} & {fmt(jpeg)} & {fmt(resize)}" + ROW_END)
        tag = {"semantic": "CleanOnly", "semantic_aug": "ViewClass",
               "semantic_align": "LogitAlign", "semantic_tc": "BothObjective"}[method]
        macro(tag + "Unseen", fmt(unseen))
        macro(tag + "Jpeg", fmt(jpeg))
        macro(tag + "Resize", fmt(resize))
    for component, tag in (("semantic_aug", "ViewClass"),
                           ("semantic_align", "LogitAlign")):
        for metric in ("Unseen", "Jpeg", "Resize"):
            delta = (np.asarray(measurements["semantic_tc"][metric]) -
                     np.asarray(measurements[component][metric]))
            mean, sd = mean_sd(delta)
            macro("AcqTcMinus" + tag + metric,
                  f"{mean:+.3f}" + r"$\pm$" + f"{sd:.3f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(GENERATED, "tab_objective_ablation.tex"), "w") as stream:
        stream.write("\n".join(lines) + "\n")


def modern_panel():
    result = read("modern_genimage.json")
    result_l = read("modern_genimage_l14.json")
    summary = result["summary"]
    summary_l = result_l["summary"]
    sources = [name for name in summary["semantic"] if name != "macro"]
    if len(sources) != 8:
        raise ValueError(f"expected eight GenImage sources, found {len(sources)}")
    names = {
        "stable_diffusion_v_1_4": "Stable Diffusion 1.4",
        "stable_diffusion_v_1_5": "Stable Diffusion 1.5",
        "Midjourney": "Midjourney",
        "BigGAN": "BigGAN",
        "glide": "GLIDE",
        "VQDM": "VQDM",
        "wukong": "Wukong",
        "ADM": "ADM",
    }
    audit = result["canonical_shortcut_auroc_from_byte_size"]
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Source & Byte-only & \method{}-B/16 & Semantic-L/14 & \method{}-L/14" + ROW_END,
             r"\midrule"]
    for source in sources:
        semantic_tc = summary["semantic_tc"][source]["auroc"]
        semantic_l = summary_l["semantic_l"][source]["auroc"]
        semantic_tc_l = summary_l["semantic_tc_l"][source]["auroc"]
        label = names.get(source, source.replace("_", r"\_"))
        lines.append(f"{label} & {audit[source]:.3f} & "
                     f"{semantic_tc['mean']:.3f}" + r"$\pm$" + f"{semantic_tc['sd']:.3f} & "
                     f"{semantic_l['mean']:.3f}" + r"$\pm$" + f"{semantic_l['sd']:.3f} & "
                     f"{semantic_tc_l['mean']:.3f}" + r"$\pm$" + f"{semantic_tc_l['sd']:.3f}" + ROW_END)
    lines += [r"\midrule"]
    semantic_tc_macro = summary["semantic_tc"]["macro"]
    semantic_l_macro = summary_l["semantic_l"]["macro"]
    semantic_tc_l_macro = summary_l["semantic_tc_l"]["macro"]
    lines.append("Macro & -- & "
                 f"{semantic_tc_macro['mean']:.3f}" + r"$\pm$" + f"{semantic_tc_macro['sd']:.3f} & "
                 f"{semantic_l_macro['mean']:.3f}" + r"$\pm$" + f"{semantic_l_macro['sd']:.3f} & "
                 f"{semantic_tc_l_macro['mean']:.3f}" + r"$\pm$" + f"{semantic_tc_l_macro['sd']:.3f}" + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(GENERATED, "tab_modern_generators.tex"), "w") as stream:
        stream.write("\n".join(lines) + "\n")
    for method, tag in (("semantic", "ModernSemantic"),
                        ("semantic_tc", "ModernSemanticTc"),
                        ("gate_full", "ModernGateFull")):
        stat = summary[method]["macro"]
        macro(tag, f"{stat['mean']:.3f}" + r"$\pm$" + f"{stat['sd']:.3f}")
    for method, tag in (("semantic_l", "ModernSemanticL"),
                        ("semantic_tc_l", "ModernAcqTcL")):
        stat = summary_l[method]["macro"]
        macro(tag, f"{stat['mean']:.3f}" + r"$\pm$" + f"{stat['sd']:.3f}")
        macro(tag + "Mean", f"{stat['mean']:.3f}")
    macro("ModernAcqTcLMinusSemanticL",
          f"{summary_l['semantic_tc_l']['macro']['mean'] - summary_l['semantic_l']['macro']['mean']:+.4f}")
    paired = []
    for seed in (0, 1, 2):
        ours = [row["fused"]["auroc"] for row in result_l["rows"]
                if row["method"] == "semantic_tc_l" and row["seed"] == seed
                and row["condition"] == "clean"]
        control = [row["fused"]["auroc"] for row in result_l["rows"]
                   if row["method"] == "semantic_l" and row["seed"] == seed
                   and row["condition"] == "clean"]
        if len(ours) != 8 or len(control) != 8:
            raise ValueError(f"expected eight L/14 modern rows for seed {seed}")
        paired.append(float(np.mean(ours) - np.mean(control)))
    macro("ModernAcqTcLMinusSemanticLMin", f"{min(paired):+.4f}")
    macro("ModernAcqTcLMinusSemanticLMax", f"{max(paired):+.4f}")
    macro("ModernDelta", f"{summary['gate_full']['macro']['mean'] - summary['semantic']['macro']['mean']:+.4f}")
    macro("ModernGateFullMinusTc",
          f"{summary['gate_full']['macro']['mean'] - summary['semantic_tc']['macro']['mean']:+.4f}")
    macro("ModernHeaderMin", f"{min(audit.values()):.3f}")
    macro("ModernHeaderMax", f"{max(audit.values()):.3f}")
    macro("ModernSourceN", str(len(sources)))
    for method, method_tag in (("semantic", "ModernSemantic"),
                               ("semantic_tc", "ModernSemanticTc"),
                               ("gate_full", "ModernGateFull")):
        for condition, condition_tag in (("jpeg70", "Jpeg"),
                                         ("resize_half", "Resize")):
            stat = result["robustness"][method][condition]["macro"]
            macro(method_tag + condition_tag,
                  f"{stat['mean']:.3f}" + r"$\pm$" + f"{stat['sd']:.3f}")
    for method, method_tag in (("semantic_l", "ModernSemanticL"),
                               ("semantic_tc_l", "ModernAcqTcL")):
        for condition, condition_tag in (("jpeg70", "Jpeg"),
                                         ("resize_half", "Resize")):
            stat = result_l["robustness"][method][condition]["macro"]
            macro(method_tag + condition_tag,
                  f"{stat['mean']:.3f}" + r"$\pm$" + f"{stat['sd']:.3f}")


def gate_diagnostics():
    result = read("gate_diagnostics.json")
    rows = result["rows"]
    labels = {"clean": "Clean", "jpeg70": "JPEG 70", "resize_half": "Resize"}
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Input & Fused & Semantic & Spectral & $\alpha_{\rm real}$ & $\alpha_{\rm fake}$" + ROW_END,
             r"\midrule"]
    for condition in ("clean", "jpeg70", "resize_half"):
        selected = [row for row in rows if row["condition"] == condition]
        if len(selected) != 3:
            raise ValueError(f"expected three {condition} rows, found {len(selected)}")
        fused = [row["fused"]["auroc"] for row in selected]
        semantic = [row["semantic"]["auroc"] for row in selected]
        spectral = [row["spectral"]["auroc"] for row in selected]
        real = [row["gate_real_mean"] for row in selected]
        fake = [row["gate_fake_mean"] for row in selected]
        lines.append(f"{labels[condition]} & {fmt(fused)} & {fmt(semantic)} & "
                     f"{fmt(spectral)} & {fmt(real, 4)} & {fmt(fake, 4)}" + ROW_END)
        tag = {"clean": "Clean", "jpeg70": "Jpeg", "resize_half": "Resize"}[condition]
        macro("GateAlpha" + tag + "Real", fmt(real, 4))
        macro("GateAlpha" + tag + "Fake", fmt(fake, 4))
    lines += [r"\bottomrule", r"\end{tabular}"]
    condition_means = []
    spectral_means = []
    for condition in ("clean", "jpeg70", "resize_half"):
        selected = [row for row in rows if row["condition"] == condition]
        condition_means.extend([
            np.mean([row["gate_real_mean"] for row in selected]),
            np.mean([row["gate_fake_mean"] for row in selected]),
        ])
        spectral_means.append(np.mean([row["spectral"]["auroc"] for row in selected]))
    macro("GateAlphaMinimumMean", f"{min(condition_means):.4f}")
    macro("GateSpectralMin", f"{min(spectral_means):.3f}")
    macro("GateSpectralMax", f"{max(spectral_means):.3f}")
    with open(os.path.join(GENERATED, "tab_gate_diagnostics.tex"), "w") as stream:
        stream.write("\n".join(lines) + "\n")


def modern_pgd():
    result = read("modern_pgd.json")
    rows = result["rows"]
    for method, tag in (("semantic_tc", "SemanticTc"),
                        ("gate_full", "GateFull")):
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 2:
            raise ValueError(f"expected two modern PGD rows for {method}, found {len(selected)}")
        macro("ModernPgd" + tag + "Clean",
              f"{np.mean([row['clean_auroc'] for row in selected]):.3f}")
        macro("ModernPgd" + tag + "Attack",
              f"{np.mean([row['attacked_auroc'] for row in selected]):.3f}")
        macro("ModernPgd" + tag + "Psnr",
              f"{np.mean([row['psnr'] for row in selected]):.1f}")
    macro("ModernPgdMaxAttack",
          f"{max(row['attacked_auroc'] for row in rows):.3f}")
    macro("ModernPgdMinPsnr", f"{min(row['psnr'] for row in rows):.1f}")


def runtime_table():
    result = read("inference_benchmark.json")
    labels = {"semantic": "Semantic baseline", "semantic_tc": r"\method{}-B/16",
              "semantic_tc_l": r"\method{}-L/14", "gate_full": r"\fusionmethod{}"}
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Model & ms/image & image/s & peak MiB" + ROW_END, r"\midrule"]
    for row in result["rows"]:
        lines.append(f"{labels[row['method']]} & "
                     f"{row['milliseconds_per_image']:.3f} & "
                     f"{row['images_per_second']:.0f} & "
                     f"{row['peak_allocated_bytes'] / 2**20:.0f}" + ROW_END)
        tag = {"semantic": "RuntimeSemantic", "semantic_tc": "RuntimeAcqTc",
               "semantic_tc_l": "RuntimeAcqTcL",
               "gate_full": "RuntimeGateFull"}[row["method"]]
        macro(tag + "Ms", f"{row['milliseconds_per_image']:.3f}")
        macro(tag + "Throughput", f"{row['images_per_second']:.0f}")
        macro(tag + "Memory", f"{row['peak_allocated_bytes'] / 2**20:.0f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(GENERATED, "tab_runtime.tex"), "w") as stream:
        stream.write("\n".join(lines) + "\n")


def diagnostic_macros():
    audit = read("audit_repeats.json")
    summaries = [row["summary"] for row in audit["subsets"]]
    macro("AuditRepeatN", str(summaries[0]["n"]))
    macro("AuditRepeatMaxSd", f"{max(row['sd'] for row in summaries):.3f}")
    fs_rows = [row for row in audit["subsets"]
               if row["subset"].startswith("ForenSynths/")]
    stable = [row for row in fs_rows
              if row["summary"]["below_0.70_rate"] in (0.0, 1.0)]
    macro("AuditRepeatFsStableN", str(len(stable)))
    macro("AuditRepeatFsN", str(len(fs_rows)))

    pgd = read("pgd_check.json")
    rows = pgd["rows"]
    one40 = next(row for row in rows if row["steps"] == 40 and row["restarts"] == 1)
    one80 = next(row for row in rows if row["steps"] == 80 and row["restarts"] == 1)
    multi = next(row for row in rows if row["steps"] == 40 and row["restarts"] > 1)
    macro("PgdCheckEpsilon", f"{pgd['epsilon_over_255']:g}")
    macro("PgdCheckForty", f"{one40['auroc']:.3f}")
    macro("PgdCheckEighty", f"{one80['auroc']:.3f}")
    macro("PgdCheckRestarts", f"{multi['auroc']:.3f}")
    macro("PgdCheckRestartN", str(multi["restarts"]))
    macro("PgdCheckEvasion", f"{multi['targeted_evasion_rate']:.3f}")
    macro("PgdCheckConditionalEvasion", f"{multi['conditional_evasion_rate']:.3f}")

    linearity = read("linearity_check.json")
    for key, tag in (("scale_0.005", "Five"), ("scale_0.010", "Ten"),
                     ("scale_0.020", "Twenty")):
        test = linearity["tests"][key]
        macro("LinearityFull" + tag, f"{test['full_r2']:.3f}")
        macro("LinearityPath" + tag, f"{test['linear_path_r2']:.3f}")
    macro("LinearityImages", str(linearity["images"]))
    macro("LinearityTrainProbes", str(linearity["train_probes"]))
    macro("LinearityTestProbes", str(linearity["test_probes_per_scale"]))


def write_macros():
    path = os.path.join(GENERATED, "review_macros.tex")
    with open(path, "w") as stream:
        stream.write("% AUTO-GENERATED by artifact/src/render_review_results.py\n")
        for name, value in sorted(MACROS.items()):
            stream.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")


def main():
    os.makedirs(GENERATED, exist_ok=True)
    transform_control()
    objective_ablation()
    modern_panel()
    gate_diagnostics()
    modern_pgd()
    runtime_table()
    diagnostic_macros()
    write_macros()
    print(f"[render-review] wrote {len(MACROS)} macros and four tables")


if __name__ == "__main__":
    main()
