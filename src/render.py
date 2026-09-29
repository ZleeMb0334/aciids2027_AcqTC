"""Core results -> manuscript/generated/*.tex.

Final-review experiments are rendered by render_review_results.py. Nothing in
manuscript/generated is hand-edited.
"""
import glob, json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
GEN = os.path.join(os.path.dirname(ROOT), "manuscript", "generated")
os.makedirs(GEN, exist_ok=True)
MACROS = {}


def mac(name, value):
    MACROS[name] = value


def load(pat):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, pat))):
        # Finder/iCloud conflict copies such as ``result 2.json`` are archival
        # duplicates, never independent repetitions.
        if re.search(r" \d+\.[^.]+$", os.path.basename(f)):
            continue
        try:
            with open(f) as fh:
                out.append((os.path.basename(f), json.load(fh)))
        except Exception:
            pass
    return out


def fmt(x, n=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


# ------------------------------------------------- Table 1: benchmark audit

# Canonical display names: the raw strings are dataset directory names, and
# printing them verbatim gave "stable_diffusion_v_1_4" and lowercase "biggan"
# in tables that elsewhere read "SD 1.4" and "BigGAN".
DISPLAY = {
    "progan": "ProGAN", "biggan": "BigGAN", "cyclegan": "CycleGAN",
    "stylegan": "StyleGAN", "stargan": "StarGAN", "gaugan": "GauGAN",
    "deepfake": "DeepFake", "seeingdark": "SeeingDark", "crn": "CRN",
    "imle": "IMLE", "san": "SAN", "glide": "GLIDE", "wukong": "Wukong",
    "adm": "ADM", "vqdm": "VQDM", "midjourney": "Midjourney",
    "stable_diffusion_v_1_4": "SD 1.4", "stable_diffusion_v_1_5": "SD 1.5",
}


def disp(name):
    return DISPLAY.get(name, DISPLAY.get(name.lower(), name.replace("_", r"\_")))


def table_audit():
    d = load("benchmark_audit.json")
    if not d:
        return
    rows = d[0][1]
    gi = [r for r in rows if r["subset"].startswith("GenImage")]
    fs = [r for r in rows if r["subset"].startswith("ForenSynths")]
    fs.sort(key=lambda r: r["shortcut_auroc"])
    lines = [r"\begin{tabular}{llccl}", r"\toprule",
             r"Benchmark & Subset & Real / fake container & "
             r"Shortcut AUROC & Strongest cue \\", r"\midrule"]
    for i, r in enumerate(fs):
        sub = disp(r["subset"].split("/", 1)[1])
        fr = "/".join(sorted(r["format_real"])) or "?"
        ff = "/".join(sorted(r["format_fake"])) or "?"
        cue = r["best_single_feature"].replace("_", r"\_")
        lines.append(f"{'ForenSynths' if i == 0 else ''} & {sub} & {fr} / {ff} & "
                     f"{r['shortcut_auroc']:.3f} & {cue} \\\\")
    lines.append(r"\midrule")
    # every GenImage subset is identical to three decimals; one row states that
    gmin, gmax = min(r["shortcut_auroc"] for r in gi), max(r["shortcut_auroc"] for r in gi)
    gfr = "/".join(sorted({f for r in gi for f in r["format_real"]}))
    gff = "/".join(sorted({f for r in gi for f in r["format_fake"]}))
    gcue = sorted({r["best_single_feature"] for r in gi})
    rng = f"{gmin:.3f}" if abs(gmax - gmin) < 5e-4 else f"{gmin:.3f}--{gmax:.3f}"
    lines.append(f"GenImage & all {len(gi)} generators & {gfr} / {gff} & {rng} & "
                 + ", ".join(c.replace("_", r"\_") for c in gcue) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_audit.tex"), "w").write("\n".join(lines))
    mac("AuditNSubsets", str(len(rows)))
    mac("AuditGenImageMin", fmt(min(r["shortcut_auroc"] for r in gi)))
    mac("AuditGenImageMax", fmt(max(r["shortcut_auroc"] for r in gi)))
    mac("AuditGenImageN", str(len(gi)))
    clean = [r for r in fs if r["shortcut_auroc"] < 0.70]
    conf = [r for r in fs if r["shortcut_auroc"] >= 0.70]
    mac("AuditFSCleanN", str(len(clean)))
    mac("AuditFSConfN", str(len(conf)))
    mac("AuditFSCleanMax", fmt(max(r["shortcut_auroc"] for r in clean)))
    mac("AuditFSConfMin", fmt(min(r["shortcut_auroc"] for r in conf)))
    mac("AuditFSCleanList", ", ".join(sorted(disp(r["subset"].split("/")[1]) for r in clean)))
    mac("AuditFSConfList", ", ".join(sorted(disp(r["subset"].split("/")[1]) for r in conf)))


# ------------------------------------------- Table 2: protocol ladder (GATE1)
def table_protocol():
    got = {}
    for acq in ("raw", "matched", "history"):
        d = load(f"gate1_{acq}.json")
        if d:
            got[acq] = {r["generator"]: r for r in d[0][1]}
    if not got:
        return
    gens = sorted(set().union(*[set(v) for v in got.values()]))
    lines = [r"\begin{tabular}{l" + "cc" * len(got) + r"}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{a}}}" for a in got) + r" \\",
             # rules under each spanning header show which columns it covers
             " ".join(rf"\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(got)))]
    lines.append("Generator & " + " & ".join("clean & A1" for _ in got) + r" \\")
    lines.append(r"\midrule")
    for g in gens:
        cells = []
        for acq in got:
            r = got[acq].get(g)
            cells += [fmt(r["auroc_clean"]) if r else "--",
                      fmt(r.get("a1_s1.0_auroc")) if r else "--"]
        lines.append(disp(g) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_protocol.tex"), "w").write("\n".join(lines))
    for acq in got:
        v = [r["auroc_clean"] for r in got[acq].values()]
        d_ = [r["auroc_clean"] - r.get("a1_s1.0_auroc", r["auroc_clean"])
              for r in got[acq].values()]
        mac(f"Proto{acq.capitalize()}MeanClean", fmt(float(np.mean(v))))
        mac(f"Proto{acq.capitalize()}MaxDrop", fmt(float(np.max(d_))))
        mac(f"Proto{acq.capitalize()}N", str(len(v)))


# ------------------------------------------- Table 3/4: main ForenSynths runs
GENS = ["progan", "deepfake", "biggan", "gaugan", "cyclegan", "stylegan"]
GEN_LABELS = {"progan": "ProGAN", "deepfake": "DeepFake", "biggan": "BigGAN",
              "gaugan": "GauGAN", "cyclegan": "CycleGAN", "stylegan": "StyleGAN"}
NAMES = {"semantic": "Semantic baseline",
         "semantic_l": "Semantic-L/14 (matched)",
         "semantic_tc": "\\method{}-B/16 (ours)",
         "semantic_tc_l": "\\textbf{\\method{}-L/14 (ours)}",
         "noise": "Spectral branch only",
         "gate": "Gated fusion", "gate_robust": "\\;+ removal/resampling consistency",
         "gate_srr": "\\;+ consistency + entropy hinge",
         "gate_full": "\\fusionmethod{} (fusion ablation)",
         "cnndetection": "CNNDetection", "ufd": "UnivFD", "npr": "NPR",
         "freqnet": "FreqNet",
         "concat": "Fusion: concat", "static": "Fusion: static weight",
         "mean": "Fusion: mean"}
SHORT = {"semantic": "Semantic", "semantic_l": "Semantic-L",
         "semantic_tc": "AcqTC-B/16", "semantic_tc_l": "AcqTC-L/14",
         "noise": "Spectral", "gate": "Gate",
         "gate_robust": "+RC", "gate_srr": "+RC+EH", "gate_full": "SpecRel",
         "cnndetection": "CNNDet", "ufd": "UFD", "npr": "NPR", "freqnet": "FreqNet",
         "concat": "Concat", "static": "Static", "mean": "Mean"}


def collect_runs():
    # Later result tags replace earlier rows for the same method and seed.  This
    # supports targeted reruns without double-counting a replacement checkpoint.
    runs = {}
    for pat in ("fs__s*.json", "baselines__s*.json", "gatefull__s*.json",
                "semtc__s*.json", "acqtcl__s*.json"):
        for fn, rows in load(pat):
            for r in rows:
                r = dict(r)
                if fn.startswith("acqtcl__") and r["method"] == "semantic_tc":
                    r["method"] = "semantic_tc_l"
                seed = r.get("seed")
                if seed is None:
                    seed = int(fn.split("__s", 1)[1].split(".", 1)[0])
                runs.setdefault(r["method"], {})[int(seed)] = r
    for fn, rows in load("matchL__s*.json"):
        for r in rows:
            if r["method"] != "semantic":
                continue
            r = dict(r)
            r["method"] = "semantic_l"
            seed = int(r.get("seed", fn.split("__s", 1)[1].split(".", 1)[0]))
            runs.setdefault("semantic_l", {})[seed] = r
    # The final review reruns the gated-family PGD cells with the fused
    # classification loss alone.  Overlay those attack measurements while
    # preserving the clean, transfer, and post-processing fields of each run.
    reevaluated = load("pgd_reeval.json")
    if reevaluated:
        for row in reevaluated[-1][1].get("rows", []):
            target = runs.get(row["method"], {}).get(int(row["seed"]))
            if target is None:
                continue
            for eps in (1, 2, 4, 8):
                value = row.get(f"eps{eps}")
                if value:
                    target[f"a3_eps{eps}_auroc"] = value["auroc"]
                    target[f"a3_eps{eps}_psnr"] = value["psnr"]
    return {m: [by_seed[s] for s in sorted(by_seed)]
            for m, by_seed in runs.items()}


def agg(rs, key, sub=None):
    v = [(r["per_generator"][sub][key] if sub else r.get(key)) for r in rs]
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def table_main():
    runs = collect_runs()
    if not runs:
        return
    order = [m for m in ("cnndetection", "ufd", "npr", "freqnet",
                         "semantic_tc", "semantic_l", "semantic_tc_l") if m in runs]
    include_benign = all(
        all(r.get("a4_jpeg70_auroc") is not None and
            r.get("a4_resize_auroc") is not None for r in runs[m])
        for m in order)
    other_generators = ["biggan", "gaugan", "cyclegan", "stylegan"]
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Method & DeepFake & Other four & Unseen mean$\pm$SD"
             r" & JPEG 70 & Resize \\", r"\midrule"]
    RETRAINED = set()
    undertrained = []
    for m in order:
        if m == "semantic_tc" and any(x in order for x in
                                       ("cnndetection", "ufd", "npr", "freqnet")):
            lines.append(r"\midrule")
        rs = runs[m]
        deepfake, _ = agg(rs, "auroc", "deepfake")
        other = float(np.mean([
            np.mean([r["per_generator"][g]["auroc"] for g in other_generators])
            for r in rs]))
        mu, _ = agg(rs, "auroc_macro")
        unseen, unseen_sd = agg(rs, "auroc_unseen_macro")
        # A retrained baseline scoring near chance measures our training budget,
        # not the published method. Flag it in the table rather than letting it
        # stand as that method's performance.
        flag = ""
        if m in RETRAINED and mu is not None and mu < 0.60:
            flag = r"$^{\dagger}$"
            undertrained.append(m)
        unseen_cell = (f"{fmt(unseen)}" + r"\sd{" + f"{fmt(unseen_sd)}" + r"}"
                       if unseen is not None else "--")
        if include_benign:
            jpeg, _ = agg(rs, "a4_jpeg70_auroc")
            resize, _ = agg(rs, "a4_resize_auroc")
        else:
            jpeg = resize = None
        lines.append(NAMES.get(m, m) + flag +
                     f" & {fmt(deepfake)} & {fmt(other)} & {unseen_cell}"
                     f" & {fmt(jpeg)} & {fmt(resize)}" + r" \\")
        if mu is not None:
            mac(f"Main{m.title().replace('_','')}Macro", fmt(mu))
    mac("UndertrainedList", ", ".join(NAMES.get(m, m) for m in undertrained)
        if undertrained else "none")
    mac("UndertrainedN", str(len(undertrained)))
    external = [m for m in ("cnndetection", "ufd", "npr", "freqnet") if m in runs]
    if external and "semantic_tc" in runs:
        best = max(external, key=lambda m: agg(runs[m], "auroc_unseen_macro")[0])
        best_score = agg(runs[best], "auroc_unseen_macro")[0]
        ours = agg(runs["semantic_tc"], "auroc_unseen_macro")[0]
        mac("ExternalBestName", NAMES[best])
        mac("ExternalBestUnseen", fmt(best_score))
        mac("AcqTcExternalMargin", f"{ours - best_score:+.3f}")
        if "semantic_tc_l" in runs:
            ours_l = agg(runs["semantic_tc_l"], "auroc_unseen_macro")[0]
            mac("AcqTcLExternalMargin", f"{ours_l - best_score:+.3f}")
        for key, tag in (("a4_jpeg70_auroc", "Jpeg"),
                         ("a4_resize_auroc", "Resize")):
            available = [m for m in external if agg(runs[m], key)[0] is not None]
            if not available:
                continue
            best_benign = max(available, key=lambda m: agg(runs[m], key)[0])
            best_benign_score = agg(runs[best_benign], key)[0]
            mac(f"ExternalBest{tag}Name", NAMES[best_benign])
            mac(f"ExternalBest{tag}", fmt(best_benign_score))
            if "semantic_tc_l" in runs:
                ours_l_benign = agg(runs["semantic_tc_l"], key)[0]
                if ours_l_benign is not None:
                    mac(f"AcqTcLExternal{tag}Margin",
                        f"{ours_l_benign - best_benign_score:+.3f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_main.tex"), "w").write("\n".join(lines))
    n = max(len(v) for v in runs.values())
    mac("MainNSeeds", str(n))
    # spread as well as mean: the resize effect varies a lot by seed and a bare
    # mean would hide that.
    for m, tag in (("gate", "Gate"), ("gate_full", "GateFull"),
                   ("semantic", "Semantic"), ("gate_robust", "GateRobust")):
        rs = runs.get(m, [])
        v = [r.get("a4_resize_auroc") for r in rs if r.get("a4_resize_auroc") is not None]
        if len(v) > 1:
            mac(tag + "ResizeSd", fmt(float(np.std(v, ddof=1))))
            mac(tag + "ResizeMin", fmt(float(min(v))))
            mac(tag + "ResizeMax", fmt(float(max(v))))
    mac("SeedPhrase",
        "Every run uses a single seed; the margins between the strongest methods "
        "are smaller than we would trust at $n=1$, and we say so where they matter."
        if n < 2 else
        f"Runs use {n} seeds unless stated otherwise; tables report means with "
        f"across-seed standard deviations.")


ATK = [("Clean", "auroc_progan", None),
       ("PGD 2/255", "a3_eps2_auroc", "a3_eps2_psnr"),
       ("PGD 4/255", "a3_eps4_auroc", "a3_eps4_psnr"),
       ("JPEG 70", "a4_jpeg70_auroc", "a4_jpeg70_psnr"),
       ("Half-scale resize", "a4_resize_auroc", "a4_resize_psnr")]


def table_attacks():
    runs = collect_runs()
    if not runs:
        return
    order = [m for m in ("semantic", "noise", "gate_full", "semantic_tc") if m in runs]
    lines = [r"\begin{tabular}{l" + "c" * len(order) + r"}", r"\toprule",
             "Attack (PSNR) & " + " & ".join(SHORT.get(m, m) for m in order) + r" \\",
             r"\midrule"]
    for label, k, pk in ATK:
        cells, ps = [], []
        for m in order:
            mu, _ = agg(runs[m], k)
            cells.append(fmt(mu))
            if pk:
                p, _ = agg(runs[m], pk)
                if p is not None:
                    ps.append(p)
        tag = f" ({np.mean(ps):.0f}\\,dB)" if ps else ""
        lines.append(f"{label}{tag} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_attacks.tex"), "w").write("\n".join(lines))


def matched_macros():
    """Macros for the matched-backbone controls (ViT-L/14 and the ViT-B/16 probe)."""
    for pat, suffix in (("matchL__s*.json", "L"), ("probeB__s*.json", "Probe")):
        grouped = {}
        for _, rows in load(pat):
            for r in rows:
                base = {"gate_full": "GateFull", "semantic": "Semantic"}.get(r["method"])
                if not base:
                    continue
                grouped.setdefault(base, []).append(r)
        for base, rs in grouped.items():
            for k, s_ in (("auroc_unseen_macro", "Unseen"), ("auroc_macro", "Macro"),
                          ("a4_jpeg70_auroc", "Jpeg"), ("a4_resize_auroc", "Resize"),
                          ("gap", "Gap")):
                mu, _ = agg(rs, k)
                if mu is not None:
                    mac(base + suffix + s_, fmt(mu))
            mac(base + suffix + "N", str(len(rs)))


def table_matched():
    """Matched-backbone controls: our method on ViT-L/14, and a linear probe on
    our ViT-B/16, so capacity is held fixed within each block."""
    runs = collect_runs()
    extra = {}
    for pat, tag in (("matchL__s*.json", "L"), ("probeB__s*.json", "P")):
        for _, rows in load(pat):
            for r in rows:
                extra.setdefault(tag + ":" + r["method"], []).append(r)
    if not extra:
        return
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Backbone & Method & Unseen macro & JPEG 70 & resize \\", r"\midrule"]

    def row(label, method, rs):
        cells = []
        for key in ("auroc_unseen_macro", "a4_jpeg70_auroc", "a4_resize_auroc"):
            mu, sd = agg(rs, key)
            if mu is None:
                cells.append("--")
            elif len(rs) > 1:
                cells.append(f"{fmt(mu)}" + r"\sd{" + f"{fmt(sd)}" + r"}")
            else:
                cells.append(fmt(mu))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    if "ufd" in runs:
        row(r"ViT-L/14 & UniversalFakeDetect (probe)", "ufd", runs["ufd"])
    if "L:gate_full" in extra:
        row(r" & \fusionmethod{}", "gate_full", extra["L:gate_full"])
    if "L:semantic" in extra:
        row(r" & semantic branch only", "semantic", extra["L:semantic"])
    lines.append(r"\midrule")
    if "P:semantic" in extra:
        row(r"ViT-B/16 & linear probe (UFD-style)", "semantic", extra["P:semantic"])
    if "semantic" in runs:
        row(r" & LayerNorm-tuned", "semantic", runs["semantic"])
    if "gate_full" in runs:
        row(r" & \fusionmethod{}", "gate_full", runs["gate_full"])
    if "semantic_tc" in runs:
        row(r" & \method{} (ours)", "semantic_tc", runs["semantic_tc"])
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_matched.tex"), "w").write("\n".join(lines))


def table_cost():
    runs = collect_runs()
    rows = []
    for m, rs in runs.items():
        b = next((r.get("breakdown") for r in rs if r.get("breakdown")), None)
        tp = next((r.get("trainable") for r in rs if r.get("trainable")), None)
        tot = next((r.get("params") for r in rs if r.get("params")), None)
        if b and m in {"semantic", "semantic_l", "semantic_tc", "semantic_tc_l"}:
            # Semantic deployment drops the instantiated diagnostic spectral
            # branch, which is bypassed by FusionDetector.forward.
            tp = b["ln_affine"] + b["sem_head"]
            tot = b["backbone_total"] + b["sem_head"]
        if tp:
            rows.append((m, tp, tot, b))
    if not rows:
        return
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Method & Active trainable & Deployed total & \% \\", r"\midrule"]
    for m, tp, tot, b in sorted(rows, key=lambda x: x[1]):
        pc = 100.0 * tp / tot if tot else float("nan")
        lines.append(f"{NAMES.get(m, m)} & {tp:,} & {tot:,} & {pc:.2f} \\\\")
        if m == "gate":
            mac("MainTrainablePct", f"{pc:.2f}")
        if b and m == "semantic_tc":
            mac("LNParams", f"{b['ln_affine']:,}")
            mac("LNPct", f"{b['ln_pct_of_backbone']:.3f}")
            mac("AcqTcTrainable", f"{tp:,}")
            mac("AcqTcTrainablePct", f"{pc:.3f}")
        if b and m == "semantic_tc_l":
            mac("AcqTcLLNParams", f"{b['ln_affine']:,}")
            mac("AcqTcLTrainable", f"{tp:,}")
            mac("AcqTcLTrainablePct", f"{pc:.3f}")
            mac("AcqTcLTotalM", f"{tot / 1e6:.1f}")
        if m == "gate_full" and tot:
            mac("MainTotalM", f"{tot / 1e6:.1f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_cost.tex"), "w").write("\n".join(lines))


# ------------------------------------------------- claim-level macros + figures
def table_reliance():
    """Reliance + linearisation numbers from reliance__*.json."""
    d = load("reliance__*.json")
    if not d:
        return
    rows = d[-1][1]
    if not rows:
        return
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Method & $N_{\mathrm{eff}}$ & $\|\mathbf{w}\|_1$ & top-5 share \\",
             r"\midrule"]
    for r in rows:
        if r["method"] == "semantic":
            continue
        w = np.abs(np.asarray(r["w"], dtype=float))
        p = w / max(w.sum(), 1e-12)
        neff = 1.0 / max(float(np.square(p).sum()), 1e-12)
        r["N_eff"] = neff
        lines.append(f"{NAMES.get(r['method'], r['method'])} & {neff:.1f} & "
                     f"{r.get('w_l1', float('nan')):.2f} & "
                     f"{100*r['top5_share']:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(GEN, "tab_reliance.tex"), "w").write("\n".join(lines))
    def eps50(r):
        """Smallest attack budget (in /255) at which AUROC crosses 0.5,
        linearly interpolated between the two bracketing budgets."""
        ad = r.get("adaptive", {})
        pts = sorted((float(k.replace("eps", "")), v["auroc"]) for k, v in ad.items())
        if not pts:
            return None
        for (e0, a0), (e1, a1) in zip(pts, pts[1:]):
            if a0 >= 0.5 > a1:
                t = (a0 - 0.5) / max(a0 - a1, 1e-9)
                return e0 + t * (e1 - e0)
        return pts[0][0] if pts[0][1] < 0.5 else None

    by = {r["method"]: r for r in rows}
    for r in rows:
        r["eps50"] = eps50(r)
    for m, tag in (("gate", "Gate"), ("gate_robust", "GateRobust"),
                   ("gate_srr", "GateSrr"), ("semantic", "Semantic"),
                   ("noise", "Noise")):
        if m in by:
            mac("Hw" + tag, fmt(by[m]["H_w"]))
            if "N_eff" not in by[m]:
                w = np.abs(np.asarray(by[m]["w"], dtype=float))
                p = w / max(w.sum(), 1e-12)
                by[m]["N_eff"] = 1.0 / max(float(np.square(p).sum()), 1e-12)
            mac("Neff" + tag, fmt(by[m]["N_eff"], 1))
            mac("RelTop" + tag, fmt(100 * by[m]["top5_share"], 1))
            mac("WLone" + tag, fmt(by[m].get("w_l1", float("nan")), 2))
            if by[m].get("eps50") is not None:
                mac("Eps" + tag, fmt(by[m]["eps50"], 3))
    r0 = rows[0]
    lin = r0.get("linearity", {})
    mac("LinRsqFull", fmt(lin.get("full_s0.02")))
    mac("LinRsqLin", fmt(lin.get("linear_path_s0.02")))
    mac("HwMax", fmt(r0["H_max"]))
    mac("RelTopFive", fmt(100 * r0["top5_share"], 1))
    hs = [r["H_w"] for r in rows]
    mac("HwMin", fmt(min(hs))); mac("HwMaxObs", fmt(max(hs)))


def claims_and_figures():
    runs = collect_runs()
    if not runs:
        return
    def one(m, k):
        mu, _ = agg(runs.get(m, []), k)
        return mu
    def paired_seed_diff(a, b, key):
        aa = {int(r["seed"]): r for r in runs.get(a, []) if r.get("seed") is not None}
        bb = {int(r["seed"]): r for r in runs.get(b, []) if r.get("seed") is not None}
        return np.asarray([aa[s][key] - bb[s][key]
                           for s in sorted(set(aa) & set(bb))], dtype=float)
    for m, tag in (("semantic", "Semantic"), ("semantic_l", "SemanticL"),
                   ("semantic_tc", "AcqTc"),
                   ("semantic_tc_l", "AcqTcL"),
                   ("noise", "Noise"), ("gate", "Gate"),
                   ("gate_robust", "GateRobust"), ("gate_srr", "GateSrr"),
                   ("gate_full", "GateFull"), ("cnndetection", "CNNDet"),
                   ("ufd", "UFD"), ("npr", "NPR"), ("freqnet", "FreqNet"),
                   ("concat", "Concat"), ("static", "Static"), ("mean", "Mean")):
        if m not in runs:
            continue
        for k, suf in (("auroc_progan", "In"), ("auroc_unseen_macro", "Unseen"),
                       ("gap", "Gap"), ("a1_auroc", "AOne"),
                       ("a4_jpeg70_auroc", "Jpeg"), ("a2_resample_auroc", "Resample"),
                       ("a3_eps4_auroc", "Pgd"), ("a3_eps4_psnr", "PgdPsnr"),
                       ("a4_resize_auroc", "Resize"), ("auroc_macro", "Macro"),
                       ("H_w", "Hw")):
            v = one(m, k)
            if v is not None:
                mac(tag + suf, fmt(v, 1 if suf == "PgdPsnr" else 3))
    # rank correlation between the two axes
    ms = [m for m in runs if one(m, "auroc_unseen_macro") is not None
          and one(m, "a3_eps4_auroc") is not None]
    if len(ms) >= 4:
        from scipy.stats import spearmanr
        g = [one(m, "auroc_unseen_macro") for m in ms]
        rb = [one(m, "a3_eps4_auroc") for m in ms]
        rho, pv = spearmanr(g, rb)
        mac("OrthoRho", fmt(float(rho), 2)); mac("OrthoP", fmt(float(pv), 2))
        mac("OrthoN", str(len(ms)))
    for a, b, tag in (("gate_full", "semantic", "FullSem"),
                      ("gate_srr", "gate_robust", "SrrRc"),
                      ("semantic_tc_l", "semantic_l", "AcqTcLSemanticL"),
                      ("semantic_tc_l", "freqnet", "AcqTcLFreqNet")):
        d = paired_seed_diff(a, b, "auroc_unseen_macro")
        if len(d):
            mac(tag + "Diff", fmt(float(d.mean()), 4))
            mac(tag + "DiffMin", fmt(float(d.min()), 4))
            mac(tag + "DiffMax", fmt(float(d.max()), 4))
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                             "xtick.labelsize": 8, "ytick.labelsize": 8,
                             "legend.fontsize": 7.5})
        IMG = os.path.join(os.path.dirname(ROOT), "manuscript", "images")
        os.makedirs(IMG, exist_ok=True)
        MARK = {"cnndetection": ("o", "#7f7f7f"), "ufd": ("s", "#9467bd"),
                "npr": ("D", "#8c564b"), "freqnet": ("h", "#000000"),
                "semantic": ("^", "#0072b2"),
                "noise": ("v", "#d55e00"), "gate": ("*", "#009e73"),
                "gate_robust": ("P", "#cc79a7"), "gate_full": ("X", "#e69f00"),
                "gate_srr": ("<", "#56b4e9")}
        if ms:
            fig, ax = plt.subplots(figsize=(3.5, 2.7))
            for m in ms:
                mk, c = MARK.get(m, ("o", "k"))
                ax.scatter(one(m, "auroc_unseen_macro"), one(m, "a3_eps4_auroc"),
                           marker=mk, s=70, color=c, edgecolor="k", linewidth=.4,
                           label=SHORT.get(m, m), zorder=3)
            ax.axhline(0.5, ls=":", c="grey", lw=.8, zorder=1)
            ax.set_xlabel("cross-generator AUROC (unseen)")
            ax.set_ylabel("AUROC under adaptive PGD")
            ax.set_ylim(-0.06, 1.06); ax.grid(alpha=.25, zorder=0)
            ax.legend(loc="upper left", frameon=False, handletextpad=.2,
                      borderpad=.2, labelspacing=.25, ncol=2,
                      columnspacing=.6)
            fig.tight_layout(pad=0.3)
            fig.savefig(os.path.join(IMG, "orthogonality.pdf"))
            plt.close(fig)
        # H and attack behaviour must both come from the BAND-ABLATION measurement
        # on the deployed model; the training-time linear weights are a different
        # quantity and mixing them understates the spread across methods.
        rel = load("reliance__*.json")
        relrows = rel[-1][1] if rel else []
        hs = []
        for r in relrows:
            if r["method"] == "semantic":
                continue
            ad = r.get("adaptive", {})
            pts = sorted((float(k.replace("eps", "")), v["auroc"]) for k, v in ad.items())
            e50 = None
            for (e0, a0), (e1, a1) in zip(pts, pts[1:]):
                if a0 >= 0.5 > a1:
                    e50 = e0 + (a0 - 0.5) / max(a0 - a1, 1e-9) * (e1 - e0)
                    break
            if e50 is None and pts and pts[0][1] < 0.5:
                e50 = pts[0][0]
            if e50 is not None:
                w = np.abs(np.asarray(r["w"], dtype=float))
                p = w / max(w.sum(), 1e-12)
                neff = 1.0 / max(float(np.square(p).sum()), 1e-12)
                hs.append((neff, e50, r["method"], "eps50"))
        if len(hs) >= 1:
            fig, ax = plt.subplots(figsize=(3.5, 2.7))
            for h, p_, m, k in hs:
                mk, c = MARK.get(m, ("o", "k"))
                ax.scatter(h, p_, marker=mk, s=70, color=c, edgecolor="k",
                           linewidth=.4, label=SHORT.get(m, m), zorder=3)
            xs_ = np.array([h for h, _, _, _ in hs])
            neff_span = float(xs_.max() - xs_.min()) if len(hs) > 1 else 0.0
            mac("NeffSpan", fmt(neff_span, 3))
            ys_all = np.array([p for _, p, _, _ in hs])
            yspan = float(ys_all.max() - ys_all.min()) if len(hs) > 1 else 0.0
            mac("AtkSpan", fmt(yspan, 3))
            ax.set_title("cross-detector diagnostic; scales are uncontrolled", fontsize=7.5)
            ax.set_xlabel(r"effective band count $N_{\mathrm{eff}}$")
            ax.set_ylabel(r"PGD $\epsilon$ at AUROC $=0.5$ ($/255$)")
            ax.grid(alpha=.25, zorder=0)
            ax.legend(loc="best", frameon=False, handletextpad=.2,
                      borderpad=.2, labelspacing=.25)
            fig.tight_layout(pad=0.3)
            fig.savefig(os.path.join(IMG, "reliance.pdf"))
            plt.close(fig)
    except Exception as e:
        print("  ! figures:", e)


def figure_audit():
    d = load("benchmark_audit.json")
    if not d:
        return
    rows = d[0][1]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = sorted(rows, key=lambda r: (r["subset"].split("/")[0], r["shortcut_auroc"]))
    names = [r["subset"].split("/")[1] for r in rows]
    vals = [r["shortcut_auroc"] for r in rows]
    cols = ["#c0392b" if v >= 0.70 else "#2980b9" for v in vals]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.axvline(0.5, ls="--", c="k", lw=.9)
    ax.axvline(0.70, ls=":", c="grey", lw=.9)
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("shortcut AUROC (file metadata only, no pixels)")
    nfs = sum(1 for r in rows if r["subset"].startswith("ForenSynths"))
    ax.text(1.01, nfs / 2, "ForenSynths", rotation=90, va="center", ha="left", fontsize=7)
    ax.text(1.01, nfs + (len(rows) - nfs) / 2, "GenImage", rotation=90, va="center",
            ha="left", fontsize=7)
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    IMG = os.path.join(os.path.dirname(ROOT), "manuscript", "images")
    os.makedirs(IMG, exist_ok=True)
    fig.savefig(os.path.join(IMG, "audit.pdf"))
    plt.close(fig)


# Macros the manuscript references. Any that a run has not produced yet are
# emitted as "??" so the document still compiles and the gap is visible in the
# PDF rather than becoming an undefined-control-sequence error.
REQUIRED = [
    "AuditNSubsets", "AuditGenImageMin", "AuditGenImageMax", "AuditGenImageN",
    "AuditFSCleanN", "AuditFSConfN", "AuditFSCleanMax", "AuditFSConfMin",
    "AuditFSCleanList", "AuditFSConfList",
    "ProtoRawMeanClean", "ProtoMatchedMeanClean", "ProtoHistoryMeanClean",
    "ProtoRawMaxDrop", "ProtoMatchedMaxDrop", "ProtoHistoryMaxDrop", "ProtoRawN",
    "MainNSeeds", "MainTrainablePct", "LNParams", "LNPct",
    "AcqTcTrainable", "AcqTcTrainablePct",
    "AcqTcLExternalMargin", "AcqTcLExternalJpegMargin",
    "AcqTcLExternalResizeMargin", "ExternalBestJpeg", "ExternalBestResize",
    "AcqTcLFreqNetDiff", "AcqTcLFreqNetDiffMin", "AcqTcLFreqNetDiffMax",
    "AcqTcLSemanticLDiff",
    "AcqTcLSemanticLDiffMin", "AcqTcLSemanticLDiffMax",
    "MainTotalM", "FullSemDiff", "SrrRcDiff", "SrrRcDiffMin", "SrrRcDiffMax",
    "LinRsqFull", "LinRsqLin", "HwMax", "RelTopFive",
    "HwGate", "HwGateRobust", "RelTopGate", "RelTopGateRobust",
]
for _m in ("Semantic", "AcqTc", "Noise", "Gate", "GateRobust", "GateSrr", "GateFull"):
    for _s in ("In", "Unseen", "Gap", "Macro", "AOne", "Jpeg", "Resample",
               "Resize", "Pgd", "PgdPsnr"):
        REQUIRED.append(_m + _s)
for _m in ("CNNDet", "UFD", "NPR", "FreqNet"):
    for _s in ("In", "Unseen", "Gap", "Macro"):
        REQUIRED.append(_m + _s)
for _m in ("SemanticL", "AcqTcL"):
    for _s in ("In", "Unseen", "Gap", "Macro", "Jpeg", "Resize"):
        REQUIRED.append(_m + _s)
# Only our own models expose a spectral band path.
for _m in ("Semantic", "Noise", "Gate", "GateRobust", "GateSrr", "GateFull"):
    REQUIRED.append(_m + "Hw")
for _m in ("Noise", "Gate", "GateRobust", "GateSrr"):
    REQUIRED.append("Neff" + _m)


def write_macros():
    missing = [k for k in REQUIRED if k not in MACROS]
    for k in missing:
        MACROS[k] = r"\textbf{??}"
    if missing:
        print(f"[render] {len(missing)} macros not yet produced: "
              + ", ".join(sorted(missing)[:8]) + (" ..." if len(missing) > 8 else ""))
    with open(os.path.join(GEN, "macros.tex"), "w") as f:
        f.write("% AUTO-GENERATED by artifact/src/render.py -- do not edit\n")
        for k, v in sorted(MACROS.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"[render] {len(MACROS)} macros -> generated/macros.tex")


if __name__ == "__main__":
    table_audit(); table_protocol(); table_main(); table_attacks(); table_cost()
    table_matched(); matched_macros()
    figure_audit(); table_reliance(); claims_and_figures()
    write_macros()
    for f in sorted(glob.glob(os.path.join(GEN, "*.tex"))):
        print("  ", os.path.basename(f), os.path.getsize(f), "bytes")
