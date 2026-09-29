"""Post-process generated tables: grey-subscript SDs, and citations on the
method and dataset names.

Both render scripts emit plain "mean$\\pm$sd" cells and bare method names. Rather
than thread citations and formatting through every f-string, we normalise the
generated .tex here, so the two renderers stay simple and the styling is applied
in exactly one place.
"""
import glob, os, re

GEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "..", "manuscript", "generated")
GEN = os.path.normpath(GEN)

# method / dataset -> citation key. Longest names first so that e.g. "CNNDetection"
# is not matched inside another string.
CITE = [
    ("UniversalFakeDetect", "ojha2023universal"),
    ("CNNDetection",        "wang2020cnn"),
    ("ForenSynths",         "wang2020cnn"),
    ("FatFormer",           "liu2024fatformer"),
    ("GenImage",            "zhu2023genimage"),
    ("FreqNet",             "tan2024freqnet"),
    ("UnivFD",              "ojha2023universal"),
    ("SPAI",                "karageorgiou2025spai"),
    ("GenD",                "yermakov2026gend"),
    ("NPR",                 "tan2024rethinking"),
    ("ProGAN",              "wang2020cnn"),
    ("StyleGAN",            "wang2020cnn"),
    ("BigGAN",              "wang2020cnn"),
    ("Midjourney",          "zhu2023genimage"),
    ("Wukong",              "zhu2023genimage"),
    ("GLIDE",               "zhu2023genimage"),
]


def fix(path):
    s = open(path).read()
    orig = s
    # arrows and grey SD subscripts crowd the default LNCS column separation
    if "tabcolsep" not in s:
        s = s.replace(r"\begin{tabular}",
                      "\\setlength{\\tabcolsep}{7pt}%\n\\begin{tabular}", 1)
    # 1. mean $\pm$ sd  ->  mean \sd{sd}
    s = re.sub(r"([0-9]\.[0-9]+)\s*\$\\pm\$\s*([0-9]*\.?[0-9]+)", r"\1\\sd{\2}", s)
    # 2. citations, once per name per file, and never inside an existing \cite
    for name, key in CITE:
        if key in s:
            continue
        s = re.sub(r"(?<![\\{\w])" + re.escape(name) + r"(?![\w}])",
                   name + r"~\\cite{" + key + "}", s, count=1)
    if s != orig:
        open(path, "w").write(s)
        return True
    return False


# ---------------------------------------------------------------- direction
# Arrow marks the desired direction for each metric column. Columns that are
# descriptive (a spread, a count, a container name) get no arrow, because an
# arrow there would assert a preference the paper does not make.
UP = [r"Unseen", r"Unseen macro", r"DeepFake", r"Other four", r"JPEG 70",
      r"Resize", r"resize", r"Fused", r"Semantic", r"Spectral", r"SpecRel",
      r"AcqTC-B/16", r"Byte-only", r"image/s", r"Unseen mean",
      r"\method{}", r"AcqTC", r"SpecRel", r"Semantic-L/14", r"\method{}-B/16",
      r"\method{}-L/14", r"Fused", r"Spectral", r"Semantic"]
DOWN = [r"Shortcut AUROC", r"ms/image", r"peak MiB", r"top-5 share",
        r"\|\mathbf{w}\|_1"]
NEUTRAL = {"Method", "Benchmark", "Subset", "Backbone", "Cutoff", "Retained",
           "Fusion", "Objective", "Input", "Model", "Source", "Attack (PSNR)",
           "Real / fake container", "Strongest cue", "Active trainable",
           "Deployed total", r"\%", r"$N_{\mathrm{eff}}$"}


def add_arrows(s):
    """Append an arrow to header cells whose metric has a preferred direction."""
    m = re.search(r"\\toprule\s*\n(.*?)\\\\", s, re.S)
    if not m:
        return s
    header = m.group(1)
    cells = header.split("&")
    out = []
    for c in cells:
        t = c.strip()
        # per-cell idempotence: a header that already carries an arrow is left
        # alone, but a newly added column beside it still gets one
        if "uparrow" in t or "downarrow" in t:
            out.append(c)
            continue
        bare = re.sub(r"\\(text)?bf\{|\}|\\emph\{", "", t).strip()
        # a column headed by a method name (possibly a LaTeX macro such as
        # \method{}) is still an AUROC column and takes the same arrow
        if re.search(r"\\method|AcqTC|SpecRel|Fused", t) and bare not in NEUTRAL:
            out.append(c.rstrip() + r"\,$\uparrow$ ")
        elif bare in NEUTRAL or not bare:
            out.append(c)
        elif any(bare == u or bare.startswith(u) for u in UP):
            out.append(c.rstrip() + r"\,$\uparrow$ ")
        elif any(bare == d or bare.startswith(d) for d in DOWN):
            out.append(c.rstrip() + r"\,$\downarrow$ ")
        else:
            out.append(c)
    return s[:m.start(1)] + "&".join(out) + s[m.end(1):]


# ------------------------------------------------------------------ best cell
# Bold the best value in each numeric column, but only for tables that compare
# methods on a common footing. Tables that describe a dataset, sweep a knob, or
# report a diagnostic have no "winner" and are left alone.
COMPARATIVE = {"tab_main.tex", "tab_matched.tex", "tab_fusion_forms.tex",
               "tab_objective_ablation.tex", "tab_transform_control.tex",
               "tab_modern_generators.tex", "tab_attacks.tex"}
NUM = re.compile(r"^\s*\$?([0-9]*\.[0-9]+)")


def bold_best(s, higher_is_better=True):
    lines = s.split("\n")
    body = [i for i, l in enumerate(lines)
            if "&" in l and "\\\\" in l and "toprule" not in l
            and "midrule" not in l and "bottomrule" not in l
            and "multicolumn" not in l]
    if len(body) < 2:
        return s
    rows = [lines[i].split("&") for i in body]
    ncol = min(len(r) for r in rows)
    for c in range(1, ncol):
        vals = []
        for r in rows:
            m = NUM.match(r[c].strip())
            vals.append(float(m.group(1)) if m else None)
        good = [v for v in vals if v is not None]
        if len(good) < 2:
            continue
        target = max(good) if higher_is_better else min(good)
        # a column where everything ties carries no information as a highlight
        if good.count(target) == len(good):
            continue
        for r, v in zip(rows, vals):
            if v == target:
                cell = r[c]
                lead = cell[:len(cell) - len(cell.lstrip())]
                tail = " \\\\" if cell.rstrip().endswith("\\\\") else ""
                r[c] = (lead + r"\textbf{"
                        + cell.strip().rstrip("\\").strip() + "} " + tail)
    for i, r in zip(body, rows):
        lines[i] = "&".join(r)
    return "\n".join(lines)


if __name__ == "__main__":
    n = 0
    for f in sorted(glob.glob(os.path.join(GEN, "tab_*.tex"))):
        changed = fix(f)
        body = open(f).read()
        new = add_arrows(body)
        if os.path.basename(f) in COMPARATIVE:
            new = bold_best(new, higher_is_better=True)
        if new != body:
            open(f, "w").write(new)
            changed = True
        if changed:
            n += 1
            print("  fixed", os.path.basename(f))
    print(f"[fix] {n} table files normalised")
