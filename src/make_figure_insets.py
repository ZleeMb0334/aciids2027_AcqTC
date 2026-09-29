"""Figure-1 insets, each drawn from a saved artifact result.

Every panel in figures/insets/ is a real measurement: nothing here is schematic.
Run after render.py so the inputs exist.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(os.path.dirname(ROOT), "manuscript", "images", "insets")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6,
                     "axes.linewidth": .6, "xtick.major.width": .6,
                     "ytick.major.width": .6, "figure.dpi": 300,
                     "savefig.transparent": True, "figure.facecolor": "none",
                     "axes.facecolor": "none", "savefig.facecolor": "none"})
TEAL, ORANGE, BLUE, RED, GREY = "#1f8474", "#cd672c", "#2d6087", "#be473e", "#8a97a3"


def load(p):
    try:
        return json.load(open(os.path.join(RES, p)))
    except Exception:
        return None


# ---------------------------------------------------------------- 1. the audit
d = load("benchmark_audit.json")
if d:
    rows = sorted(d, key=lambda r: r["shortcut_auroc"])
    v = [r["shortcut_auroc"] for r in rows]
    c = [RED if x >= 0.70 else TEAL for x in v]
    fig, ax = plt.subplots(figsize=(1.55, 1.05))
    ax.barh(np.arange(len(v)), v, color=c, height=.82)
    ax.axvline(0.5, ls=":", c="k", lw=.6)
    ax.set_xlim(0.4, 1.02); ax.set_yticks([])
    ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xticklabels(["0.5", "", "1.0"])
    ax.set_xlabel("header-only AUROC", labelpad=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.15); fig.savefig(f"{OUT}/audit.png", transparent=True); plt.close(fig)
    print("audit.png", len(v), "subsets")

# ------------------------------------------------- 2. spectra, real vs fake
import re
mac = os.path.join(os.path.dirname(ROOT), "manuscript", "generated", "fig_framework_data.tex")
if os.path.exists(mac):
    t = open(mac).read()
    def coords(name):
        m = re.search(r"\\newcommand\{\\" + name + r"\}\{(.*?)\}\n", t, re.S)
        if not m:
            return None
        return np.array([float(x.split(",")[1]) for x in
                         re.findall(r"\(([^)]*)\)", m.group(1))])
    R, F = coords("fwSpecReal"), coords("fwSpecFake")
    if R is not None and F is not None:
        fig, ax = plt.subplots(figsize=(1.55, 1.05))
        ax.plot(R, color=TEAL, lw=1.2, label="real")
        ax.plot(F, color=ORANGE, lw=1.2, label="fake")
        ax.set_xticks([0, len(R) - 1]); ax.set_xticklabels(["0", r"$\pi$"])
        ax.set_yticks([]); ax.set_xlabel("radial freq.", labelpad=1)
        ax.legend(frameon=False, loc="upper right", handlelength=1.0, borderpad=.1)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(pad=0.15); fig.savefig(f"{OUT}/spectrum.png", transparent=True); plt.close(fig)
        print("spectrum.png")

# ------------------------------- 3. per-generator AUROC: branches vs fusion
GENS = ["deepfake", "biggan", "gaugan", "cyclegan", "stylegan"]
runs = {}
for pat in ("fs__s*.json", "baselines__s*.json"):
    for f in sorted(glob.glob(os.path.join(RES, pat))):
        for r in json.load(open(f)):
            runs.setdefault(r["method"], []).append(r)
def per(m):
    rs = runs.get(m, [])
    out = []
    for g in GENS:
        v = [r["per_generator"][g]["auroc"] for r in rs if g in r.get("per_generator", {})]
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)
if runs.get("gate_full") and runs.get("noise"):
    fig, ax = plt.subplots(figsize=(1.95, 1.05))
    x = np.arange(len(GENS)); w = 0.27
    ax.bar(x - w, per("noise"), w, color=ORANGE, label="spectral")
    ax.bar(x, per("semantic"), w, color=BLUE, label="semantic")
    ax.bar(x + w, per("gate_full"), w, color=TEAL, label="fusion")
    ax.set_ylim(0.4, 1.02); ax.set_yticks([0.5, 0.75, 1.0])
    ax.set_xticks(x); ax.set_xticklabels([g[:4] for g in GENS], rotation=45, ha="right")
    ax.set_ylabel("AUROC", labelpad=1)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, .97),
              handlelength=.7, columnspacing=.5, borderpad=.05, handletextpad=.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.15); fig.savefig(f"{OUT}/generalise.png", transparent=True); plt.close(fig)
    print("generalise.png")

# ------------------------------------------- 4. robustness vs attack budget
rel = load("reliance__fs__s0.json")
if rel:
    fig, ax = plt.subplots(figsize=(1.85, 1.05))
    style = {"gate": (TEAL, "-"), "gate_robust": (ORANGE, "--"),
             "semantic": (BLUE, "-."), "noise": (GREY, ":")}
    for r in rel:
        if r["method"] not in style:
            continue
        pts = sorted((float(k.replace("eps", "")), v["auroc"])
                     for k, v in r["adaptive"].items())
        c, ls = style[r["method"]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, lw=1.1,
                marker="o", ms=1.8, label={"gate": "gate", "gate_robust": "gate+RC",
                       "semantic": "semantic", "noise": "spectral"}[r["method"]])
    ax.axhline(0.5, ls=":", c="k", lw=.6)
    ax.set_xscale("log"); ax.set_xlabel(r"budget $\epsilon$ ($/255$)", labelpad=1)
    ax.set_ylabel("AUROC", labelpad=1); ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(frameon=False, loc="upper right", handlelength=1.0, borderpad=.1,
              labelspacing=.12, handletextpad=.35, borderaxespad=.15)
    ax.set_xlim(right=6.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.15); fig.savefig(f"{OUT}/robust.png", transparent=True); plt.close(fig)
    print("robust.png")
print("insets ->", OUT)
