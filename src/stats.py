"""Seed-level descriptive comparisons for the main detector runs.

Generator columns within a seed share one trained model.  Treating those
columns as independent replicates produces anti-conservative tests.  With three
training seeds, we report the mean, standard deviation, plus range of paired
unseen-generator differences.
"""
import glob, json, os
import numpy as np

GENS = ["progan", "deepfake", "biggan", "gaugan", "cyclegan", "stylegan"]
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load(tag="fs"):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, f"{tag}__s*.json"))):
        seed = int(f.split("__s")[1].split(".")[0])
        for r in json.load(open(f)):
            out.setdefault(r["method"], {})[seed] = r
    return out


def paired_unseen(a, b, runs):
    ds = []
    for s in sorted(set(runs.get(a, {})) & set(runs.get(b, {}))):
        ra, rb = runs[a][s], runs[b][s]
        ds.append(ra["auroc_unseen_macro"] - rb["auroc_unseen_macro"])
    return np.asarray(ds, dtype=float)


if __name__ == "__main__":
    # Namespace by tag: matchL runs a DIFFERENT backbone under the same method
    # names, so merging them into one key silently compares ViT-L/14 against
    # ViT-B/16 and reports the capacity gap as a method effect.
    runs = {}
    for tag in ("fs", "baselines", "abl", "gatefull"):  # later tag replaces seed 0
        for m, v in load(tag).items():
            runs.setdefault(m, {}).update(v)
    runsL = {}                                      # ViT-L/14 block
    for m, v in load("matchL").items():
        runsL[m] = v
    runsP = {}                                      # ViT-B/16 linear probe
    for m, v in load("probeB").items():
        runsP[m] = v
    def summarise(d, title):
        print(f"=== {title} ===")
        for m, byseed in sorted(d.items()):
            v = np.array([r["auroc_unseen_macro"] for r in byseed.values()])
            sd = v.std(ddof=1) if len(v) > 1 else float("nan")
            print(f"  {m:14s} n={len(v)}  mean={v.mean():.4f}  sd={sd:.4f}")

    summarise(runsL, "ViT-L/14 block")
    summarise(runsP, "ViT-B/16 linear probe")
    print("=== ViT-B/16 block (unseen macro over seeds) ===")
    for m, byseed in sorted(runs.items()):
        v = np.array([r["auroc_unseen_macro"] for r in byseed.values()])
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        print(f"  {m:14s} n={len(v)}  mean={v.mean():.4f}  sd={sd:.4f}")
    print("\n=== paired unseen-macro differences by seed, ViT-B/16 block ===")
    for a, b in [("gate_full","semantic"),("gate","semantic"),("gate_full","gate"),
                 ("gate_full","gate_robust"),("gate_srr","gate_robust"),
                 ("gate","concat"),("gate","static"),("gate","mean"),
                 ("semantic","cnndetection")]:
        d = paired_unseen(a, b, runs)
        if len(d) < 2:
            print(f"  {a:11s} vs {b:13s}: insufficient paired seeds (n={len(d)})")
            continue
        print(f"  {a:11s} vs {b:13s}: n={len(d)} mean={d.mean():+.4f} "
              f"sd={d.std(ddof=1):.4f} range=[{d.min():+.4f},{d.max():+.4f}]")
    print("\n=== paired within the ViT-L/14 block ===")
    for a, b in [("gate_full", "semantic")]:
        d = paired_unseen(a, b, runsL)
        if len(d) >= 2:
            print(f"  {a:11s} vs {b:13s}: n={len(d)} mean={d.mean():+.4f} "
                  f"sd={d.std(ddof=1):.4f} range=[{d.min():+.4f},{d.max():+.4f}]")
        else:
            print(f"  {a:11s} vs {b:13s}: insufficient paired seeds (n={len(d)})")
