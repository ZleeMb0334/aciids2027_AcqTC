"""Acquisition-shortcut audit across benchmarks.

For each (benchmark, generator) subset: how much of the real/fake label is
predictable from the ACQUISITION CHANNEL alone -- container format, chroma
subsampling, JPEG quantisation table, pixel dimensions, and bits-per-pixel --
with no access to pixel content?

A high AUROC here means published detection numbers on that subset are, to that
extent, not measurements of generator-artifact detection.
"""
import io, json, os, sys, glob, collections
import numpy as np
from PIL import Image, JpegImagePlugin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "/workspace/df-fusion/results"
FS = "/workspace/df-fusion/data/forensynths/CNN_synth_testset"
N_PER_CLASS = 400


def meta_features(raw: bytes):
    """Features computable WITHOUT decoding pixel content."""
    im = Image.open(io.BytesIO(raw))
    w, h = im.size
    fmt = im.format or "?"
    nbytes = len(raw)
    f = {"w": w, "h": h, "px": w * h, "aspect": w / max(h, 1),
         "bpp": nbytes * 8.0 / max(w * h, 1),
         "is_jpeg": float(fmt == "JPEG"), "is_png": float(fmt == "PNG"),
         "square": float(w == h), "pow2": float((w & (w - 1)) == 0)}
    q = getattr(im, "quantization", None)
    if q:
        t = np.array(q[0], dtype=float)
        f.update({"q_mean": t.mean(), "q_max": t.max(), "q_dc": t.flatten()[0]})
        try:
            f["subsampling"] = float(JpegImagePlugin.get_sampling(im))
        except Exception:
            f["subsampling"] = -1.0
    else:
        f.update({"q_mean": -1.0, "q_max": -1.0, "q_dc": -1.0, "subsampling": -1.0})
    return f, fmt


def audit(name, reals, fakes, read):
    rng = np.random.default_rng(0)
    reals = list(rng.permutation(reals))[:N_PER_CLASS]
    fakes = list(rng.permutation(fakes))[:N_PER_CLASS]
    if len(reals) < 50 or len(fakes) < 50:
        return None
    X, y, fmts = [], [], {0: collections.Counter(), 1: collections.Counter()}
    for lab, items in ((0, reals), (1, fakes)):
        for it in items:
            try:
                f, fmt = meta_features(read(it))
            except Exception:
                continue
            X.append(f); y.append(lab); fmts[lab][fmt] += 1
    keys = sorted(X[0].keys())
    Xa = np.array([[x[k] for k in keys] for x in X]); ya = np.array(y)
    if len(np.unique(ya)) < 2:
        return None
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
    p = cross_val_predict(clf, Xa, ya, cv=5, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(ya, p))
    # which single feature is most responsible
    singles = {}
    for j, k in enumerate(keys):
        try:
            singles[k] = float(roc_auc_score(ya, Xa[:, j]))
        except Exception:
            singles[k] = 0.5
    best = max(singles, key=lambda k: abs(singles[k] - 0.5))
    return {"subset": name, "n_real": int((ya == 0).sum()), "n_fake": int((ya == 1).sum()),
            "shortcut_auroc": auc, "format_real": dict(fmts[0]), "format_fake": dict(fmts[1]),
            "best_single_feature": best, "best_single_auroc": singles[best],
            "feature_aurocs": singles}


rows = []

# ---------------------------------------------------------------- ForenSynths
def collect(d):
    out = []
    for sub in ("", "*/"):
        out += glob.glob(os.path.join(FS, d, sub, "0_real", "*"))
    return out


for gen in sorted(os.listdir(FS)):
    p = os.path.join(FS, gen)
    if not os.path.isdir(p):
        continue
    reals = glob.glob(f"{p}/0_real/*") + glob.glob(f"{p}/*/0_real/*")
    fakes = glob.glob(f"{p}/1_fake/*") + glob.glob(f"{p}/*/1_fake/*")
    r = audit(f"ForenSynths/{gen}", reals, fakes, lambda x: open(x, "rb").read())
    if r:
        rows.append(r)
        print(f"  ForenSynths/{gen:12s} shortcut AUROC={r['shortcut_auroc']:.4f} "
              f"best={r['best_single_feature']}({r['best_single_auroc']:.3f}) "
              f"fmt real={r['format_real']} fake={r['format_fake']}", flush=True)

# ------------------------------------------------------------------ GenImage
import data as D
idx = D.build_index("test", cache=True)
store = D.ShardStore(idx["files"])
recs = idx["records"]
for g in sorted({r["generator"] for r in recs}):
    sel = [i for i, r in enumerate(recs) if r["generator"] == g]
    reals = [i for i in sel if recs[i]["label"] == 0]
    fakes = [i for i in sel if recs[i]["label"] == 1]
    r = audit(f"GenImage/{g}", reals, fakes,
              lambda i: store.bytes_at(recs[int(i)]["shard"], recs[int(i)]["row"]))
    if r:
        rows.append(r)
        print(f"  GenImage/{g:24s} shortcut AUROC={r['shortcut_auroc']:.4f} "
              f"best={r['best_single_feature']}({r['best_single_auroc']:.3f}) "
              f"fmt real={r['format_real']} fake={r['format_fake']}", flush=True)

with open(f"{OUT}/benchmark_audit.json", "w") as f:
    json.dump(rows, f, indent=1)
print(f"[audit] wrote benchmark_audit.json ({len(rows)} subsets)", flush=True)
