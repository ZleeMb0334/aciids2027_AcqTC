"""Repeated acquisition audit for sampling and fold sensitivity."""
import argparse
import io
import json
import os
import glob

import numpy as np
from PIL import Image, JpegImagePlugin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import data as GI


OUT = os.environ.get("DF_OUT", "/workspace/df-fusion/results")
FS = os.environ.get("DF_FS", "/workspace/df-fusion/data/forensynths")
N_PER_CLASS = 400
POOL_PER_CLASS = 1000
FS_GENERATORS = [
    "biggan", "crn", "cyclegan", "deepfake", "gaugan", "imle",
    "progan", "san", "seeingdark", "stargan", "stylegan",
]


def meta_features(raw):
    image = Image.open(io.BytesIO(raw))
    width, height = image.size
    fmt = image.format or "?"
    values = {
        "w": width,
        "h": height,
        "px": width * height,
        "aspect": width / max(height, 1),
        "bpp": len(raw) * 8.0 / max(width * height, 1),
        "is_jpeg": float(fmt == "JPEG"),
        "is_png": float(fmt == "PNG"),
        "square": float(width == height),
        "pow2": float((width & (width - 1)) == 0),
    }
    quantization = getattr(image, "quantization", None)
    if quantization:
        table = np.asarray(quantization[0], dtype=float)
        values.update({"q_mean": table.mean(), "q_max": table.max(), "q_dc": table.ravel()[0]})
        try:
            values["subsampling"] = float(JpegImagePlugin.get_sampling(image))
        except Exception:
            values["subsampling"] = -1.0
    else:
        values.update({"q_mean": -1.0, "q_max": -1.0, "q_dc": -1.0, "subsampling": -1.0})
    return values


def feature_pool(reals, fakes, read):
    rng = np.random.default_rng(2027)
    chosen = {
        0: list(rng.permutation(reals))[:POOL_PER_CLASS],
        1: list(rng.permutation(fakes))[:POOL_PER_CLASS],
    }
    rows = []
    labels = []
    for label in (0, 1):
        for item in chosen[label]:
            try:
                rows.append(meta_features(read(item)))
                labels.append(label)
            except Exception:
                continue
    keys = sorted(rows[0])
    matrix = np.asarray([[row[key] for key in keys] for row in rows], dtype=float)
    return matrix, np.asarray(labels, dtype=int), keys


def repeat_scores(matrix, labels):
    values = []
    for sample_seed in range(5):
        rng = np.random.default_rng(sample_seed)
        indices = []
        for label in (0, 1):
            pool = np.flatnonzero(labels == label)
            indices.extend(int(i) for i in rng.choice(
                pool, size=min(N_PER_CLASS, len(pool)), replace=False))
        indices = np.asarray(indices)
        x, y = matrix[indices], labels[indices]
        for fold_seed in range(5):
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
            classifier = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=fold_seed)
            probability = cross_val_predict(
                classifier, x, y, cv=folds, method="predict_proba")[:, 1]
            values.append({
                "sample_seed": sample_seed,
                "fold_seed": fold_seed,
                "auroc": float(roc_auc_score(y, probability)),
            })
    scores = np.asarray([row["auroc"] for row in values])
    summary = {
        "mean": float(scores.mean()),
        "sd": float(scores.std(ddof=1)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "n": int(len(scores)),
        "below_0.60_rate": float(np.mean(scores < 0.60)),
        "below_0.70_rate": float(np.mean(scores < 0.70)),
        "below_0.80_rate": float(np.mean(scores < 0.80)),
    }
    return values, summary


def fs_paths(generator):
    root = os.path.join(FS, "CNN_synth_testset", generator)
    reals = glob.glob(f"{root}/0_real/*") + glob.glob(f"{root}/*/0_real/*")
    fakes = glob.glob(f"{root}/1_fake/*") + glob.glob(f"{root}/*/1_fake/*")
    return reals, fakes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(OUT, "audit_repeats.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if os.path.isfile(args.out) and not args.force:
        print(f"[audit-repeat] keeping existing {args.out}", flush=True)
        return
    output = {
        "protocol": {
            "pool_per_class": POOL_PER_CLASS,
            "sample_per_class": N_PER_CLASS,
            "sample_seeds": 5,
            "fold_seeds": 5,
            "folds": 5,
            "grouping": "file-level stratification; source-batch identifiers unavailable",
            "classifier": "sklearn GradientBoostingClassifier(n_estimators=100,max_depth=3)",
        },
        "subsets": [],
    }
    for generator in FS_GENERATORS:
        reals, fakes = fs_paths(generator)
        if min(len(reals), len(fakes)) < 50:
            continue
        matrix, labels, keys = feature_pool(reals, fakes, lambda path: open(path, "rb").read())
        values, summary = repeat_scores(matrix, labels)
        output["subsets"].append({
            "subset": f"ForenSynths/{generator}",
            "features": keys,
            "pool_real": int(np.sum(labels == 0)),
            "pool_fake": int(np.sum(labels == 1)),
            "summary": summary,
            "runs": values,
        })
        print(f"[audit-repeat] ForenSynths/{generator}: "
              f"{summary['mean']:.4f}+/-{summary['sd']:.4f}", flush=True)

    index = GI.build_index("test", cache=True)
    store = GI.ShardStore(index["files"])
    records = index["records"]
    for generator in GI.GENS:
        selected = [i for i, row in enumerate(records) if row["generator"] == generator]
        reals = [i for i in selected if records[i]["label"] == 0]
        fakes = [i for i in selected if records[i]["label"] == 1]
        matrix, labels, keys = feature_pool(
            reals, fakes,
            lambda i: store.bytes_at(records[int(i)]["shard"], records[int(i)]["row"]))
        values, summary = repeat_scores(matrix, labels)
        output["subsets"].append({
            "subset": f"GenImage/{generator}",
            "features": keys,
            "pool_real": int(np.sum(labels == 0)),
            "pool_fake": int(np.sum(labels == 1)),
            "summary": summary,
            "runs": values,
        })
        print(f"[audit-repeat] GenImage/{generator}: "
              f"{summary['mean']:.4f}+/-{summary['sd']:.4f}", flush=True)

    path = args.out
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[audit-repeat] wrote {path}")


if __name__ == "__main__":
    main()
