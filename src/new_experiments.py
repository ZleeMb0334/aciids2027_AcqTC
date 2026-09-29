"""Final-review experiments for the acquisition-screened detector study.

Subcommands:
  modern     Evaluate preserved ForenSynths checkpoints on canonicalized
             GenImage sources, including diffusion models and Midjourney.
  modern-pgd Run targeted PGD on two canonicalized unseen sources.
  gate       Measure gate behaviour on clean, JPEG, and resized ProGAN inputs.
  pgd        Check PGD step-count and restart sensitivity at a small budget.
  linearity  Test the local spectral linearization on held-out perturbations.

Every command writes a self-describing JSON result. No manuscript value is
emitted unless model inference completed successfully.
"""
import argparse
import io
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from torch.utils.data import DataLoader, Dataset

import data as GI
import data_fs as FS
from attacks import a3_pgd, a4_jpeg, a4_resize, psnr
from models import FusionDetector
from spectral import BandProfile


DEVICE = "cuda"
HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.dirname(HERE)
PROJECT = os.path.dirname(ARTIFACT)
DEFAULT_RESULTS = (os.path.join(PROJECT, "results")
                   if os.path.isdir(os.path.join(PROJECT, "results"))
                   else os.path.join(ARTIFACT, "results"))


def parse_seeds(value):
    return [int(x) for x in value.split(",") if x.strip()]


def fusion_name(method):
    for suffix in ("_robust", "_srr", "_full", "_tc"):
        method = method.replace(suffix, "")
    return method


def resolve_checkpoint(results, method, seed):
    candidates = []
    if method == "semantic_l":
        candidates.append(os.path.join(
            results, f"ckpt_matchL__semantic__s{seed}.pt"))
    if method == "semantic_tc_l":
        candidates.append(os.path.join(
            results, f"ckpt_acqtcl__semantic_tc__s{seed}.pt"))
    if method == "gate_full" and seed == 0:
        candidates.append(os.path.join(results, "ckpt_gatefull__gate_full__s0.pt"))
    candidates += [
        os.path.join(results, f"ckpt_semtc__{method}__s{seed}.pt"),
        os.path.join(results, f"ckpt_fs__{method}__s{seed}.pt"),
        os.path.join(results, f"ckpt_{method}__{method}__s{seed}.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"checkpoint unavailable for {method}, seed {seed}: {candidates}")


def load_model(path, method):
    saved = torch.load(path, map_location="cpu")
    run_args = saved.get("args", {})
    arch = run_args.get("arch", "ViT-B-16-quickgelu")
    tune = run_args.get("tune", "ln")
    size = int(run_args.get("size", 224))
    model = FusionDetector(
        semantic_kw={"tune": tune, "arch": arch},
        noise_kw={"residual": "srm", "size": size},
        fusion=("semantic" if method in {"semantic_l", "semantic_tc_l"}
                else fusion_name(method)),
    ).to(DEVICE)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, {"arch": arch, "tune": tune, "size": size}


def metric_bundle(y, score):
    y = np.asarray(y)
    score = np.asarray(score)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
    }


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "n": int(len(values)),
    }


def jpeg_bytes(im, quality=None, qtables=None, subsampling=0):
    buf = io.BytesIO()
    if qtables is None:
        im.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    else:
        im.save(buf, format="JPEG", qtables=qtables, subsampling=subsampling)
    return buf.getvalue()


class CanonicalGenImageDS(Dataset):
    """History-matched canonical protocol for cross-family evaluation.

    PNG synthetic files first receive a quantization table sampled from real
    files of the same generator subset. Every image is then resized to a common
    128-pixel square, encoded as JPEG-95, and resized to the detector input.
    The label-independent final chain removes container and geometry fields.
    """

    def __init__(self, index, subset, qpool, model_size=224, canonical_size=128):
        self.store = GI.ShardStore(index["files"])
        self.source_indices = list(subset)
        self.recs = [index["records"][i] for i in subset]
        self.qpool = qpool
        self.model_size = model_size
        self.canonical_size = canonical_size

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        raw = self.store.bytes_at(rec["shard"], rec["row"])
        src = Image.open(io.BytesIO(raw))
        was_jpeg = src.format == "JPEG"
        im = src.convert("RGB")
        if not was_jpeg:
            if not self.qpool:
                raise RuntimeError("history matching requires real JPEG quantization tables")
            qt, subsampling = self.qpool[(i * 2654435761) % len(self.qpool)]
            im = Image.open(io.BytesIO(jpeg_bytes(im, qtables=qt, subsampling=subsampling))).convert("RGB")
        im = im.resize((self.canonical_size, self.canonical_size), Image.BICUBIC)
        canonical = jpeg_bytes(im, quality=95, subsampling=0)
        im = Image.open(io.BytesIO(canonical)).convert("RGB")
        im = im.resize((self.model_size, self.model_size), Image.BICUBIC)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        return (x, rec["label"], GI.GENS.index(rec["generator"]),
                self.source_indices[i], len(canonical))


def balanced_genimage_subset(index, generator, n, seed=0):
    rng = np.random.default_rng(seed)
    recs = index["records"]
    selected = []
    for label in (0, 1):
        pool = [i for i, row in enumerate(recs)
                if row["generator"] == generator and row["label"] == label]
        take = min(n // 2, len(pool))
        selected.extend(int(i) for i in rng.choice(pool, size=take, replace=False))
    return sorted(selected)


@torch.no_grad()
def evaluate_parts(model, loader, transform=None):
    values = {key: [] for key in
              ("y", "logit", "semantic", "spectral", "gate", "index", "bytes")}
    for batch in loader:
        x, y = batch[0].to(DEVICE, non_blocking=True), batch[1].numpy()
        if transform is not None:
            x = transform(x)
        pred = model(x, return_parts=True)
        values["y"].append(y)
        values["logit"].append(pred["logit"].float().cpu().numpy())
        values["semantic"].append(pred["sem_logit"].float().cpu().numpy())
        values["spectral"].append(pred["noise_logit"].float().cpu().numpy())
        values["gate"].append(pred["gate"].float().cpu().numpy())
        values["index"].append(batch[3].numpy())
        if len(batch) > 4:
            values["bytes"].append(batch[4].numpy())
    return {key: np.concatenate(parts) if parts else np.array([])
            for key, parts in values.items()}


def canonical_shortcut_auroc(y, byte_size):
    features = np.asarray(byte_size, dtype=float).reshape(-1, 1)
    labels = np.asarray(y, dtype=int)
    classifier = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
    prob = cross_val_predict(classifier, features, labels, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, prob))


def run_modern(args):
    methods = [name for name in args.methods.split(",") if name]
    semantic_tc_available = all(
        os.path.isfile(os.path.join(
            args.results, f"ckpt_semtc__semantic_tc__s{seed}.pt"))
        for seed in parse_seeds(args.seeds))
    if semantic_tc_available and "semantic_tc" not in methods:
        if not any(name in {"semantic_l", "semantic_tc_l"} for name in methods):
            methods.append("semantic_tc")
    index = GI.build_index("test", cache=True)
    datasets = {}
    metadata_audit = {}
    for generator in GI.GENS:
        subset = balanced_genimage_subset(index, generator, args.n, args.sample_seed)
        qpool = GI.sample_qtable_pool(index, subset, n=256, seed=args.sample_seed)
        dataset = CanonicalGenImageDS(index, subset, qpool, args.size, args.canonical_size)
        datasets[generator] = dataset
        print(f"[modern] {generator}: n={len(dataset)} qtables={len(qpool)}", flush=True)

    rows = []
    predictions = {}
    robustness_generators = set(args.robustness_generators.split(","))
    unknown = robustness_generators - set(datasets)
    if unknown:
        raise ValueError(f"unknown robustness generators: {sorted(unknown)}")
    transforms = {
        "clean": None,
        "jpeg70": lambda x: a4_jpeg(x, 70),
        "resize_half": lambda x: a4_resize(x, 0.5),
    }
    for method in methods:
        for seed in parse_seeds(args.seeds):
            path = resolve_checkpoint(args.results, method, seed)
            model, config = load_model(path, method)
            for generator, dataset in datasets.items():
                loader = DataLoader(dataset, batch_size=args.bs, num_workers=args.workers,
                                    pin_memory=True)
                conditions = transforms if generator in robustness_generators else {"clean": None}
                for condition, transform in conditions.items():
                    values = evaluate_parts(model, loader, transform)
                    if condition == "clean" and generator not in metadata_audit:
                        metadata_audit[generator] = canonical_shortcut_auroc(
                            values["y"], values["bytes"])
                    row = {
                        "method": method,
                        "seed": seed,
                        "generator": generator,
                        "condition": condition,
                        "n": int(len(values["y"])),
                        "checkpoint": os.path.basename(path),
                        "config": config,
                        "fused": metric_bundle(values["y"], values["logit"]),
                        "semantic_logit": metric_bundle(values["y"], values["semantic"]),
                        "spectral_logit": metric_bundle(values["y"], values["spectral"]),
                        "gate_mean": float(values["gate"].mean()),
                        "gate_real_mean": float(values["gate"][values["y"] == 0].mean()),
                        "gate_fake_mean": float(values["gate"][values["y"] == 1].mean()),
                    }
                    rows.append(row)
                    prefix = re.sub(
                        r"[^A-Za-z0-9]+", "_",
                        f"{method}_s{seed}_{generator}_{condition}")
                    for key in ("index", "y", "logit", "semantic", "spectral", "gate", "bytes"):
                        predictions[f"{prefix}__{key}"] = values[key]
                    print(f"[modern] {method} s{seed} {generator} {condition}: "
                          f"AUROC={row['fused']['auroc']:.4f} gate={row['gate_mean']:.3f}",
                          flush=True)
            del model
            torch.cuda.empty_cache()

    summary = {}
    for method in methods:
        summary[method] = {}
        for generator in GI.GENS:
            selected = [row for row in rows
                        if row["method"] == method and row["generator"] == generator
                        and row["condition"] == "clean"]
            summary[method][generator] = {
                "auroc": summarize([row["fused"]["auroc"] for row in selected]),
                "ap": summarize([row["fused"]["ap"] for row in selected]),
            }
        seed_macros = []
        for seed in parse_seeds(args.seeds):
            selected = [row["fused"]["auroc"] for row in rows
                        if row["method"] == method and row["seed"] == seed
                        and row["condition"] == "clean"]
            seed_macros.append(float(np.mean(selected)))
        summary[method]["macro"] = summarize(seed_macros)

    robustness = {}
    for method in methods:
        robustness[method] = {}
        for condition in ("jpeg70", "resize_half"):
            robustness[method][condition] = {}
            for generator in sorted(robustness_generators):
                selected = [row["fused"]["auroc"] for row in rows
                            if row["method"] == method
                            and row["generator"] == generator
                            and row["condition"] == condition]
                robustness[method][condition][generator] = summarize(selected)
            seed_macros = []
            for seed in parse_seeds(args.seeds):
                selected = [row["fused"]["auroc"] for row in rows
                            if row["method"] == method and row["seed"] == seed
                            and row["condition"] == condition]
                seed_macros.append(float(np.mean(selected)))
            robustness[method][condition]["macro"] = summarize(seed_macros)

    output = {
        "experiment": "canonicalized GenImage transfer",
        "protocol": {
            "sample_seed": args.sample_seed,
            "n_per_generator": args.n,
            "canonical_size": args.canonical_size,
            "model_size": args.size,
            "history_matching": "sample real JPEG quantization tables within each generator subset",
            "common_chain": "resize to canonical square, JPEG-95, resize to detector input",
            "robustness_generators": sorted(robustness_generators),
            "postprocessing": ["JPEG-70", "half-scale downsample-upsample"],
            "methods": methods,
        },
        "canonical_shortcut_auroc_from_byte_size": metadata_audit,
        "rows": rows,
        "summary": summary,
        "robustness": robustness,
    }
    path = os.path.join(args.results, args.output_stem + ".json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    np.savez_compressed(os.path.join(args.results, args.output_stem + "_predictions.npz"),
                        **predictions)
    print(f"[modern] wrote {path}")


def pgd_external_evaluate(model, loader, eps, steps, seed):
    """Attack synthetic examples while retaining clean real scores."""
    yy, clean_scores, attacked_scores, indices, distortion = [], [], [], [], []
    for batch_index, batch in enumerate(loader):
        x = batch[0].to(DEVICE, non_blocking=True)
        labels = batch[1].numpy()
        with torch.no_grad():
            clean = model(x).float().cpu().numpy()
        fake_mask = torch.tensor(labels == 1, device=DEVICE)
        changed = x.clone()
        if fake_mask.any():
            torch.manual_seed(seed * 100000 + batch_index)
            changed[fake_mask] = a3_pgd(
                model, x[fake_mask], 0.0, eps=eps, steps=steps,
                alpha=eps / 8, gate_aware=False)
            distortion.append(psnr(x[fake_mask], changed[fake_mask]))
        with torch.no_grad():
            attacked = model(changed).float().cpu().numpy()
        yy.append(labels)
        clean_scores.append(clean)
        attacked_scores.append(attacked)
        indices.append(batch[3].numpy())
    y = np.concatenate(yy)
    clean = np.concatenate(clean_scores)
    attacked = np.concatenate(attacked_scores)
    return ({
        "clean_auroc": float(roc_auc_score(y, clean)),
        "attacked_auroc": float(roc_auc_score(y, attacked)),
        "psnr": float(np.mean(np.concatenate(distortion))),
    }, {
        "index": np.concatenate(indices),
        "y": y,
        "clean_score": clean,
        "attacked_score": attacked,
    })


def run_modern_pgd(args):
    """Evaluate one-sided adaptive evasion on canonicalized unseen sources."""
    outpath = os.path.join(args.results, "modern_pgd.json")
    rows = []
    if os.path.isfile(outpath):
        with open(outpath) as stream:
            rows = json.load(stream).get("rows", [])
    completed = {(row["method"], int(row["seed"]), row["generator"])
                 for row in rows}
    index = GI.build_index("test", cache=True)
    generators = [name for name in args.generators.split(",") if name]
    unknown = set(generators) - set(GI.GENS)
    if unknown:
        raise ValueError(f"unknown generators: {sorted(unknown)}")
    datasets = {}
    for generator in generators:
        subset = balanced_genimage_subset(index, generator, args.n, args.sample_seed)
        qpool = GI.sample_qtable_pool(index, subset, n=256, seed=args.sample_seed)
        datasets[generator] = CanonicalGenImageDS(
            index, subset, qpool, args.size, args.canonical_size)

    for method in [name for name in args.methods.split(",") if name]:
        for seed in parse_seeds(args.seeds):
            path = resolve_checkpoint(args.results, method, seed)
            model, config = load_model(path, method)
            for generator, dataset in datasets.items():
                key = (method, seed, generator)
                if key in completed:
                    print(f"[modern-pgd] keeping {method} s{seed} {generator}", flush=True)
                    continue
                loader = DataLoader(dataset, batch_size=args.bs,
                                    num_workers=args.workers, pin_memory=True)
                metrics, predictions = pgd_external_evaluate(
                    model, loader, args.eps / 255.0, args.steps, seed)
                row = {
                    "method": method,
                    "seed": seed,
                    "generator": generator,
                    "n": len(dataset),
                    "checkpoint": os.path.basename(path),
                    "config": config,
                    **metrics,
                }
                rows.append(row)
                with open(outpath, "w") as stream:
                    json.dump({
                        "experiment": "canonicalized GenImage adaptive PGD",
                        "protocol": {
                            "sample_seed": args.sample_seed,
                            "canonical_size": args.canonical_size,
                            "history_matching": "real-file JPEG quantization tables",
                            "common_chain": "resize, JPEG-95, detector resize",
                            "target": 0,
                            "epsilon_over_255": args.eps,
                            "steps": args.steps,
                            "alpha": "epsilon/8",
                            "restarts": 1,
                            "random_initialization": True,
                            "perturbed_class": "synthetic only",
                        },
                        "rows": rows,
                    }, stream, indent=2)
                predpath = os.path.join(
                    args.results,
                    f"modern_pgd_predictions__{method}__s{seed}__{generator}.npz")
                np.savez_compressed(predpath, **predictions)
                print(f"[modern-pgd] {method} s{seed} {generator}: "
                      f"{metrics['clean_auroc']:.4f} -> "
                      f"{metrics['attacked_auroc']:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
    print(f"[modern-pgd] wrote {outpath}")


def fs_loader(generator, n, seed, size, bs, workers):
    dataset = FS.FSDataset(FS.subset(generator, n, seed), size, False, seed)
    return DataLoader(dataset, batch_size=bs, num_workers=workers, pin_memory=True)


def run_gate(args):
    transforms = {
        "clean": None,
        "jpeg70": lambda x: a4_jpeg(x, 70),
        "resize_half": lambda x: a4_resize(x, 0.5),
    }
    rows = []
    predictions = {}
    for seed in parse_seeds(args.seeds):
        path = resolve_checkpoint(args.results, "gate_full", seed)
        model, config = load_model(path, "gate_full")
        for generator in args.generators.split(","):
            loader = fs_loader(generator, args.n, args.sample_seed, args.size, args.bs, args.workers)
            for name, transform in transforms.items():
                values = evaluate_parts(model, loader, transform)
                y = values["y"]
                row = {
                    "seed": seed,
                    "generator": generator,
                    "condition": name,
                    "n": int(len(y)),
                    "checkpoint": os.path.basename(path),
                    "config": config,
                    "fused": metric_bundle(y, values["logit"]),
                    "semantic": metric_bundle(y, values["semantic"]),
                    "spectral": metric_bundle(y, values["spectral"]),
                    "gate_mean": float(values["gate"].mean()),
                    "gate_sd": float(values["gate"].std(ddof=1)),
                    "gate_real_mean": float(values["gate"][y == 0].mean()),
                    "gate_fake_mean": float(values["gate"][y == 1].mean()),
                }
                rows.append(row)
                prefix = re.sub(r"[^A-Za-z0-9]+", "_", f"s{seed}_{generator}_{name}")
                for key in ("index", "y", "logit", "semantic", "spectral", "gate"):
                    predictions[f"{prefix}__{key}"] = values[key]
                print(f"[gate] s{seed} {generator} {name}: "
                      f"AUROC={row['fused']['auroc']:.4f} gate={row['gate_mean']:.3f}",
                      flush=True)
        del model
        torch.cuda.empty_cache()

    output = {"experiment": "gate diagnostics", "rows": rows}
    path = os.path.join(args.results, "gate_diagnostics.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    np.savez_compressed(os.path.join(args.results, "gate_diagnostics_predictions.npz"),
                        **predictions)
    print(f"[gate] wrote {path}")


@torch.no_grad()
def scores(model, loader):
    yy, ss = [], []
    for x, y, _, _ in loader:
        yy.append(y.numpy())
        ss.append(model(x.to(DEVICE, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(yy), np.concatenate(ss)


def validation_threshold(model, seed, size, bs, workers, fpr=0.05):
    items = FS.train_subset(8000, seed=seed)
    perm = np.random.default_rng(seed).permutation(len(items))
    nval = len(perm) // 10
    val = [items[i] for i in perm[:nval]]
    loader = DataLoader(FS.FSDataset(val, size, False, seed), batch_size=bs,
                        num_workers=workers, pin_memory=True)
    y, score = scores(model, loader)
    threshold = float(np.quantile(score[y == 0], 1.0 - fpr))
    return threshold, {"n": int(len(y)), "target_fpr": fpr,
                       "real_fpr": float(np.mean(score[y == 0] >= threshold))}


def pgd_evaluate(model, loader, threshold, eps, steps, restarts, seed):
    yy, clean_scores, attacked_scores = [], [], []
    for batch_index, (x, y, _, _) in enumerate(loader):
        x = x.to(DEVICE, non_blocking=True)
        labels = y.numpy()
        with torch.no_grad():
            clean = model(x).float().cpu().numpy()
        fake_mask = torch.tensor(labels == 1, device=DEVICE)
        attacked = clean.copy()
        if fake_mask.any():
            xf = x[fake_mask]
            best_score = None
            for restart in range(restarts):
                torch.manual_seed(seed * 100000 + batch_index * 100 + restart)
                candidate = a3_pgd(model, xf, 0.0, eps=eps, steps=steps,
                                   alpha=eps / 8, gate_aware=False)
                with torch.no_grad():
                    candidate_score = model(candidate).float().cpu().numpy()
                if best_score is None:
                    best_score = candidate_score
                else:
                    best_score = np.minimum(best_score, candidate_score)
            attacked[labels == 1] = best_score
        yy.append(labels)
        clean_scores.append(clean)
        attacked_scores.append(attacked)
    y = np.concatenate(yy)
    clean = np.concatenate(clean_scores)
    attacked = np.concatenate(attacked_scores)
    fake = y == 1
    initially_detected = fake & (clean >= threshold)
    return {
        "auroc": float(roc_auc_score(y, attacked)),
        "clean_auroc": float(roc_auc_score(y, clean)),
        "clean_fake_tpr": float(np.mean(clean[fake] >= threshold)),
        "attacked_fake_tpr": float(np.mean(attacked[fake] >= threshold)),
        "targeted_evasion_rate": float(np.mean(attacked[fake] < threshold)),
        "conditional_evasion_rate": (
            float(np.mean(attacked[initially_detected] < threshold))
            if initially_detected.any() else None),
        "initially_detected_fake_n": int(initially_detected.sum()),
    }


def run_pgd(args):
    path = resolve_checkpoint(args.results, "gate_full", args.seed)
    model, config = load_model(path, "gate_full")
    threshold, threshold_info = validation_threshold(
        model, args.seed, args.size, args.bs, args.workers, args.fpr)
    loader = fs_loader("progan", args.n, args.sample_seed, args.size, args.bs, args.workers)
    settings = [(steps, 1) for steps in (10, 20, 40, 80)] + [(40, args.restarts)]
    rows = []
    for steps, restarts in settings:
        result = pgd_evaluate(model, loader, threshold, args.eps / 255.0,
                              steps, restarts, args.seed)
        result.update({"steps": steps, "restarts": restarts})
        rows.append(result)
        print(f"[pgd] steps={steps} restarts={restarts}: "
              f"AUROC={result['auroc']:.4f} evasion={result['targeted_evasion_rate']:.3f}",
              flush=True)
    output = {
        "experiment": "PGD optimization check",
        "method": "gate_full",
        "seed": args.seed,
        "checkpoint": os.path.basename(path),
        "config": config,
        "test_n": args.n,
        "test_sample_seed": args.sample_seed,
        "epsilon_over_255": args.eps,
        "alpha": "epsilon/8",
        "target": 0,
        "random_initialization": True,
        "projection": "L-infinity ball intersected with [0,1]",
        "threshold": threshold,
        "threshold_info": threshold_info,
        "rows": rows,
    }
    outpath = os.path.join(args.results, "pgd_check.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[pgd] wrote {outpath}")


def pgd_table_evaluate(model, loader, eps, seed):
    yy, score, indices, distortion = [], [], [], []
    for batch_index, (x, y, _, sample_index) in enumerate(loader):
        x = x.to(DEVICE, non_blocking=True)
        labels = y.numpy()
        fake_mask = torch.tensor(labels == 1, device=DEVICE)
        changed = x.clone()
        if fake_mask.any():
            torch.manual_seed(seed * 100000 + batch_index)
            changed[fake_mask] = a3_pgd(
                model, x[fake_mask], 0.0, eps=eps, steps=40,
                alpha=eps / 8, gate_aware=False)
            distortion.append(psnr(x[fake_mask], changed[fake_mask]))
        with torch.no_grad():
            score.append(model(changed).float().cpu().numpy())
        yy.append(labels)
        indices.append(sample_index.numpy())
    y = np.concatenate(yy)
    scores_ = np.concatenate(score)
    result = {
        "auroc": float(roc_auc_score(y, scores_)),
        "psnr": float(np.mean(np.concatenate(distortion))),
    }
    return result, y, scores_, np.concatenate(indices)


def run_pgd_table(args):
    outpath = os.path.join(args.results, "pgd_reeval.json")
    rows = []
    if os.path.isfile(outpath):
        with open(outpath) as stream:
            rows = json.load(stream).get("rows", [])
    completed = {(row["method"], int(row["seed"])) for row in rows}
    for method in args.methods.split(","):
        for seed in parse_seeds(args.seeds):
            if (method, seed) in completed:
                print(f"[pgd-table] keeping {method} s{seed}", flush=True)
                continue
            predictions = {}
            path = resolve_checkpoint(args.results, method, seed)
            model, config = load_model(path, method)
            loader = fs_loader("progan", args.n, seed, args.size, args.bs, args.workers)
            result = {
                "method": method,
                "seed": seed,
                "checkpoint": os.path.basename(path),
                "config": config,
                "n": args.n,
                "loss": "fused-logit binary cross-entropy toward label 0",
                "steps": 40,
                "alpha": "epsilon/8",
                "restarts": 1,
                "random_initialization": True,
            }
            for numerator in (1, 2, 4, 8):
                metrics, labels, scores_, sample_indices = pgd_table_evaluate(
                    model, loader, numerator / 255.0, seed)
                result[f"eps{numerator}"] = metrics
                prefix = f"{method}_s{seed}_eps{numerator}"
                predictions[prefix + "__y"] = labels
                predictions[prefix + "__score"] = scores_
                predictions[prefix + "__index"] = sample_indices
                print(f"[pgd-table] {method} s{seed} eps={numerator}/255: "
                      f"AUROC={result[f'eps{numerator}']['auroc']:.4f}", flush=True)
            rows.append(result)
            with open(outpath, "w") as f:
                json.dump({"experiment": "corrected fused-loss PGD table",
                           "rows": rows}, f, indent=2)
            prediction_path = os.path.join(
                args.results, f"pgd_predictions__{method}__s{seed}.npz")
            np.savez_compressed(prediction_path, **predictions)
            del model
            torch.cuda.empty_cache()
    print(f"[pgd-table] wrote {outpath}")


def apply_frequency_shift(x, band_map, shift):
    transformed = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
    magnitude = transformed.abs().clamp(min=1e-8)
    phase = torch.angle(transformed)
    field = torch.zeros_like(band_map, dtype=torch.float32)
    for band, amount in enumerate(shift):
        field[band_map == band] = float(amount)
    changed = torch.exp(torch.log(magnitude) + field[None, None]) * torch.exp(1j * phase)
    return torch.fft.ifft2(torch.fft.ifftshift(changed, dim=(-2, -1))).real.clamp(0, 1)


def r2_score(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator) if denominator > 0 else None


@torch.no_grad()
def probe(model, x, band_profile, shifts):
    base = model(x, return_parts=True)
    base_phi = model.noi(x, return_phi=True)[2]
    base_linear = model.noi.phi_lin(base_phi).mean()
    dphi, full_delta, linear_delta = [], [], []
    for shift in shifts:
        changed = apply_frequency_shift(x, band_profile.band, shift)
        pred = model(changed, return_parts=True)
        phi = model.noi(changed, return_phi=True)[2]
        dphi.append((phi - base_phi).mean(0).float().cpu().numpy())
        full_delta.append(float((pred["logit"] - base["logit"]).mean()))
        linear_delta.append(float(model.noi.phi_lin(phi).mean() - base_linear))
    return np.asarray(dphi), np.asarray(full_delta), np.asarray(linear_delta)


def run_linearity(args):
    path = resolve_checkpoint(args.results, "gate_full", args.seed)
    model, config = load_model(path, "gate_full")
    loader = fs_loader("progan", args.n, args.sample_seed, args.size, args.n, 0)
    x = next(iter(loader))[0].to(DEVICE)
    bands = BandProfile(args.size, 32, 2).to(DEVICE)
    rng = np.random.default_rng(args.probe_seed)
    train_shifts = rng.normal(size=(args.train_probes, bands.B)) * args.fit_scale
    dphi, delta, _ = probe(model, x, bands, train_shifts)
    weight = np.linalg.lstsq(dphi, delta, rcond=None)[0]
    rank = int(np.linalg.matrix_rank(dphi))
    linear_weight = model.noi.phi_lin.weight.detach().squeeze(0).cpu().numpy()
    tests = {}
    for scale in (0.005, 0.01, 0.02):
        shifts = rng.normal(size=(args.test_probes, bands.B)) * scale
        test_phi, actual, actual_linear = probe(model, x, bands, shifts)
        predicted = test_phi @ weight
        predicted_linear = test_phi @ linear_weight
        tests[f"scale_{scale:.3f}"] = {
            "full_r2": r2_score(actual, predicted),
            "full_corr2": float(np.corrcoef(actual, predicted)[0, 1] ** 2),
            "full_rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            "linear_path_r2": r2_score(actual_linear, predicted_linear),
            "linear_path_corr2": float(np.corrcoef(actual_linear, predicted_linear)[0, 1] ** 2),
        }
        print(f"[linearity] scale={scale:.3f}: "
              f"R2={tests[f'scale_{scale:.3f}']['full_r2']:.3f} "
              f"linear={tests[f'scale_{scale:.3f}']['linear_path_r2']:.3f}", flush=True)
    output = {
        "experiment": "held-out local spectral linearization",
        "measurement_version": 3,
        "method": "gate_full",
        "seed": args.seed,
        "checkpoint": os.path.basename(path),
        "config": config,
        "images": int(len(x)),
        "sample_seed": args.sample_seed,
        "probe_seed": args.probe_seed,
        "fit_scale": args.fit_scale,
        "train_probes": args.train_probes,
        "test_probes_per_scale": args.test_probes,
        "design_rank": rank,
        "bands": int(bands.B),
        "tests": tests,
    }
    outpath = os.path.join(args.results, "linearity_check.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[linearity] wrote {outpath}")


def run_benchmark(args):
    loader = fs_loader("progan", args.bs, args.sample_seed, args.size, args.bs, 0)
    x = next(iter(loader))[0].to(DEVICE)
    rows = []
    methods = ["semantic", "semantic_tc", "gate_full"]
    if os.path.exists(os.path.join(
            args.results, f"ckpt_acqtcl__semantic_tc__s{args.seed}.pt")):
        methods.append("semantic_tc_l")
    for method in methods:
        path = resolve_checkpoint(args.results, method, args.seed)
        model, config = load_model(path, method)
        if method in {"semantic", "semantic_tc", "semantic_tc_l"}:
            del model.noi
            inference = lambda batch: model.sem(batch)[1]
        else:
            inference = model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for _ in range(args.warmup):
                inference(x)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iterations):
                inference(x)
            end.record()
            torch.cuda.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        row = {
            "method": method,
            "seed": args.seed,
            "checkpoint": os.path.basename(path),
            "config": config,
            "batch_size": int(len(x)),
            "warmup_iterations": args.warmup,
            "timed_iterations": args.iterations,
            "total_ms": elapsed_ms,
            "milliseconds_per_image": elapsed_ms / (args.iterations * len(x)),
            "images_per_second": args.iterations * len(x) / (elapsed_ms / 1000.0),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        rows.append(row)
        print(f"[benchmark] {method}: {row['milliseconds_per_image']:.3f} ms/image, "
              f"{row['images_per_second']:.1f} image/s", flush=True)
        del inference, model
        torch.cuda.empty_cache()
        time.sleep(1)
    output = {
        "experiment": "single-GPU inference benchmark",
        "device": torch.cuda.get_device_name(),
        "precision": "float32; autocast disabled",
        "rows": rows,
    }
    outpath = os.path.join(args.results, "inference_benchmark.json")
    with open(outpath, "w") as stream:
        json.dump(output, stream, indent=2)
    print(f"[benchmark] wrote {outpath}")


def common(parser):
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=0)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    modern = sub.add_parser("modern")
    common(modern)
    modern.add_argument("--n", type=int, default=1000)
    modern.add_argument("--canonical-size", type=int, default=128)
    modern.add_argument("--methods", default="semantic,gate_full")
    modern.add_argument("--seeds", default="0,1,2")
    modern.add_argument("--output-stem", default="modern_genimage")
    modern.add_argument("--robustness-generators",
                        default="Midjourney,stable_diffusion_v_1_5")
    modern.set_defaults(function=run_modern)

    modern_pgd = sub.add_parser("modern-pgd")
    common(modern_pgd)
    modern_pgd.add_argument("--n", type=int, default=1000)
    modern_pgd.add_argument("--canonical-size", type=int, default=128)
    modern_pgd.add_argument("--methods", default="semantic_tc,gate_full")
    modern_pgd.add_argument("--seeds", default="0")
    modern_pgd.add_argument("--generators",
                            default="Midjourney,stable_diffusion_v_1_5")
    modern_pgd.add_argument("--eps", type=float, default=4.0)
    modern_pgd.add_argument("--steps", type=int, default=40)
    modern_pgd.set_defaults(function=run_modern_pgd)

    gate = sub.add_parser("gate")
    common(gate)
    gate.add_argument("--n", type=int, default=1000)
    gate.add_argument("--generators", default="progan")
    gate.add_argument("--seeds", default="0,1,2")
    gate.set_defaults(function=run_gate)

    pgd = sub.add_parser("pgd")
    common(pgd)
    pgd.add_argument("--n", type=int, default=128)
    pgd.add_argument("--seed", type=int, default=0)
    pgd.add_argument("--eps", type=float, default=0.25,
                     help="L-infinity budget numerator in units of /255")
    pgd.add_argument("--restarts", type=int, default=5)
    pgd.add_argument("--fpr", type=float, default=0.05)
    pgd.set_defaults(function=run_pgd)

    pgd_table = sub.add_parser("pgd-table")
    common(pgd_table)
    pgd_table.add_argument("--n", type=int, default=1000)
    pgd_table.add_argument("--methods", default="gate,gate_robust,gate_srr,gate_full")
    pgd_table.add_argument("--seeds", default="0,1,2")
    pgd_table.set_defaults(function=run_pgd_table)

    linearity = sub.add_parser("linearity")
    common(linearity)
    linearity.add_argument("--n", type=int, default=32)
    linearity.add_argument("--seed", type=int, default=0)
    linearity.add_argument("--probe-seed", type=int, default=2027)
    linearity.add_argument("--fit-scale", type=float, default=0.01)
    linearity.add_argument("--train-probes", type=int, default=128)
    linearity.add_argument("--test-probes", type=int, default=64)
    linearity.set_defaults(function=run_linearity)

    benchmark = sub.add_parser("benchmark")
    common(benchmark)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--warmup", type=int, default=20)
    benchmark.add_argument("--iterations", type=int, default=100)
    benchmark.set_defaults(function=run_benchmark)

    args = parser.parse_args()
    os.makedirs(args.results, exist_ok=True)
    args.function(args)


if __name__ == "__main__":
    main()
