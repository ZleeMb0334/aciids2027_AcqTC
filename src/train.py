"""One run = one (method, fold, seed) -> checkpoint + tidy score CSV.

Every knob that a table column varies is a CLI flag, so an ablation row is a
different command line, never a different code path.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from models import FusionDetector, count_trainable, trainable_breakdown
from losses import focal_bce, paired_supcon, removal_consistency
from attacks import SpectralReference, a1_fingerprint_removal


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def make_loader(idx, subset, bs, train, seed, pairing="class", workers=8):
    ds = D.GenImageDS(idx, subset, train=train, seed=seed)
    if train:
        samp = D.PairedBatchSampler(ds.recs, bs, mode=pairing, seed=seed)
        return DataLoader(ds, batch_sampler=samp, num_workers=workers, pin_memory=True,
                          persistent_workers=workers > 0)
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0)


@torch.no_grad()
def score(model, loader, dev):
    model.eval()
    out = {k: [] for k in ("logit", "gate", "sem", "noi", "y", "gen")}
    for x, y, g, _ in loader:
        x = x.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x, return_parts=True)
        out["logit"].append(p["logit"].float().cpu().numpy())
        out["gate"].append(p["gate"].float().cpu().numpy())
        out["sem"].append(p["sem_logit"].float().cpu().numpy())
        out["noi"].append(p["noise_logit"].float().cpu().numpy())
        out["y"].append(y.numpy()); out["gen"].append(g.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="held-out generator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fusion", default="gate",
                    choices=["gate", "concat", "static", "mean", "semantic", "noise"])
    ap.add_argument("--tune", default="ln", choices=["ln", "full", "probe"])
    ap.add_argument("--arch", default="ViT-B-16-quickgelu")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--residual", default="srm", choices=["srm", "gauss", "none"])
    ap.add_argument("--pairing", default="class", choices=["class", "balanced"])
    ap.add_argument("--robust", type=int, default=1, help="removal-resistant training")
    ap.add_argument("--curriculum", type=int, default=1)
    ap.add_argument("--consistency", type=int, default=1)
    ap.add_argument("--lam-metric", type=float, default=0.5)
    ap.add_argument("--lam-robust", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-train-per-gen", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default="/workspace/df-fusion/results")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    run = f"{a.tag}__{a.fold}__s{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(f"{a.out}/ckpt", exist_ok=True)
    t0 = time.time()

    idx = D.load_indices()
    sp = D.logo_split(idx, a.fold, a.n_train_per_gen, a.n_test, seed=a.seed)
    ltr = make_loader(idx["train"], sp["train"], a.bs, True, a.seed, a.pairing, a.workers)
    lva = make_loader(idx["train"], sp["val"], a.bs, False, a.seed, workers=a.workers)
    lin = make_loader(idx["test"], sp["test_in"], a.bs, False, a.seed, workers=a.workers)
    lout = make_loader(idx["test"], sp["test_out"], a.bs, False, a.seed, workers=a.workers)
    print(f"[{run}] train={len(sp['train'])} val={len(sp['val'])} "
          f"test_in={len(sp['test_in'])} test_out={len(sp['test_out'])}", flush=True)

    model = FusionDetector(
        semantic_kw={"arch": a.arch, "pretrained": a.pretrained, "tune": a.tune},
        noise_kw={"residual": a.residual}, fusion=a.fusion).to(dev)
    tp, ap_, pct = count_trainable(model)
    brk = trainable_breakdown(model)
    print(f"[{run}] trainable {tp}/{ap_} = {pct:.3f}%  LN={brk['ln_affine']} "
          f"({brk['ln_pct_of_backbone']:.3f}% of backbone)", flush=True)

    # Spectral reference estimated on TRAINING data only (no test leakage).
    refpath = f"{a.out}/ckpt/ref__{a.fold}__s{a.seed}.pt"
    if os.path.exists(refpath):
        ref = SpectralReference.load(refpath, dev)
    else:
        ref = SpectralReference.estimate(lva, dev, max_batches=30)
        ref.save(refpath)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * max(1, len(ltr)))
    hist = []
    for ep in range(a.epochs):
        model.train()
        frac = ep / max(1, a.epochs - 1)
        smax = min(1.0, frac / 0.4) if a.curriculum else 1.0
        lam_r = a.lam_robust * (smax if a.curriculum else 1.0)
        agg = {"cls": 0., "met": 0., "rob": 0., "drift": 0., "gate": 0., "n": 0}
        for x, y, g, _ in ltr:
            x = x.to(dev, non_blocking=True); y = y.float().to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(x, return_parts=True)
                lc = focal_bce(p["logit"], y)
                lm = paired_supcon(p["z"].float(), y)
                loss = lc + a.lam_metric * lm
                lr_ = torch.zeros((), device=dev)
                if a.robust and (y > 0.5).any():
                    xf = x[y > 0.5]
                    s = float(np.random.uniform(0, smax))
                    with torch.no_grad():
                        xr = a1_fingerprint_removal(xf, ref, strength=s)
                    if a.consistency:
                        lr_, dr = removal_consistency(model.noi, xf, xr, a.beta)
                    else:
                        _, l2 = model.noi(xr)
                        lr_ = F.binary_cross_entropy_with_logits(l2, torch.ones_like(l2))
                        dr = torch.zeros((), device=dev)
                    loss = loss + lam_r * lr_
                    agg["drift"] += float(dr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
            opt.step(); sched.step()
            agg["cls"] += float(lc); agg["met"] += float(lm); agg["rob"] += float(lr_)
            agg["gate"] += float(p["gate"].mean()); agg["n"] += 1
        v = score(model, lva, dev)
        row = {"epoch": ep, "smax": smax,
               "cls": agg["cls"] / agg["n"], "metric": agg["met"] / agg["n"],
               "robust": agg["rob"] / agg["n"], "drift": agg["drift"] / agg["n"],
               "gate_mean": agg["gate"] / agg["n"],
               "val_auroc": auroc(v["y"], v["logit"]),
               "val_auroc_sem": auroc(v["y"], v["sem"]),
               "val_auroc_noi": auroc(v["y"], v["noi"])}
        hist.append(row)
        print(f"[{run}] ep{ep} " + " ".join(f"{k}={row[k]:.4f}" for k in
              ("cls", "metric", "robust", "gate_mean", "val_auroc", "val_auroc_sem",
               "val_auroc_noi")), flush=True)

    res = {"run": run, "args": vars(a), "git": git_hash(),
           "trainable": tp, "params": ap_, "trainable_pct": pct, "breakdown": brk,
           "history": hist, "wall_s": time.time() - t0,
           "gpu": torch.cuda.get_device_name(0)}
    for name, ld in (("test_in", lin), ("test_out", lout)):
        s = score(model, ld, dev)
        np.savez_compressed(f"{a.out}/scores__{run}__{name}.npz", **s)
        res[f"auroc_{name}"] = auroc(s["y"], s["logit"])
        res[f"auroc_{name}_sem"] = auroc(s["y"], s["sem"])
        res[f"auroc_{name}_noi"] = auroc(s["y"], s["noi"])
        res[f"gate_{name}"] = float(np.mean(s["gate"]))
    torch.save({"model": model.state_dict(), "args": vars(a)}, f"{a.out}/ckpt/{run}.pt")
    with open(f"{a.out}/run__{run}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{run}] DONE in={res['auroc_test_in']:.4f} out={res['auroc_test_out']:.4f} "
          f"({res['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
