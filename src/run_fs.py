"""Main experiment on ForenSynths: train on ProGAN, test cross-generator, and
measure BOTH axes (generalisation and attack robustness) for every method.
"""
import argparse, json, os, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_fs as FS
from models import FusionDetector, count_trainable, trainable_breakdown
from losses import focal_bce, paired_supcon, removal_consistency, reliance_entropy_loss
from attacks import (SpectralReference, a1_fingerprint_removal, a2_radial_match,
                     a2_whiten, a2_resample, a3_pgd, a4_jpeg, a4_resize, a4_blur, psnr)
from spectral import reliance_entropy

ap = argparse.ArgumentParser()
ap.add_argument("--methods", default="semantic,noise,gate,gate_robust,gate_srr,gate_full")
ap.add_argument("--lam-srr", type=float, default=1.0)
ap.add_argument("--srr-target", type=float, default=0.90)
ap.add_argument("--arch", default="ViT-B-16-quickgelu")
ap.add_argument("--tune", default="ln", choices=["ln", "probe", "full"])
ap.add_argument("--srr-mode", default="weight", choices=["weight", "grad"])
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--n-train", type=int, default=8000)
ap.add_argument("--n-test", type=int, default=2000)
ap.add_argument("--n-attack", type=int, default=1000, help="images used for attack curves")
ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--tag", default="fs")
ap.add_argument("--out", default="/workspace/df-fusion/results")
a = ap.parse_args()
dev = "cuda"
review_control = a.tag in {"semtc", "abl", "objabl", "acqtcl"}
torch.manual_seed(a.seed); np.random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)

tr_items = FS.train_subset(a.n_train, seed=a.seed)
rng = np.random.default_rng(a.seed)
perm = rng.permutation(len(tr_items))
nv = len(perm) // 10
va_items = [tr_items[i] for i in perm[:nv]]
tr_items = [tr_items[i] for i in perm[nv:]]
dtr = FS.FSDataset(tr_items, a.size, True, a.seed)
dva = FS.FSDataset(va_items, a.size, False, a.seed)
ltr = DataLoader(dtr, batch_sampler=FS.FSPairedSampler(tr_items, dtr.cls, a.bs, "class", a.seed),
                 num_workers=12, pin_memory=True)
lva = DataLoader(dva, batch_size=a.bs, num_workers=8, pin_memory=True)
tests = {g: DataLoader(FS.FSDataset(FS.subset(g, a.n_test, a.seed), a.size, False, a.seed),
                       batch_size=a.bs, num_workers=8, pin_memory=True) for g in FS.CLEAN}
print(f"[fs] train={len(tr_items)} val={len(va_items)} test gens={list(tests)}", flush=True)

ref = SpectralReference.estimate(lva, dev, max_batches=30)
ref.save(f"{a.out}/{a.tag}_ref__s{a.seed}.pt")

# Merge with any previous rows for this (tag, seed) so a run can top up missing
# methods without discarding the ones already measured.
RESULT = f"{a.out}/{a.tag}__s{a.seed}.json"
rows = []
if os.path.exists(RESULT):
    try:
        with open(RESULT) as f:
            rows = [r for r in json.load(f) if r["method"] not in a.methods.split(",")]
        print(f"[fs] merging with {len(rows)} existing rows", flush=True)
    except Exception:
        rows = []

for method in a.methods.split(","):
    t0 = time.time()
    removal = method.endswith(("_robust", "_srr", "_full"))
    srr = method.endswith("_srr")
    transform = method.endswith(("_full", "_tc", "_aug", "_align"))
    fusion = (method.replace("_robust", "").replace("_srr", "")
              .replace("_full", "").replace("_tc", "")
              .replace("_aug", "").replace("_align", ""))
    m = FusionDetector(semantic_kw={"tune": a.tune, "arch": a.arch},
                       noise_kw={"residual": "srm", "size": a.size}, fusion=fusion).to(dev)
    tp, tot, pct = count_trainable(m)
    brk = trainable_breakdown(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * max(1, len(ltr)))
    for ep in range(a.epochs):
        m.train()
        smax = min(1.0, (ep / max(1, a.epochs - 1)) / 0.4)
        agg = {"l": 0., "g": 0., "n": 0}
        for x, y, _, _ in ltr:
            x, y = x.to(dev, non_blocking=True), y.float().to(dev, non_blocking=True)
            p = m(x, return_parts=True)
            loss = focal_bce(p["logit"], y) + 0.5 * paired_supcon(p["z"].float(), y)
            if removal and (y > 0.5).any():
                xf = x[y > 0.5]
                s = float(np.random.uniform(0, smax))
                with torch.no_grad():
                    xr = a1_fingerprint_removal(xf, ref, strength=s)
                    xs = a2_resample(xf, strength=s)
                l1, _ = removal_consistency(m.noi, xf, xr)
                l2, _ = removal_consistency(m.noi, xf, xs)
                loss = loss + smax * 0.5 * (l1 + l2)
            if transform:
                # Apply the same transform-consistency objective to the whole
                # class-balanced batch. The semantic_tc control uses this block
                # without training the removal-consistent spectral objective.
                with torch.no_grad():
                    q = int(np.random.choice([90, 70, 50]))
                    xb1 = a4_jpeg(x, q)
                    xb2 = a4_resize(x, float(np.random.choice([0.5, 0.75])))
                for xb in (xb1, xb2):
                    pb = m(xb, return_parts=True)
                    view_classification = (
                        0.0 if method.endswith("_align")
                        else focal_bce(pb["logit"], y))
                    logit_alignment = (
                        0.0 if method.endswith("_aug") else
                        0.5 * ((pb["logit"] - p["logit"].detach()) ** 2).mean())
                    loss = loss + smax * 0.25 * (
                        view_classification + logit_alignment)
            if srr:
                w = m.noi.reliance(x if a.srr_mode == "grad" else None, mode=a.srr_mode)
                deficit, Hb = reliance_entropy_loss(w, target=a.srr_target)
                loss = loss + smax * a.lam_srr * deficit
                agg["H"] = agg.get("H", 0.) + float(Hb)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad], 1.0)
            opt.step(); sch.step()
            agg["l"] += float(loss); agg["g"] += float(p["gate"].mean()); agg["n"] += 1
        extra = f" H={agg['H']/agg['n']:.3f}" if "H" in agg else ""
        print(f"[fs] {method} ep{ep} loss={agg['l']/agg['n']:.4f} "
              f"gate={agg['g']/agg['n']:.3f}{extra}", flush=True)
    m.eval()

    def run(ld, fn=None, eps=None):
        ys, ss, ps = [], [], []
        for x, y, _, _ in ld:
            x = x.to(dev); yb = y.numpy()
            if eps is not None:
                fk = torch.tensor(yb == 1, device=dev)
                xa = x.clone()
                if fk.any():
                    xa[fk] = a3_pgd(m, x[fk], 0.0, eps=eps, steps=40,
                                    gate_aware=False)
                    # budget is measured ONLY over the images actually perturbed;
                    # averaging in the untouched reals would inflate PSNR.
                    ps.append(psnr(x[fk], xa[fk]))
                x = xa
            elif fn is not None:
                with torch.no_grad():
                    xa = fn(x)
                ps.append(psnr(x, xa)); x = xa
            with torch.no_grad():
                ss.append(m(x).float().cpu().numpy())
            ys.append(yb)
        y, s = np.concatenate(ys), np.concatenate(ss)
        return (float(roc_auc_score(y, s)), float(average_precision_score(y, s)),
                float(np.mean(np.concatenate(ps))) if ps else float("inf"))

    r = {"method": method, "seed": a.seed, "trainable": tp, "params": tot,
         "trainable_pct": pct, "breakdown": brk, "train_s": time.time() - t0,
         "evaluation_profile": ("transfer+jpeg70+resize"
                                if review_control else "full")}
    per = {}
    for g, ld in tests.items():
        au, apx, _ = run(ld)
        per[g] = {"auroc": au, "ap": apx, "shortcut_auroc": FS.SHORTCUT_AUROC[g]}
    r["per_generator"] = per
    r["auroc_macro"] = float(np.mean([v["auroc"] for v in per.values()]))
    r["auroc_progan"] = per["progan"]["auroc"]
    r["auroc_unseen_macro"] = float(np.mean([v["auroc"] for k, v in per.items() if k != "progan"]))
    r["gap"] = r["auroc_progan"] - r["auroc_unseen_macro"]

    atk_ld = DataLoader(FS.FSDataset(FS.subset("progan", a.n_attack, a.seed), a.size,
                                     False, a.seed), batch_size=a.bs, num_workers=8,
                        pin_memory=True)
    transform_suite = [
        ("a4_jpeg70", lambda z: a4_jpeg(z, 70)),
        ("a4_resize", lambda z: a4_resize(z, 0.5)),
    ]
    if not review_control:
        transform_suite = [
            ("a1", lambda z: a1_fingerprint_removal(z, ref, 1.0)),
            ("a2_radial", lambda z: a2_radial_match(z, ref, 1.0)),
            ("a2_whiten", lambda z: a2_whiten(z, ref, 1.0)),
            ("a2_resample", lambda z: a2_resample(z, 1.0)),
        ] + transform_suite + [
            ("a4_blur", lambda z: a4_blur(z, 1.0)),
        ]
    for nm, fn in transform_suite:
        au, _, pp = run(atk_ld, fn)
        r[f"{nm}_auroc"], r[f"{nm}_psnr"] = au, pp
    if not review_control:
        for e in (1 / 255, 2 / 255, 4 / 255, 8 / 255):
            au, _, pp = run(atk_ld, eps=e)
            r[f"a3_eps{e*255:.0f}_auroc"], r[f"a3_eps{e*255:.0f}_psnr"] = au, pp
    w = m.noi.phi_lin.weight.detach().cpu().numpy().ravel()
    r["H_w"] = reliance_entropy(w); r["H_max"] = float(np.log(w.size))
    rows.append(r)
    attack_note = (f"a1={r['a1_auroc']:.4f} res={r['a2_resample_auroc']:.4f} "
                   f"pgd4={r['a3_eps4_auroc']:.4f} "
                   if not review_control else "")
    print(f"[fs] {method:14s} progan={r['auroc_progan']:.4f} "
          f"unseen={r['auroc_unseen_macro']:.4f} gap={r['gap']:+.4f} | "
          f"{attack_note}jpeg={r['a4_jpeg70_auroc']:.4f} "
          f"resize={r['a4_resize_auroc']:.4f} H={r['H_w']:.3f} "
          f"[{r['train_s']:.0f}s]", flush=True)
    print("      per-gen: " + "  ".join(f"{g}={v['auroc']:.3f}" for g, v in per.items()), flush=True)
    torch.save({"model": m.state_dict(), "args": vars(a)},
               f"{a.out}/ckpt_{a.tag}__{method}__s{a.seed}.pt")
    with open(RESULT, "w") as f:
        json.dump(rows, f, indent=1)
print("[fs] done", flush=True)
