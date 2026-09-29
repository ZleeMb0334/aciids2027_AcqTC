"""E3: diagnose the local spectral-cost model on trained detectors.

For each detector:
  (1) measure the reliance vector w by band-wise spectral perturbation,
  (2) test the local-linearity assumption,
  (3) record effective spectral support plus the adaptive attack transition.

Cross-detector score scales, margins, plus band costs are uncontrolled.  The
output is therefore diagnostic.
"""
import argparse, glob, json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_fs as FS
from models import FusionDetector
from spectral import BandProfile, reliance_entropy, predicted_cost
from attacks import a3_pgd, psnr

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="fs")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--delta", type=float, default=0.15, help="log-magnitude probe size")
ap.add_argument("--out", default="/workspace/df-fusion/results")
a = ap.parse_args()
dev = "cuda"
bp = BandProfile(a.size, 32, 2).to(dev)
B = bp.B


def perturb_band(x, b, delta):
    """Scale the magnitude of every frequency in band b by exp(delta), phase kept."""
    m = (bp.band == b).to(x.device)
    F_ = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
    mag, ph = F_.abs().clamp(min=1e-8), torch.angle(F_)
    newlog = torch.log(mag) + delta * m.unsqueeze(0).unsqueeze(0)
    out = torch.fft.ifft2(torch.fft.ifftshift(torch.exp(newlog) * torch.exp(1j * ph),
                                              dim=(-2, -1))).real
    return out.clamp(0, 1)


items = FS.subset("progan", a.n, a.seed)
ld = DataLoader(FS.FSDataset(items, a.size, False, a.seed), batch_size=32, num_workers=8)
rows = []
for ck in sorted(glob.glob(f"{a.out}/ckpt_{a.tag}__*__s{a.seed}.pt")):
    method = os.path.basename(ck).split("__")[1]
    # method names encode the training recipe; the architecture is the prefix.
    fusion = method
    for suffix in ("_robust", "_srr", "_full"):
        fusion = fusion.replace(suffix, "")
    m = FusionDetector(semantic_kw={"tune": "ln"},
                       noise_kw={"residual": "srm", "size": a.size}, fusion=fusion).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev)["model"]); m.eval()

    base, ys = [], []
    xs = []
    with torch.no_grad():
        for x, y, _, _ in ld:
            x = x.to(dev)
            base.append(m(x).float().cpu().numpy()); ys.append(y.numpy())
            xs.append(x.cpu())
    base = np.concatenate(base); ys = np.concatenate(ys)

    # (1) reliance vector by band perturbation
    w = np.zeros(B)
    with torch.no_grad():
        for b in range(B):
            d = []
            for xb in xs:
                xb = xb.to(dev)
                d.append((m(perturb_band(xb, b, a.delta)) - m(xb)).float().cpu().numpy())
            w[b] = float(np.mean(np.concatenate(d))) / a.delta
    Hw = reliance_entropy(w)

    # (2) local linearity.  The inverse transform plus clipping changes the
    # realised band displacement.  Predictions therefore use measured dphi.
    # The explicit linear path provides a self-check whose R^2 should be one.
    rng = np.random.default_rng(0)
    lin = {}
    w_lin = m.noi.phi_lin.weight.detach().squeeze(0).cpu().numpy()
    for scale in (0.02, 0.05, 0.15):
        pred_full, pred_lin, d_full, d_lin = [], [], [], []
        with torch.no_grad():
            for _ in range(64):
                v = rng.normal(size=B) * scale
                xb = xs[int(rng.integers(0, len(xs)))].to(dev)
                F_ = torch.fft.fftshift(torch.fft.fft2(xb.float()), dim=(-2, -1))
                mag, ph = F_.abs().clamp(min=1e-8), torch.angle(F_)
                shift = torch.zeros_like(bp.band, dtype=torch.float32)
                for b in range(B):
                    shift[bp.band == b] = float(v[b])
                nl = torch.log(mag) + shift.unsqueeze(0).unsqueeze(0)
                xp = torch.fft.ifft2(torch.fft.ifftshift(
                    torch.exp(nl) * torch.exp(1j * ph), dim=(-2, -1))).real.clamp(0, 1)
                f0 = m(xb)
                f1 = m(xp)
                _, _, p0 = m.noi(xb, return_phi=True)
                _, _, p1 = m.noi(xp, return_phi=True)
                dphi = (p1 - p0).mean(0).cpu().numpy()          # realised change
                pred_full.append(float(np.dot(w, dphi)))
                pred_lin.append(float(np.dot(w_lin, dphi)))
                d_full.append(float((f1 - f0).mean()))
                d_lin.append(float((m.noi.phi_lin(p1) - m.noi.phi_lin(p0)).mean()))
        for tag, xarr, yarr in (("full", pred_full, d_full),
                                ("linear_path", pred_lin, d_lin)):
            xa_, ya_ = np.array(xarr), np.array(yarr)
            lin[f"{tag}_s{scale}"] = (float(np.corrcoef(xa_, ya_)[0, 1] ** 2)
                                      if xa_.std() > 0 and ya_.std() > 0 else float("nan"))
    r2 = lin.get("full_s0.02", float("nan"))

    # (3) local predicted cost plus measured adaptive transition
    kappa = bp.counts.cpu().numpy()
    pred = predicted_cost(w, margin=1.0, kappa=kappa)
    meas = {}
    for eps in (0.125 / 255, 0.25 / 255, 0.5 / 255, 1 / 255, 2 / 255, 4 / 255):
        ss, pp = [], []
        for xb, yb in zip(xs, np.split(ys, np.cumsum([len(x) for x in xs])[:-1])):
            xb = xb.to(dev); fk = torch.tensor(yb == 1, device=dev)
            xa = xb.clone()
            if fk.any():
                xa[fk] = a3_pgd(m, xb[fk], 0.0, eps=eps, steps=40,
                                gate_aware=False)
                pp.append(psnr(xb[fk], xa[fk]))
            with torch.no_grad():
                ss.append(m(xa).float().cpu().numpy())
        au = float(roc_auc_score(ys, np.concatenate(ss)))
        meas[f"eps{eps*255:.3f}"] = {"auroc": au,
                                     "psnr": float(np.mean(np.concatenate(pp))) if pp else None}
    # Record scale separately because cross-detector costs depend on the margin,
    # sensitivity norm, plus band costs.
    r_ = {"method": method, "seed": a.seed, "measurement_version": 2,
          "H_w": Hw, "H_max": float(np.log(B)),
          "w_l1": float(np.abs(w).sum()), "w_l2": float(np.linalg.norm(w)),
          "w": w.tolist(), "linearity_r2": r2, "linearity": lin,
          "predicted_cost": pred, "clean_auroc": float(roc_auc_score(ys, base)),
          "adaptive": meas,
          "top5_share": float(np.sort(np.abs(w))[::-1][:5].sum() / np.abs(w).sum())}
    rows.append(r_)
    print(f"[rel] {method:14s} H(w)={Hw:.3f}/{np.log(B):.3f} top5={100*r_['top5_share']:.1f}% "
          f"R2full={lin.get('full_s0.02', float('nan')):.3f} "
          f"R2lin={lin.get('linear_path_s0.02', float('nan')):.3f} predcost={pred:.4g} "
          + f"|w|_1={r_['w_l1']:.3g} "
          + " ".join(f"{k}:{v['auroc']:.3f}" for k, v in meas.items()), flush=True)
    with open(f"{a.out}/reliance__{a.tag}__s{a.seed}.json", "w") as f:
        json.dump(rows, f, indent=1)
print("[rel] done", flush=True)
