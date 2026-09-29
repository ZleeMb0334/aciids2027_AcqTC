"""Diagnose the GATE-1 negative: why does A1 not move the detector?

Separates four candidate explanations, each with its own measurement:
  H1  the attack is too weak / edits the wrong frequencies
  H2  the detector's reliance is spread, so a sparse edit cannot move it
  H3  the detector is reading a trivial acquisition shortcut (format /
      compression / size), not a generator fingerprint at all
  H4  the data is wrong for this attack (GenImage re-encodes; the GAN
      fingerprint lives in uncompressed ForenSynths-style data)
"""
import io, json, os, sys, collections
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
from models import NoiseBranch
from attacks import SpectralReference, a1_fingerprint_removal, psnr
from spectral import BandProfile, log_spectrum, ResidualExtractor, reliance_entropy
from sklearn.metrics import roc_auc_score

dev = "cuda"
OUT = "/workspace/df-fusion/results"
idx = D.build_index("test", cache=False)
recs = idx["records"]
res = {}

# ---------------------------------------------------------------- H3: shortcut
print("=== H3: acquisition shortcut (format / size / compression) ===", flush=True)
h3 = {}
for g in sorted({r["generator"] for r in recs}):
    sel = [i for i, r in enumerate(recs) if r["generator"] == g]
    if len({recs[i]["label"] for i in sel}) < 2:
        continue
    store = D.ShardStore(idx["files"])
    info = {0: collections.Counter(), 1: collections.Counter()}
    sizes = {0: [], 1: []}
    bpp = {0: [], 1: []}
    rng = np.random.default_rng(0)
    for lab in (0, 1):
        s = [i for i in sel if recs[i]["label"] == lab]
        for i in rng.choice(s, size=min(300, len(s)), replace=False):
            r = recs[int(i)]
            b = store.bytes_at(r["shard"], r["row"])
            im = Image.open(io.BytesIO(b))
            info[lab][im.format] += 1
            sizes[lab].append(im.size)
            bpp[lab].append(len(b) * 8.0 / (im.size[0] * im.size[1]))
    h3[g] = {
        "format_real": dict(info[0]), "format_fake": dict(info[1]),
        "size_real": collections.Counter(sizes[0]).most_common(2),
        "size_fake": collections.Counter(sizes[1]).most_common(2),
        "bpp_real": float(np.mean(bpp[0])), "bpp_fake": float(np.mean(bpp[1])),
    }
    print(f"  {g:24s} fmt real={dict(info[0])} fake={dict(info[1])}  "
          f"bpp real={np.mean(bpp[0]):.3f} fake={np.mean(bpp[1]):.3f}  "
          f"size real={h3[g]['size_real'][0]} fake={h3[g]['size_fake'][0]}", flush=True)
res["H3_shortcut"] = h3

# ---------------------------------------------------------------- H1: attack
print("=== H1: does A1 edit the frequencies it claims to? ===", flush=True)
g = "BigGAN"
sel = [i for i, r in enumerate(recs) if r["generator"] == g]
rng = np.random.default_rng(0)
tr = [int(i) for i in rng.choice(sel, size=min(3000, len(sel)), replace=False)]
ds = D.GenImageDS(idx, tr, train=False, seed=0)
dl = DataLoader(ds, batch_size=64, num_workers=8)
ref = SpectralReference.estimate(dl, dev, max_batches=25)
bp = BandProfile(224, 32, 2).to(dev)
h1 = {}
for topk in (1024, 8192, 25088, 50176):
    mask = ref.peak_bands(topk)
    yy, xx = torch.meshgrid(torch.arange(224) - 112, torch.arange(224) - 112, indexing="ij")
    rad = (torch.sqrt(xx.float()**2 + yy.float()**2) / 112).to(dev)
    h1[topk] = {"mean_radius_of_selected": float(rad[mask].mean()),
                "frac_of_spectrum": topk / (224 * 224)}
    x, y, *_ = next(iter(dl))
    xf = x[y == 1][:16].to(dev)
    if xf.numel() == 0:
        continue
    xa = a1_fingerprint_removal(xf, ref, 1.0, topk=topk)
    sb = torch.log(torch.fft.fftshift(torch.fft.fft2(xf.mean(1)), dim=(-2, -1)).abs() + 1e-8)
    sa = torch.log(torch.fft.fftshift(torch.fft.fft2(xa.mean(1)), dim=(-2, -1)).abs() + 1e-8)
    gap_before = float((sb.mean(0) - ref.real)[mask].abs().mean())
    gap_after = float((sa.mean(0) - ref.real)[mask].abs().mean())
    h1[topk].update({"psnr": float(np.mean(psnr(xf, xa))),
                     "fingerprint_gap_before": gap_before,
                     "fingerprint_gap_after": gap_after,
                     "gap_closed_pct": 100 * (1 - gap_after / max(gap_before, 1e-8))})
    print(f"  topk={topk:6d} ({100*topk/50176:5.1f}% of spectrum) "
          f"mean_radius={h1[topk]['mean_radius_of_selected']:.3f} "
          f"psnr={h1[topk]['psnr']:6.2f}dB gap {gap_before:.4f}->{gap_after:.4f} "
          f"({h1[topk]['gap_closed_pct']:.1f}% closed)", flush=True)
res["H1_attack_strength"] = h1

# ---------------------------------------------------------------- H2: reliance
print("=== H2: band-ablation reliance of the trained detector ===", flush=True)
lab = np.array([recs[i]["label"] for i in sel])
tr2, te2 = [], []
for l in (0, 1):
    s = [sel[j] for j in np.where(lab == l)[0]]
    s = list(np.random.default_rng(1).permutation(s))[:2000]
    tr2 += s[:len(s)//2]; te2 += s[len(s)//2:]
ltr = DataLoader(D.GenImageDS(idx, [int(i) for i in tr2], train=True, seed=0),
                 batch_size=64, shuffle=True, num_workers=8, drop_last=True)
lte = DataLoader(D.GenImageDS(idx, [int(i) for i in te2], train=False, seed=0),
                 batch_size=64, num_workers=8)
m = NoiseBranch(residual="srm").to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
for ep in range(4):
    m.train()
    for x, y, _, _ in ltr:
        x, y = x.to(dev), y.float().to(dev)
        _, lg = m(x)
        loss = F.binary_cross_entropy_with_logits(lg, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
m.eval()

@torch.no_grad()
def phis_and_scores():
    P, S, Y = [], [], []
    for x, y, _, _ in lte:
        x = x.to(dev)
        h, lg, phi = m(x, return_phi=True)
        P.append(phi.float().cpu().numpy()); S.append(lg.float().cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(S), np.concatenate(Y)

P, S, Y = phis_and_scores()
base = roc_auc_score(Y, S)
# empirical reliance: |d score / d phi_b| estimated by the linear head + a
# permutation ablation of each band (model-agnostic, exact for the linear path)
w_lin = m.phi_lin.weight.detach().cpu().numpy().ravel()
rng2 = np.random.default_rng(0)
abl = np.zeros(P.shape[1])
with torch.no_grad():
    Pt = torch.tensor(P, device=dev)
    for b in range(P.shape[1]):
        Pp = Pt.clone(); Pp[:, b] = Pt[rng2.permutation(len(Pt)), b]
        d = (m.phi_lin(Pp) - m.phi_lin(Pt)).squeeze(-1).abs().mean()
        abl[b] = float(d)
res["H2_reliance"] = {"auroc": float(base),
                      "H_linear": reliance_entropy(w_lin),
                      "H_ablation": reliance_entropy(abl),
                      "H_max": float(np.log(P.shape[1])),
                      "top5_bands": [int(i) for i in np.argsort(-abl)[:5]],
                      "top5_share": float(np.sort(abl)[::-1][:5].sum() / abl.sum())}
print(f"  auroc={base:.4f} H_lin={res['H2_reliance']['H_linear']:.3f} "
      f"H_abl={res['H2_reliance']['H_ablation']:.3f} (max {np.log(P.shape[1]):.3f}) "
      f"top5 bands hold {100*res['H2_reliance']['top5_share']:.1f}% of reliance", flush=True)

# separability of phi alone (is the band profile even the signal?)
from sklearn.linear_model import LogisticRegression
perm = np.random.default_rng(0).permutation(len(P))
P, S, Y = P[perm], S[perm], Y[perm]
n = len(P) // 2
lr = LogisticRegression(max_iter=2000).fit(P[:n], Y[:n])
res["H2_reliance"]["phi_only_auroc"] = float(roc_auc_score(Y[n:], lr.decision_function(P[n:])))
print(f"  band-profile-only logistic AUROC = {res['H2_reliance']['phi_only_auroc']:.4f}", flush=True)

with open(f"{OUT}/diag_gate1.json", "w") as f:
    json.dump(res, f, indent=1, default=str)
print("[diag] wrote diag_gate1.json", flush=True)
