"""Evaluate official baseline implementations on OUR splits and attack suite.

Provenance is recorded per baseline in the output so the paper can state exactly
what is 'official weights' versus 'official code, retrained here'.
"""
import argparse, hashlib, importlib.util, json, os, sys, time
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_fs as FS
from attacks import (SpectralReference, a1_fingerprint_removal, a2_radial_match,
                     a2_whiten, a2_resample, a3_pgd, a4_jpeg, a4_resize, a4_blur, psnr)

BASE = "/workspace/df-fusion/baselines"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Wrap(nn.Module):
    """Adapts an official baseline to our interface: input in [0,1], output logit."""

    def __init__(self, net, mean, std, pre=None):
        super().__init__()
        self.net, self.pre = net, pre
        self.register_buffer("m", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x, return_parts=False):
        z = (x - self.m) / self.s
        if self.pre is not None:
            z = self.pre(z)
        o = self.net(z)
        o = o.squeeze(-1) if o.dim() > 1 and o.shape[-1] == 1 else o.view(o.shape[0], -1)[:, 0]
        if return_parts:
            return {"logit": o, "gate": torch.full_like(o, 0.5),
                    "sem_logit": o, "noise_logit": o}
        return o


def load_cnndetection(dev, which="blur_jpg_prob0.5"):
    path = f"{BASE}/CNNDetection/networks/resnet.py"
    spec = importlib.util.spec_from_file_location("official_cnndetection_resnet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.resnet50(num_classes=1)
    ckpt = f"{BASE}/CNNDetection/weights/{which}.pth"
    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd["model"])
    return Wrap(net, IMAGENET_MEAN, IMAGENET_STD).to(dev).eval(), {
        "paper": "Wang et al., CVPR 2020 (CNNDetection)",
        "code": "github.com/PeterWang512/CNNDetection",
        "code_commit": "ea0b5622365e3a9cd31d1b54b6b5971131a839ab",
        "weights": f"official {which}.pth", "weights_sha256": file_sha256(ckpt),
        "provenance": "official code + official weights"}


def load_ufd(dev, ckpt=None):
    """Ojha et al. CVPR 2023: frozen CLIP ViT-L/14 + linear probe (official arch)."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
    visual = model.visual
    del model
    for p in visual.parameters():
        p.requires_grad_(False)

    class UFD(nn.Module):
        def __init__(self, v, d=768):
            super().__init__()
            self.v, self.fc = v, nn.Linear(d, 1)

        def forward(self, x):
            if x.shape[-1] != 224:
                x = F.interpolate(x, (224, 224), mode="bicubic", align_corners=False)
            # NOT under no_grad: the backbone parameters are frozen, but input
            # gradients must flow for the white-box attack to be white-box.
            f = self.v(x)
            return self.fc(f.float())

    net = UFD(visual)
    official = (f"{BASE}/UniversalFakeDetect/official_weights/"
                "pretrained_weights/fc_weights.pth")
    if ckpt is None and os.path.exists(official):
        ckpt = official
    prov = {"paper": "Ojha et al., CVPR 2023 (UniversalFakeDetect)",
            "code": "github.com/WisconsinAIVision/UniversalFakeDetect",
            "code_commit": "030495aea3300a8b54c0ec37ec7fe1dd7e63c619",
            "weights": "official fc_weights.pth" if ckpt else "uninitialised probe",
            "provenance": ("official code + official weights" if ckpt else
                           "official architecture; probe awaiting local training")}
    if ckpt and os.path.exists(ckpt):
        net.fc.load_state_dict(torch.load(ckpt, map_location="cpu"))
        prov["weights_sha256"] = file_sha256(ckpt)
        if os.path.realpath(ckpt) != os.path.realpath(official):
            prov["weights"] = f"trained here: {os.path.basename(ckpt)}"
            prov["provenance"] = "official architecture + recipe, probe trained here"
    return Wrap(net, CLIP_MEAN, CLIP_STD).to(dev), prov


def load_npr(dev, ckpt=None):
    """Tan et al. CVPR 2024: Neighboring Pixel Relationships, official code."""
    path = f"{BASE}/NPR-DeepfakeDetection/networks/resnet.py"
    spec = importlib.util.spec_from_file_location("official_npr_resnet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.resnet50(num_classes=1)
    prov = {"paper": "Tan et al., CVPR 2024 (NPR)",
            "code": "github.com/chuangchuangtan/NPR-DeepfakeDetection",
            "code_commit": "781ced3f7ca2cdc69ec9dd4ef27e8d0b3c07752a",
            "weights": "official NPR.pth",
            "provenance": "official code + official weights"}
    if ckpt is None:
        ckpt = f"{BASE}/NPR-DeepfakeDetection/NPR.pth"
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("model", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        net.load_state_dict(state)
        prov["weights_sha256"] = file_sha256(ckpt)
        if os.path.basename(ckpt) != "NPR.pth":
            prov["weights"] = f"trained here: {os.path.basename(ckpt)}"
            prov["provenance"] = "official code, trained here on ProGAN"
    return Wrap(net, IMAGENET_MEAN, IMAGENET_STD).to(dev), prov


def load_freqnet(dev, ckpt=None):
    """Tan et al. AAAI 2024: FreqNet, authors' code and released checkpoint."""
    path = f"{BASE}/FreqNet-DeepfakeDetection/networks/freqnet.py"
    spec = importlib.util.spec_from_file_location("official_freqnet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.freqnet(num_classes=1)
    if ckpt is None:
        ckpt = f"{BASE}/FreqNet-DeepfakeDetection/4-classes-freqnet-v2.pth"
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"FreqNet checkpoint unavailable: {ckpt}")
    net.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return Wrap(net, IMAGENET_MEAN, IMAGENET_STD).to(dev).eval(), {
        "paper": "Tan et al., AAAI 2024 (FreqNet)",
        "code": "github.com/chuangchuangtan/FreqNet-DeepfakeDetection",
        "code_commit": "2c6bd020cda1c0001e7a7d30ebd2dd5c55b5a8b6",
        "weights": f"official {os.path.basename(ckpt)}",
        "weights_sha256": file_sha256(ckpt),
        "provenance": "official code + official weights"}


LOADERS = {"cnndetection": load_cnndetection, "ufd": load_ufd,
           "npr": load_npr, "freqnet": load_freqnet}


def evaluate(m, ld, dev, fn=None, eps=None, capture=False):
    ys, ss, ps, ids = [], [], [], []
    for x, y, _, sample_id in ld:
        x = x.to(dev); yb = y.numpy()
        if eps is not None:
            fk = torch.tensor(yb == 1, device=dev)
            xa = x.clone()
            if fk.any():
                xa[fk] = a3_pgd(m, x[fk], 0.0, eps=eps, steps=40)
                ps.append(psnr(x[fk], xa[fk]))
            x = xa
        elif fn is not None:
            with torch.no_grad():
                xa = fn(x)
            ps.append(psnr(x, xa)); x = xa
        with torch.no_grad():
            ss.append(m(x).float().cpu().numpy())
        ys.append(yb)
        if capture:
            ids.append(np.asarray(sample_id))
    y, s = np.concatenate(ys), np.concatenate(ss)
    result = (float(roc_auc_score(y, s)), float(average_precision_score(y, s)),
              float(np.mean(np.concatenate(ps))) if ps else float("inf"))
    return result + (y, s, np.concatenate(ids)) if capture else result


def train_probe(name, dev, n_train, bs, epochs, size, seed, out):
    """Train the trainable part of a baseline on the standard ProGAN split."""
    items = FS.train_subset(n_train, seed)
    ds = FS.FSDataset(items, size, True, seed)
    ld = DataLoader(ds, batch_sampler=FS.FSPairedSampler(items, ds.cls, bs, "class", seed),
                    num_workers=12, pin_memory=True)
    # An empty checkpoint disables the released weights for local retraining.
    m, prov = LOADERS[name](dev, "") if name in {"ufd", "npr"} else LOADERS[name](dev)
    params = [p for p in m.parameters() if p.requires_grad]
    # UFD learns a linear probe on a frozen encoder and converges in a few epochs.
    # NPR trains a ResNet-50 from scratch (its own trainer uses pretrained=False)
    # and its published recipe uses the full 720k-image ProGAN train split; at our
    # budget it needs many more passes, so we give it a longer schedule and report
    # honestly if it still falls short of its published range.
    lr = 1e-3 if name == "ufd" else 1e-4
    if name == "npr":
        epochs = max(epochs, 40)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0 if name == "ufd" else 1e-4)
    m.train()
    for ep in range(epochs):
        tot = n = 0
        for x, y, _, _ in ld:
            x, y = x.to(dev), y.float().to(dev)
            loss = F.binary_cross_entropy_with_logits(m(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss); n += 1
        print(f"[base] {name} ep{ep} loss={tot/max(n,1):.4f}", flush=True)
    m.eval()
    p = f"{out}/base_{name}__s{seed}.pt"
    torch.save(m.net.fc.state_dict() if name == "ufd" else m.net.state_dict(), p)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="cnndetection,ufd,npr,freqnet")
    ap.add_argument("--train", default="", help="which need training here")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-attack", type=int, default=1000, help="images used for attack curves")
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/df-fusion/results")
    ap.add_argument("--clean-only", action="store_true",
                    help="evaluate the matched clean transfer panel without attacks")
    ap.add_argument("--benign-only", action="store_true",
                    help="evaluate clean transfer plus JPEG/resize on ProGAN; skip removal and PGD")
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    tests = {g: DataLoader(FS.FSDataset(FS.subset(g, a.n_test, a.seed), a.size, False, a.seed),
                           batch_size=a.bs, num_workers=8, pin_memory=True) for g in FS.CLEAN}
    ref = None
    if not (a.clean_only or a.benign_only):
        va = FS.train_subset(2000, a.seed)
        lva = DataLoader(FS.FSDataset(va, a.size, False, a.seed),
                         batch_size=a.bs, num_workers=8)
        ref = SpectralReference.estimate(lva, dev, max_batches=25)
    # Merge with existing rows for this seed: running a single baseline must not
    # discard the others (this clobbered cnndetection and ufd at seed 0 once).
    RESULT = f"{a.out}/baselines__s{a.seed}.json"
    rows = []
    existing_by_method = {}
    if os.path.exists(RESULT):
        try:
            with open(RESULT) as f:
                existing = json.load(f)
                existing_by_method = {r["method"]: r for r in existing}
                rows = [r for r in existing
                        if r["method"] not in a.baselines.split(",")]
            print(f"[base] merging with {len(rows)} existing rows", flush=True)
        except Exception:
            rows = []
    need = set(a.train.split(","))
    for name in a.baselines.split(","):
        t0 = time.time()
        ck = None
        if name in need:
            ck = train_probe(name, dev, a.n_train, a.bs, a.epochs, a.size, a.seed, a.out)
        m, prov = LOADERS[name](dev, ck) if name != "cnndetection" else LOADERS[name](dev)
        m.eval()
        r = {"method": name, "seed": a.seed, "provenance": prov,
             "params": sum(p.numel() for p in m.parameters()),
             "train_s": time.time() - t0}
        reuse_clean = a.benign_only and name in existing_by_method
        per = existing_by_method[name]["per_generator"] if reuse_clean else {}
        captured = []
        if not reuse_clean:
            for g, ld in tests.items():
                if a.clean_only or a.benign_only:
                    au, apx, _, labels, scores, sample_ids = evaluate(
                        m, ld, dev, capture=True)
                    captured.append((g, labels, scores, sample_ids))
                else:
                    au, apx, _ = evaluate(m, ld, dev)
                per[g] = {"auroc": au, "ap": apx,
                          "shortcut_auroc": FS.SHORTCUT_AUROC[g]}
        r["per_generator"] = per
        r["auroc_progan"] = per["progan"]["auroc"]
        r["auroc_unseen_macro"] = float(np.mean([v["auroc"] for k, v in per.items() if k != "progan"]))
        r["auroc_macro"] = float(np.mean([v["auroc"] for v in per.values()]))
        r["gap"] = r["auroc_progan"] - r["auroc_unseen_macro"]
        if captured:
            np.savez_compressed(
                f"{a.out}/baseline_predictions__{name}__s{a.seed}.npz",
                generator=np.concatenate([
                    np.repeat(g, len(labels)) for g, labels, _, _ in captured]),
                label=np.concatenate([labels for _, labels, _, _ in captured]),
                score=np.concatenate([scores for _, _, scores, _ in captured]),
                sample_index=np.concatenate([ids for _, _, _, ids in captured]))
            if a.clean_only:
                rows.append(r)
                print(f"[base] {name:14s} progan={r['auroc_progan']:.4f} "
                      f"unseen={r['auroc_unseen_macro']:.4f} gap={r['gap']:+.4f}",
                      flush=True)
                print("      per-gen: " + "  ".join(
                    f"{g}={v['auroc']:.3f}" for g, v in per.items()), flush=True)
                with open(RESULT, "w") as f:
                    json.dump(rows, f, indent=1)
                continue
        atk = DataLoader(FS.FSDataset(FS.subset("progan", a.n_attack, a.seed), a.size,
                                      False, a.seed), batch_size=a.bs, num_workers=8,
                         pin_memory=True)
        transform_suite = [
            ("a4_jpeg70", lambda z: a4_jpeg(z, 70)),
            ("a4_resize", lambda z: a4_resize(z, 0.5)),
        ]
        if not a.benign_only:
            transform_suite = [
                ("a1", lambda z: a1_fingerprint_removal(z, ref, 1.0)),
                ("a2_radial", lambda z: a2_radial_match(z, ref, 1.0)),
                ("a2_whiten", lambda z: a2_whiten(z, ref, 1.0)),
                ("a2_resample", lambda z: a2_resample(z, 1.0)),
            ] + transform_suite + [
                ("a4_blur", lambda z: a4_blur(z, 1.0)),
            ]
        benign_captured = []
        for nm, fn in transform_suite:
            if a.benign_only:
                au, _, pp, labels, scores, sample_ids = evaluate(
                    m, atk, dev, fn, capture=True)
                benign_captured.append((nm, labels, scores, sample_ids))
            else:
                au, _, pp = evaluate(m, atk, dev, fn)
            r[f"{nm}_auroc"], r[f"{nm}_psnr"] = au, pp
        if benign_captured:
            np.savez_compressed(
                f"{a.out}/baseline_benign_predictions__{name}__s{a.seed}.npz",
                condition=np.concatenate([
                    np.repeat(nm, len(labels))
                    for nm, labels, _, _ in benign_captured]),
                label=np.concatenate([labels for _, labels, _, _ in benign_captured]),
                score=np.concatenate([scores for _, _, scores, _ in benign_captured]),
                sample_index=np.concatenate([
                    sample_ids for _, _, _, sample_ids in benign_captured]))
        if not a.benign_only:
            for e in (1 / 255, 2 / 255, 4 / 255, 8 / 255):
                au, _, pp = evaluate(m, atk, dev, eps=e)
                r[f"a3_eps{e*255:.0f}_auroc"], r[f"a3_eps{e*255:.0f}_psnr"] = au, pp
        rows.append(r)
        print(f"[base] {name:14s} progan={r['auroc_progan']:.4f} "
              f"unseen={r['auroc_unseen_macro']:.4f} gap={r['gap']:+.4f} | "
              f"jpeg={r['a4_jpeg70_auroc']:.4f} resize={r['a4_resize_auroc']:.4f}" +
              ("" if a.benign_only else
               f" a1={r['a1_auroc']:.4f} pgd4={r['a3_eps4_auroc']:.4f}"), flush=True)
        print("      per-gen: " + "  ".join(f"{g}={v['auroc']:.3f}" for g, v in per.items()), flush=True)
        with open(RESULT, "w") as f:
            json.dump(rows, f, indent=1)
    print("[base] done", flush=True)
