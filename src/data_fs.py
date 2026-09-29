"""ForenSynths (Wang et al. 2020 'CNNDetection') data layer.

Chosen over GenImage for the robustness experiments because the benchmark audit
(audit_benchmarks.py) shows GenImage is perfectly separable from acquisition
metadata alone (shortcut AUROC 1.000 in all 8 subsets), whereas the ForenSynths
subsets used here score 0.47-0.64. The higher-shortcut ForenSynths subsets
(stargan, imle, san, crn, seeingdark; 0.86-0.99) are excluded for the same,
measured reason.

Protocol follows the original: train on ProGAN, test cross-generator; images are
cropped, never resized, and never re-encoded.
"""
import glob
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

FS = os.environ.get("DF_FS", "/workspace/df-fusion/data/forensynths")
TEST = os.path.join(FS, "CNN_synth_testset")
VAL = os.path.join(FS, "progan_val")

# shortcut AUROC from results/benchmark_audit.json (metadata-only separability)
SHORTCUT_AUROC = {
    "progan": 0.4736, "deepfake": 0.5184, "biggan": 0.5210, "gaugan": 0.5440,
    "cyclegan": 0.6308, "stylegan": 0.6396, "stargan": 0.8876, "imle": 0.8637,
    "san": 0.9480, "crn": 0.9586, "seeingdark": 0.9876,
}
CLEAN = ["progan", "deepfake", "biggan", "gaugan", "cyclegan", "stylegan"]  # screened group
CONFOUNDED = ["stargan", "imle", "san", "crn", "seeingdark"]
GAN_LIKE = {"progan", "biggan", "cyclegan", "stylegan", "stargan", "gaugan"}


def _list(root, gen):
    r = sorted(glob.glob(f"{root}/{gen}/0_real/*") + glob.glob(f"{root}/{gen}/*/0_real/*"))
    f = sorted(glob.glob(f"{root}/{gen}/1_fake/*") + glob.glob(f"{root}/{gen}/*/1_fake/*"))
    return r, f


def subset(gen, n=None, seed=0, root=TEST):
    r, f = _list(root, gen)
    rng = np.random.default_rng(seed)
    if n:
        r = list(rng.permutation(r))[: n // 2]
        f = list(rng.permutation(f))[: n // 2]
    return [(p, 0, gen) for p in r] + [(p, 1, gen) for p in f]


def train_subset(n=None, seed=0):
    """ProGAN validation split -- the standard training pool."""
    r, f = _list(VAL, "")
    if not r:
        r = sorted(glob.glob(f"{VAL}/*/0_real/*")); f = sorted(glob.glob(f"{VAL}/*/1_fake/*"))
    rng = np.random.default_rng(seed)
    if n:
        r = list(rng.permutation(r))[: n // 2]
        f = list(rng.permutation(f))[: n // 2]
    return [(p, 0, "progan_val") for p in r] + [(p, 1, "progan_val") for p in f]


def _crop(im, size, rng=None):
    w, h = im.size
    if w < size or h < size:
        s = size / min(w, h)
        im = im.resize((max(size, int(w * s + .5)), max(size, int(h * s + .5))), Image.BICUBIC)
        w, h = im.size
    if rng is None:
        l, t = (w - size) // 2, (h - size) // 2
    else:
        l, t = int(rng.integers(0, w - size + 1)), int(rng.integers(0, h - size + 1))
    return im.crop((l, t, l + size, t + size))


class FSDataset(Dataset):
    def __init__(self, items, size=224, train=False, seed=0):
        self.items, self.size, self.train = items, size, train
        self.rng = np.random.default_rng(seed)
        self.cls = [os.path.basename(os.path.dirname(os.path.dirname(p)))
                    for p, _, _ in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, g = self.items[i]
        im = Image.open(p).convert("RGB")
        im = _crop(im, self.size, self.rng if self.train else None)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        if self.train and self.rng.random() < 0.5:
            x = torch.flip(x, dims=[2])
        return x, y, 0, i


class FSPairedSampler(Sampler):
    """Batches with matched real/fake halves drawn from the same LSUN class --
    the closest available approximation to source-level pairing on ProGAN."""

    def __init__(self, items, cls, batch_size, mode="class", seed=0):
        self.bs, self.mode = batch_size, mode
        self.rng = np.random.default_rng(seed)
        self.by = {}
        for i, ((p, y, g), c) in enumerate(zip(items, cls)):
            self.by.setdefault((c, y), []).append(i)
        self.real = [i for i, (p, y, g) in enumerate(items) if y == 0]
        self.fake = [i for i, (p, y, g) in enumerate(items) if y == 1]
        self.classes = sorted({c for (c, y) in self.by
                               if (c, 0) in self.by and (c, 1) in self.by})
        if mode == "class" and len(self.classes) < 4:
            self.mode = "balanced"
        self.n = max(1, min(len(self.real), len(self.fake)) * 2 // batch_size)

    def __len__(self):
        return self.n

    def __iter__(self):
        half = self.bs // 2
        for _ in range(self.n):
            if self.mode == "class":
                b = []
                for c in self.rng.choice(self.classes, size=half, replace=True):
                    b.append(int(self.rng.choice(self.by[(c, 0)])))
                    b.append(int(self.rng.choice(self.by[(c, 1)])))
            else:
                b = [int(i) for i in self.rng.choice(self.real, half, replace=False)] + \
                    [int(i) for i in self.rng.choice(self.fake, half, replace=False)]
            yield b
