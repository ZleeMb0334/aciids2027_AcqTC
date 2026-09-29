"""GenImage-arrow data layer: shard indexing, LOGO folds, pair-aware batches.

Design notes that matter for the science:
  * Images are decoded from the ORIGINAL bytes and only cropped -- never
    re-encoded and never resized on the training path. Re-encoding would alter
    exactly the high-frequency evidence the noise branch reads.
  * The LOGO split is by generator: the held-out generator's images (real AND
    fake) are never seen in training.
  * Reals are the GenImage ImageNet reals shipped alongside each generator, so
    real and fake come from the same source distribution. Pairing is at the
    ImageNet-class level, not the source-image level -- stated as such.
"""
import glob
import io
import json
import os
import re
import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

GENS = ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4',
        'stable_diffusion_v_1_5', 'VQDM', 'wukong']
GAN_GENS = {'BigGAN'}
DIFF_GENS = set(GENS) - GAN_GENS
DATA = os.environ.get("DF_DATA", "/workspace/df-fusion/data")


# ------------------------------------------------------------------- indexing
def _shard_files(split):
    return sorted(glob.glob(os.path.join(DATA, "genimage", "data", split, "*", "*.arrow")))


def build_index(split, cache=True):
    """[{shard, row, label, generator, cls}] for every downloaded row."""
    cpath = os.path.join(DATA, f"index_{split}.json")
    if cache and os.path.exists(cpath):
        with open(cpath) as f:
            return json.load(f)
    files, recs = _shard_files(split), []
    for si, f in enumerate(files):
        gen = os.path.basename(os.path.dirname(f))
        with pa.memory_map(f) as src:
            t = pa.ipc.open_stream(src).read_all()
            labs = t["label"].to_pylist()
            paths = t["image_path"].to_pylist()
        for r, (lab, p) in enumerate(zip(labs, paths)):
            m = re.search(r"(n\d{8})", p)          # ImageNet wnid when present
            recs.append({"shard": si, "row": r, "label": int(lab),
                         "generator": gen, "cls": m.group(1) if m else ""})
    out = {"files": files, "records": recs}
    if cache:
        with open(cpath, "w") as f:
            json.dump(out, f)
    return out


class ShardStore:
    """Lazily memory-maps arrow shards; one instance per worker process."""

    def __init__(self, files):
        self.files, self._t = files, {}

    def bytes_at(self, shard, row):
        t = self._t.get(shard)
        if t is None:
            src = pa.memory_map(self.files[shard])
            t = pa.ipc.open_stream(src).read_all()
            self._t[shard] = t
        return t["image"][row].as_py()


# ------------------------------------------------------------------- dataset
def _center_crop(im, size):
    w, h = im.size
    if w < size or h < size:
        s = size / min(w, h)
        im = im.resize((max(size, int(w * s + .5)), max(size, int(h * s + .5))), Image.BICUBIC)
        w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return im.crop((l, t, l + size, t + size))


def _random_crop(im, size, rng):
    w, h = im.size
    if w < size or h < size:
        return _center_crop(im, size)
    l = rng.integers(0, w - size + 1)
    t = rng.integers(0, h - size + 1)
    return im.crop((int(l), int(t), int(l) + size, int(t) + size))


def _rematch(im, quality, qtables=None, subsampling=0):
    """Re-encode to JPEG. Applied IDENTICALLY to both classes, so container
    format and bitrate carry no label information afterwards."""
    buf = io.BytesIO()
    if qtables is not None:
        im.save(buf, format="JPEG", qtables=qtables, subsampling=subsampling)
    else:
        im.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def sample_qtable_pool(index, subset, n=256, seed=0):
    """Collect quantization tables + chroma subsampling from the REAL images.

    These describe the acquisition channel the reals went through. Replaying that
    channel on the fakes equalises compression *history*, not merely container
    format -- without it, matched-format data still leaks 'double-compressed vs
    single-compressed' as a label cue.
    """
    from PIL import JpegImagePlugin
    store = ShardStore(index["files"])
    rng = np.random.default_rng(seed)
    reals = [i for i in subset if index["records"][i]["label"] == 0]
    pool = []
    for i in rng.choice(reals, size=min(n, len(reals)), replace=False):
        r = index["records"][int(i)]
        try:
            im = Image.open(io.BytesIO(store.bytes_at(r["shard"], r["row"])))
            if im.format != "JPEG":
                continue
            q = getattr(im, "quantization", None)
            if not q:
                continue
            ss = JpegImagePlugin.get_sampling(im)
            pool.append(({k: list(v) for k, v in q.items()}, int(ss)))
        except Exception:
            continue
    return pool


class GenImageDS(Dataset):
    """acq='raw'      : images exactly as shipped (the standard, shortcut-laden protocol)
       acq='matched'  : crop to `size` from NATIVE resolution (never upscaled, so no
                        resampling cue) then re-encode both classes to JPEG at
                        `quality`. Removes the format / bitrate / resolution shortcut.
       Default size 128 is the largest crop that no GenImage image has to be
       upscaled to reach (BigGAN fakes are 128x128)."""

    def __init__(self, index, subset, size=128, train=False, seed=0,
                 acq="matched", quality=95, qpool=None):
        self.store = ShardStore(index["files"])
        self.recs = [index["records"][i] for i in subset]
        self.size, self.train = size, train
        self.acq, self.quality = acq, quality
        self.qpool = qpool or []
        self.rng = np.random.default_rng(seed)
        if acq == "history" and not self.qpool:
            raise ValueError("acq='history' needs qpool from sample_qtable_pool()")

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        b = self.store.bytes_at(r["shard"], r["row"])
        im = Image.open(io.BytesIO(b))
        was_jpeg = (im.format == "JPEG")
        im = im.convert("RGB")
        if self.acq == "history" and not was_jpeg:
            # Replay the reals' acquisition channel on the fake, at NATIVE
            # resolution and before cropping, exactly as the reals experienced it.
            qt, ss = self.qpool[(i * 2654435761) % len(self.qpool)]
            im = _rematch(im, None, {k: v for k, v in qt.items()}, ss)
        im = _random_crop(im, self.size, self.rng) if self.train else _center_crop(im, self.size)
        if self.acq in ("matched", "history"):
            im = _rematch(im, self.quality)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        if self.train and self.rng.random() < 0.5:
            x = torch.flip(x, dims=[2])
        return x, r["label"], GENS.index(r["generator"]), i


# ------------------------------------------------------------------- sampler
class PairedBatchSampler(Sampler):
    """Batches that always contain matched real/fake groups.

    mode='class'  : real and fake drawn from the same ImageNet class (the
                    closest available approximation to source-level pairing).
    mode='balanced': class-agnostic 50/50 real/fake batches (the ablation).
    """

    def __init__(self, recs, batch_size, mode="class", seed=0, drop_last=True):
        self.bs, self.mode, self.drop_last = batch_size, mode, drop_last
        self.rng = np.random.default_rng(seed)
        self.real = [i for i, r in enumerate(recs) if r["label"] == 0]
        self.fake = [i for i, r in enumerate(recs) if r["label"] == 1]
        by = {}
        for i, r in enumerate(recs):
            by.setdefault((r["cls"], r["label"]), []).append(i)
        self.by = by
        self.classes = sorted({c for (c, l) in by if c and (c, 0) in by and (c, 1) in by})
        if mode == "class" and len(self.classes) < 8:
            self.mode = "balanced"          # not enough paired classes; degrade loudly
        self.n = min(len(self.real), len(self.fake)) * 2 // batch_size

    def __len__(self):
        return max(1, self.n)

    def __iter__(self):
        half = self.bs // 2
        for _ in range(len(self)):
            if self.mode == "class":
                cs = self.rng.choice(self.classes, size=half, replace=True)
                b = []
                for c in cs:
                    b.append(int(self.rng.choice(self.by[(c, 0)])))
                    b.append(int(self.rng.choice(self.by[(c, 1)])))
            else:
                b = [int(i) for i in self.rng.choice(self.real, half, replace=False)] + \
                    [int(i) for i in self.rng.choice(self.fake, half, replace=False)]
            yield b


# ------------------------------------------------------------------- LOGO
def logo_split(index, held_out, n_train_per_gen=None, n_test=None, seed=0,
               val_frac=0.1):
    """Leave-one-generator-out. Returns dict of index lists.

    train/val come from the TRAIN split of the other generators;
    test_in  = test split of the training generators (in-distribution)
    test_out = test split of the held-out generator (the generalisation target)
    """
    rng = np.random.default_rng(seed)
    tr_idx = [i for i, r in enumerate(index["train"]["records"])
              if r["generator"] != held_out]
    te_in = [i for i, r in enumerate(index["test"]["records"])
             if r["generator"] != held_out]
    te_out = [i for i, r in enumerate(index["test"]["records"])
              if r["generator"] == held_out]

    def cap(idx, recs, n):
        if n is None:
            return idx
        out = []
        for g in GENS:
            for lab in (0, 1):
                s = [i for i in idx if recs[i]["generator"] == g and recs[i]["label"] == lab]
                if s:
                    out += list(rng.choice(s, size=min(n // 2, len(s)), replace=False))
        return sorted(int(i) for i in out)

    tr_idx = cap(tr_idx, index["train"]["records"], n_train_per_gen)
    te_in = cap(te_in, index["test"]["records"], n_test)
    te_out = cap(te_out, index["test"]["records"], n_test)
    tr_idx = list(rng.permutation(tr_idx))
    nv = int(len(tr_idx) * val_frac)
    return {"train": tr_idx[nv:], "val": tr_idx[:nv], "test_in": te_in, "test_out": te_out}


def load_indices():
    return {"train": build_index("train"), "test": build_index("test")}
