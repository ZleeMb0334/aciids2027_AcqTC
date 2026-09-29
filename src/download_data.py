"""Download a fixed, budgeted subset of GenImage-arrow.

Budget is set in BUDGET below and is part of the experimental record: the exact
shard list downloaded is written to data/DOWNLOAD_MANIFEST.json so the subsample
is reproducible without re-deriving it.
"""
import json, os, sys, urllib.request, hashlib
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("DF_DATA", "/workspace/df-fusion/data")
REPO = "nebula/GenImage-arrow"
TRAIN_SHARDS_PER_GEN = int(os.environ.get("TRAIN_SHARDS", 6))   # ~4k rows each
GENS = ['ADM', 'BigGAN', 'glide', 'Midjourney', 'stable_diffusion_v_1_4',
        'stable_diffusion_v_1_5', 'VQDM', 'wukong']


def hf_url(repo, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def fetch(url, dest, expect_sha=None, retries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest, "cached"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for a in range(retries):
        try:
            tmp = dest + ".part"
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            if expect_sha:
                h = hashlib.sha256()
                with open(tmp, "rb") as f:
                    for b in iter(lambda: f.read(1 << 22), b""):
                        h.update(b)
                if h.hexdigest() != expect_sha:
                    os.remove(tmp)
                    raise ValueError("sha256 mismatch")
            os.rename(tmp, dest)
            return dest, "ok"
        except Exception as e:
            if a == retries - 1:
                return dest, f"FAIL {e}"
    return dest, "FAIL"


def main():
    os.makedirs(ROOT, exist_ok=True)
    m = json.load(urllib.request.urlopen(hf_url(REPO, "manifest.json"), timeout=120))
    jobs, record = [], {"repo": REPO, "train_shards_per_gen": TRAIN_SHARDS_PER_GEN, "shards": {}}
    for split in ("test", "train"):
        for g in GENS:
            b = m["bundles"][split][g]
            all_s = b["shards"]
            if split == "test":
                shards = all_s
            else:
                # Shards are ordered real-then-fake, so the first-N slice would be
                # all-real. Sample evenly across the list to span both classes.
                n = min(TRAIN_SHARDS_PER_GEN, len(all_s))
                idx = [round(i * (len(all_s) - 1) / max(1, n - 1)) for i in range(n)]
                shards = [all_s[i] for i in sorted(set(idx))]
            record["shards"][f"{split}/{g}"] = [
                {"path": s["path"], "rows": s["rows"], "sha256": s["sha256"]} for s in shards]
            for s in shards:
                jobs.append((hf_url(REPO, s["path"]), os.path.join(ROOT, "genimage", s["path"]), s["sha256"]))
    with open(os.path.join(ROOT, "DOWNLOAD_MANIFEST.json"), "w") as f:
        json.dump(record, f, indent=1)
    print(f"[download] {len(jobs)} shards", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for dest, st in ex.map(lambda j: fetch(*j), jobs):
            done += 1
            if st != "ok" and st != "cached":
                print(f"[{done}/{len(jobs)}] {st} {dest}", flush=True)
            elif done % 10 == 0:
                print(f"[{done}/{len(jobs)}] ok", flush=True)
    print("[download] genimage done", flush=True)


if __name__ == "__main__":
    main()
