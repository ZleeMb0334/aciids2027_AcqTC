"""Record cryptographic identities for checkpoints used by reported results."""
import argparse
import glob
import hashlib
import json
import os


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="/workspace/df-fusion/results")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    output = {
        "algorithm": "SHA-256 over checkpoint bytes",
        "scope": "all experiment checkpoints and baseline weights retained in the results directory",
        "files": [],
    }
    patterns = ("ckpt_*.pt", "base_*.pt")
    paths = sorted({path for pattern in patterns
                    for path in glob.glob(os.path.join(args.results, pattern))})
    for path in paths:
        row = {
            "file": os.path.basename(path),
            "bytes": os.path.getsize(path),
            "sha256": sha256(path),
        }
        output["files"].append(row)
        print(f"[checkpoint-manifest] {row['file']}: {row['sha256']}", flush=True)
    out = args.out or os.path.join(args.results, "checkpoint_integrity.json")
    with open(out, "w") as stream:
        json.dump(output, stream, indent=2)
    print(f"[checkpoint-manifest] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
