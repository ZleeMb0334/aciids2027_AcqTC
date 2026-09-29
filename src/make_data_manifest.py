"""Create compact content-integrity records for the evaluated datasets."""
import argparse
import gzip
import hashlib
import json
import os


def tree_digest(path, subset, manifest):
    digest = hashlib.sha256()
    count = 0
    total = 0
    for root, directories, files in os.walk(path):
        directories.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            relative = os.path.relpath(full, path).replace(os.sep, "/")
            size = os.path.getsize(full)
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(str(size).encode("ascii") + b"\0")
            file_digest = hashlib.sha256()
            with open(full, "rb") as f:
                for block in iter(lambda: f.read(4 << 20), b""):
                    digest.update(block)
                    file_digest.update(block)
            manifest.write(
                f"{subset}\t{relative}\t{size}\t{file_digest.hexdigest()}\n")
            count += 1
            total += size
    return {"files": count, "bytes": total, "tree_sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get(
        "DF_FS", "/workspace/df-fusion/data/forensynths"))
    parser.add_argument("--out", default="/workspace/df-fusion/results/data_integrity.json")
    parser.add_argument("--file-manifest", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    default_manifest = os.path.join(
        os.path.dirname(args.out), "forensynths_file_manifest.tsv.gz")
    requested_manifest = args.file_manifest or default_manifest
    if (os.path.isfile(args.out) and os.path.isfile(requested_manifest)
            and not args.force):
        print(f"[manifest] keeping existing {args.out}", flush=True)
        return
    targets = {
        "progan_val": os.path.join(args.root, "progan_val"),
    }
    test_root = os.path.join(args.root, "CNN_synth_testset")
    for name in sorted(os.listdir(test_root)):
        path = os.path.join(test_root, name)
        if os.path.isdir(path):
            targets[f"CNN_synth_testset/{name}"] = path
    output = {
        "algorithm": "SHA-256 over sorted relative path, file size, and file bytes",
        "root_layout": "ForenSynths validation and CNN_synth_testset directories",
        "file_manifest_columns": ["subset", "relative_path", "bytes", "sha256"],
        "subsets": {},
    }
    manifest_path = requested_manifest
    with gzip.open(manifest_path, "wt", encoding="utf-8", newline="\n") as manifest:
        manifest.write("subset\trelative_path\tbytes\tsha256\n")
        for name, path in targets.items():
            output["subsets"][name] = tree_digest(path, name, manifest)
            print(f"[manifest] {name}: {output['subsets'][name]}", flush=True)
    output["file_manifest"] = os.path.basename(manifest_path)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[manifest] wrote {args.out}")


if __name__ == "__main__":
    main()
