"""Correct semantic-path parameter metadata while preserving original counts.

Early result rows counted an instantiated diagnostic spectral branch even when
FusionDetector bypassed it. This migration records those harness-level counts
and replaces the public fields with the active deployment-path counts.
"""
import argparse
import glob
import json
import os


def semantic_only(method):
    return method == "semantic" or method.startswith("semantic_")


def migrate(path):
    with open(path) as stream:
        payload = json.load(stream)
    if not isinstance(payload, list):
        return 0
    changed = 0
    for row in payload:
        breakdown = row.get("breakdown")
        if not breakdown or not semantic_only(row.get("method", "")):
            continue
        row_changed = False
        active_trainable = breakdown["ln_affine"] + breakdown["sem_head"]
        deployed_params = breakdown["backbone_total"] + breakdown["sem_head"]
        if row.get("trainable") != active_trainable or row.get("params") != deployed_params:
            row.setdefault("instantiated_trainable", row.get("trainable"))
            row.setdefault("instantiated_params", row.get("params"))
            row["trainable"] = active_trainable
            row["params"] = deployed_params
            row["trainable_pct"] = 100.0 * active_trainable / deployed_params
            row["parameter_scope"] = "active semantic deployment path"
            row_changed = True
        # Remove a redundant explicit default written by some runs; the method
        # configuration and transform set are already stored in the run args.
        if row.get("tc_views") == ["jpeg", "resize"]:
            del row["tc_views"]
            row_changed = True
        changed += int(row_changed)
    if changed:
        with open(path, "w") as stream:
            json.dump(payload, stream, indent=1)
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    total = 0
    files = sorted(glob.glob(os.path.join(args.results, "*.json")))
    for filename in files:
        count = migrate(filename)
        if count:
            print(f"[params] {os.path.basename(filename)}: {count} row(s)")
            total += count
    print(f"[params] corrected {total} semantic row(s)")
