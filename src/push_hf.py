"""Push artifact code + results + manuscript to the HF dataset repo.

Run from the LAPTOP. The token is read locally and never copied to the server.
"""
import os, sys, subprocess
from huggingface_hub import HfApi

REPO = os.environ.get("HF_REPO", "lexuanbach/df-semantic-noise-fusion")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # artifact/
PROJ = os.path.dirname(ROOT)
TOKFILE = os.path.expanduser("~/Desktop/05_Notes/2026/0-Working/hf-token.md")


def token():
    return open(TOKFILE).read().strip().split()[-1]


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else "sync"
    api = HfApi(token=token())
    api.create_repo(REPO, repo_type="dataset", exist_ok=True, private=True)
    for folder, prefix, patterns in (
        (os.path.join(ROOT, "src"), "artifact/src", ["*.py"]),
        (os.path.join(ROOT, "configs"), "artifact/configs", ["*"]),
        (os.path.join(ROOT, "results"), "artifact/results", ["*.json", "*.csv", "*.npz", "*.tex"]),
        (os.path.join(PROJ, "manuscript"), "manuscript", ["*.tex", "*.bib", "*.md", "*.pdf"]),
        (PROJ, "", ["PLAN.md", "STATUS.md"]),
    ):
        if not os.path.isdir(folder):
            continue
        try:
            api.upload_folder(folder_path=folder, path_in_repo=prefix, repo_id=REPO,
                              repo_type="dataset", commit_message=msg,
                              allow_patterns=patterns,
                              ignore_patterns=["*.pt", "__pycache__/*", "*.aux", "*.log"])
        except Exception as e:
            print(f"  ! {prefix}: {e}")
        else:
            print(f"  + {prefix}")
    print(f"pushed to https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
