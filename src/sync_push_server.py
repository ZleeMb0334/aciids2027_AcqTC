"""Mirror server results to the laptop and push to HF on a loop.

The GPU containers have been wiped without notice before, so nothing may live
only on the server. Usage:  python3 sync_push_server.py [interval_seconds]
"""
import os, subprocess, sys, time

HOST = os.environ.get("DF_HOST", "h100")
REMOTE = os.environ.get("DF_REMOTE", "/workspace/df-fusion/results/")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL = os.path.join(ROOT, "results") + "/"
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 600


def once(i):
    os.makedirs(LOCAL, exist_ok=True)
    r = subprocess.run(["rsync", "-az", "--exclude", "ckpt/", "--exclude", "*.pt", "-e",
                        "ssh -o BatchMode=yes", f"{HOST}:{REMOTE}", LOCAL],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[sync {i}] rsync failed: {r.stderr.strip()[:200]}", flush=True)
        return
    n = sum(len(f) for _, _, f in os.walk(LOCAL))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "src", "push_hf.py"),
                        f"server sync {time.strftime('%Y-%m-%d %H:%M')}"],
                       capture_output=True, text=True)
    print(f"[sync {i}] {n} files | push {'ok' if p.returncode == 0 else 'FAILED'}", flush=True)
    if p.returncode:
        print(p.stdout[-500:], p.stderr[-500:], flush=True)


if __name__ == "__main__":
    i = 0
    while True:
        i += 1
        try:
            once(i)
        except Exception as e:
            print(f"[sync {i}] error {e}", flush=True)
        time.sleep(INTERVAL)
