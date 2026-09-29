# Environment capture

`requirements-lock.txt` is generated with `python3 -m pip freeze` inside the
H100 container used for the final-review experiments. `system.json` records
Python, PyTorch, CUDA, cuDNN, GPU, operating-system, and platform information.

This capture supports the added September 2026 experiments. Older checkpoints
were produced in the same project lineage, although an immutable container
digest was unavailable. The lock therefore cannot establish bit-for-bit
identity with every earlier training run.
