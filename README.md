# AcqTC artifact

This snapshot contains experiment source, reported JSON results, the
generators for manuscript tables and macros, plus diagnostic figures.
The render.py and render_review_results.py scripts read results and write
manuscript/generated files.

## Snapshot scope

Public datasets, most model checkpoints, and the original server container are
excluded from this workspace. The download_data.py script creates
data/DOWNLOAD_MANIFEST.json during GenImage download;
`results/genimage_download_manifest.json` preserves the evaluated manifest.
The ForenSynths release
comes from the official CNNDetection release. Retrieve its test and validation
archives with `dataset/test/download_testset.sh` and
`dataset/val/download_valset.sh` from
https://github.com/PeterWang512/CNNDetection. Place the extracted directories at
`data/forensynths/CNN_synth_testset` and `data/forensynths/progan_val`.
`results/data_integrity.json` records file counts, byte counts, and a stable
tree SHA-256 for the evaluated copy. The compressed
`results/forensynths_file_manifest.tsv.gz` records every relative path, size,
and file SHA-256. Four server constraints were preserved in
the run notes:

- Python 3.10
- numpy 1.26.4
- protobuf 3.20.3
- transformers 4.44.2

The code also requires PyTorch, torchvision, Pillow, scikit-learn, SciPy,
Matplotlib, plus open_clip_torch. `environment/requirements-lock.txt` captures
the final-review container used for the added experiments. It may differ from
the environment that produced older checkpoints. The included JSON and NPZ
files support result auditing. Bit-for-bit retraining of earlier runs still
requires reconstruction of the omitted environment.

OpenAI CLIP weights must use the QuickGELU variants
ViT-B-16-quickgelu or ViT-L-14-quickgelu.

## Reproducing the analysis

    python3 src/audit_benchmarks.py
    python3 src/gate1.py --acq raw
    python3 src/gate1.py --acq matched
    python3 src/gate1.py --acq history
    python3 src/run_fs.py --seed 0
    python3 src/baselines_eval.py --seed 0 --train '' --clean-only
    python3 src/baselines_eval.py --seed 0 --train '' --benign-only
    python3 src/reliance.py --seed 0
    python3 src/run_fs.py --seed 0 --tag semtc --methods semantic_tc
    python3 src/run_fs.py --seed 0 --tag objabl --methods semantic_aug,semantic_align
    python3 src/run_fs.py --seed 0 --tag acqtcl --methods semantic_tc --arch ViT-L-14-quickgelu
    python3 src/run_fs.py --seed 0 --tag abl --methods concat,static,mean
    python3 src/audit_repeats.py --force
    python3 src/new_experiments.py gate
    python3 src/new_experiments.py modern
    python3 src/new_experiments.py modern --methods semantic_l,semantic_tc_l --output-stem modern_genimage_l14
    python3 src/new_experiments.py modern-pgd
    python3 src/new_experiments.py pgd
    python3 src/new_experiments.py pgd-table
    python3 src/new_experiments.py linearity
    python3 src/new_experiments.py benchmark
    python3 src/make_data_manifest.py --force
    python3 src/make_checkpoint_manifest.py
    python3 src/make_framework_data.py
    python3 src/stats.py
    python3 src/review_analysis.py
    python3 src/render.py
    python3 src/render_review_results.py
    python3 src/render_result_plots.py

Repeat the main detector, baseline, semantic-control, and fusion-control
commands for seeds 1 and 2.
The `semantic_tc` identifier denotes AcqTC. `semantic_aug` retains transformed-
view classification without logit alignment; `semantic_align` retains logit
alignment without transformed-view classification. Semantic-only variants
bypass the inactive spectral module in the shared experiment harness.
The run_fs.py script merges new methods into a method/seed result file. A
targeted rerun of the full model may appear in a gatefull result file. The
renderer keeps one row per method/seed; the specific gatefull row replaces its
fs counterpart. The `semtc` and `abl` review controls evaluate transfer,
JPEG-70, and half-scale resizing; they skip unrelated attack families.

## Baseline provenance

| baseline | repository commit | weights used |
|---|---|---|
| CNNDetection | `ea0b5622` | official blur/JPEG checkpoint |
| UniversalFakeDetect | `030495ae` | official linear-probe checkpoint |
| NPR | `781ced3f` | official `NPR.pth` checkpoint |
| FreqNet | `2c6bd020` | official `4-classes-freqnet-v2.pth` checkpoint |

Each baseline JSON row records this provenance. The fixed-checkpoint repetitions
vary the sampled evaluation files; their SD quantifies test-sampling variation.
The `baseline_predictions__<method>__s<seed>.npz` archives preserve generator,
label, score, and sample index for all 12,000 evaluated files. The benign-only
pass reuses each preserved clean row, then evaluates JPEG-70 and half-scale
resizing on the common 1,000-image ProGAN transformation set.
The `baseline_benign_predictions__<method>__s<seed>.npz` archives retain the
corresponding transformed scores and sample indices.

## Review corrections retained in the code

1. Adaptive-attack PSNR is averaged over perturbed fake images.
2. Local-linearity checks use realised band displacement after inverse FFT plus
   clipping. Version 2 also compares fused-score sensitivity with fused-score
   change. The included reliance JSON predates version 2; its fit values are
   excluded from manuscript claims.
3. Adaptive sweeps begin at 0.125/255 in reliance.py.
4. Result aggregation deduplicates method/seed pairs.
5. The theoretical summary uses effective band count
   \(N_{\mathrm{eff}}=1/\sum_b q_b^2\). Shannon entropy remains only in the
   reported exploratory hinge ablation.
6. The reported PGD objective minimizes fused-logit binary cross-entropy. The
   optional gate-weight term in `attacks.py` is excluded from reported attacks.
7. Semantic-only parameter rows report the active deployment path. The
   migration preserves earlier harness-level counts in `instantiated_trainable`
   and `instantiated_params`; those counts included a bypassed diagnostic
   spectral branch.
8. Generated-result loading ignores Finder/iCloud conflict copies whose names
   end in a numeric suffix before the extension. Such copies are archival and
   never count as repetitions.

## Final-review additions

- `review_analysis.json` contains cutoff sensitivity and fusion-form summaries.
- `audit_repeats.json` measures shortcut-index sensitivity over five sampling
  seeds and five shuffled fold seeds.
- `modern_genimage.json` evaluates preserved checkpoints on a common
  history-matched, canonicalized GenImage protocol. History matching replays a
  real-file quantization table on PNG synthetic files. The final encoding chain
  is shared by both labels and documented in the result file.
- `modern_genimage_l14.json` applies the same frozen protocol to AcqTC-L/14 and
  its matched semantic L/14 control. Its NPZ companion retains all predictions.
- `modern_pgd.json` applies one-sided, fused-loss PGD to Midjourney and Stable
  Diffusion 1.5 under the same canonicalization protocol. Its NPZ companions
  preserve sample identifiers plus clean and attacked scores.
- `gate_diagnostics.json` records gate and branch behavior under clean, JPEG,
  and resize conditions.
- `pgd_reeval.json` contains the corrected fused-loss attack sweep.
- `pgd_check.json` records the step-count and restart check.
- `linearity_check.json` records a held-out test of the local linearization.
- `inference_benchmark.json` records synchronized H100 latency, throughput,
  and peak allocated memory.
- `objabl__s0..2.json` separates transformed-view classification from logit
  alignment in the AcqTC objective.
- `acqtcl__s0..2.json` contains the three-seed ViT-L/14 scale study. It uses
  the same objective and schedule as AcqTC-B/16.
- `baselines__s0..2.json` compares four released detectors; twelve NPZ files
  retain their per-example predictions.
- Twelve `baseline_benign_predictions__*.npz` files preserve JPEG-70 and resize
  predictions for the external detectors.
- `checkpoint_integrity.json` records checkpoint sizes and SHA-256 digests.
- Corresponding compressed NPZ files preserve per-example predictions where
  the final-review scripts generate them.
