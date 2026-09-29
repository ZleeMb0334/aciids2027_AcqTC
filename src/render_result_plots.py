#!/usr/bin/env python3
"""Render non-redundant diagnostic plots from the released result files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifact" / "results"
OUTPUT = ROOT / "manuscript" / "images" / "results_diagnostics.pdf"
SEEDS = (0, 1, 2)


def load_internal(stem: str, method: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        records = json.loads((RESULTS / f"{stem}__s{seed}.json").read_text())
        rows.append(next(record for record in records if record["method"] == method))
    return rows


def load_external(method: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        records = json.loads((RESULTS / f"baselines__s{seed}.json").read_text())
        rows.append(next(record for record in records if record["method"] == method))
    return rows


def per_source(records: list[dict], sources: list[str]) -> np.ndarray:
    return np.asarray(
        [[record["per_generator"][source]["auroc"] for source in sources]
         for record in records],
        dtype=float,
    )


def render() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.4,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )

    fig, (ax_legacy, ax_modern) = plt.subplots(
        1,
        2,
        figsize=(6.25, 2.20),
        gridspec_kw={"width_ratios": (1.02, 1.18)},
        constrained_layout=True,
    )

    legacy_sources = ["deepfake", "biggan", "gaugan", "cyclegan", "stylegan"]
    legacy_labels = ["DeepFake", "BigGAN", "GauGAN", "CycleGAN", "StyleGAN"]
    series = [
        ("FreqNet", load_external("freqnet"), "#7A7A7A", "o", "--"),
        ("Semantic-L/14", load_internal("matchL", "semantic"), "#2F6690", "s", ":"),
        ("AcqTC-L/14", load_internal("acqtcl", "semantic_tc"), "#00897B", "D", "-"),
    ]
    x = np.arange(len(legacy_sources))
    _lo, _hi = [], []
    for label, records, colour, marker, linestyle in series:
        values = per_source(records, legacy_sources)
        _mu, _sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        _lo.append(float((_mu - _sd).min())); _hi.append(float((_mu + _sd).max()))
        ax_legacy.errorbar(
            x,
            _mu,
            yerr=_sd,
            label=label,
            color=colour,
            marker=marker,
            markersize=3.7,
            linewidth=1.25,
            elinewidth=0.8,
            capsize=2.0,
            linestyle=linestyle,
        )
    ax_legacy.set_xticks(x, legacy_labels, rotation=20, ha="right",
                         rotation_mode="anchor")
    # limits follow the data so no error-bar cap is clipped
    _l, _h = min(_lo), max(_hi)
    _s = max(_h - _l, 1e-6)
    ax_legacy.set_ylim(_l - 0.09 * _s, min(1.004, _h + 0.09 * _s))
    ax_legacy.set_yticks([t for t in (0.94, 0.96, 0.98, 1.00)
                          if _l - 0.09 * _s <= t <= _h + 0.09 * _s])
    ax_legacy.set_ylabel("AUROC", labelpad=2)
    ax_legacy.set_title("(a) Source-wise legacy transfer", loc="left",
                        fontweight="bold", pad=7)
    ax_legacy.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax_legacy.legend(frameon=False, loc="lower right", handlelength=2.2)

    modern = json.loads((RESULTS / "modern_genimage_l14.json").read_text())
    modern_sources = [
        "ADM", "BigGAN", "glide", "Midjourney",
        "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong",
    ]
    modern_labels = ["ADM", "BigGAN", "GLIDE", "Midjourney", "SD 1.4", "SD 1.5", "VQDM", "Wukong"]
    lookup = {
        (row["method"], row["seed"], row["generator"]): row["fused"]["auroc"]
        for row in modern["rows"]
        if row["condition"] == "clean"
    }
    paired = np.asarray(
        [[lookup[("semantic_tc_l", seed, source)] - lookup[("semantic_l", seed, source)]
          for source in modern_sources] for seed in SEEDS]
    )
    gains = paired.mean(axis=0)
    errors = paired.std(axis=0, ddof=1)
    colours = ["#00897B" if gain >= 0 else "#C44E52" for gain in gains]
    ax_modern.bar(
        np.arange(len(modern_sources)), gains, yerr=errors, width=0.68,
        color=colours, edgecolor="white", linewidth=0.45,
        error_kw={"elinewidth": 0.8, "capsize": 2.0, "capthick": 0.8},
    )
    ax_modern.axhline(0.0, color="#555555", linewidth=0.75)
    ax_modern.set_xticks(np.arange(len(modern_sources)), modern_labels,
                         rotation=20, ha="right", rotation_mode="anchor")
    ax_modern.set_ylabel(r"AUROC gain", labelpad=2)
    # limits follow the data (bar +/- error) so no cap is ever clipped
    top = float((gains + errors).max())
    bot = float(min(0.0, (gains - errors).min()))
    span = max(top - bot, 1e-6)
    ax_modern.set_ylim(bot - 0.06 * span, top + 0.10 * span)
    ax_modern.set_title("(b) Objective gain on GenImage", loc="left",
                        fontweight="bold", pad=7)
    ax_modern.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax_modern.set_axisbelow(True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # constrained_layout does not account for rotated tick labels reaching into
    # the neighbouring axes' ylabel, so the column gap is set explicitly
    try:
        fig.get_layout_engine().set(w_pad=0.06, h_pad=0.03, wspace=0.10)
    except Exception:
        fig.subplots_adjust(wspace=0.34)

    fig.savefig(
        OUTPUT,
        bbox_inches="tight",
        metadata={"Title": "AcqTC result diagnostics", "Creator": "artifact/src/render_result_plots.py"},
    )
    plt.close(fig)


if __name__ == "__main__":
    render()
