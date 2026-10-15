"""Compare Recognition runs: inequality within an episode, status by rank, inequality over training.

    python algorithms/recognition/plot_runs.py \
        Individual=outputs/recognition/sep3_individual \
        Shared=outputs/recognition/sep3_shared \
        --out outputs/recognition/compare.png

Each run is drawn in its own colour; lines are the mean over seeds, bands span the
lowest to highest seed.
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical palette (first three slots pass all-pairs colour-vision checks)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def load_run(run_dir):
    curves = [read_csv(p) for p in sorted(glob.glob(os.path.join(run_dir, "eval_curve_seed*.csv")))]
    metrics = [read_csv(p) for p in sorted(glob.glob(os.path.join(run_dir, "metrics_seed*.csv")))]
    if not curves or not metrics:
        raise SystemExit(f"{run_dir}: no eval_curve_seed*.csv / metrics_seed*.csv found")
    return curves, metrics


def smooth_finite(x, y, window):
    """Rolling mean over the finite entries of y (episode metrics are NaN between episode ends)."""
    keep = np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(y) >= window:
        y = np.convolve(y, np.ones(window) / window, mode="valid")
        x = x[window - 1:]
    return x, y


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="semibold", pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def band(ax, x, ys, colour, label):
    ys = np.stack(ys)
    ax.fill_between(x, ys.min(0), ys.max(0), color=colour, alpha=0.15, linewidth=0)
    ax.plot(x, ys.mean(0), color=colour, linewidth=2, label=label, marker="o" if len(x) == 1 else None)
    ax.annotate(label, (x[-1], ys.mean(0)[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Label=run_dir")
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument("--smooth", type=int, default=5, help="rolling window for the training curve")
    args = parser.parse_args()
    if len(args.runs) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} runs per figure; split the comparison into several figures")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURFACE)
    for (label, run_dir), colour in zip((r.split("=", 1) for r in args.runs), SERIES):
        curves, metrics = load_run(run_dir)
        n_seeds = len(curves)
        name = f"{label} ({n_seeds} seed{'s' if n_seeds > 1 else ''})"

        band(axes[0], curves[0]["step"], [c["gini_mean"] for c in curves], colour, name)

        ranks = sorted((k for k in curves[0] if k.startswith("status_rank")), key=lambda k: int(k[11:]))
        final = np.array([[c[k][-1] for k in ranks] for c in curves])  # (seeds, N)
        x = np.arange(1, len(ranks) + 1)
        axes[1].fill_between(x, final.min(0), final.max(0), color=colour, alpha=0.15, linewidth=0)
        axes[1].plot(x, final.mean(0), color=colour, linewidth=2, marker="o", markersize=5, label=name)

        smoothed = [smooth_finite(m["env_step"], m["status_gini"], args.smooth) for m in metrics]
        length = min(len(s[1]) for s in smoothed)
        band(axes[2], smoothed[0][0][:length] / 1e6, [s[1][:length] for s in smoothed], colour, name)

    style(axes[0], "Status inequality within an episode", "step of the episode", "status Gini")
    style(axes[1], "Status by rank at episode end", "rank (1 = best known)", "status (sum of recognition)")
    style(axes[2], "Status inequality over training", "training steps (millions)", "status Gini at episode end")
    axes[1].set_xticks(np.arange(1, 11))
    for ax in axes:
        ax.margins(x=0.02)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
