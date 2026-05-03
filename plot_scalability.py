"""Plot scalability results from CSV.

Generates:
  Figure 1: Runtime vs n (log-scale y), lines for Bounded vs Ours vs HP,
            panels by density
  Figure 2: Runtime vs n, panels by in-degree bound, lines for each checker
            (key figure for Cor 5.5: Bounded shows polynomial scaling)

Run:  python plot_scalability.py [--input results/scalability.csv]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Try usetex for proper LaTeX rendering; fall back gracefully
try:
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    # Test that LaTeX works
    fig_test = plt.figure()
    fig_test.text(0.5, 0.5, r"$k$")
    fig_test.savefig("/dev/null", format="png")
    plt.close(fig_test)
    USETEX = True
except Exception:
    plt.rcParams["text.usetex"] = False
    USETEX = False

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
})

CHECKERS = [
    ("Frontier (Prop.~4.1)",     "bounded_time", "#2ca02c", "D"),
    ("Def.~3.3 (no frontier)",   "ours_time",    "#1f77b4", "o"),
    ("HPm (2015)",                "hp_time",      "#d62728", "s"),
    ("BV (2018)",                "bv_time",      "#9467bd", "^"),
    ("CNESS (2021)",             "cness_time",   "#ff7f0e", "v"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path):
    """Load and preprocess scalability results."""
    df = pd.read_csv(path)
    for col in ["bounded_time", "ours_time", "hp_time", "bv_time", "cness_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _detect_timeout(df):
    """Detect timeout value from the data."""
    for col in ["bounded_time", "ours_time", "hp_time", "bv_time", "cness_time"]:
        if col in df.columns:
            mx = df[col].max()
            if mx >= 9.0:
                return round(mx)
    return None


def _style_axis(ax):
    """Apply clean academic styling to a single axis."""
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, which="major", linewidth=0.3, color="#cccccc",
                  zorder=0)
    ax.xaxis.grid(False)


def _plot_checker_lines(ax, sub, checkers=None):
    """Plot mean + 95% CI shading for each checker."""
    if checkers is None:
        checkers = CHECKERS
    for label, col, color, marker in checkers:
        if col not in sub.columns:
            continue
        grouped = sub.groupby("n")[col]
        mean = grouped.mean()
        sem = grouped.sem()
        ci_lo = mean - 1.96 * sem
        ci_hi = mean + 1.96 * sem

        ax.plot(mean.index, mean.values, f"-{marker}",
                color=color, label=label, markersize=2.5, linewidth=1.0,
                zorder=3)
        ax.fill_between(mean.index, ci_lo.values, ci_hi.values,
                        alpha=0.12, color=color, linewidth=0, zorder=2)


def _add_timeout_line(fig, axes, timeout):
    """Add a single dashed red timeout line across all panels (no label, no legend)."""
    if timeout is None:
        return
    for ax in axes:
        ax.axhline(y=timeout, color="#d62728", linewidth=0.7,
                    linestyle="--", zorder=1)


def _add_legend(fig, axes):
    """Add shared horizontal legend below panels."""
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CHECKERS),
               frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.5, handletextpad=0.4)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_runtime_vs_n(df, output_dir="results"):
    """Figure 1: Runtime vs n, panels by density."""
    densities = sorted(df["density"].unique())
    timeout = _detect_timeout(df)

    fig, axes = plt.subplots(1, len(densities), figsize=(5.5, 2.6),
                             sharey=True)
    if len(densities) == 1:
        axes = [axes]

    for ax, density in zip(axes, densities):
        sub = df[df["density"] == density]
        _plot_checker_lines(ax, sub)
        _style_axis(ax)
        ax.set_xlabel(r"Variables ($n$)")
        ax.set_title(rf"Density $= {density}$")

    axes[0].set_ylabel("Runtime (s)")
    _add_timeout_line(fig, axes, timeout)
    _add_legend(fig, axes)

    fig.tight_layout(w_pad=1.2)
    fig.subplots_adjust(bottom=0.27)
    for ext in ["pdf", "png"]:
        p = os.path.join(output_dir, f"scalability_runtime.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext == "png" else None)
        print(f"Saved {p}")
    plt.close(fig)


def plot_runtime_by_indegree(df, output_dir="results", density=0.4):
    """Figure 2: Runtime vs n, panels by in-degree bound.

    Key figure for Cor 4.5: in bounded-degree panels, Frontier (Prop 4.4)
    shows polynomial scaling while brute-force and HP grow exponentially.
    Filters to a single density for cleaner signal.
    """
    df = df[df["density"] == density]
    indegrees = sorted(df["max_indegree"].unique(),
                       key=lambda x: (1 if str(x) == "inf" else 0,
                                      float("inf") if str(x) == "inf"
                                      else float(x)))
    timeout = _detect_timeout(df)

    n_panels = len(indegrees)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5, 2.25), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, indeg in zip(axes, indegrees):
        sub = df[df["max_indegree"] == indeg]
        _plot_checker_lines(ax, sub)
        _style_axis(ax)
        ax.set_xlabel(r"Variables ($n$)")
        if str(indeg) == "inf":
            title = "Unbounded"
        else:
            title = rf"$k \leq {int(float(indeg))}$"
        ax.set_title(title)

    axes[0].set_ylabel("Runtime (s)")
    _add_timeout_line(fig, axes, timeout)
    _add_legend(fig, axes)

    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(bottom=0.32)
    for ext in ["pdf", "png"]:
        p = os.path.join(output_dir, f"scalability_indegree.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=200 if ext == "png" else None)
        print(f"Saved {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df):
    """Print summary statistics."""
    print("\nSummary Statistics:")
    print("=" * 60)

    checkers = [("Bounded", "bounded"), ("Ours", "ours"), ("HP Modified", "hp"),
                 ("BV", "bv"), ("CNESS", "cness")]
    for label, prefix in checkers:
        col = f"{prefix}_time"
        if col not in df.columns:
            continue
        print(f"\n{label}:")
        for n in sorted(df["n"].unique()):
            sub = df[df["n"] == n]
            times = sub[col]
            to_col = f"{prefix}_timeout"
            n_to = sub[to_col].sum() if to_col in sub.columns else 0
            print(f"  n={n:3d}: median={times.median():8.4f}s  "
                  f"mean={times.mean():8.4f}s  "
                  f"max={times.max():8.4f}s  "
                  f"timeouts={int(n_to)}/{len(sub)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scalability results")
    parser.add_argument("--input", "-i", default="results/scalability.csv",
                        help="Input CSV path")
    parser.add_argument("--output-dir", "-o", default="results",
                        help="Output directory for plots")
    args = parser.parse_args()

    df = load_data(args.input)
    print_summary(df)
    plot_runtime_vs_n(df, args.output_dir)
    plot_runtime_by_indegree(df, args.output_dir)
