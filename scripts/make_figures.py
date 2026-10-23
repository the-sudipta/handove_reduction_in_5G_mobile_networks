"""Figures for the README: replication vs paper, and the corrected study.

    python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import pandas as pd                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
COLORS = {"A3": "#1f77b4", "LSTM": "#ff7f0e", "RL": "#2ca02c", "Proposed": "#d62728",
          "DQN-raw": "#9467bd", "Eq4-greedy": "#7f7f7f"}


def grouped_bars(ax, table, value, error=None, methods=None):
    """One group of bars per user count, one bar per method."""
    methods = methods or [m for m in COLORS if m in set(table["method"])]
    users = sorted(table["users"].unique())
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        rows = table[table["method"] == method].set_index("users").reindex(users)
        x = [u + (i - (len(methods) - 1) / 2) * width for u in range(len(users))]
        ax.bar(x, rows[value], width, yerr=None if error is None else rows[error],
               color=COLORS[method], label=method, capsize=2)
    ax.set_xticks(range(len(users)), users)
    ax.set_xlabel("Number of users")


def replication_figure():
    table = pd.read_csv(ROOT / "replication" / "results" / "seed_sweep_median_vs_table4.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, column, title in [(axes[0], "handovers_paper", "Published (Table 4)"),
                              (axes[1], "handovers_median10", "Authors' code re-run (median of 10 seeds)")]:
        grouped_bars(ax, table, column, methods=["A3", "LSTM", "RL", "Proposed"])
        ax.set_title(title)
    axes[0].set_ylabel("Handovers")
    axes[1].legend()
    fig.suptitle("Scale1x: the released code reproduces Table 4")
    fig.tight_layout()
    fig.savefig(FIGURES / "replication_table4.png", dpi=150)


def corrected_figures():
    summary = pd.read_csv(ROOT / "results" / "corrected" / "summary.csv")
    panels = [("ho_per_user_min", "Handovers per user-minute"),
              ("ping_pong_ratio", "Ping-pong ratio"),
              ("failure_ratio", "Handover failure ratio"),
              ("throughput_mbps", "Mean user throughput (Mbps)")]
    for scale, table in summary.groupby("scale"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (metric, label) in zip(axes.flat, panels):
            grouped_bars(ax, table, metric, metric + "_ci")
            ax.set_ylabel(label)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Corrected simulator, Scale{scale}x (mean ± 95% CI over seeds)")
        fig.tight_layout()
        fig.savefig(FIGURES / f"corrected_scale{scale}x.png", dpi=150)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    replication_figure()
    corrected_figures()
    print(f"Figures written to {FIGURES}")
