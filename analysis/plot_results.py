"""
Visualization and plotting for the Nim pilot study.

Generates all key figures:
  1. Bar chart: Win/Loss accuracy across 4 test sets per architecture
  2. Generalization heatmap: accuracy vs (num_heaps, max_heap_size)
  3. Learning curves: train/val accuracy over epochs
  4. Grundy value confusion matrix (CGT-Net)
  5. Ablation: CGT-Net with vs without auxiliary Grundy loss

Usage:
    python -m analysis.plot_results
"""

from __future__ import annotations
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, List, Optional

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODEL_NAMES = ["mlp_baseline", "deepsets_baseline", "cgt_net", "dqn"]
MODEL_LABELS = ["MLP", "DeepSets", "CGT-Net", "DQN"]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
TEST_SETS = ["test_id", "test_ood_heaps", "test_ood_sizes", "test_ood_both"]
TEST_LABELS = ["ID", "OOD-Heaps", "OOD-Sizes", "OOD-Both"]


def load_results() -> Dict:
    path = os.path.join(RESULTS_DIR, "pilot_results.json")
    with open(path) as f:
        return json.load(f)


def load_histories() -> Dict:
    path = os.path.join(RESULTS_DIR, "training_histories.json")
    with open(path) as f:
        return json.load(f)


def plot_accuracy_comparison(results: Dict):
    """Bar chart: Win/Loss accuracy across all test sets for each model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    n_models = len(MODEL_NAMES)
    n_tests = len(TEST_SETS)
    bar_width = 0.18
    x = np.arange(n_tests)

    for i, (model, label, color) in enumerate(zip(MODEL_NAMES, MODEL_LABELS, COLORS)):
        if model not in results:
            continue
        means = []
        stds = []
        for ts in TEST_SETS:
            if ts in results[model]:
                stats = results[model][ts].get("win_loss_accuracy", {})
                if isinstance(stats, dict):
                    means.append(stats.get("mean", 0))
                    stds.append(stats.get("std", 0))
                else:
                    means.append(stats)
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        ax.bar(
            x + i * bar_width,
            means,
            bar_width,
            yerr=stds,
            label=label,
            color=color,
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Test Set")
    ax.set_ylabel("Win/Loss Accuracy")
    ax.set_title("Pilot Study: Generalization Across Test Sets")
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(TEST_LABELS)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_comparison.png"), bbox_inches="tight")
    plt.close()
    print("Saved accuracy_comparison.png")


def plot_learning_curves(histories: Dict):
    """Learning curves: train/val accuracy over epochs for supervised models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    supervised = ["mlp_baseline", "deepsets_baseline", "cgt_net"]
    labels = ["MLP", "DeepSets", "CGT-Net"]

    for ax, model, label, color in zip(axes, supervised, labels, COLORS):
        if model not in histories:
            continue

        for seed_i, h in enumerate(histories[model]):
            alpha = 0.3 if seed_i > 0 else 1.0
            lw = 1.0 if seed_i > 0 else 2.0
            epochs = range(1, len(h["train_acc"]) + 1)
            ax.plot(epochs, h["train_acc"], color=color, alpha=alpha, lw=lw)
            ax.plot(
                epochs,
                h["val_acc"],
                color=color,
                alpha=alpha,
                lw=lw,
                linestyle="--",
            )

        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3)
        ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Accuracy")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", lw=2, label="Train"),
        Line2D([0], [0], color="gray", lw=2, linestyle="--", label="Val"),
    ]
    axes[-1].legend(handles=legend_elements, loc="lower right")

    plt.suptitle("Training Curves (solid=train, dashed=val)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"), bbox_inches="tight")
    plt.close()
    print("Saved learning_curves.png")


def plot_generalization_heatmap(results: Dict):
    """
    Heatmap showing accuracy degradation as we increase heaps and heap sizes.
    Uses the 4 test sets as a 2x2 grid: (small/large heaps) x (few/many heaps).
    """
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(16, 4))

    for ax, model, label in zip(axes, MODEL_NAMES, MODEL_LABELS):
        if model not in results:
            ax.set_visible(False)
            continue

        grid = np.zeros((2, 2))
        mapping = {
            (0, 0): "test_id",
            (0, 1): "test_ood_heaps",
            (1, 0): "test_ood_sizes",
            (1, 1): "test_ood_both",
        }

        for (r, c), ts in mapping.items():
            if ts in results[model]:
                stats = results[model][ts].get("win_loss_accuracy", {})
                grid[r, c] = stats.get("mean", 0) if isinstance(stats, dict) else stats

        sns.heatmap(
            grid,
            ax=ax,
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            vmin=0.4,
            vmax=1.0,
            xticklabels=["2-3 heaps", "4-6 heaps"],
            yticklabels=["size 0-7", "size 8-15"],
            cbar=False,
        )
        ax.set_title(label)

    plt.suptitle("Generalization Heatmap: Accuracy by Configuration", y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, "generalization_heatmap.png"), bbox_inches="tight"
    )
    plt.close()
    print("Saved generalization_heatmap.png")


def plot_grundy_analysis(results: Dict, histories: Dict):
    """CGT-Net specific: Grundy value prediction accuracy across test sets."""
    if "cgt_net" not in results:
        print("No CGT-Net results found, skipping Grundy analysis.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ds_names = []
    grundy_means = []
    grundy_stds = []
    wl_means = []

    for ts, label in zip(TEST_SETS, TEST_LABELS):
        if ts in results["cgt_net"]:
            ds_names.append(label)
            g_stats = results["cgt_net"][ts].get("grundy_accuracy", {})
            w_stats = results["cgt_net"][ts].get("win_loss_accuracy", {})

            if isinstance(g_stats, dict):
                grundy_means.append(g_stats.get("mean", 0))
                grundy_stds.append(g_stats.get("std", 0))
            else:
                grundy_means.append(g_stats if g_stats else 0)
                grundy_stds.append(0)

            if isinstance(w_stats, dict):
                wl_means.append(w_stats.get("mean", 0))
            else:
                wl_means.append(w_stats if w_stats else 0)

    x = np.arange(len(ds_names))
    width = 0.35

    ax.bar(x - width / 2, wl_means, width, label="Win/Loss Acc", color=COLORS[2])
    ax.bar(
        x + width / 2,
        grundy_means,
        width,
        yerr=grundy_stds,
        label="Grundy Value Acc",
        color=COLORS[3],
        capsize=3,
    )

    ax.set_xlabel("Test Set")
    ax.set_ylabel("Accuracy")
    ax.set_title("CGT-Net: Win/Loss vs Grundy Value Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "grundy_analysis.png"), bbox_inches="tight")
    plt.close()
    print("Saved grundy_analysis.png")


def plot_dqn_training(histories: Dict):
    """DQN training progress: win rate and loss over episodes."""
    if "dqn" not in histories:
        print("No DQN histories found, skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for seed_i, h in enumerate(histories["dqn"]):
        alpha = 0.4 if seed_i > 0 else 1.0
        lw = 1.0 if seed_i > 0 else 2.0

        if h["win_rates"]:
            x_wr = np.arange(len(h["win_rates"])) * 1000
            ax1.plot(x_wr, h["win_rates"], color=COLORS[3], alpha=alpha, lw=lw)

        if h["losses"]:
            smoothed = np.convolve(h["losses"], np.ones(100) / 100, mode="valid")
            ax2.plot(smoothed, color=COLORS[3], alpha=alpha, lw=lw)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Win Rate vs Random")
    ax1.set_title("DQN Win Rate")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Loss (smoothed)")
    ax2.set_title("DQN Training Loss")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dqn_training.png"), bbox_inches="tight")
    plt.close()
    print("Saved dqn_training.png")


def plot_summary_table(results: Dict):
    """Generate a summary table as an image."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    headers = ["Model", "ID Acc", "OOD-Heaps", "OOD-Sizes", "OOD-Both", "Move Opt%"]
    rows = []

    for model, label in zip(MODEL_NAMES, MODEL_LABELS):
        if model not in results:
            continue
        row = [label]
        for ts in TEST_SETS:
            if ts in results[model]:
                stats = results[model][ts].get("win_loss_accuracy", {})
                if isinstance(stats, dict):
                    m, s = stats.get("mean", 0), stats.get("std", 0)
                    row.append(f"{m:.1%} ± {s:.1%}")
                else:
                    row.append(f"{stats:.1%}")
            else:
                row.append("N/A")

        omr = results[model].get("optimal_move_rate", {})
        if isinstance(omr, dict) and "mean" in omr:
            row.append(f"{omr['mean']:.1%}")
        else:
            row.append("N/A")
        rows.append(row)

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        bg = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i, j].set_facecolor(bg)

    ax.set_title("Pilot Study Results Summary", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "summary_table.png"), bbox_inches="tight")
    plt.close()
    print("Saved summary_table.png")


def generate_all_plots():
    """Generate all plots from saved results."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    results = load_results()
    histories = load_histories()

    print("\nGenerating plots...")
    plot_accuracy_comparison(results)
    plot_learning_curves(histories)
    plot_generalization_heatmap(results)
    plot_grundy_analysis(results, histories)
    plot_dqn_training(histories)
    plot_summary_table(results)
    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    generate_all_plots()
