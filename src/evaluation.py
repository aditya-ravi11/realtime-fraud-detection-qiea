"""
Metrics computation and plot generation for the fraud detection project.

All plots use seaborn-v0_8-whitegrid style, 300 DPI, tight layout.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_DPI = 300


def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute classification metrics.

    Returns dict with precision, recall, f1, auc_roc.
    """
    metrics = {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC_ROC": roc_auc_score(y_true, y_prob),
    }
    return metrics


def plot_qiea_convergence(convergence_history, mean_fitness_history, save_path):
    """
    Plot QIEA convergence: best fitness line + mean fitness shaded band.
    """
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))

    generations = range(1, len(convergence_history) + 1)

    ax.plot(generations, convergence_history, "b-", linewidth=2, label="Best Fitness")
    ax.fill_between(
        generations,
        mean_fitness_history,
        convergence_history,
        alpha=0.2,
        color="blue",
        label="Mean-to-Best Band",
    )
    ax.plot(
        generations,
        mean_fitness_history,
        "b--",
        linewidth=1,
        alpha=0.7,
        label="Mean Population Fitness",
    )

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (AUC-ROC - Parsimony Penalty)", fontsize=12)
    ax.set_title("QIEA Feature Selection Convergence", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_roc_curves(roc_data, save_path):
    """
    Overlay ROC curves.

    Parameters
    ----------
    roc_data : dict
        Keys are labels (e.g., "QIEA + RF"), values are (y_true, y_prob) tuples.
        If the key contains "ALL", plot with dashed line.
    """
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))

    for (label, (y_true, y_prob)), color in zip(roc_data.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        linestyle = "--" if "ALL" in label else "-"
        ax.plot(
            fpr, tpr, linestyle=linestyle, color=color, linewidth=2,
            label=f"{label} (AUC = {auc:.4f})"
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves: QIEA-Selected Features", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_confusion_matrices(cm_all, cm_qiea, label_all, label_qiea, save_path):
    """
    Side-by-side confusion matrices.
    """
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cm, title in [
        (axes[0], cm_all, f"ALL Features\n({label_all})"),
        (axes[1], cm_qiea, f"QIEA Features\n({label_qiea})"),
    ]:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            annot_kws={"size": 14},
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(title, fontsize=13)

    fig.suptitle("Confusion Matrices Comparison", fontsize=15, y=1.02)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_feature_importance(feature_names, importances, qiea_mask, save_path, top_n=15):
    """
    Horizontal bar chart of top features by RF importance.
    Green = QIEA-selected, grey = not selected.
    """
    plt.style.use(PLOT_STYLE)

    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    sorted_idx = sorted_idx[::-1]  # Reverse for horizontal bar (top at top)

    names = [feature_names[i] for i in sorted_idx]
    imps = importances[sorted_idx]
    colors = ["#2ecc71" if qiea_mask[i] == 1 else "#95a5a6" for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), imps, color=colors, edgecolor="white")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(
        "Feature Importance (Random Forest) with QIEA Selection Overlay", fontsize=13
    )

    # Legend
    green_patch = mpatches.Patch(color="#2ecc71", label="QIEA Selected")
    grey_patch = mpatches.Patch(color="#95a5a6", label="Not Selected")
    ax.legend(handles=[green_patch, grey_patch], fontsize=11, loc="lower right")

    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_latency_comparison(results_df, save_path):
    """
    Grouped bar chart: FS method x classifier -> latency.

    results_df must have columns: FS_Method, Model, Latency_ms.
    """
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(12, 7))

    methods = results_df["FS_Method"].unique()
    models = results_df["Model"].unique()
    x = np.arange(len(methods))
    width = 0.25
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for i, model in enumerate(models):
        subset = results_df[results_df["Model"] == model]
        # Ensure order matches methods
        latencies = []
        for m in methods:
            row = subset[subset["FS_Method"] == m]
            latencies.append(row["Latency_ms"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, latencies, width, label=model, color=colors[i % len(colors)])

    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_xlabel("Feature Selection Method", fontsize=12)
    ax.set_ylabel("Average Inference Latency (ms/transaction)", fontsize=12)
    ax.set_title(
        "Per-Transaction Inference Latency by Feature Selection Method", fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_streaming_latency(batch_latencies, batch_sizes, save_path):
    """
    Plot streaming throughput and latency.

    Parameters
    ----------
    batch_latencies : list of float
        Per-batch latency in ms.
    batch_sizes : list of int
        Number of transactions per batch.
    """
    plt.style.use(PLOT_STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: cumulative transactions over time
    cumulative_txns = np.cumsum(batch_sizes)
    cumulative_time = np.cumsum(batch_latencies) / 1000  # convert to seconds
    ax1.plot(cumulative_time, cumulative_txns, "b-", linewidth=2)
    ax1.set_xlabel("Elapsed Time (s)", fontsize=12)
    ax1.set_ylabel("Transactions Processed", fontsize=12)
    ax1.set_title("Streaming Throughput", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Right: per-batch latency
    ax2.plot(range(1, len(batch_latencies) + 1), batch_latencies, "r-", linewidth=1, alpha=0.7)
    ax2.axhline(
        y=np.median(batch_latencies), color="blue", linestyle="--",
        label=f"Median: {np.median(batch_latencies):.2f} ms"
    )
    ax2.axhline(
        y=np.percentile(batch_latencies, 95), color="orange", linestyle="--",
        label=f"P95: {np.percentile(batch_latencies, 95):.2f} ms"
    )
    ax2.set_xlabel("Batch Number", fontsize=12)
    ax2.set_ylabel("Latency (ms)", fontsize=12)
    ax2.set_title("Per-Batch Inference Latency", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Kafka Streaming Simulation Results", fontsize=14, y=1.02)
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_system_architecture(save_path):
    """
    Generate a system architecture diagram using matplotlib patches.
    """
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Box definitions: (x, y, width, height, label, color)
    boxes = [
        (0.5, 2.0, 2.0, 2.0, "Data Sources\n(Credit Card\nTransactions)", "#3498db"),
        (3.5, 2.0, 2.0, 2.0, "Apache Kafka\n(Message\nBroker)", "#e67e22"),
        (6.5, 2.0, 2.0, 2.0, "Spark\nStructured\nStreaming", "#e74c3c"),
        (9.5, 2.0, 2.0, 2.0, "Feature\nSelection\n(QIEA)", "#9b59b6"),
        (12.5, 2.0, 2.0, 2.0, "MLlib\nModels\n(LR/RF/GBT)", "#2ecc71"),
    ]

    # Alert box on top right
    alert_box = (13.0, 5.0, 1.5, 0.8, "Alerts", "#e74c3c")

    for x, y, w, h, label, color in boxes:
        fancy = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", alpha=0.85, linewidth=2,
        )
        ax.add_patch(fancy)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=10, fontweight="bold", color="white",
        )

    # Alert box
    x, y, w, h, label, color = alert_box
    fancy = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="white", alpha=0.85, linewidth=2,
    )
    ax.add_patch(fancy)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center", fontsize=10, fontweight="bold", color="white",
    )

    # Arrows between boxes
    arrow_props = dict(
        arrowstyle="-|>", color="#2c3e50", linewidth=2, mutation_scale=20
    )
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        y1 = boxes[i][1] + boxes[i][3] / 2
        x2 = boxes[i + 1][0]
        y2 = boxes[i + 1][1] + boxes[i + 1][3] / 2
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props,
        )

    # Arrow from Models to Alerts
    ax.annotate(
        "", xy=(13.75, 5.0), xytext=(13.5, 4.0),
        arrowprops=dict(arrowstyle="-|>", color="#2c3e50", linewidth=2, mutation_scale=20),
    )

    ax.set_title(
        "System Architecture: Real-Time Fraud Detection Pipeline",
        fontsize=14, fontweight="bold", pad=20,
    )

    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
