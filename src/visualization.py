"""
Chart-building helpers used across the Streamlit dashboard.
All charts are built from real data passed in -- nothing is fabricated.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FAFAFA", "figure.facecolor": "#FFFFFF"})

PALETTE = ["#4C6FFF", "#22C1A4", "#F2A93B", "#EF6461", "#8A6FD6", "#3AB0FF"]


def class_distribution_chart(df: pd.DataFrame, target_col: str):
    counts = df[target_col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PALETTE, legend=False, ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig


def text_length_distribution_chart(df: pd.DataFrame, text_col: str):
    lengths = df[text_col].astype(str).str.split().apply(len)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(lengths, bins=25, color=PALETTE[0], ax=ax, kde=True)
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Text Length Distribution (words per entry)")
    fig.tight_layout()
    return fig


def word_frequency_chart(clean_texts: pd.Series, top_n: int = 20):
    from collections import Counter
    all_words = " ".join(clean_texts.astype(str)).split()
    counts = Counter(all_words).most_common(top_n)
    words, freqs = zip(*counts) if counts else ([], [])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(x=list(freqs), y=list(words), hue=list(words), palette=PALETTE, legend=False, ax=ax)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    ax.set_title(f"Top {top_n} Most Frequent Words")
    fig.tight_layout()
    return fig


def confusion_matrix_chart(cm, classes):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax, cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


def model_comparison_chart(results: dict, metric: str = "f1_macro"):
    names = list(results.keys())
    values = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = sns.barplot(x=names, y=values, hue=names, palette=PALETTE, legend=False, ax=ax)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=15, ha="right")
    for bar, v in zip(bars.patches, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def metric_group_comparison_chart(results: dict):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    names = list(results.keys())
    data = {m: [results[n][m] for n in names] for m in metrics}
    df = pd.DataFrame(data, index=names)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df.plot(kind="bar", ax=ax, color=PALETTE[:len(metrics)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison")
    plt.xticks(rotation=15, ha="right")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def class_probability_chart(class_probabilities: dict):
    names = list(class_probabilities.keys())
    values = list(class_probabilities.values())
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = sns.barplot(x=values, y=names, hue=names, palette=PALETTE, legend=False, ax=ax)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability")
    ax.set_title("Class Probability Breakdown")
    for bar, v in zip(bars.patches, values):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig
