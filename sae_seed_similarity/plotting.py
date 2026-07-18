"""Plotting functions designed for scripts and interactive notebooks."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def save_figure(figure: plt.Figure, output_dir: Path, name: str) -> None:
    """Save one publication-ready figure as both PNG and SVG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def heatmap(matrix: pd.DataFrame, title: str, label: str) -> plt.Figure:
    figure, axis = plt.subplots(
        figsize=(max(5, 0.8 * len(matrix)), max(4, 0.7 * len(matrix)))
    )
    sns.heatmap(
        matrix, vmin=0, vmax=1, cmap="viridis", annot=len(matrix) <= 10, ax=axis
    )
    axis.set_title(title)
    axis.set_xlabel(label)
    axis.set_ylabel(label)
    return figure


def scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str,
    hue: str | None = None,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=frame, x=x, y=y, hue=hue, alpha=0.55, ax=axis)
    axis.set_title(title)
    return figure


def distribution(
    frame: pd.DataFrame,
    x: str,
    group: str,
    *,
    title: str,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=frame, x=group, y=x, inner="quartile", cut=0, ax=axis)
    axis.set_title(title)
    return figure
