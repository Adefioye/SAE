"""Plotting functions designed for scripts and interactive notebooks."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
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


def paper_encoder_decoder_joint(frame: pd.DataFrame, *, threshold: float) -> plt.Figure:
    """Paper Figure 1: independent decoder and encoder matched cosines."""
    values = frame.copy()
    values["Assignment agreement"] = np.where(
        values["same_counterpart"], "Same counterpart", "Different counterparts"
    )
    palette = {
        "Same counterpart": sns.color_palette("colorblind")[0],
        "Different counterparts": sns.color_palette("colorblind")[1],
    }
    grid = sns.JointGrid(
        data=values,
        x="decoder_matched_cosine",
        y="encoder_matched_cosine",
        height=7,
        ratio=5,
        space=0.08,
    )
    for label in ("Different counterparts", "Same counterpart"):
        subset = values[values["Assignment agreement"] == label]
        if subset.empty:
            continue
        grid.ax_joint.scatter(
            subset["decoder_matched_cosine"],
            subset["encoder_matched_cosine"],
            s=8,
            alpha=0.28,
            color=palette[label],
            label=label,
            rasterized=True,
        )
        sns.histplot(
            data=subset,
            x="decoder_matched_cosine",
            bins=50,
            stat="density",
            element="step",
            fill=False,
            color=palette[label],
            ax=grid.ax_marg_x,
        )
        sns.histplot(
            data=subset,
            y="encoder_matched_cosine",
            bins=50,
            stat="density",
            element="step",
            fill=False,
            color=palette[label],
            ax=grid.ax_marg_y,
        )
    grid.ax_joint.axvline(threshold, color="0.35", linestyle="--", linewidth=1)
    grid.ax_joint.axhline(threshold, color="0.35", linestyle="--", linewidth=1)
    grid.ax_joint.set(
        title="Encoder and decoder Hungarian alignment",
        xlabel="Decoder matched cosine",
        ylabel="Encoder matched cosine",
        xlim=(0, 1.01),
        ylim=(0, 1.01),
    )
    grid.ax_joint.legend(loc="lower right")
    return grid.figure


def paper_threshold_sweep(
    frame: pd.DataFrame, *, selected_threshold: float
) -> plt.Figure:
    """Paper Figure A1: overlap definitions across cosine thresholds."""
    figure, axis = plt.subplots(figsize=(7, 5))
    for column, label, style in (
        ("cosine_threshold_fraction", "Both matched cosines", "-"),
        ("shared_fraction", "Agreement + both cosines", "-"),
        ("max_cosine_fraction", "Both maximum cosines", "--"),
    ):
        axis.plot(frame["threshold"], frame[column], label=label, linestyle=style)
    axis.axvline(
        selected_threshold,
        color="0.35",
        linestyle=":",
        linewidth=1.5,
        label=f"Selected threshold = {selected_threshold:g}",
    )
    axis.set(
        title="Shared-latent fraction across thresholds",
        xlabel="Cosine threshold",
        ylabel="Fraction of SAE-A latents",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axis.legend(loc="lower left")
    return figure


def paper_hungarian_vs_max_cosine(frame: pd.DataFrame, *, direction: str) -> plt.Figure:
    """Paper Figure A2 for either decoder or encoder directions."""
    if direction not in {"decoder", "encoder"}:
        raise ValueError("direction must be 'decoder' or 'encoder'")
    values = frame.copy()
    values["Assignment agreement"] = np.where(
        values["same_counterpart"], "Same counterpart", "Different counterparts"
    )
    figure, axis = plt.subplots(figsize=(6.5, 6))
    sns.scatterplot(
        data=values,
        x=f"{direction}_matched_cosine",
        y=f"{direction}_max_cosine",
        hue="Assignment agreement",
        hue_order=["Same counterpart", "Different counterparts"],
        alpha=0.35,
        s=14,
        linewidth=0,
        ax=axis,
        rasterized=True,
    )
    axis.plot([0, 1], [0, 1], color="0.35", linewidth=1)
    axis.set(
        title=f"{direction.capitalize()} Hungarian vs maximum cosine",
        xlabel="Hungarian matched cosine",
        ylabel="Maximum cosine (non-bijective)",
        xlim=(0, 1.01),
        ylim=(0, 1.01),
    )
    return figure


def paper_shared_orphan_distribution(frame: pd.DataFrame) -> plt.Figure:
    """Compare matched-cosine distributions for shared and orphan latents."""
    values = frame.copy()
    values["Latent class"] = np.where(values["is_shared"], "Shared", "Orphan")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for axis, column, label in (
        (axes[0], "decoder_matched_cosine", "Decoder matched cosine"),
        (axes[1], "encoder_matched_cosine", "Encoder matched cosine"),
    ):
        sns.histplot(
            data=values,
            x=column,
            hue="Latent class",
            hue_order=["Shared", "Orphan"],
            bins=50,
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            ax=axis,
        )
        axis.set(xlabel=label, xlim=(0, 1.01))
    figure.suptitle("Matched-cosine distributions by paper latent class")
    return figure


def paper_shared_orphan_fractions(summary: pd.DataFrame) -> plt.Figure:
    """Shared/orphan part-to-whole comparison for each SAE pair."""
    values = summary.melt(
        id_vars=["sae_a", "sae_b"],
        value_vars=["shared_fraction", "orphan_fraction"],
        var_name="latent_class",
        value_name="fraction",
    )
    values["SAE pair"] = values["sae_a"] + " / " + values["sae_b"]
    values["Latent class"] = values["latent_class"].map(
        {"shared_fraction": "Shared", "orphan_fraction": "Orphan"}
    )
    pivot = values.pivot(index="SAE pair", columns="Latent class", values="fraction")
    pivot = pivot.reindex(columns=["Shared", "Orphan"])
    figure, axis = plt.subplots(figsize=(max(6, 1.2 * len(pivot)), 4.8))
    pivot.plot(kind="bar", stacked=True, ylim=(0, 1), ax=axis)
    axis.set(
        title="Shared and orphan latent fractions",
        xlabel="SAE pair",
        ylabel="Fraction of latents",
    )
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Latent class", loc="upper right")
    return figure


def paper_similarity_vs_frequency(frame: pd.DataFrame) -> plt.Figure:
    """Two-seed adaptation of the paper's similarity/firing-frequency figure."""
    values = frame.copy()
    values["Latent class"] = np.where(values["is_shared"], "Shared", "Orphan")
    figure, axis = plt.subplots(figsize=(7, 5.5))
    sns.scatterplot(
        data=values,
        x="average_matched_cosine",
        y="activation_frequency_a",
        hue="Latent class",
        hue_order=["Shared", "Orphan"],
        alpha=0.35,
        s=14,
        linewidth=0,
        ax=axis,
        rasterized=True,
    )
    positive = values["activation_frequency_a"] > 0
    if positive.any():
        axis.set_yscale("log")
    axis.set(
        title="Matched similarity vs SAE-A firing frequency",
        xlabel="Mean of encoder and decoder matched cosine",
        ylabel="Activation frequency in SAE A",
        xlim=(0, 1.01),
    )
    return figure
