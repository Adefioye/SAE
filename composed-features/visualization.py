"""Analysis and Matplotlib plots for correlated-amplitude Experiment 1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import torch
from torch import Tensor

# Importing experiment_1 also makes the upstream source directory importable.
from experiment_1 import toy_source as _toy_source  # noqa: F401
from plot_fns import w_cossim


@torch.no_grad()
def evaluate_sae(model, sae, *, samples: int = 10_000) -> dict[str, float]:
    """Evaluate reconstruction and sparsity on fresh toy-model activations."""
    features = model.get_batch(samples)
    model(features)
    hidden = model.h
    feature_acts = sae.encode(hidden)
    reconstructed = sae.decode(feature_acts)
    squared_error = (reconstructed - hidden).pow(2).sum(dim=-1)
    denom = hidden.pow(2).sum(dim=-1).clamp_min(1e-12)
    return {
        "mse": float(squared_error.mean()),
        "normalized_mse": float((squared_error / denom).mean()),
        "mean_l0": float((feature_acts > 0).sum(dim=-1).float().mean()),
        "mean_l1": float(feature_acts.abs().sum(dim=-1).mean()),
    }


@torch.no_grad()
def data_feature_responses(model, sae) -> Tensor:
    """SAE responses to each individual ground-truth input feature."""
    return sae.encode(model.W).detach().cpu()


@torch.no_grad()
def decoder_cosine_similarity(model, sae) -> Tensor:
    """Cosine similarity between TMS feature vectors and SAE decoder vectors."""
    # Reuse the original plot helper, preserving SAE feature order explicitly.
    order = torch.arange(sae.W_dec.shape[0], device=sae.W_dec.device)
    similarities, _ = w_cossim(model.W, sae.W_dec, sort=order)
    return similarities


def _rotation_from_first_feature(model) -> Tensor:
    theta = torch.atan2(model.W[0, 1], model.W[0, 0])
    return torch.stack(
        (torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta))
    ).reshape(2, 2)


def plot_training_loss(losses: Sequence[float]):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(losses, color="black")
    ax.set(xlabel="TMS training step", ylabel="reconstruction loss", yscale="log")
    fig.tight_layout()
    return fig, ax


def plot_feature_batch(features: Tensor, *, samples: int = 200):
    """Show that each row has one active x, one active y, and equal amplitudes."""
    shown = features[:samples].detach().cpu()
    fig, ax = plt.subplots(figsize=(7, 3))
    image = ax.imshow(shown.T, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set(
        xlabel="sample",
        ylabel="ground-truth feature",
        yticks=range(4),
        yticklabels=("x1", "x2", "y1", "y2"),
    )
    fig.colorbar(image, ax=ax, label="amplitude")
    fig.tight_layout()
    return fig, ax


def plot_sae_matrices(model, sae):
    similarities = decoder_cosine_similarity(model, sae)
    responses = data_feature_responses(model, sae)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
    im0 = axes[0].imshow(similarities, cmap="RdYlBu_r", vmin=-1, vmax=1)
    axes[0].set(title="Decoder cosine similarity", xlabel="SAE feature", ylabel="data feature")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)
    im1 = axes[1].imshow(responses, cmap="viridis", vmin=0)
    axes[1].set(title="SAE activation", xlabel="SAE feature", ylabel="data feature")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    for ax in axes:
        ax.set_yticks(range(4), labels=("x1", "x2", "y1", "y2"))
    return fig, axes


def plot_geometry(model, sae, *, title: str | None = None):
    """Overlay the 2-D TMS feature vectors and SAELens decoder directions."""
    rotation = _rotation_from_first_feature(model).to(model.W.device)
    data_vectors = (model.W @ rotation).detach().cpu()
    decoder_vectors = (sae.W_dec @ rotation).detach().cpu()
    colors = ("#1b9e77", "#1b9e77", "#d95f02", "#d95f02")

    fig, ax = plt.subplots(figsize=(4, 4))
    for index, vector in enumerate(data_vectors):
        ax.annotate(
            "",
            xy=vector,
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "lw": 3, "color": colors[index]},
        )
        ax.text(vector[0], vector[1], ("x1", "x2", "y1", "y2")[index])
    for index, vector in enumerate(decoder_vectors):
        ax.annotate(
            "",
            xy=vector,
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "black"},
        )
        ax.text(vector[0], vector[1], f"f{index + 1}", fontsize=8)

    extent = 1.15 * max(
        float(data_vectors.abs().max()), float(decoder_vectors.abs().max()), 1e-3
    )
    ax.axhline(0, color="0.7", lw=0.7)
    ax.axvline(0, color="0.7", lw=0.7)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal", title=title)
    fig.tight_layout()
    return fig, ax


def select_best_sae(model, saes: Sequence, *, samples: int = 10_000):
    """Select the seed with the lowest normalized reconstruction error."""
    return min(saes, key=lambda sae: evaluate_sae(model, sae, samples=samples)["normalized_mse"])


def plot_sae_grid(
    model,
    results: Mapping[tuple[float, float], Sequence],
    learning_rates: Sequence[float],
    l1_coefficients: Sequence[float],
):
    """Visualize the best seed at each point in the source notebook's grid."""
    fig, axes = plt.subplots(
        len(learning_rates),
        len(l1_coefficients),
        figsize=(2 * len(l1_coefficients), 2 * len(learning_rates)),
        squeeze=False,
    )
    rotation = _rotation_from_first_feature(model).to(model.W.device)
    data_vectors = (model.W @ rotation).detach().cpu()
    for row, lr in enumerate(learning_rates):
        for column, l1 in enumerate(l1_coefficients):
            ax = axes[row, column]
            sae = select_best_sae(model, results[(lr, l1)], samples=2_000)
            decoder_vectors = (sae.W_dec @ rotation).detach().cpu()
            for vector in data_vectors:
                ax.plot((0, vector[0]), (0, vector[1]), color="#377eb8", lw=2)
            for vector in decoder_vectors:
                ax.plot((0, vector[0]), (0, vector[1]), color="black", lw=1.2)
            extent = 1.1 * max(
                float(data_vectors.abs().max()), float(decoder_vectors.abs().max()), 1e-3
            )
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
            ax.set_xticks(())
            ax.set_yticks(())
            if row == 0:
                ax.set_title(f"L1={l1:g}", fontsize=9)
            if column == 0:
                ax.set_ylabel(f"lr={lr:g}", fontsize=9)
    fig.tight_layout()
    return fig, axes
