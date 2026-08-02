"""Toy model and correlated-pair data generator"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod, sqrt
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def default_device() -> torch.device:
    """Select CUDA, MPS, or CPU in order of availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class ToyTrainingConfig:
    """Optimization settings for the toy model."""

    batches: int = 10_000
    batch_size: int = 1_000
    lr: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 0


FULL_TOY_TRAINING = ToyTrainingConfig()


@dataclass
class ToyModelConfig:
    """Configuration used by the small correlated-amplitude model."""

    hidden_size: int = 2
    feat_sets: tuple[int, int] = (2, 2)
    active_features_per_draw: int = 1
    set_magnitude_correlation: float = 1.0
    input_size: int = field(init=False)

    def __post_init__(self) -> None:
        if len(self.feat_sets) != 2 or any(size <= 0 for size in self.feat_sets):
            raise ValueError("feat_sets must contain two positive set sizes")
        if self.active_features_per_draw <= 0:
            raise ValueError("active_features_per_draw must be positive")
        if not 0 <= self.set_magnitude_correlation <= 1:
            raise ValueError("set_magnitude_correlation must be between 0 and 1")
        self.input_size = sum(self.feat_sets)


class TMS(nn.Module):
    """Tied-weight toy model of superposition."""

    def __init__(self, cfg: ToyModelConfig):
        super().__init__()
        self.cfg = cfg
        bound = 1 / sqrt(cfg.input_size)
        self.W = nn.Parameter(
            torch.empty(cfg.input_size, cfg.hidden_size).uniform_(-bound, bound)
        )
        self.b = nn.Parameter(torch.zeros(cfg.input_size))
        self.register_buffer("I", torch.ones(cfg.input_size), persistent=False)
        self.h: Tensor | None = None
        self.features: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        self.h = torch.einsum("ih,bi->bh", self.W, x)
        output = torch.einsum("ih,bh->bi", self.W, self.h) + self.b
        return F.relu(output)

    @staticmethod
    def loss(x: Tensor, reconstruction: Tensor, importance: Tensor) -> Tensor:
        return torch.mean(importance.unsqueeze(0) * (x - reconstruction).pow(2))


class ComposedFeatureTMS(TMS):
    """TMS whose samples contain one correlated-amplitude feature pair."""

    def __init__(self, cfg: ToyModelConfig):
        super().__init__(cfg)
        first_size, second_size = cfg.feat_sets
        pair_count = prod(cfg.feat_sets)
        self.register_buffer(
            "pair_probabilities",
            torch.full((pair_count,), 1 / pair_count),
            persistent=False,
        )
        pairs = torch.cartesian_prod(
            torch.arange(first_size), torch.arange(second_size)
        )
        pairs[:, 1] += first_size
        self.register_buffer("pair_indices", pairs, persistent=False)

    def get_prob_table(self) -> Tensor:
        return self.pair_probabilities.reshape(self.cfg.feat_sets)

    @torch.no_grad()
    def get_batch(self, batch_size: int) -> Tensor:
        device = self.W.device
        draws = self.cfg.active_features_per_draw
        amplitudes = torch.rand(batch_size, 2, draws, device=device)
        correlation = self.cfg.set_magnitude_correlation
        amplitudes[:, 1] = (
            (1 - correlation) * amplitudes[:, 1]
            + correlation * amplitudes[:, 0]
        )

        pair_ids = torch.multinomial(
            self.pair_probabilities,
            num_samples=batch_size * draws,
            replacement=True,
        ).reshape(batch_size, draws)
        selected_pairs = self.pair_indices[pair_ids]

        features = torch.zeros(batch_size, self.cfg.input_size, device=device)
        for draw in range(draws):
            for set_index in range(2):
                features.scatter_add_(
                    dim=1,
                    index=selected_pairs[:, draw, set_index].unsqueeze(1),
                    src=amplitudes[:, set_index, draw].unsqueeze(1),
                )
        self.features = features
        return features


def make_correlated_amplitude_model(
    training: ToyTrainingConfig = FULL_TOY_TRAINING,
    *,
    device: torch.device | str | None = None,
) -> ComposedFeatureTMS:
    """Create the four-feature, two-dimensional correlated-amplitude model."""
    torch.manual_seed(training.seed)
    cfg = ToyModelConfig(
        hidden_size=2,
        feat_sets=(2, 2),
        set_magnitude_correlation=1,
        active_features_per_draw=1,
    )
    return ComposedFeatureTMS(cfg).to(device or default_device())


def train_toy_model(
    model: ComposedFeatureTMS,
    training: ToyTrainingConfig = FULL_TOY_TRAINING,
) -> list[float]:
    """Train the toy model and return reconstruction loss at each step."""
    torch.manual_seed(training.seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.lr, weight_decay=training.weight_decay
    )
    losses: list[float] = []
    for _ in range(training.batches):
        optimizer.zero_grad(set_to_none=True)
        features = model.get_batch(training.batch_size)
        reconstruction = model(features)
        loss = model.loss(features, reconstruction, model.I)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def save_toy_model(model: ComposedFeatureTMS, path: Path | str) -> Path:
    """Save the trainable toy-model parameters."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    return destination


def load_toy_model(model: ComposedFeatureTMS, path: Path | str) -> None:
    """Load toy-model parameters into an existing model."""
    state_dict = torch.load(
        path, map_location=next(model.parameters()).device, weights_only=True
    )
    model.load_state_dict(state_dict)
