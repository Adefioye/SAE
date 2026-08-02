"""SAELens training utilities for toy-model hidden activations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from toy_model import ComposedFeatureTMS, default_device


@dataclass(frozen=True)
class SAETrainingConfig:
    """Configuration for a standard L1-regularized SAE training run."""

    d_sae: int = 4
    training_samples: int = 128_000_000
    batch_size: int = 1_024
    lr: float = 3e-4
    l1_coefficient: float = 0.3
    seed: int = 0
    lr_warm_up_steps: int = 0
    lr_decay_steps: int = 0
    adam_beta1: float = 0.0
    adam_beta2: float = 0.999
    apply_b_dec_to_input: bool = False


FULL_SAE_TRAINING = SAETrainingConfig()
GRID_LEARNING_RATES = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
GRID_L1_COEFFICIENTS = (1e-2, 3e-2, 1e-1, 3e-1, 1.0)
DEFAULT_SAE_SEEDS = (0, 1)


class TMSHiddenActivationIterator(Iterator[Tensor]):
    """Yield batches of hidden activations from a trained toy model."""

    def __init__(self, model: ComposedFeatureTMS, batch_size: int):
        self.model = model
        self.batch_size = batch_size

    def __iter__(self) -> TMSHiddenActivationIterator:
        return self

    @torch.no_grad()
    def __next__(self) -> Tensor:
        features = self.model.get_batch(self.batch_size)
        self.model(features)
        if self.model.h is None:
            raise RuntimeError("The toy model did not produce hidden activations")
        return self.model.h.detach()


def make_sae(
    config: SAETrainingConfig = FULL_SAE_TRAINING,
    *,
    d_in: int = 2,
    device: torch.device | str | None = None,
):
    """Create a standard SAELens SAE."""
    from sae_lens import StandardTrainingSAE, StandardTrainingSAEConfig

    chosen_device = str(device or default_device())
    torch.manual_seed(config.seed)
    sae_config = StandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=config.d_sae,
        device=chosen_device,
        dtype="float32",
        apply_b_dec_to_input=config.apply_b_dec_to_input,
        normalize_activations="none",
        l1_coefficient=config.l1_coefficient,
        lp_norm=1.0,
        l1_warm_up_steps=config.lr_warm_up_steps,
        decoder_init_norm=1.0,
    )
    return StandardTrainingSAE(sae_config).to(chosen_device)


def train_sae(
    model: ComposedFeatureTMS,
    config: SAETrainingConfig = FULL_SAE_TRAINING,
    *,
    device: torch.device | str | None = None,
):
    """Train one SAE on batches of toy-model hidden activations."""
    from sae_lens.config import LoggingConfig, SAETrainerConfig
    from sae_lens.training.sae_trainer import SAETrainer

    chosen_device = str(device or default_device())
    sparse_autoencoder = make_sae(
        config, d_in=model.cfg.hidden_size, device=chosen_device
    )
    trainer_config = SAETrainerConfig(
        total_training_samples=config.training_samples,
        train_batch_size_samples=config.batch_size,
        lr=config.lr,
        lr_end=config.lr,
        lr_scheduler_name="constant",
        lr_warm_up_steps=config.lr_warm_up_steps,
        lr_decay_steps=config.lr_decay_steps,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        device=chosen_device,
        n_checkpoints=0,
        checkpoint_path=None,
        save_final_checkpoint=False,
        logger=LoggingConfig(
            log_to_wandb=False,
            eval_every_n_wandb_logs=2**31 - 1,
        ),
        n_batches_for_norm_estimate=1,
    )
    activations = TMSHiddenActivationIterator(model, config.batch_size)
    trainer = SAETrainer(
        cfg=trainer_config,
        sae=sparse_autoencoder,
        data_provider=activations,
    )
    return trainer.fit()


def train_sae_seeds(
    model: ComposedFeatureTMS,
    config: SAETrainingConfig,
    seeds: Sequence[int] = DEFAULT_SAE_SEEDS,
    *,
    device: torch.device | str | None = None,
) -> list:
    """Train one independent SAE for each seed."""
    return [
        train_sae(model, replace(config, seed=seed), device=device) for seed in seeds
    ]


def train_sae_grid(
    model: ComposedFeatureTMS,
    base_config: SAETrainingConfig = FULL_SAE_TRAINING,
    *,
    learning_rates: Sequence[float] = GRID_LEARNING_RATES,
    l1_coefficients: Sequence[float] = GRID_L1_COEFFICIENTS,
    seeds: Sequence[int] = DEFAULT_SAE_SEEDS,
    device: torch.device | str | None = None,
) -> dict[tuple[float, float], list]:
    """Train SAEs across learning-rate and L1-coefficient combinations."""
    results: dict[tuple[float, float], list] = {}
    for learning_rate in learning_rates:
        for l1_coefficient in l1_coefficients:
            config = replace(
                base_config,
                lr=learning_rate,
                l1_coefficient=l1_coefficient,
            )
            results[(learning_rate, l1_coefficient)] = train_sae_seeds(
                model, config, seeds=seeds, device=device
            )
    return results


def save_sae(sparse_autoencoder, path: Path | str) -> Path:
    """Save a SAELens training model."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    sparse_autoencoder.save_model(destination)
    return destination


def load_sae(path: Path | str, *, device: torch.device | str | None = None):
    """Load a standard SAELens training model."""
    from sae_lens import StandardTrainingSAE

    return StandardTrainingSAE.load_from_disk(
        path, device=str(device or default_device()), dtype="float32"
    )
