from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class FactorialConfig:
    n_x: int = 256
    n_y: int = 256
    activation_dim: int = 256
    amplitude_correlation: float = 0.0
    singleton_probability: float = 0.0
    noise_std: float = 0.0
    seed: int = 0
    index_distribution: Literal["uniform", "zipf"] = "uniform"
    zipf_exponent: float = 1.0
    zipf_shift: float = 0.0


class SparseFactorialGenerator:
    """
    Ground-truth features:
        x_1, ..., x_nx, y_1, ..., y_ny

    A normal example contains one random x feature and one random y feature.
    Feature identities can be sampled uniformly or from a truncated Zipf law.
    Optionally, some examples contain only one constituent feature.

    Returns:
        h: observed activation vectors [batch, activation_dim]
        z: known sparse feature coefficients [batch, n_x + n_y]
    """

    def __init__(
        self,
        cfg: FactorialConfig,
        device: str = "cpu",
        true_dictionary: torch.Tensor | None = None,
        normalize_dictionary: bool = True,
    ) -> None:
        if cfg.n_x <= 0 or cfg.n_y <= 0:
            raise ValueError("n_x and n_y must be positive.")
        if cfg.activation_dim <= 0:
            raise ValueError("activation_dim must be positive.")
        if not 0.0 <= cfg.amplitude_correlation <= 1.0:
            raise ValueError("amplitude_correlation must lie in [0, 1].")
        if not 0.0 <= cfg.singleton_probability <= 1.0:
            raise ValueError("singleton_probability must lie in [0, 1].")
        if cfg.noise_std < 0:
            raise ValueError("noise_std must be nonnegative.")
        if cfg.index_distribution not in {"uniform", "zipf"}:
            raise ValueError("index_distribution must be 'uniform' or 'zipf'.")
        if cfg.zipf_exponent < 0:
            raise ValueError("zipf_exponent must be nonnegative.")
        if cfg.zipf_shift < 0:
            raise ValueError("zipf_shift must be nonnegative.")

        self.cfg = cfg
        self.device = torch.device(device)
        self.n_features = cfg.n_x + cfg.n_y

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(cfg.seed)

        if true_dictionary is None:
            # Random unit ground-truth directions.
            dictionary = torch.randn(
                self.n_features,
                cfg.activation_dim,
                generator=self.rng,
                device=self.device,
            )
        else:
            dictionary = torch.as_tensor(
                true_dictionary,
                dtype=torch.float32,
                device=self.device,
            )
            expected_shape = (self.n_features, cfg.activation_dim)
            if tuple(dictionary.shape) != expected_shape:
                raise ValueError(
                    "true_dictionary must have shape "
                    f"{expected_shape}, got {tuple(dictionary.shape)}."
                )
            if not torch.isfinite(dictionary).all():
                raise ValueError("true_dictionary must contain only finite values.")
            if torch.any(dictionary.norm(dim=1) == 0):
                raise ValueError("true_dictionary rows must be nonzero.")

        self.true_dictionary = (
            F.normalize(dictionary, dim=1)
            if normalize_dictionary
            else dictionary.clone()
        )

        self.x_probabilities = self._index_probabilities(cfg.n_x)
        self.y_probabilities = self._index_probabilities(cfg.n_y)

    def _index_probabilities(self, n_features: int) -> torch.Tensor:
        """Return the configured categorical probabilities in feature order."""
        if self.cfg.index_distribution == "uniform":
            return torch.full(
                (n_features,),
                1.0 / n_features,
                dtype=torch.float32,
                device=self.device,
            )

        ranks = torch.arange(
            1,
            n_features + 1,
            dtype=torch.float32,
            device=self.device,
        )
        weights = (ranks + self.cfg.zipf_shift).pow(
            -self.cfg.zipf_exponent
        )
        return weights / weights.sum()

    def _sample_indices(
        self,
        n_features: int,
        probabilities: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if self.cfg.index_distribution == "uniform":
            return torch.randint(
                n_features,
                size=(batch_size,),
                generator=self.rng,
                device=self.device,
            )
        return torch.multinomial(
            probabilities,
            batch_size,
            replacement=True,
            generator=self.rng,
        )

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        cfg = self.cfg

        z = torch.zeros(
            batch_size,
            self.n_features,
            device=self.device,
        )

        x_indices = self._sample_indices(
            cfg.n_x,
            self.x_probabilities,
            batch_size,
        )

        y_indices = self._sample_indices(
            cfg.n_y,
            self.y_probabilities,
            batch_size,
        ) + cfg.n_x

        # Independent positive amplitudes.
        x_amplitudes = torch.rand(
            batch_size,
            generator=self.rng,
            device=self.device,
        )

        y_amplitudes = torch.rand(
            batch_size,
            generator=self.rng,
            device=self.device,
        )

        modes = torch.rand(
            batch_size,
            generator=self.rng,
            device=self.device,
        )

        pair_mask = modes >= cfg.singleton_probability
        x_only_mask = modes < cfg.singleton_probability / 2
        y_only_mask = (
            (modes >= cfg.singleton_probability / 2)
            & (modes < cfg.singleton_probability)
        )

        rows = torch.arange(batch_size, device=self.device)

        use_x = pair_mask | x_only_mask
        use_y = pair_mask | y_only_mask

        # Match the composed-feature setup: only paired examples correlate
        # their amplitudes; singleton amplitudes retain a U[0, 1) marginal.
        y_amplitudes[pair_mask] = (
            cfg.amplitude_correlation * x_amplitudes[pair_mask]
            + (1 - cfg.amplitude_correlation) * y_amplitudes[pair_mask]
        )

        z[rows[use_x], x_indices[use_x]] = x_amplitudes[use_x]
        z[rows[use_y], y_indices[use_y]] = y_amplitudes[use_y]

        h = z @ self.true_dictionary

        if cfg.noise_std > 0:
            h = h + cfg.noise_std * torch.randn(
                h.shape,
                generator=self.rng,
                device=self.device,
            )

        return h, z
