from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FactorialConfig:
    n_x: int = 256
    n_y: int = 256
    activation_dim: int = 256
    singleton_probability: float = 0.0
    noise_std: float = 0.0
    seed: int = 0


class SparseFactorialGenerator:
    """
    Ground-truth features:
        x_1, ..., x_nx, y_1, ..., y_ny

    A normal example contains one random x feature and one random y feature.
    Optionally, some examples contain only one constituent feature.

    Returns:
        h: observed activation vectors [batch, activation_dim]
        z: known sparse feature coefficients [batch, n_x + n_y]
    """

    def __init__(
        self,
        cfg: FactorialConfig,
        device: str = "cpu",
    ) -> None:
        if not 0.0 <= cfg.singleton_probability <= 1.0:
            raise ValueError("singleton_probability must lie in [0, 1].")

        self.cfg = cfg
        self.device = torch.device(device)
        self.n_features = cfg.n_x + cfg.n_y

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(cfg.seed)

        # Random unit ground-truth directions.
        dictionary = torch.randn(
            self.n_features,
            cfg.activation_dim,
            generator=self.rng,
            device=self.device,
        )

        self.true_dictionary = F.normalize(dictionary, dim=1)

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg

        z = torch.zeros(
            batch_size,
            self.n_features,
            device=self.device,
        )

        x_indices = torch.randint(
            cfg.n_x,
            size=(batch_size,),
            generator=self.rng,
            device=self.device,
        )

        y_indices = torch.randint(
            cfg.n_y,
            size=(batch_size,),
            generator=self.rng,
            device=self.device,
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