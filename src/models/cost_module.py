"""Cost Module (Energy Function) for the JEPA cognitive architecture.

The Cost Module evaluates the "energy" or "desirability" of a latent state
produced by the JEPA encoder.  Unlike RL reward functions, the cost module
is trained with **supervised learning** to predict the instantaneous quality
of a state.

In our radar beamforming application:
    - cost = 1 − received_power
    - Low received power (target off-beam) → high cost
    - High received power (target on-beam) → low cost

The CEM planner minimizes the predicted cost over a planning horizon to
select optimal beam steering actions.

References
----------
LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence." §4.4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CostConfig:
    """Hyperparameters for the cost module."""

    latent_dim: int = 32
    hidden_dims: list[int] | None = None

    def __post_init__(self) -> None:
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


class CostModule(nn.Module):
    """Learned energy function mapping latent states to scalar costs.

    Architecture: z (32D) → [Linear → ELU] × N → Linear → Sigmoid → cost ∈ [0, 1]

    The sigmoid output ensures costs are bounded in [0, 1], which stabilises
    the CEM planner and makes the cost landscape well-behaved for optimization.

    Training target: ``cost = 1 − received_power``, supervised with MSE loss.
    """

    def __init__(self, config: CostConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.latent_dim
        for h_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ELU(inplace=True))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        # Sigmoid ensures cost ∈ [0, 1] → well-behaved for CEM optimization
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predicts the cost (energy) of a latent state.

        Args:
            z: Latent state tensor, shape ``(batch, latent_dim)`` or ``(latent_dim,)``.

        Returns:
            Scalar cost ∈ [0, 1], shape ``(batch, 1)`` or ``(1,)``.
            0 = perfect beam lock, 1 = complete signal loss.
        """
        return self.net(z)

    def compute_loss(
        self,
        z: torch.Tensor,
        received_power: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes supervised MSE loss for training the cost module.

        Args:
            z: Latent states, shape ``(batch, latent_dim)``.  **Must be detached**
                from the JEPA encoder to prevent the cost loss from corrupting
                the world model representations.
            received_power: Ground-truth received power ∈ [0, 1], shape ``(batch,)``
                or ``(batch, 1)``.

        Returns:
            Tuple of (loss, info_dict).
        """
        predicted_cost = self.net(z.detach())  # Double-detach for safety
        target_cost = (1.0 - received_power).view(-1, 1)
        loss = F.mse_loss(predicted_cost, target_cost)

        info = {
            "cost_loss": loss.item(),
            "predicted_cost_mean": predicted_cost.mean().item(),
            "target_cost_mean": target_cost.mean().item(),
        }
        return loss, info
