"""Action-Conditioned Joint Embedding Predictive Architecture (JEPA) World Model.

Implements Yann LeCun's cognitive architecture from "A Path Towards Autonomous
Machine Intelligence" (2022) for learning a predictive world model in latent
space.  The architecture consists of:

    1. **Online Encoder** f_θ:  Maps raw observations to latent states.
    2. **Predictor** g_φ:  Predicts the next latent state given the current
       latent state and an action.
    3. **Target Encoder** f̄_θ (EMA):  Provides stable training targets via
       Exponential Moving Average of the online encoder.  No gradients flow
       through this network.

Training uses VICReg (Variance-Invariance-Covariance Regularization) loss to
prevent representation collapse without requiring negative samples or
contrastive pairs.

References
----------
- LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence."
- Bardes, A., Ponce, J., & LeCun, Y. (2022). "VICReg: Variance-Invariance-
  Covariance Regularization for Self-Supervised Learning." ICLR 2022.
- Assran, M. et al. (2023). "Self-Supervised Learning from Images with a Joint
  Embedding Predictive Architecture." CVPR 2023.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class JEPAConfig:
    """Hyperparameters for the JEPA world model."""

    obs_dim: int = 10
    action_dim: int = 2
    latent_dim: int = 32
    encoder_hidden: list[int] | None = None
    predictor_hidden: list[int] | None = None
    ema_decay: float = 0.996
    vicreg_lambda: float = 25.0  # Invariance weight
    vicreg_mu: float = 25.0      # Variance weight
    vicreg_nu: float = 1.0       # Covariance weight

    def __post_init__(self) -> None:
        if self.encoder_hidden is None:
            self.encoder_hidden = [128, 64]
        if self.predictor_hidden is None:
            self.predictor_hidden = [128, 64]


# ═══════════════════════════════════════════════════════════════════════════════
# Network Components
# ═══════════════════════════════════════════════════════════════════════════════


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: list[int],
    use_layernorm: bool = True,
    final_layernorm: bool = True,
) -> nn.Sequential:
    """Constructs a multi-layer perceptron with ELU activations and LayerNorm.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        hidden_dims: List of hidden layer widths.
        use_layernorm: If True, applies LayerNorm after each hidden layer.
        final_layernorm: If True, applies LayerNorm to the output layer.

    Returns:
        An ``nn.Sequential`` module implementing the MLP.
    """
    layers: list[nn.Module] = []
    current_dim = in_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(h_dim))
        layers.append(nn.ELU(inplace=True))
        current_dim = h_dim
    layers.append(nn.Linear(current_dim, out_dim))
    if final_layernorm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class JEPAEncoder(nn.Module):
    """Perception module: maps raw radar observations to latent states.

    Architecture: obs (10D) → [Linear → LayerNorm → ELU] × N → Linear → LayerNorm → z (32D)

    The final LayerNorm ensures latent representations have bounded norms,
    which stabilizes VICReg training and CEM planning.
    """

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.net = _build_mlp(obs_dim, latent_dim, hidden_dims, final_layernorm=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode an observation into a latent state.

        Args:
            obs: Observation tensor of shape ``(batch, obs_dim)`` or ``(obs_dim,)``.

        Returns:
            Latent state tensor of shape ``(batch, latent_dim)`` or ``(latent_dim,)``.
        """
        return self.net(obs)


class JEPAPredictor(nn.Module):
    """World model core: predicts the next latent state given current state and action.

    Architecture: [z_t || a_t] → [Linear → LayerNorm → ELU] × N → Linear → z_{t+1}

    Note: The predictor does NOT have a final LayerNorm so that it can freely
    match the target encoder's output distribution.  This asymmetry between
    encoder (with final LN) and predictor (without) is standard in JEPA.
    """

    def __init__(
        self, latent_dim: int, action_dim: int, hidden_dims: list[int]
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.net = _build_mlp(
            latent_dim + action_dim, latent_dim, hidden_dims, final_layernorm=False
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next latent state.

        Args:
            z: Current latent state, shape ``(batch, latent_dim)`` or ``(latent_dim,)``.
            action: Action taken, shape ``(batch, action_dim)`` or ``(action_dim,)``.

        Returns:
            Predicted next latent state, same shape as ``z``.
        """
        x = torch.cat([z, action], dim=-1)
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# VICReg Loss
# ═══════════════════════════════════════════════════════════════════════════════


def vicreg_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    lambda_inv: float = 25.0,
    mu_var: float = 25.0,
    nu_cov: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Computes the VICReg (Variance-Invariance-Covariance) loss.

    This loss function prevents representation collapse in self-supervised
    learning without requiring negative samples or contrastive pairs.

    Args:
        z_pred: Predicted next latent from the predictor, shape ``(N, D)``.
            Gradients flow through this tensor.
        z_target: Target next latent from the target encoder, shape ``(N, D)``.
            **Must be detached** (no gradients).
        lambda_inv: Weight for the invariance (MSE) term.
        mu_var: Weight for the variance (anti-collapse) term.
        nu_cov: Weight for the covariance (decorrelation) term.

    Returns:
        Tuple of (total_loss, info_dict) where info_dict contains individual
        component values for logging.
    """
    batch_size, latent_dim = z_pred.shape

    # ── 1. Invariance Loss ──────────────────────────────────────────────────
    # MSE between predicted and target latents.
    # This is the primary learning signal: "predict what the target encoder sees."
    inv_loss = F.mse_loss(z_pred, z_target)

    # ── 2. Variance Loss ────────────────────────────────────────────────────
    # Hinge loss ensuring the standard deviation of each latent dimension
    # stays above γ=1 across the batch.  Prevents collapse to a constant.
    # Only applied to z_pred (target encoder has no gradients).
    std_pred = torch.sqrt(z_pred.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1.0 - std_pred))

    # ── 3. Covariance Loss ──────────────────────────────────────────────────
    # Penalizes off-diagonal elements of the covariance matrix.
    # Forces each latent dimension to encode unique information.
    # Only applied to z_pred (target encoder has no gradients).
    z_centered = z_pred - z_pred.mean(dim=0)
    cov_matrix = (z_centered.T @ z_centered) / max(batch_size - 1, 1)
    # Sum of squared off-diagonal elements, normalized by dimensionality
    off_diag_sq = cov_matrix.pow(2).sum() - cov_matrix.diagonal().pow(2).sum()
    cov_loss = off_diag_sq / latent_dim

    # ── Total ───────────────────────────────────────────────────────────────
    total = lambda_inv * inv_loss + mu_var * var_loss + nu_cov * cov_loss

    info = {
        "invariance": inv_loss.item(),
        "variance": var_loss.item(),
        "covariance": cov_loss.item(),
        "total": total.item(),
        "std_mean": std_pred.mean().item(),
        "std_min": std_pred.min().item(),
    }

    return total, info


# ═══════════════════════════════════════════════════════════════════════════════
# JEPA World Model (Ties Everything Together)
# ═══════════════════════════════════════════════════════════════════════════════


class JEPAWorldModel(nn.Module):
    """Complete JEPA world model with online encoder, predictor, and EMA target encoder.

    This module manages:
    - The online encoder (receives gradients)
    - The predictor (receives gradients)
    - The target encoder (NO gradients; updated via EMA)
    - VICReg loss computation
    - EMA weight updates

    Usage::

        model = JEPAWorldModel(config)
        # Training forward pass:
        z_pred, z_target, loss, info = model.compute_loss(obs, action, next_obs)
        loss.backward()
        optimizer.step()
        model.update_target_encoder()

        # Inference (planning):
        z = model.encode(obs)
        z_next = model.predict(z, action)
    """

    def __init__(self, config: JEPAConfig) -> None:
        super().__init__()
        self.config = config

        # Online encoder (trained via backprop)
        self.encoder = JEPAEncoder(
            config.obs_dim, config.latent_dim, config.encoder_hidden
        )

        # Predictor (trained via backprop)
        self.predictor = JEPAPredictor(
            config.latent_dim, config.action_dim, config.predictor_hidden
        )

        # Target encoder (NO gradients — updated via EMA only)
        self.target_encoder = JEPAEncoder(
            config.obs_dim, config.latent_dim, config.encoder_hidden
        )
        # Initialize target encoder with identical weights
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        # Freeze target encoder — no gradients ever
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.ema_decay = config.ema_decay

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Updates the target encoder weights via Exponential Moving Average.

        θ̄ ← τ · θ̄ + (1 − τ) · θ

        This is the core mechanism that prevents representation collapse.
        The target encoder provides slowly-evolving, stable training targets
        that the online encoder and predictor must learn to predict.
        """
        for online_p, target_p in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target_p.data.mul_(self.ema_decay).add_(
                online_p.data, alpha=1.0 - self.ema_decay
            )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encodes observations using the ONLINE encoder (for planning/inference).

        Args:
            obs: Observation tensor, shape ``(batch, obs_dim)`` or ``(obs_dim,)``.

        Returns:
            Latent state, shape ``(batch, latent_dim)`` or ``(latent_dim,)``.
        """
        return self.encoder(obs)

    def predict(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts the next latent state (for multi-step planning rollouts).

        Args:
            z: Current latent state.
            action: Action to take.

        Returns:
            Predicted next latent state.
        """
        return self.predictor(z, action)

    def compute_loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Computes the VICReg training loss for a batch of transitions.

        This is the main training entry point.  The forward pass is:

        1. z_t     = encoder(obs)              [gradient flows]
        2. ẑ_{t+1} = predictor(z_t, action)    [gradient flows]
        3. z̄_{t+1} = target_encoder(next_obs)  [NO gradient — detached]
        4. loss    = VICReg(ẑ_{t+1}, z̄_{t+1})

        Args:
            obs: Current observations, shape ``(batch, obs_dim)``.
            action: Actions taken, shape ``(batch, action_dim)``.
            next_obs: Next observations, shape ``(batch, obs_dim)``.

        Returns:
            Tuple of (z_pred, z_target, loss, info_dict).
        """
        # Online encoder: produces z_t with gradients
        z_current = self.encoder(obs)

        # Predictor: produces predicted z_{t+1} with gradients
        z_pred = self.predictor(z_current, action)

        # Target encoder: produces ground-truth z_{t+1} WITHOUT gradients
        with torch.no_grad():
            z_target = self.target_encoder(next_obs)

        # VICReg loss
        loss, info = vicreg_loss(
            z_pred,
            z_target,
            lambda_inv=self.config.vicreg_lambda,
            mu_var=self.config.vicreg_mu,
            nu_cov=self.config.vicreg_nu,
        )

        return z_pred, z_target, loss, info

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Returns parameters that should be optimized (encoder + predictor only).

        The target encoder is explicitly excluded — it is updated via EMA.
        """
        return list(self.encoder.parameters()) + list(self.predictor.parameters())
