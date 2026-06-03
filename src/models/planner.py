"""Cross-Entropy Method (CEM) Planner for the JEPA cognitive architecture.

Instead of learning a reactive policy via RL, the agent **plans** by
imagining future trajectories through the learned world model and selecting
the action sequence that minimizes predicted cost over a planning horizon.

Algorithm (per timestep):
    1. Encode current observation → z_0
    2. Initialize action distribution: a ~ N(μ, σ²)
    3. For I iterations:
        a. Sample P action sequences of length H from the distribution
        b. Roll out each sequence through the world model:
           ẑ_{k+1} = predictor(ẑ_k, a_k)  for k = 0..H-1
        c. Evaluate total cost = Σ cost_module(ẑ_k)
        d. Select top-K (elite) sequences with lowest cost
        e. Refit μ, σ to the elite set
    4. Execute a*_0 = μ[0] (first action of optimal sequence)

Warm-starting: The previous solution is shifted left by one timestep to
provide temporal consistency between planning steps.

References
----------
- Rubinstein, R. (1999). "The Cross-Entropy Method for Combinatorial and
  Continuous Optimization." Methodology and Computing in Applied Probability.
- Chua, K. et al. (2018). "Deep Reinforcement Learning in a Handful of
  Trials using Probabilistic Dynamics Models." NeurIPS 2018.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PlannerConfig:
    """Hyperparameters for the CEM planner."""

    horizon: int = 15           # Planning horizon (timesteps)
    population: int = 256       # Number of candidate action sequences
    elite_fraction: float = 0.1 # Fraction of top sequences to keep
    n_iterations: int = 3       # CEM optimization iterations
    action_dim: int = 2         # Dimensionality of actions
    action_low: float = -1.0    # Lower action bound
    action_high: float = 1.0    # Upper action bound
    momentum: float = 0.1       # Momentum for distribution update (smoothing)
    init_std: float = 0.5       # Initial standard deviation for sampling


class CEMPlanner:
    """Cross-Entropy Method planner for model-predictive control.

    The planner uses the JEPA world model to simulate futures and the cost
    module to evaluate them.  It optimizes over action sequences to find
    the one that minimizes cumulative predicted cost.

    Attributes:
        world_model: Trained JEPAWorldModel (encoder + predictor).
        cost_module: Trained CostModule (energy function).
        config: PlannerConfig hyperparameters.
    """

    def __init__(
        self,
        world_model: nn.Module,
        cost_module: nn.Module,
        config: PlannerConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.world_model = world_model
        self.cost_module = cost_module
        self.config = config
        self.device = torch.device(device)

        self._n_elite = max(int(config.population * config.elite_fraction), 1)

        # Previous solution for warm-starting
        self._prev_mean: torch.Tensor | None = None

    def reset(self) -> None:
        """Resets the warm-start state (call at the beginning of each episode)."""
        self._prev_mean = None

    @torch.no_grad()
    def plan(self, z_current: torch.Tensor) -> torch.Tensor:
        """Plans the optimal next action via CEM optimization.

        Args:
            z_current: Current latent state, shape ``(latent_dim,)``.

        Returns:
            Optimal action for the current timestep, shape ``(action_dim,)``.
        """
        cfg = self.config
        H = cfg.horizon
        P = cfg.population
        D = cfg.action_dim

        # ── Initialize distribution ─────────────────────────────────────────
        if self._prev_mean is not None:
            # Warm start: shift previous solution left by 1
            mean = torch.zeros(H, D, device=self.device)
            mean[:-1] = self._prev_mean[1:]
            mean[-1] = 0.0  # Pad last step with zero (no prior info)
        else:
            mean = torch.zeros(H, D, device=self.device)

        std = torch.full((H, D), cfg.init_std, device=self.device)

        # ── CEM Optimization Loop ───────────────────────────────────────────
        for iteration in range(cfg.n_iterations):
            # Sample P action sequences of length H
            # Shape: (P, H, D)
            noise = torch.randn(P, H, D, device=self.device)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            actions = actions.clamp(cfg.action_low, cfg.action_high)

            # Roll out world model for each sequence
            # Start from the same z_current for all candidates
            z = z_current.unsqueeze(0).expand(P, -1)  # (P, latent_dim)
            total_costs = torch.zeros(P, device=self.device)

            for t in range(H):
                # Predict next latent
                z = self.world_model.predict(z, actions[:, t, :])  # (P, latent_dim)
                # Evaluate cost at this future state
                step_cost = self.cost_module(z).squeeze(-1)  # (P,)
                # Discount future costs slightly to prioritize immediate gains
                discount = 0.99 ** t
                total_costs += discount * step_cost

            # Select elite sequences (lowest cost)
            _, elite_indices = total_costs.topk(self._n_elite, largest=False)
            elite_actions = actions[elite_indices]  # (n_elite, H, D)

            # Refit distribution from elites
            new_mean = elite_actions.mean(dim=0)  # (H, D)
            new_std = elite_actions.std(dim=0) + 1e-6  # (H, D)

            # Momentum-based update for stability
            if iteration == 0 and self._prev_mean is None:
                # First ever iteration: use elite statistics directly
                mean = new_mean
                std = new_std
            else:
                mean = (1 - cfg.momentum) * new_mean + cfg.momentum * mean
                std = (1 - cfg.momentum) * new_std + cfg.momentum * std

        # Save solution for warm-starting next timestep
        self._prev_mean = mean.clone()

        # Return the first action of the optimized sequence
        return mean[0].clamp(cfg.action_low, cfg.action_high)

    @torch.no_grad()
    def plan_with_info(
        self, z_current: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Plans the optimal action and returns diagnostic information.

        Args:
            z_current: Current latent state, shape ``(latent_dim,)``.

        Returns:
            Tuple of (optimal_action, info_dict).
        """
        cfg = self.config
        H = cfg.horizon
        P = cfg.population
        D = cfg.action_dim

        if self._prev_mean is not None:
            mean = torch.zeros(H, D, device=self.device)
            mean[:-1] = self._prev_mean[1:]
            mean[-1] = 0.0
        else:
            mean = torch.zeros(H, D, device=self.device)

        std = torch.full((H, D), cfg.init_std, device=self.device)

        best_cost = float("inf")

        for iteration in range(cfg.n_iterations):
            noise = torch.randn(P, H, D, device=self.device)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            actions = actions.clamp(cfg.action_low, cfg.action_high)

            z = z_current.unsqueeze(0).expand(P, -1)
            total_costs = torch.zeros(P, device=self.device)

            for t in range(H):
                z = self.world_model.predict(z, actions[:, t, :])
                step_cost = self.cost_module(z).squeeze(-1)
                discount = 0.99 ** t
                total_costs += discount * step_cost

            _, elite_indices = total_costs.topk(self._n_elite, largest=False)
            elite_actions = actions[elite_indices]

            new_mean = elite_actions.mean(dim=0)
            new_std = elite_actions.std(dim=0) + 1e-6

            if iteration == 0 and self._prev_mean is None:
                mean = new_mean
                std = new_std
            else:
                mean = (1 - cfg.momentum) * new_mean + cfg.momentum * mean
                std = (1 - cfg.momentum) * new_std + cfg.momentum * std

            iter_best = total_costs[elite_indices[0]].item()
            if iter_best < best_cost:
                best_cost = iter_best

        self._prev_mean = mean.clone()

        info = {
            "best_cost": best_cost,
            "mean_std": std.mean().item(),
            "action_0": mean[0, 0].item(),
            "action_1": mean[0, 1].item(),
        }

        return mean[0].clamp(cfg.action_low, cfg.action_high), info
