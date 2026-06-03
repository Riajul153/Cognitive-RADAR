"""JEPA Beam Tracking Agent — ties together the world model, cost module, and CEM planner.

This is the top-level agent class that provides a simple ``act(obs) → action``
interface for deployment and evaluation, and manages the full training pipeline:

    Phase A: Data collection (oracle / noisy / random policies)
    Phase B: Offline world model and cost module training
    Phase C: Online planning with continuous fine-tuning
"""

from __future__ import annotations

import os
import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..models.jepa import JEPAWorldModel, JEPAConfig
from ..models.cost_module import CostModule, CostConfig
from ..models.planner import CEMPlanner, PlannerConfig


class TransitionDataset:
    """In-memory dataset of (obs, action, next_obs, received_power) transitions.

    Stores transitions as numpy arrays and provides a PyTorch DataLoader
    for efficient batched training.
    """

    def __init__(self, capacity: int = 2_000_000) -> None:
        self.capacity = capacity
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.next_obs: list[np.ndarray] = []
        self.powers: list[float] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        received_power: float,
    ) -> None:
        """Adds a single transition to the dataset."""
        self.obs.append(obs.copy())
        self.actions.append(action.copy())
        self.next_obs.append(next_obs.copy())
        self.powers.append(float(received_power))
        # Evict oldest if over capacity
        if len(self.obs) > self.capacity:
            self.obs.pop(0)
            self.actions.pop(0)
            self.next_obs.pop(0)
            self.powers.pop(0)

    def add_batch(
        self,
        transitions: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    ) -> None:
        """Adds a batch of transitions."""
        for obs, action, next_obs, power in transitions:
            self.add(obs, action, next_obs, power)

    def __len__(self) -> int:
        return len(self.obs)

    def to_dataloader(
        self, batch_size: int = 512, shuffle: bool = True
    ) -> DataLoader:
        """Creates a PyTorch DataLoader from the stored transitions."""
        obs_t = torch.tensor(np.array(self.obs), dtype=torch.float32)
        act_t = torch.tensor(np.array(self.actions), dtype=torch.float32)
        next_t = torch.tensor(np.array(self.next_obs), dtype=torch.float32)
        pow_t = torch.tensor(np.array(self.powers), dtype=torch.float32)

        dataset = TensorDataset(obs_t, act_t, next_t, pow_t)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=True,
        )


class JEPABeamTrackingAgent:
    """Top-level JEPA agent for adaptive beamforming.

    Attributes:
        world_model: The JEPA world model (encoder + predictor + target encoder).
        cost_module: The cost/energy module.
        planner: The CEM planner.
        dataset: Transition dataset for training.
    """

    def __init__(
        self,
        jepa_config: JEPAConfig,
        cost_config: CostConfig,
        planner_config: PlannerConfig,
        device: str = "auto",
    ) -> None:
        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build modules
        self.world_model = JEPAWorldModel(jepa_config).to(self.device)
        self.cost_module = CostModule(cost_config).to(self.device)
        self.planner = CEMPlanner(
            self.world_model, self.cost_module, planner_config, device=self.device
        )

        # Dataset
        self.dataset = TransitionDataset()

        # Training state
        self.total_train_steps = 0
        self.best_success_rate = -1.0
        self.best_mean_gain = -1.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Selects an action using the CEM planner.

        Args:
            obs: Raw observation from the environment, shape ``(obs_dim,)``.

        Returns:
            Action array, shape ``(action_dim,)``.
        """
        self.world_model.eval()
        self.cost_module.eval()

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        z = self.world_model.encode(obs_t)
        action_t = self.planner.plan(z)

        return action_t.cpu().numpy()

    def act_with_info(self, obs: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Selects an action and returns planning diagnostics."""
        self.world_model.eval()
        self.cost_module.eval()

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        z = self.world_model.encode(obs_t)
        action_t, info = self.planner.plan_with_info(z)

        return action_t.cpu().numpy(), info

    def reset_planner(self) -> None:
        """Resets CEM warm-start state (call at episode boundaries)."""
        self.planner.reset()

    def train_offline(
        self,
        n_epochs: int = 100,
        batch_size: int = 512,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        cost_loss_weight: float = 0.1,
        writer: Any = None,
        log_freq: int = 50,
    ) -> dict[str, list[float]]:
        """Trains the world model and cost module offline on the collected dataset.

        Args:
            n_epochs: Number of training epochs over the dataset.
            batch_size: Mini-batch size.
            lr: Learning rate for AdamW.
            weight_decay: Weight decay for regularization.
            cost_loss_weight: Scaling factor for cost module loss relative to
                JEPA VICReg loss.
            writer: Optional TensorBoard SummaryWriter.
            log_freq: Log to TensorBoard every N gradient steps.

        Returns:
            Dictionary of training history lists.
        """
        self.world_model.train()
        self.cost_module.train()

        # Optimizers — separate for world model and cost module
        wm_optimizer = optim.AdamW(
            self.world_model.get_trainable_params(),
            lr=lr,
            weight_decay=weight_decay,
        )
        cost_optimizer = optim.AdamW(
            self.cost_module.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Cosine annealing LR scheduler
        dataloader = self.dataset.to_dataloader(batch_size=batch_size)
        total_steps = n_epochs * len(dataloader)
        wm_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            wm_optimizer, T_max=total_steps, eta_min=lr * 0.01
        )
        cost_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            cost_optimizer, T_max=total_steps, eta_min=lr * 0.01
        )

        history: dict[str, list[float]] = {
            "jepa_loss": [],
            "cost_loss": [],
            "invariance": [],
            "variance": [],
            "covariance": [],
            "std_min": [],
        }

        step = 0
        for epoch in range(n_epochs):
            epoch_jepa_loss = 0.0
            epoch_cost_loss = 0.0
            n_batches = 0

            for obs_b, act_b, next_b, power_b in dataloader:
                obs_b = obs_b.to(self.device)
                act_b = act_b.to(self.device)
                next_b = next_b.to(self.device)
                power_b = power_b.to(self.device)

                # ── Step 1: Train JEPA world model ──────────────────────
                wm_optimizer.zero_grad()
                z_pred, z_target, jepa_loss, jepa_info = self.world_model.compute_loss(
                    obs_b, act_b, next_b
                )
                jepa_loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    self.world_model.get_trainable_params(), max_norm=1.0
                )
                wm_optimizer.step()
                wm_scheduler.step()

                # EMA update of target encoder
                self.world_model.update_target_encoder()

                # ── Step 2: Train Cost Module ───────────────────────────
                # Encode observations with the (now-updated) encoder, DETACHED
                with torch.no_grad():
                    z_for_cost = self.world_model.encode(obs_b)

                cost_optimizer.zero_grad()
                cost_loss, cost_info = self.cost_module.compute_loss(
                    z_for_cost, power_b
                )
                cost_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.cost_module.parameters(), max_norm=1.0
                )
                cost_optimizer.step()
                cost_scheduler.step()

                # ── Logging ─────────────────────────────────────────────
                epoch_jepa_loss += jepa_info["total"]
                epoch_cost_loss += cost_info["cost_loss"]
                n_batches += 1
                step += 1
                self.total_train_steps += 1

                if writer and step % log_freq == 0:
                    writer.add_scalar("jepa/total_loss", jepa_info["total"], step)
                    writer.add_scalar("jepa/invariance", jepa_info["invariance"], step)
                    writer.add_scalar("jepa/variance", jepa_info["variance"], step)
                    writer.add_scalar("jepa/covariance", jepa_info["covariance"], step)
                    writer.add_scalar("jepa/std_min", jepa_info["std_min"], step)
                    writer.add_scalar("jepa/std_mean", jepa_info["std_mean"], step)
                    writer.add_scalar("cost/loss", cost_info["cost_loss"], step)
                    writer.add_scalar(
                        "cost/predicted_mean", cost_info["predicted_cost_mean"], step
                    )
                    writer.add_scalar("train/lr", wm_scheduler.get_last_lr()[0], step)

            # Epoch summary
            avg_jepa = epoch_jepa_loss / max(n_batches, 1)
            avg_cost = epoch_cost_loss / max(n_batches, 1)
            history["jepa_loss"].append(avg_jepa)
            history["cost_loss"].append(avg_cost)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch + 1:3d}/{n_epochs}: "
                    f"JEPA={avg_jepa:.4f}  Cost={avg_cost:.4f}  "
                    f"inv={jepa_info['invariance']:.4f}  "
                    f"var={jepa_info['variance']:.4f}  "
                    f"cov={jepa_info['covariance']:.4f}  "
                    f"std_min={jepa_info['std_min']:.3f}"
                )

        return history

    def save_checkpoint(self, path: str, metadata: dict | None = None) -> None:
        """Saves a full checkpoint of the agent state.

        Args:
            path: File path for the checkpoint ``.pt`` file.
            metadata: Optional metadata dict to include in the checkpoint.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "world_model_state": self.world_model.state_dict(),
            "cost_module_state": self.cost_module.state_dict(),
            "total_train_steps": self.total_train_steps,
            "best_success_rate": self.best_success_rate,
            "best_mean_gain": self.best_mean_gain,
        }
        if metadata:
            checkpoint["metadata"] = metadata
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> dict:
        """Loads a checkpoint and restores agent state.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The full checkpoint dict (including any metadata).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.world_model.load_state_dict(checkpoint["world_model_state"])
        self.cost_module.load_state_dict(checkpoint["cost_module_state"])
        self.total_train_steps = checkpoint.get("total_train_steps", 0)
        self.best_success_rate = checkpoint.get("best_success_rate", -1.0)
        self.best_mean_gain = checkpoint.get("best_mean_gain", -1.0)
        return checkpoint
