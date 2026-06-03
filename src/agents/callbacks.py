"""Training callbacks for evaluation, checkpoints, and paper-ready metric logging."""

from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


def _is_locked(gain: float, error_deg: float, success_cfg: dict[str, Any]) -> bool:
    return (
        gain >= float(success_cfg.get("min_gain", 0.85))
        and error_deg <= float(success_cfg.get("max_error_deg", 5.0))
    )


def evaluate_tracking_policy(
    model,
    env,
    n_eval_episodes: int,
    success_cfg: dict[str, Any],
    deterministic: bool = True,
) -> dict[str, float]:
    """Run evaluation episodes and aggregate tracking / success metrics."""
    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    all_gains: list[float] = []
    all_errors: list[float] = []
    locked_fractions: list[float] = []

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_len = 0
        locked_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = env.step(action)
            
            # eval_env is a VecEnv, usually with 1 environment
            done = bool(dones[0])
            info = infos[0]
            reward = rewards[0]
            
            ep_return += float(reward)
            ep_len += 1
            gain = float(info.get("normalized_gain", 0.0))
            err = float(info.get("tracking_error_deg", 90.0))
            all_gains.append(gain)
            all_errors.append(err)
            if info.get("tracking_locked", _is_locked(gain, err, success_cfg)):
                locked_steps += 1

        ep_returns.append(ep_return)
        ep_lengths.append(ep_len)
        locked_fractions.append(locked_steps / max(ep_len, 1))

    return {
        "mean_reward": float(np.mean(ep_returns)),
        "std_reward": float(np.std(ep_returns)),
        "mean_gain": float(np.mean(all_gains)),
        "std_gain": float(np.std(all_gains)),
        "mean_error_deg": float(np.mean(all_errors)),
        "std_error_deg": float(np.std(all_errors)),
        "mean_locked_fraction": float(np.mean(locked_fractions)),
        "success_rate": float(
            np.mean([lf >= float(success_cfg.get("min_locked_fraction", 0.8)) for lf in locked_fractions])
        ),
    }


class TrackingMetricsCallback(BaseCallback):
    """Logs training-time tracking metrics to TensorBoard and CSV."""

    def __init__(
        self,
        log_dir: str,
        log_freq: int = 100,
        csv_flush_freq: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.csv_flush_freq = csv_flush_freq
        self.csv_path = os.path.join(log_dir, "training_tracking.csv")
        self._csv_rows: list[dict[str, Any]] = []
        self._csv_initialized = False

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        infos = self.locals.get("infos", [{}])
        valid = [i for i in infos if i and "normalized_gain" in i]
        if not valid:
            return True

        gain = float(np.mean([float(i["normalized_gain"]) for i in valid]))
        error_deg = float(np.mean([float(i["tracking_error_deg"]) for i in valid]))
        locked = float(np.mean([float(i.get("tracking_locked", False)) for i in valid]))

        self.logger.record("tracking/gain", gain)
        self.logger.record("tracking/error_deg", error_deg)
        self.logger.record("tracking/locked", locked)
        self.logger.record("tracking/n_envs", float(len(valid)))

        rewards = self.locals.get("rewards", [0.0])
        row = {
            "timestep": self.num_timesteps,
            "gain": gain,
            "error_deg": error_deg,
            "locked": locked,
            "reward": float(np.mean(rewards)),
        }
        self._csv_rows.append(row)

        if len(self._csv_rows) >= self.csv_flush_freq:
            self._flush_csv()
        return True

    def _flush_csv(self) -> None:
        if not self._csv_rows:
            return
        write_header = not self._csv_initialized and not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._csv_rows[0].keys()))
            if write_header:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerows(self._csv_rows)
        self._csv_rows.clear()

    def _on_training_end(self) -> None:
        self._flush_csv()


class BeamTrackingEvalCallback(EvalCallback):
    """EvalCallback that also logs gain/error/success and can save best-by-gain models."""

    def __init__(
        self,
        *args,
        success_config: dict[str, Any] | None = None,
        save_best_by_gain: bool = True,
        eval_tracking_csv: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.success_config = success_config or {}
        self.save_best_by_gain = save_best_by_gain
        self.eval_tracking_csv = eval_tracking_csv
        self.best_mean_gain = -1.0
        self.best_success_rate = -1.0
        self._eval_csv_initialized = False

        if self.save_best_by_gain and self.best_model_save_path:
            self.best_gain_save_path = os.path.join(
                os.path.dirname(self.best_model_save_path.rstrip("/\\")),
                "best_by_gain",
            )
            self.best_success_save_path = os.path.join(
                os.path.dirname(self.best_model_save_path.rstrip("/\\")),
                "best_by_success",
            )
            os.makedirs(self.best_gain_save_path, exist_ok=True)
            os.makedirs(self.best_success_save_path, exist_ok=True)
        else:
            self.best_gain_save_path = None
            self.best_success_save_path = None

    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return continue_training

        metrics = evaluate_tracking_policy(
            self.model,
            self.eval_env,
            self.n_eval_episodes,
            self.success_config,
            deterministic=True,
        )

        self.logger.record("eval/mean_gain", metrics["mean_gain"])
        self.logger.record("eval/mean_error_deg", metrics["mean_error_deg"])
        self.logger.record("eval/mean_locked_fraction", metrics["mean_locked_fraction"])
        self.logger.record("eval/success_rate", metrics["success_rate"])

        if self.verbose:
            print(
                f"Eval tracking @ {self.num_timesteps}: "
                f"gain={metrics['mean_gain']:.4f}, "
                f"error={metrics['mean_error_deg']:.2f}°, "
                f"success={metrics['success_rate']:.1%}, "
                f"locked_frac={metrics['mean_locked_fraction']:.1%}"
            )

        if self.eval_tracking_csv:
            self._append_eval_csv(self.num_timesteps, metrics)

        if self.best_gain_save_path and metrics["mean_gain"] > self.best_mean_gain:
            self.best_mean_gain = metrics["mean_gain"]
            path = os.path.join(self.best_gain_save_path, "best_model")
            self.model.save(path)
            if self.verbose:
                print(f"New best mean gain {self.best_mean_gain:.4f} -> {path}.zip")

        if metrics["success_rate"] > self.best_success_rate:
            self.best_success_rate = metrics["success_rate"]
            if self.best_success_save_path:
                path = os.path.join(self.best_success_save_path, "best_model")
                self.model.save(path)
                if self.verbose:
                    print(f"New best success rate {self.best_success_rate:.1%} -> {path}.zip")

        return continue_training

    def _append_eval_csv(self, timestep: int, metrics: dict[str, float]) -> None:
        row = {"timestep": timestep, **metrics}
        write_header = not self._eval_csv_initialized and not os.path.exists(self.eval_tracking_csv)
        with open(self.eval_tracking_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._eval_csv_initialized = True
            writer.writerow(row)


def save_run_manifest(log_dir: str, config: dict[str, Any], device: str) -> None:
    """Writes config snapshot and success-criteria doc for reproducibility."""
    manifest_dir = os.path.join(log_dir, "run_manifest")
    os.makedirs(manifest_dir, exist_ok=True)

    with open(os.path.join(manifest_dir, "config.yaml"), "w") as f:
        import yaml

        yaml.dump(config, f, default_flow_style=False)

    success = config.get("environment", {}).get("success", {})
    doc = {
        "episode_termination": {
            "terminated_on_track_loss": success.get("enable_track_loss_termination", False),
            "truncated_on_time_limit": True,
            "episode_length_steps": config.get("environment", {}).get("episode_length", 500),
        },
        "success_criteria_for_metrics": {
            "tracking_locked_per_step": (
                f"gain >= {success.get('min_gain', 0.85)} AND "
                f"error_deg <= {success.get('max_error_deg', 5.0)}"
            ),
            "episode_success": (
                f"locked_step_fraction >= {success.get('min_locked_fraction', 0.8)} "
                "over full episode"
            ),
            "note": "By default episodes do NOT end early on track loss; they run 5s then truncate.",
        },
        "device": device,
    }
    with open(os.path.join(manifest_dir, "success_criteria.json"), "w") as f:
        json.dump(doc, f, indent=2)
