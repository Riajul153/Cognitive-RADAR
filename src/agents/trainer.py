"""DRL agent training pipeline using Stable-Baselines3 with real-time dashboard streaming."""

from __future__ import annotations

import os
import yaml
import gymnasium as gym
from typing import Any
import numpy as np
import torch

from stable_baselines3 import SAC, TD3, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class BeamTrackingMonitor(gym.Wrapper):
    """Monitor wrapper that forwards dashboard snapshots to the parent process."""

    def get_dashboard_state(self) -> dict[str, Any]:
        return self.env.unwrapped.get_dashboard_state()


from ..environment.beam_tracking_env import BeamTrackingEnv
from ..utils.dashboard_client import DashboardClient
from .callbacks import (
    BeamTrackingEvalCallback,
    TrackingMetricsCallback,
    save_run_manifest,
)


def resolve_training_device(device_cfg: str = "auto") -> str:
    """Resolve SB3 device string: auto | cuda | cpu."""
    choice = (device_cfg or "auto").lower().strip()
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot see a GPU. "
                f"Installed torch: {torch.__version__}. "
                "Install a CUDA build, e.g.:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
        return "cuda"
    if choice == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device '{device_cfg}'. Use auto, cuda, or cpu.")


class DashboardStreamingCallback(BaseCallback):
    """Stable-Baselines3 callback to stream live training variables to the dashboard."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        n_rows: int = 8,
        n_cols: int = 8,
        stream_freq: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.stream_freq = stream_freq
        self.client = DashboardClient(host, port, n_rows=n_rows, n_cols=n_cols)

    def _on_training_start(self) -> None:
        self.client.start()

    def _on_step(self) -> bool:
        if self.n_calls % self.stream_freq == 0:
            try:
                rewards = self.locals.get("rewards", [0.0])
                step_reward = float(np.mean(rewards))

                # SubprocVecEnv: fetch state from worker 0 via env_method
                snap = self.training_env.env_method("get_dashboard_state", indices=[0])[0]
                n_envs = self.training_env.num_envs
                ep_len = max(snap["episode_length"], 1)

                self.client.send_state(
                    step=snap["step"],
                    episode=int(self.num_timesteps // (ep_len * n_envs) + 1),
                    algorithm=self.model.__class__.__name__,
                    target_pos=np.array(snap["target_pos"]),
                    target_angles=tuple(snap["target_angles"]),
                    beam_angles=tuple(snap["beam_angles"]),
                    gain=snap["gain"],
                    error_deg=snap["error_deg"],
                    reward=step_reward,
                    phases=np.array(snap["phases"]),
                    beam_pattern=snap["beam_pattern"],
                )
            except Exception as e:
                # Ensure callback errors never crash the training run
                if self.verbose:
                    print(f"Dashboard stream error: {e}")
        return True

    def _on_training_end(self) -> None:
        self.client.stop()


class BeamTrackingTrainer:
    """Manages the training of deep RL agents for antenna beam tracking."""

    def __init__(self, config_or_path: dict[str, Any] | str):
        """Initializes the trainer.

        Args:
            config_or_path: Dictionary or file path to YAML configuration.
        """
        if isinstance(config_or_path, str):
            with open(config_or_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_or_path

        # Setup training parameters
        self.train_params = self.config["training"]
        self.ant_params = self.config["antenna"]
        self.tgt_params = self.config["target"]
        self.env_params = self.config["environment"]
        self.dash_params = self.config.get("dashboard", {})

        # Ensure output directories exist
        self.model_dir = self.train_params.get("model_dir", "models/")
        self.log_dir = self.train_params.get("log_dir", "logs/")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "best_model"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "best_by_gain"), exist_ok=True)

        self.success_params = self.env_params.get("success", {})
        self.env = None
        self.model = None

    def _env_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for BeamTrackingEnv (per-worker seed set by VecEnv)."""
        return {
            "n_rows": self.ant_params["n_rows"],
            "n_cols": self.ant_params["n_cols"],
            "frequency_hz": self.ant_params["frequency_hz"],
            "element_spacing": self.ant_params["element_spacing"],
            "target_config": self.tgt_params,
            "reward_config": self.env_params["reward"],
            "success_config": self.success_params,
            "dt": self.env_params["dt"],
            "episode_length": self.env_params["episode_length"],
            "action_mode": self.env_params.get("action_mode", "parametric"),
            "parametric_type": self.env_params.get("parametric_type", "incremental"),
            "max_angular_step_deg": float(self.env_params.get("max_angular_step_deg", 2.0)),
            "elevation_max_deg": float(self.env_params.get("elevation_max_deg", 60.0)),
            "azimuth_max_deg": float(self.env_params.get("azimuth_max_deg", 60.0)),
            "rng_seed": None,
        }

    def create_train_vec_env(self) -> gym.Env:
        """Creates vectorized training environments for parallel rollouts."""
        n_envs = int(self.train_params.get("n_envs", 10))
        vec_type = str(self.train_params.get("vec_env_type", "subproc")).lower()
        vec_cls = SubprocVecEnv if vec_type == "subproc" else DummyVecEnv

        return make_vec_env(
            BeamTrackingEnv,
            n_envs=n_envs,
            seed=int(self.train_params.get("env_seed", 42)),
            env_kwargs=self._env_kwargs(),
            monitor_dir=os.path.join(self.log_dir, "train_monitor"),
            vec_env_cls=vec_cls,
            wrapper_class=BeamTrackingMonitor,
        )

    def create_eval_env(self) -> gym.Env:
        """Single evaluation environment (deterministic metrics)."""
        kwargs = self._env_kwargs()
        kwargs["rng_seed"] = 99
        raw_env = BeamTrackingEnv(**kwargs)
        monitor_path = os.path.join(self.log_dir, "eval_monitor.csv")
        monitored_env = Monitor(raw_env, filename=monitor_path)
        return BeamTrackingMonitor(monitored_env)



    @staticmethod
    def _callback_freq(env_steps: int, n_envs: int) -> int:
        """Convert desired environment-step interval to VecEnv callback frequency."""
        return max(int(env_steps) // max(n_envs, 1), 1)

    def train(self) -> None:
        """Trains the SAC or TD3 agent based on configuration."""
        n_envs = int(self.train_params.get("n_envs", 10))
        self.env = self.create_train_vec_env()
        eval_env = self.create_eval_env()

        # Policy network kwargs
        policy_kwargs = {
            "net_arch": self.train_params.get("net_arch", [256, 256, 128])
        }

        algo = self.train_params.get("algorithm", "SAC").upper()
        device = resolve_training_device(self.train_params.get("device", "auto"))
        save_run_manifest(self.log_dir, self.config, device)
        if device == "cuda":
            print(f"Training on GPU: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
        else:
            print(f"Training on CPU (torch {torch.__version__})")
            if "+cpu" in torch.__version__:
                print(
                    "Tip: install CUDA PyTorch to use your GPU:\n"
                    "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
                )

        # Instantiate RL agent
        resume_path = self.train_params.get("resume_path", None)
        reset_num_timesteps = True
        
        if resume_path:
            print(f"Resuming training from checkpoint: {resume_path}")
            reset_num_timesteps = False
            if algo == "SAC":
                self.model = SAC.load(resume_path, env=self.env, device=device, tensorboard_log=self.log_dir)
            elif algo == "TD3":
                self.model = TD3.load(resume_path, env=self.env, device=device, tensorboard_log=self.log_dir)
            elif algo == "DQN":
                self.model = DQN.load(resume_path, env=self.env, device=device, tensorboard_log=self.log_dir)
            else:
                raise ValueError(f"Unsupported algorithm: {algo} for resuming.")
        elif algo == "SAC":
            self.model = SAC(
                "MlpPolicy",
                self.env,
                device=device,
                learning_rate=self.train_params.get("learning_rate", 3e-4),
                buffer_size=self.train_params.get("buffer_size", 1000000),
                batch_size=self.train_params.get("batch_size", 256),
                ent_coef=self.train_params.get("ent_coef", "auto"),
                gamma=self.train_params.get("gamma", 0.99),
                tau=self.train_params.get("tau", 0.005),
                train_freq=self.train_params.get("train_freq", 1),
                gradient_steps=self.train_params.get("gradient_steps", 1),
                policy_kwargs=policy_kwargs,
                tensorboard_log=self.log_dir,
                verbose=1,
            )
        elif algo == "TD3":
            # TD3 needs action noise for exploration
            n_actions = self.env.action_space.shape[-1]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.1 * np.ones(n_actions)
            )
            self.model = TD3(
                "MlpPolicy",
                self.env,
                device=device,
                learning_rate=self.train_params.get("learning_rate", 3e-4),
                buffer_size=self.train_params.get("buffer_size", 1000000),
                batch_size=self.train_params.get("batch_size", 256),
                action_noise=action_noise,
                gamma=self.train_params.get("gamma", 0.99),
                tau=self.train_params.get("tau", 0.005),
                train_freq=self.train_params.get("train_freq", (1, "episode")),
                gradient_steps=self.train_params.get("gradient_steps", -1),
                policy_kwargs=policy_kwargs,
                tensorboard_log=self.log_dir,
                verbose=1,
            )
        elif algo == "DQN":
            self.model = DQN(
                "MlpPolicy",
                self.env,
                device=device,
                learning_rate=self.train_params.get("learning_rate", 1e-4),
                buffer_size=self.train_params.get("buffer_size", 500000),
                batch_size=self.train_params.get("batch_size", 256),
                exploration_fraction=self.train_params.get("exploration_fraction", 0.1),
                exploration_initial_eps=self.train_params.get("exploration_initial_eps", 1.0),
                exploration_final_eps=self.train_params.get("exploration_final_eps", 0.05),
                train_freq=self.train_params.get("train_freq", 1),
                gradient_steps=self.train_params.get("gradient_steps", 1),
                target_update_interval=self.train_params.get("target_update_interval", 1000),
                policy_kwargs=policy_kwargs,
                tensorboard_log=self.log_dir,
                verbose=1,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algo}. Must be SAC, TD3, or DQN.")

        eval_freq_env = int(self.train_params.get("eval_freq", 50000))
        checkpoint_freq_env = int(self.train_params.get("checkpoint_freq", 100000))
        log_freq_env = int(self.train_params.get("log_freq", 100))

        eval_freq = self._callback_freq(eval_freq_env, n_envs)
        checkpoint_freq = self._callback_freq(checkpoint_freq_env, n_envs)
        log_freq = self._callback_freq(log_freq_env, n_envs)

        eval_success_cfg = {**self.success_params, "eval_seed_offset": 1000}

        eval_callback = BeamTrackingEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.model_dir, "best_model"),
            log_path=os.path.join(self.log_dir, "evaluations"),
            eval_freq=eval_freq,
            n_eval_episodes=self.train_params.get("n_eval_episodes", 20),
            deterministic=True,
            render=False,
            verbose=1,
            success_config=eval_success_cfg,
            save_best_by_gain=self.train_params.get("save_best_by_gain", True),
            eval_tracking_csv=os.path.join(self.log_dir, "eval_tracking.csv"),
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=os.path.join(self.log_dir, "checkpoints"),
            name_prefix=f"{algo}_beam_tracking",
        )

        metrics_callback = TrackingMetricsCallback(
            log_dir=self.log_dir,
            log_freq=log_freq,
            csv_flush_freq=self.train_params.get("csv_flush_freq", 2000),
        )

        dashboard_callback = DashboardStreamingCallback(
            host=self.dash_params.get("websocket_host", "localhost"),
            port=self.dash_params.get("websocket_port", 8765),
            n_rows=self.ant_params["n_rows"],
            n_cols=self.ant_params["n_cols"],
            stream_freq=self.dash_params.get("stream_freq", 5),
        )

        callbacks = [
            eval_callback,
            checkpoint_callback,
            metrics_callback,
            dashboard_callback,
        ]

        total_steps = self.train_params.get("total_timesteps", 3_000_000)
        vec_type = str(self.train_params.get("vec_env_type", "subproc"))
        print(f"Starting {algo} training for {total_steps:,} timesteps...")
        print(f"  Parallel envs: {n_envs} ({vec_type} VecEnv) -> ~{n_envs}x samples per policy step")
        print(f"  Checkpoints every {checkpoint_freq_env:,} env steps -> {self.log_dir}/checkpoints/")
        print(f"  Eval every {eval_freq_env:,} env steps; best model -> {self.model_dir}/best_model/")
        print(f"  Best-by-gain model -> {self.model_dir}/best_by_gain/")
        print(f"  TensorBoard -> tensorboard --logdir {self.log_dir}")
        if not self.success_params.get("enable_track_loss_termination", False):
            print(
                "  Episodes: fixed 5s horizon (truncation only). "
                "Success = metrics, not early termination."
            )
        self.model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            tb_log_name=f"{algo}_run",
            reset_num_timesteps=reset_num_timesteps,
        )

        # Save final model
        final_model_path = os.path.join(self.model_dir, f"{algo}_final_model")
        self.model.save(final_model_path)
        print(f"Training completed. Final model saved to {final_model_path}.")
        
        self.env.close()
        eval_env.close()
