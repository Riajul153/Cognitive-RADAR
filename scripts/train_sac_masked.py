import os
import argparse
import numpy as np
import yaml
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from src.antenna.array import UniformPlanarArray
from src.antenna.beamformer import Beamformer
from src.environment.beam_tracking_env import BeamTrackingEnv
from src.agents.trainer import BeamTrackingTrainer
from src.antenna.steering import angular_distance
from src.agents.masked_sac import MaskedSAC
from src.agents.callbacks import TrackingMetricsCallback, evaluate_tracking_policy

# ── Custom Antenna with Physical Nulls (from Run A) ────────────────────
class DipoleBeamformer(Beamformer):
    def _dipole_element_factor(self, theta, phi):
        cos_gamma = np.sin(theta) * np.cos(phi)
        sin_gamma = np.sqrt(1.0 - cos_gamma**2 + 1e-12)
        E_field = np.cos((np.pi / 2.0) * cos_gamma) / sin_gamma
        return E_field**2

    def compute_beam_pattern(self, theta_grid, phi_grid):
        af_power = super().compute_beam_pattern(theta_grid, phi_grid)
        ef_power = self._dipole_element_factor(theta_grid, phi_grid)
        total_pattern = af_power * ef_power
        return total_pattern / (np.max(total_pattern) + 1e-12)

    def compute_gain_at_target(self, target_theta, target_phi):
        af_power = super().compute_gain_at_target(target_theta, target_phi)
        ef_power = self._dipole_element_factor(target_theta, target_phi)
        return float(af_power * ef_power)

class FixedRewardDipoleEnv(BeamTrackingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beamformer = DipoleBeamformer(self.array)
        self.monopulse.beamformer = self.beamformer

    def step(self, action):
        self.current_step += 1
        self.last_action = action.copy() if hasattr(action, "copy") else action

        # Decode & Apply Action
        phases = self._decode_action(action)
        self.beamformer.set_weights(phases)

        # Propagate Target
        self.target.step()
        t_pos = self.target.get_position()
        t_theta, t_phi = self.target.get_angular_position()

        # Compute Physics Gain (Monopulse sees this)
        physics_gain = self.beamformer.compute_gain_at_target(t_theta, t_phi)
        self.last_gain = physics_gain
        
        # Compute Reward Gain (AF only)
        af_gain = float(super(type(self.beamformer), self.beamformer).compute_gain_at_target(t_theta, t_phi))

        b_theta = self.beam_theta_cmd
        b_phi = self.beam_phi_cmd
        error_rad = angular_distance(t_theta, t_phi, b_theta, b_phi)
        self.last_error = error_rad
        error_deg = float(np.degrees(error_rad))

        is_first_step = (self.current_step == 1)
        reward, reward_info = self.reward_computer.compute(
            af_gain, phases, is_first_step=is_first_step
        )
        
        tracking_locked = af_gain >= self.min_gain and error_deg <= self.max_error_deg
        if tracking_locked:
            self._locked_steps += 1
            self._consecutive_locked += 1
            self._consecutive_loss_steps = 0
        else:
            self._consecutive_locked = 0
            self._consecutive_loss_steps += 1

        locked_fraction = self._locked_steps / max(self.current_step, 1)

        terminated = False
        if self.enable_track_loss_termination and self._consecutive_loss_steps >= self.track_loss_steps:
            terminated = True
        truncated = self.current_step >= self.episode_length

        obs = self._get_obs()
        
        episode_success = locked_fraction >= self.min_locked_fraction
        info = {
            "target_pos": t_pos,
            "target_angles": (t_theta, t_phi),
            "beam_angles": (b_theta, b_phi),
            "normalized_gain": physics_gain,
            "tracking_error_deg": error_deg,
            "tracking_locked": tracking_locked,
            "consecutive_locked_steps": self._consecutive_locked,
            "episode_locked_fraction": locked_fraction,
            "episode_success": episode_success if truncated or terminated else False,
            "terminated_track_loss": terminated and self.enable_track_loss_termination,
        }
        return obs, float(reward), terminated, truncated, info

def make_env(config):
    def _init():
        env_kwargs = {
            "n_rows": config["antenna"].get("n_rows", 8),
            "n_cols": config["antenna"].get("n_cols", 8),
            "frequency_hz": config["antenna"].get("frequency_hz", 10e9),
            "element_spacing": config["antenna"].get("element_spacing", 0.5),
            "target_config": config.get("target", {}),
            "dt": config["environment"].get("dt", 0.01),
            "episode_length": config["environment"].get("episode_length", 500),
            "action_mode": config["environment"].get("action_mode", "parametric"),
            "parametric_type": config["environment"].get("parametric_type", "incremental"),
            "max_angular_step_deg": config["environment"].get("max_angular_step_deg", 2.0),
            "elevation_max_deg": config["environment"].get("elevation_max_deg", 60.0),
            "azimuth_max_deg": config["environment"].get("azimuth_max_deg", 60.0),
            "reward_config": config["environment"].get("reward", {}),
            "success_config": config["environment"].get("success", {}),
        }
        return FixedRewardDipoleEnv(**env_kwargs)
    return _init


class MaskedSACTrainer(BeamTrackingTrainer):
    """Custom trainer that instantiates the MaskedSAC algorithm."""
    
    def _setup_envs(self):
        self.n_envs = self.config["training"].get("n_envs", 1)
        env_fns = [make_env(self.config) for _ in range(self.n_envs)]
        self.env = SubprocVecEnv(env_fns)
        self.env = VecMonitor(self.env)
        
        self.eval_env = SubprocVecEnv([make_env(self.config)])
        self.eval_env = VecMonitor(self.eval_env)
        
        # INCREASED FRAME STACK to 8
        frame_stack = self.config["training"].get("frame_stack", 8)
        print(f"Applying FrameStack = {frame_stack}")
        self.env = VecFrameStack(self.env, n_stack=frame_stack)
        self.eval_env = VecFrameStack(self.eval_env, n_stack=frame_stack)

    def train(self):
        self._setup_envs()
        print(f"=== Starting Masked SAC Training on Dipole Environment ===")
        print(f"Log Dir:   {self.log_dir}")
        print(f"Model Dir: {self.model_dir}")

        train_cfg = self.config["training"]
        
        # Instantiate the custom MaskedSAC!
        self.model = MaskedSAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=float(train_cfg["learning_rate"]),
            buffer_size=int(train_cfg["buffer_size"]),
            learning_starts=10000,
            batch_size=int(train_cfg["batch_size"]),
            tau=float(train_cfg["tau"]),
            gamma=float(train_cfg["gamma"]),
            train_freq=int(train_cfg["train_freq"]),
            gradient_steps=int(train_cfg["gradient_steps"]),
            ent_coef=train_cfg["ent_coef"],
            policy_kwargs=dict(net_arch=train_cfg["net_arch"]),
            tensorboard_log=self.log_dir,
            verbose=1,
            device=train_cfg["device"],
        )

        tracking_cb = TrackingMetricsCallback(
            log_dir=self.log_dir,
            log_freq=train_cfg["log_freq"],
            csv_flush_freq=train_cfg.get("csv_flush_freq", 1000),
            verbose=1,
        )

        checkpoint_cb = CheckpointCallback(
            save_freq=max(train_cfg["checkpoint_freq"] // self.n_envs, 1),
            save_path=os.path.join(self.log_dir, "checkpoints"),
            name_prefix="masked_sac",
            verbose=1,
        )
        
        from stable_baselines3.common.callbacks import EvalCallback
        eval_cb = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(self.model_dir, "best_model"),
            log_path=os.path.join(self.log_dir, "evaluations"),
            eval_freq=max(train_cfg["eval_freq"] // self.n_envs, 1),
            n_eval_episodes=train_cfg["n_eval_episodes"],
            deterministic=True,
            render=False,
        )

        callbacks = CallbackList([tracking_cb, checkpoint_cb, eval_cb])

        self.model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            callback=callbacks,
            tb_log_name="MaskedSAC_run",
        )

        print("Training completed. Saving final model...")
        final_path = os.path.join(self.model_dir, "MaskedSAC_final_model")
        self.model.save(final_path)
        print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sac_dipole_masked.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    trainer = MaskedSACTrainer(config)
    trainer.train()
