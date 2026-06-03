
import os
import argparse
import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from gymnasium import spaces

from src.antenna.array import UniformPlanarArray
from src.antenna.beamformer import Beamformer
from src.environment.beam_tracking_env import BeamTrackingEnv
from src.agents.trainer import BeamTrackingTrainer
from src.antenna.steering import angular_distance

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
    def __init__(self, config):
        super().__init__(config)
        self.beamformer = DipoleBeamformer(self.array)
        self.monopulse.beamformer = self.beamformer

    def step(self, action):
        self.current_step += 1
        self.last_action = action.copy() if hasattr(action, "copy") else action

        # 1. Decode & Apply Action
        phases = self._decode_action(action)
        self.beamformer.set_weights(phases)

        # 2. Propagate Target
        self.target.step()
        t_pos = self.target.get_position()
        t_theta, t_phi = self.target.get_angular_position()

        # 3. Compute Physics Gain (Monopulse sees this)
        physics_gain = self.beamformer.compute_gain_at_target(t_theta, t_phi)
        self.last_gain = physics_gain
        
        # 4. Compute Reward Gain (AF only, ignores hardware nulls!)
        af_gain = float(super(type(self.beamformer), self.beamformer).compute_gain_at_target(t_theta, t_phi))

        b_theta = self.beam_theta_cmd
        b_phi = self.beam_phi_cmd

        error_rad = angular_distance(t_theta, t_phi, b_theta, b_phi)
        self.last_error = error_rad
        error_deg = float(np.degrees(error_rad))

        # 5. Compute Reward using AF_GAIN
        is_first_step = (self.current_step == 1)
        reward, reward_info = self.reward_computer.compute(
            af_gain, phases, is_first_step=is_first_step
        )
        
        # 6. Success bookkeeping using AF_GAIN
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

        obs = self._get_obs() # Monopulse processor intrinsically uses physics_gain
        
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
        return FixedRewardDipoleEnv(config)
    return _init

class DipoleTrainer(BeamTrackingTrainer):
    def _setup_envs(self):
        env_fns = [make_env(self.config) for _ in range(self.n_envs)]
        self.env = SubprocVecEnv(env_fns)
        self.env = VecMonitor(self.env)
        self.eval_env = SubprocVecEnv([make_env(self.config)])
        self.eval_env = VecMonitor(self.eval_env)
        if self.config["training"].get("frame_stack", 1) > 1:
            from stable_baselines3.common.vec_env import VecFrameStack
            self.env = VecFrameStack(self.env, n_stack=self.config["training"]["frame_stack"])
            self.eval_env = VecFrameStack(self.eval_env, n_stack=self.config["training"]["frame_stack"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sac_parametric_dipole_fixed_reward.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    trainer = DipoleTrainer(config)
    trainer.train()
