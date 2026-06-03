
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

class CrossedDipoleBeamformer(Beamformer):
    def _dipole_element_factor_x(self, theta, phi):
        cos_gamma = np.sin(theta) * np.cos(phi)
        sin_gamma = np.sqrt(1.0 - cos_gamma**2 + 1e-12)
        E_field = np.cos((np.pi / 2.0) * cos_gamma) / sin_gamma
        return E_field**2

    def _dipole_element_factor_y(self, theta, phi):
        cos_gamma = np.sin(theta) * np.sin(phi)
        sin_gamma = np.sqrt(1.0 - cos_gamma**2 + 1e-12)
        E_field = np.cos((np.pi / 2.0) * cos_gamma) / sin_gamma
        return E_field**2

    def compute_beam_pattern(self, theta_grid, phi_grid):
        af_power = super().compute_beam_pattern(theta_grid, phi_grid)
        ef_power_x = self._dipole_element_factor_x(theta_grid, phi_grid)
        ef_power_y = self._dipole_element_factor_y(theta_grid, phi_grid)
        # Sum powers of X and Y dipoles (polarization/spatial diversity)
        total_pattern = af_power * (ef_power_x + ef_power_y)
        return total_pattern / (np.max(total_pattern) + 1e-12)

    def compute_gain_at_target(self, target_theta, target_phi):
        af_power = super().compute_gain_at_target(target_theta, target_phi)
        ef_power_x = self._dipole_element_factor_x(target_theta, target_phi)
        ef_power_y = self._dipole_element_factor_y(target_theta, target_phi)
        # Combine orthogonal dipoles. Divide by 2.0 to roughly normalize max to 1.0.
        return float(af_power * (ef_power_x + ef_power_y) / 2.0)

class CrossedDipoleEnv(BeamTrackingEnv):
    def __init__(self, config):
        super().__init__(config)
        self.beamformer = CrossedDipoleBeamformer(self.array)
        self.monopulse.beamformer = self.beamformer

def make_env(config):
    def _init():
        return CrossedDipoleEnv(config)
    return _init

class CrossedDipoleTrainer(BeamTrackingTrainer):
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
    parser.add_argument("--config", type=str, default="config/sac_parametric_crossed_dipole.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    trainer = CrossedDipoleTrainer(config)
    trainer.train()
