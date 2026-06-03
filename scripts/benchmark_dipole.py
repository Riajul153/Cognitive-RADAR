
import os
import argparse
import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Import our standard modules
from src.antenna.array import UniformPlanarArray
from src.antenna.beamformer import Beamformer
from src.environment.beam_tracking_env import BeamTrackingEnv
from src.agents.trainer import BeamTrackingTrainer

# --- DIPOLE MONKEY PATCH ---
# To guarantee we don't corrupt running processes, we subclass Beamformer here.
# This touches zero existing files on disk!

class DipoleBeamformer(Beamformer):
    def _dipole_element_factor(self, theta, phi):
        # Half-wave dipole aligned with X-axis
        # cos(gamma) = sin(theta) * cos(phi)
        cos_gamma = np.sin(theta) * np.cos(phi)
        
        # Handle singularity at gamma = 0 or pi
        # Provide small epsilon to avoid divide by zero
        sin_gamma = np.sqrt(1.0 - cos_gamma**2 + 1e-12)
        
        # Electric field pattern for half-wave dipole
        E_field = np.cos((np.pi / 2.0) * cos_gamma) / sin_gamma
        return E_field**2  # Return power pattern

    def compute_beam_pattern(self, theta_grid, phi_grid):
        af_power = super().compute_beam_pattern(theta_grid, phi_grid)
        ef_power = self._dipole_element_factor(theta_grid, phi_grid)
        # Normalize so max is still roughly 1 for the dashboard
        total_pattern = af_power * ef_power
        return total_pattern / (np.max(total_pattern) + 1e-12)

    def compute_gain_at_target(self, target_theta, target_phi):
        af_power = super().compute_gain_at_target(target_theta, target_phi)
        ef_power = self._dipole_element_factor(target_theta, target_phi)
        return float(af_power * ef_power)

class DipoleBeamTrackingEnv(BeamTrackingEnv):
    def __init__(self, config):
        super().__init__(config)
        # Replace the standard isotropic beamformer with our Dipole version
        self.beamformer = DipoleBeamformer(self.array)
        self.monopulse.beamformer = self.beamformer

def make_env(config):
    def _init():
        return DipoleBeamTrackingEnv(config)
    return _init

# Subclass trainer to use the new environment
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
    parser.add_argument("--config", type=str, default="config/sac_parametric_dipole.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    trainer = DipoleTrainer(config)
    trainer.train()
