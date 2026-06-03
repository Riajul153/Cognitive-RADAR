
import os
import argparse
import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from gymnasium import spaces

# Import our standard modules
from src.antenna.array import UniformPlanarArray
from src.antenna.beamformer import Beamformer
from src.environment.beam_tracking_env import BeamTrackingEnv
from src.agents.trainer import BeamTrackingTrainer

# --- DIPOLE MONKEY PATCH (HARDWARE AWARE) ---

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

class HardwareAwareDipoleEnv(BeamTrackingEnv):
    def __init__(self, config):
        super().__init__(config)
        self.beamformer = DipoleBeamformer(self.array)
        self.monopulse.beamformer = self.beamformer
        
        # Override observation space: 10D (base) + 1D (Element Factor) = 11D
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(11,), dtype=np.float32,
        )

    def _get_obs(self):
        # Base 10D obs (already includes absolute beam angles at idx 4 and 5)
        base_obs = super()._get_obs()
        
        # Compute theoretical hardware attenuation at current commanded angle
        b_theta = self.beam_theta_cmd
        b_phi = self.beam_phi_cmd
        ef = float(self.beamformer._dipole_element_factor(b_theta, b_phi))
        
        # EF naturally ranges [0, 1]. Map to [-1, 1] to match rest of obs
        norm_ef = np.clip((ef * 2.0) - 1.0, -1.0, 1.0)
        
        obs = np.append(base_obs, [norm_ef]).astype(np.float32)
        return obs

def make_env(config):
    def _init():
        return HardwareAwareDipoleEnv(config)
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
    parser.add_argument("--config", type=str, default="config/sac_parametric_dipole_hardware_aware.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    trainer = DipoleTrainer(config)
    trainer.train()
