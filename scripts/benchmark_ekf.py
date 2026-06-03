import argparse
import numpy as np

from src.environment.beam_tracking_env import BeamTrackingEnv
from src.antenna.array import UniformPlanarArray
from src.agents.ekf_agent import EKFAgent

# ── Custom Antenna with Physical Nulls (from Run A) ────────────────────
class HalfWaveDipoleArray(UniformPlanarArray):
    """Custom array injecting physical hardware nulls into the beamformer.
    
    Models z-oriented half-wave dipoles. The element factor is:
    E(θ) = cos(π/2 * cos(θ)) / sin(θ)
    This creates a complete null (0 signal) at boresight (θ=0) and θ=180.
    Since boresight tracking is common, this is a massive challenge.
    """
    def get_steering_vector_batch(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        v_iso = super().get_steering_vector_batch(theta, phi)
        # Prevent division by zero
        theta_safe = np.clip(theta, 1e-6, np.pi - 1e-6)
        element_factor = np.cos(np.pi / 2.0 * np.cos(theta_safe)) / np.sin(theta_safe)
        element_factor = np.abs(element_factor)
        # Broadcast element factor to match shape (M, N)
        element_factor_2d = element_factor[:, np.newaxis]
        return v_iso * element_factor_2d

class DipoleBeamTrackingEnv(BeamTrackingEnv):
    """Environment initialized with the Half-Wave Dipole array."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Overwrite the isotropic array with the dipole array
        self.array = HalfWaveDipoleArray(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            frequency_hz=self.array.frequency_hz,
            element_spacing_wavelengths=self.array.spacing / self.array.wavelength
        )
        self.beamformer.array = self.array
        self.monopulse.array = self.array


def evaluate_ekf(n_episodes=50):
    print("Initializing Dipole Environment (with physical nulls)...")
    env = DipoleBeamTrackingEnv(
        n_rows=8,
        n_cols=8,
        action_mode="parametric",
        parametric_type="incremental",
        max_angular_step_deg=2.0
    )
    
    agent = EKFAgent(
        hpbw_rad=env.hpbw_rad,
        max_angular_step_rad=env.max_angular_step_rad,
        dt=env.dt
    )
    
    print(f"Evaluating classical EKF for {n_episodes} episodes...")
    
    total_gain = 0.0
    total_error = 0.0
    total_success = 0
    total_steps = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset()
        
        done = False
        truncated = False
        ep_gain = 0.0
        ep_error = 0.0
        ep_steps = 0
        
        while not (done or truncated):
            action = agent.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            obs = next_obs
            
            ep_gain += info["normalized_gain"]
            ep_error += info["tracking_error_deg"]
            ep_steps += 1
            
        total_gain += ep_gain / ep_steps
        total_error += ep_error / ep_steps
        if info.get("tracking_locked", False):
            # Using the strict success definition from the environment
            total_success += 1
            
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes} | Gain: {ep_gain/ep_steps:.3f} | Err: {ep_error/ep_steps:.1f} deg")

    metrics = {
        "mean_gain": total_gain / n_episodes,
        "mean_error": total_error / n_episodes,
        "success_rate": total_success / n_episodes
    }
    
    print("\n--- FINAL EKF BENCHMARK (DIPOLE NULLS) ---")
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Mean Gain:    {metrics['mean_gain']:.3f}")
    print(f"Mean Error:   {metrics['mean_error']:.2f} deg")
    
if __name__ == "__main__":
    evaluate_ekf(n_episodes=50)
