import numpy as np
from src.environment.beam_tracking_env import BeamTrackingEnv
from src.antenna.array import UniformPlanarArray
from src.agents.ekf_mpc_agent import EKFMPCAgent

class HalfWaveDipoleArray(UniformPlanarArray):
    def get_steering_vector_batch(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        v_iso = super().get_steering_vector_batch(theta, phi)
        theta_safe = np.clip(theta, 1e-6, np.pi - 1e-6)
        element_factor = np.cos(np.pi / 2.0 * np.cos(theta_safe)) / np.sin(theta_safe)
        element_factor = np.abs(element_factor)
        element_factor_2d = element_factor[:, np.newaxis]
        return v_iso * element_factor_2d

class DipoleBeamTrackingEnv(BeamTrackingEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.array = HalfWaveDipoleArray(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            frequency_hz=self.array.frequency_hz,
            element_spacing_wavelengths=self.array.spacing / self.array.wavelength
        )
        self.beamformer.array = self.array
        self.monopulse.array = self.array

def evaluate_ekf_mpc(n_episodes=50):
    print("Initializing Dipole Environment (with physical nulls)...")
    env = DipoleBeamTrackingEnv(
        n_rows=8,
        n_cols=8,
        action_mode="parametric",
        parametric_type="incremental",
        max_angular_step_deg=2.0
    )
    
    # Horizon 20 steps (0.2s)
    agent = EKFMPCAgent(
        hpbw_rad=env.hpbw_rad,
        max_angular_step_rad=env.max_angular_step_rad,
        dt=env.dt,
        horizon=20
    )
    
    print(f"Evaluating non-privileged EKF-MPC for {n_episodes} episodes...")
    
    total_gain = 0.0
    total_error = 0.0
    total_success = 0
    
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
            obs, reward, done, truncated, info = env.step(action)
            
            ep_gain += info["normalized_gain"]
            ep_error += info["tracking_error_deg"]
            ep_steps += 1
            
        total_gain += ep_gain / ep_steps
        total_error += ep_error / ep_steps
        if info.get("tracking_locked", False):
            total_success += 1
            
        if (ep + 1) % 5 == 0:
            print(f"Episode {ep+1}/{n_episodes} | Gain: {ep_gain/ep_steps:.3f} | Err: {ep_error/ep_steps:.1f} deg")

    metrics = {
        "success_rate": total_success / n_episodes,
        "mean_gain": total_gain / n_episodes,
        "mean_error": total_error / n_episodes,
    }
    
    print("\n--- FINAL EKF-MPC BENCHMARK (NO PRIVILEGED KNOWLEDGE) ---")
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Mean Gain:    {metrics['mean_gain']:.3f}")
    print(f"Mean Error:   {metrics['mean_error']:.2f} deg")

if __name__ == "__main__":
    evaluate_ekf_mpc(n_episodes=50)
