"""Traditional Monopulse Tracking Agent using an Extended Kalman Filter."""

import numpy as np
from typing import Any
from .kalman_filter import SingerEKF

class MonopulseTracker:
    """
    Acts as an RL agent but uses traditional radar tracking algorithms.
    Simulates monopulse error extraction by adding noise to true target angles.
    """
    def __init__(
        self,
        dt: float = 0.01,
        max_angular_step_deg: float = 2.0,
        noise_std_range: float = 5.0,     # meters
        noise_std_theta_deg: float = 0.5, # degrees
        noise_std_phi_deg: float = 0.5,   # degrees
        tau: float = 2.0,
        sigma_a: float = 30.0,
    ):
        self.dt = dt
        self.max_angular_step_rad = np.radians(max_angular_step_deg)
        self.meas_noise_std = np.array([
            noise_std_range,
            np.radians(noise_std_theta_deg),
            np.radians(noise_std_phi_deg)
        ])
        self.tau = tau
        self.sigma_a = sigma_a
        
        self.ekf = None
        
    def _init_ekf(self, initial_pos: np.ndarray):
        """Initialize EKF assuming target is stationary initially."""
        state = np.zeros(9)
        state[0:3] = initial_pos
        # Small random initial velocity to prevent numerical issues
        state[3:6] = np.random.randn(3) * 10.0
        self.ekf = SingerEKF(
            dt=self.dt,
            tau=self.tau,
            sigma_a=self.sigma_a,
            meas_noise_std=self.meas_noise_std,
            initial_state=state
        )

    def predict(self, obs: np.ndarray, state=None, episode_start=None, deterministic=True, info: dict[str, Any] = None) -> tuple[np.ndarray, Any]:
        """
        Mimics the stable-baselines3 model.predict API.
        Requires the true target position in `info` to simulate the monopulse measurement.
        """
        if info is None or "target_pos" not in info:
            # Fallback if no info is provided (e.g., very first step after env.reset() 
            # if we don't handle it properly in the evaluation loop)
            return np.zeros(2, dtype=np.float32), None
            
        target_pos = info["target_pos"]
        true_theta, true_phi = info["target_angles"]
        true_r = np.linalg.norm(target_pos)
        
        # 1. Initialize EKF if first step
        if self.ekf is None:
            self._init_ekf(target_pos)
            
        # 2. Simulate Monopulse Measurement (Add noise)
        meas_r = true_r + np.random.randn() * self.meas_noise_std[0]
        meas_theta = true_theta + np.random.randn() * self.meas_noise_std[1]
        meas_phi = true_phi + np.random.randn() * self.meas_noise_std[2]
        measurement = np.array([meas_r, meas_theta, meas_phi])
        
        # 3. Predict & Update EKF
        # Predict to current time, update with current measurement, then predict to NEXT time
        self.ekf.predict()
        self.ekf.update(measurement)
        self.ekf.predict()
        
        # 4. Extract desired beam angles from predicted state
        pred_state = self.ekf.get_state()
        px, py, pz = pred_state[0], pred_state[1], pred_state[2]
        pr = np.sqrt(px**2 + py**2 + pz**2) + 1e-9
        
        cmd_theta = float(np.arccos(np.clip(pz / pr, -1.0, 1.0)))
        cmd_phi = float(np.arctan2(py, px))
        
        # 5. Compute Incremental Action
        b_theta, b_phi = info["beam_angles"]
        d_theta = cmd_theta - b_theta
        d_phi = cmd_phi - b_phi
        
        # Wrap d_phi to [-pi, pi]
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
        
        # Normalize to [-1, 1] action space
        action = np.zeros(2, dtype=np.float32)
        action[0] = np.clip(d_theta / self.max_angular_step_rad, -1.0, 1.0)
        action[1] = np.clip(d_phi / self.max_angular_step_rad, -1.0, 1.0)
        
        return action, None
