import numpy as np

class EKFAgent:
    """Classical Extended Kalman Filter (EKF) radar tracker.
    
    This agent uses NO privileged information. It only receives the exact same
    10D observation vector as the SAC agent and outputs the same 2D incremental
    parametric action.
    
    It decodes the agent's own beam position and the monopulse error from the
    normalized 10D observation, applies a Kalman Filter with a constant-velocity
    kinematic model, and uses adaptive Measurement Noise Covariance (R) to "coast"
    when the target falls into an antenna null (low received power).
    """

    def __init__(self, hpbw_rad: float, max_angular_step_rad: float, dt: float = 0.01):
        self.hpbw_rad = hpbw_rad
        self.max_step = max_angular_step_rad
        self.dt = dt
        
        # State: [theta, theta_dot, phi, phi_dot]
        # Initialize at boresight with zero velocity
        self.x = np.zeros(4, dtype=np.float32)
        
        # State covariance matrix P
        self.P = np.eye(4, dtype=np.float32) * 1.0
        
        # State transition matrix F (Constant Velocity)
        self.F = np.array([
            [1.0, dt,  0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt ],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Process noise covariance Q
        # Assumes target can accelerate. Variance in acceleration integrated over dt.
        q_accel = 1.0  # Tuning parameter for Singer-like maneuvers
        q_pos = (dt**4) / 4 * q_accel
        q_vel = (dt**2) * q_accel
        q_cov = (dt**3) / 2 * q_accel
        
        self.Q = np.array([
            [q_pos, q_cov, 0.0,   0.0  ],
            [q_cov, q_vel, 0.0,   0.0  ],
            [0.0,   0.0,   q_pos, q_cov],
            [0.0,   0.0,   q_cov, q_vel]
        ], dtype=np.float32)
        
        # Measurement matrix H (we only measure theta and phi positions)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float32)
        
        # Nominal measurement noise variance (when SNR is high)
        self.r_nominal = (0.1 * self.hpbw_rad) ** 2
        
    def reset(self):
        """Resets the filter state at the beginning of an episode."""
        self.x = np.zeros(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Processes the 10D observation, updates the filter, and outputs the 2D action."""
        # 1. Denormalize the observation (reverse engineering the env's _get_obs)
        norm_err_el = obs[0]
        norm_err_az = obs[1]
        norm_power = obs[2]
        norm_b_theta = obs[4]
        norm_b_phi = obs[5]
        
        # Convert normalized values back to physical values
        power = (norm_power + 1.0) / 2.0  # [0, 1]
        err_el = norm_err_el * self.hpbw_rad
        err_az = norm_err_az * self.hpbw_rad
        
        b_theta = (norm_b_theta + 1.0) * (np.pi / 4.0)
        b_phi = norm_b_phi * np.pi
        
        # Form the measurement vector z
        z = np.array([
            b_theta + err_el,
            b_phi + err_az
        ], dtype=np.float32)
        
        # 2. Adaptive Measurement Noise (R)
        # If power is very low (e.g. in a null), the error signal is pure noise.
        # We exponentially increase R as power drops to force the filter to COAST.
        if power < 0.2:
            R = np.eye(2, dtype=np.float32) * 1e6  # Massive uncertainty -> coast
        else:
            # Scale R inversely with power
            R = np.eye(2, dtype=np.float32) * (self.r_nominal / (power**2 + 1e-6))
            
        # 3. Predict Step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # 4. Update Step
        y = z - (self.H @ x_pred)  # Innovation
        # Handle wrap-around for phi innovation if necessary (phi is in [-pi, pi])
        y[1] = (y[1] + np.pi) % (2.0 * np.pi) - np.pi
        
        S = self.H @ P_pred @ self.H.T + R  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman Gain
        
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        
        # 5. Controller: Proportional Steering
        # We want the beam to point exactly at the predicted target position for the NEXT step
        x_next = self.F @ self.x
        theta_cmd = x_next[0]
        phi_cmd = x_next[1]
        
        # Compute deltas required
        d_theta = theta_cmd - b_theta
        d_phi = phi_cmd - b_phi
        
        # Wrap phi delta
        d_phi = (d_phi + np.pi) % (2.0 * np.pi) - np.pi
        
        # Normalize to the agent's action space [-1, 1]
        a0 = np.clip(d_theta / self.max_step, -1.0, 1.0)
        a1 = np.clip(d_phi / self.max_step, -1.0, 1.0)
        
        return np.array([a0, a1], dtype=np.float32)
