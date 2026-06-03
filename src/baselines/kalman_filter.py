"""Extended Kalman Filter (EKF) for Singer Target Dynamics tracking."""

import numpy as np

class SingerEKF:
    """
    Extended Kalman Filter for a target undergoing Singer maneuver dynamics.
    State: [x, y, z, vx, vy, vz, ax, ay, az]
    Measurements: [range, theta, phi]
    """
    
    def __init__(self, dt: float, tau: float, sigma_a: float, meas_noise_std: np.ndarray, initial_state: np.ndarray):
        """
        Args:
            dt: Timestep (seconds).
            tau: Correlation time of the Singer model.
            sigma_a: Standard deviation of target acceleration.
            meas_noise_std: 1D array of [range_std, theta_std, phi_std].
            initial_state: 9D initial state vector.
        """
        self.dt = dt
        self.tau = tau
        self.sigma_a = sigma_a
        self.state = initial_state.copy().astype(float)
        
        # Initialize covariance matrix P
        self.P = np.eye(9) * 100.0  # Initial high uncertainty
        
        # Process noise covariance Q
        # Simplified process noise applied to acceleration components
        q = 2.0 * (sigma_a ** 2) / tau
        self.Q = np.zeros((9, 9))
        
        # Analytical exact discretization Q elements could be computed, but a 
        # standard block diagonal approximation is often sufficient for tracking.
        # We'll use a simplified piece-wise constant white acceleration model for Q
        # mapped through the Singer transitions.
        dt2 = dt**2 / 2
        dt3 = dt**3 / 6
        G = np.array([
            [dt2, 0, 0],
            [0, dt2, 0],
            [0, 0, dt2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        self.Q = G @ G.T * (sigma_a**2)

        # Measurement noise covariance R
        self.R = np.diag(meas_noise_std ** 2)

    def _singer_transition_matrix(self) -> np.ndarray:
        """Returns the 9x9 state transition matrix F for the Singer model."""
        rho = np.exp(-self.dt / self.tau)
        tau_sq = self.tau ** 2
        
        # Sub-matrices
        pos_v = self.dt
        pos_a = (rho * self.dt - 1 + np.exp(-self.dt / self.tau)) * tau_sq
        vel_a = (1 - rho) * self.tau
        
        F_1D = np.array([
            [1, self.dt, pos_a],
            [0, 1, vel_a],
            [0, 0, rho]
        ])
        
        F = np.zeros((9, 9))
        for i in range(3):
            F[i, i] = F_1D[0, 0]
            F[i, i+3] = F_1D[0, 1]
            F[i, i+6] = F_1D[0, 2]
            F[i+3, i+3] = F_1D[1, 1]
            F[i+3, i+6] = F_1D[1, 2]
            F[i+6, i+6] = F_1D[2, 2]
        return F

    def predict(self):
        """Predicts the next state and covariance."""
        F = self._singer_transition_matrix()
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        
    def _measurement_model(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes predicted measurement h(x) and Jacobian H.
        Returns:
            h_x: [range, theta, phi]
            H: 3x9 Jacobian matrix
        """
        x, y, z = state[0], state[1], state[2]
        r_xy_sq = x**2 + y**2
        r_xy = np.sqrt(r_xy_sq) + 1e-9  # Add eps to avoid division by zero
        r_sq = r_xy_sq + z**2
        r = np.sqrt(r_sq) + 1e-9
        
        # Predicted measurement
        range_pred = r
        theta_pred = np.arccos(np.clip(z / r, -1.0, 1.0))
        phi_pred = np.arctan2(y, x)
        h_x = np.array([range_pred, theta_pred, phi_pred])
        
        # Jacobian H
        H = np.zeros((3, 9))
        # d(Range)/d(x,y,z)
        H[0, 0] = x / r
        H[0, 1] = y / r
        H[0, 2] = z / r
        
        # d(Theta)/d(x,y,z)
        H[1, 0] = (x * z) / (r_sq * r_xy)
        H[1, 1] = (y * z) / (r_sq * r_xy)
        H[1, 2] = -r_xy / r_sq
        
        # d(Phi)/d(x,y,z)
        H[2, 0] = -y / r_xy_sq
        H[2, 1] = x / r_xy_sq
        H[2, 2] = 0.0
        
        return h_x, H

    def update(self, measurement: np.ndarray):
        """
        Updates the state estimate with a new measurement [range, theta, phi].
        """
        h_x, H = self._measurement_model(self.state)
        
        y = measurement - h_x
        # Wrap angular errors
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        y[2] = (y[2] + np.pi) % (2 * np.pi) - np.pi
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        
        I = np.eye(9)
        # Joseph form covariance update for numerical stability
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

    def get_state(self) -> np.ndarray:
        return self.state
