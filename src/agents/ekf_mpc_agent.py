import numpy as np
from src.agents.ekf_agent import EKFAgent
from src.agents.mpc_core import MPCSolver

class EKFMPCAgent(EKFAgent):
    """Classical baseline: EKF for estimation + MPC for control.
    
    Like the base EKFAgent, this agent has NO privileged information. It observes
    the noisy 10D hardware state and uses an Extended Kalman Filter to estimate
    the target's current angular kinematics.
    
    Unlike the base EKFAgent (which uses greedy Proportional Navigation), this
    agent generates a predicted trajectory over a rolling horizon and uses a Model
    Predictive Controller (MPC) to optimize the beam tracking trajectory.
    """
    
    def __init__(self, hpbw_rad: float, max_angular_step_rad: float, dt: float = 0.01, horizon: int = 10):
        super().__init__(hpbw_rad, max_angular_step_rad, dt)
        self.mpc = MPCSolver(horizon=horizon, max_step_rad=max_angular_step_rad)
        self.horizon = horizon
        
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Processes the 10D observation and outputs the 2D incremental action."""
        # 1. Denormalize observation
        norm_err_el = obs[0]
        norm_err_az = obs[1]
        norm_power = obs[2]
        norm_b_theta = obs[4]
        norm_b_phi = obs[5]
        
        power = (norm_power + 1.0) / 2.0  # [0, 1]
        err_el = norm_err_el * self.hpbw_rad
        err_az = norm_err_az * self.hpbw_rad
        b_theta = (norm_b_theta + 1.0) * (np.pi / 4.0)
        b_phi = norm_b_phi * np.pi
        
        z = np.array([b_theta + err_el, b_phi + err_az], dtype=np.float32)
        
        # 2. Adaptive Measurement Noise (Outage Filter)
        if power < 0.2:
            R = np.eye(2, dtype=np.float32) * 1e6
        else:
            R = np.eye(2, dtype=np.float32) * (self.r_nominal / (power**2 + 1e-6))
            
        # 3. EKF Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # 4. EKF Update
        y = z - (self.H @ x_pred)
        y[1] = (y[1] + np.pi) % (2.0 * np.pi) - np.pi
        
        S = self.H @ P_pred @ self.H.T + R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        
        # 5. MPC Prediction: Project constant velocity over horizon
        target_traj = np.zeros((self.horizon, 2))
        x_curr = self.x.copy()
        for k in range(self.horizon):
            x_curr = self.F @ x_curr
            target_traj[k] = [x_curr[0], x_curr[1]]
            
        # 6. MPC Solve
        current_beam = (b_theta, b_phi)
        u_opt = self.mpc.solve(current_beam, target_traj)
        
        # 7. Normalize action to [-1, 1] for the environment
        a0 = np.clip(u_opt[0] / self.max_step, -1.0, 1.0)
        a1 = np.clip(u_opt[1] / self.max_step, -1.0, 1.0)
        
        return np.array([a0, a1], dtype=np.float32)
