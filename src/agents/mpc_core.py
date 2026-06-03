import numpy as np
from scipy.optimize import minimize
from src.antenna.steering import angular_distance

class MPCSolver:
    """Model Predictive Control (MPC) Core for Beam Tracking.
    
    Solves a receding horizon optimization problem to find the optimal sequence of
    angular beam increments that minimizes tracking error over the prediction horizon.
    """
    
    def __init__(self, horizon: int = 10, max_step_rad: float = 2.0 * np.pi / 180.0):
        self.H = horizon
        self.max_step = max_step_rad
        # Penalty for control effort to ensure smoothness
        self.lambda_u = 0.01
        
    def _objective(self, u_flat, current_beam, target_traj):
        """
        Objective function to minimize.
        u_flat: 1D array of shape (2 * H,) containing [d_theta_0, d_phi_0, d_theta_1, ...]
        current_beam: (theta, phi)
        target_traj: shape (H, 2)
        """
        U = u_flat.reshape(self.H, 2)
        
        # Integrate actions to get beam trajectory
        # B_k = B_0 + sum_{i=0}^k U_i
        beam_traj = current_beam + np.cumsum(U, axis=0)
        
        # Calculate tracking error
        # We use a simple squared angular error approximation.
        # For phi, we must handle the circular wrap-around.
        d_theta = beam_traj[:, 0] - target_traj[:, 0]
        
        d_phi = beam_traj[:, 1] - target_traj[:, 1]
        # Wrap d_phi to [-pi, pi]
        d_phi = (d_phi + np.pi) % (2.0 * np.pi) - np.pi
        
        # Discount factor so it prioritizes immediate tracking
        gamma = 0.95
        weights = gamma ** np.arange(self.H)
        
        # Tracking cost
        tracking_cost = np.sum(weights * (d_theta**2 + d_phi**2))
        
        # Control effort cost (L2 penalty)
        control_cost = self.lambda_u * np.sum(U**2)
        
        return tracking_cost + control_cost

    def solve(self, current_beam: tuple[float, float], target_traj: np.ndarray) -> np.ndarray:
        """
        Solve the MPC problem.
        target_traj: Expected target trajectory of shape (H, 2) in radians.
        Returns: The optimal FIRST action (d_theta, d_phi) in radians.
        """
        # Ensure target_traj matches horizon
        H_actual = min(self.H, len(target_traj))
        if H_actual < self.H:
            # Pad by repeating the last prediction
            pad = np.tile(target_traj[-1:], (self.H - H_actual, 1))
            target_traj = np.vstack([target_traj, pad])
            
        current_beam_arr = np.array(current_beam)
        
        # Initial guess: Pure Pursuit (Greedy)
        # Point straight at the target at each step, bounded by max_step
        u_guess = np.zeros((self.H, 2))
        beam = current_beam_arr.copy()
        for k in range(self.H):
            delta = target_traj[k] - beam
            delta[1] = (delta[1] + np.pi) % (2.0 * np.pi) - np.pi
            u_greedy = np.clip(delta, -self.max_step, self.max_step)
            u_guess[k] = u_greedy
            beam += u_greedy
            
        u_guess_flat = u_guess.flatten()
        
        # Bounds: all actions must be within [-max_step, max_step]
        bounds = [(-self.max_step, self.max_step)] * (2 * self.H)
        
        # Solve using SLSQP
        res = minimize(
            self._objective, 
            u_guess_flat, 
            args=(current_beam_arr, target_traj),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-4, 'disp': False}
        )
        
        # Extract the first optimal action sequence
        u_opt = res.x.reshape(self.H, 2)
        
        # Return the FIRST action (receding horizon control)
        return u_opt[0]
