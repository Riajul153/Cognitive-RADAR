import numpy as np
from src.agents.mpc_core import MPCSolver
from src.target.dynamics import AerialTarget
from src.antenna.steering import cart_to_spherical

class OracleMPCAgent:
    """State-of-the-Art Upper Bound: Oracle + MPC.
    
    This agent represents the theoretical mathematical upper bound for performance.
    It has PRIVILEGED access to the true target state and target dynamics.
    
    It queries the simulation engine for the true 9D state of the target, exactly
    propagates the Singer maneuver model into the future to construct a perfect
    expected trajectory, and uses MPC to solve for the optimal beam track.
    """
    
    def __init__(self, target: AerialTarget, max_angular_step_rad: float, dt: float = 0.01, horizon: int = 10):
        self.target = target
        self.dt = dt
        self.horizon = horizon
        self.mpc = MPCSolver(horizon=horizon, max_step_rad=max_angular_step_rad)
        
        # Build exact Singer transition matrix for prediction
        # This perfectly mimics the environment's internal physics
        tau = target.correlation_time
        rho = np.exp(-dt / tau)
        
        # F operates on [pos, vel, acc] for a single axis
        self.F_axis = np.array([
            [1.0, dt,   (rho*dt - 1.0 + np.exp(-dt/tau)) * (tau**2)],
            [0.0, 1.0,  (1.0 - rho) * tau],
            [0.0, 0.0,  rho]
        ])
        
    def act(self, current_beam: tuple[float, float]) -> np.ndarray:
        """Computes the optimal action given the current true state of the environment."""
        
        # 1. Oracle Knowledge: Get exact 9D state
        true_pos = self.target.get_position()
        true_vel = self.target.get_velocity()
        true_acc = self.target.get_acceleration()
        
        # 2. Perfect Prediction: Project expected 3D trajectory
        # We project expected value, meaning process noise expectation is 0.
        target_traj_3d = np.zeros((self.horizon, 3))
        
        state_x = np.array([true_pos[0], true_vel[0], true_acc[0]])
        state_y = np.array([true_pos[1], true_vel[1], true_acc[1]])
        state_z = np.array([true_pos[2], true_vel[2], true_acc[2]])
        
        for k in range(self.horizon):
            state_x = self.F_axis @ state_x
            state_y = self.F_axis @ state_y
            state_z = self.F_axis @ state_z
            target_traj_3d[k] = [state_x[0], state_y[0], state_z[0]]
            
        # 3. Convert expected 3D trajectory to angular targets
        target_traj_ang = np.zeros((self.horizon, 2))
        for k in range(self.horizon):
            r, th, ph = cart_to_spherical(
                target_traj_3d[k, 0], 
                target_traj_3d[k, 1], 
                target_traj_3d[k, 2]
            )
            target_traj_ang[k] = [th, ph]
            
        # 4. MPC Solve
        u_opt = self.mpc.solve(current_beam, target_traj_ang)
        
        # 5. Normalize action to [-1, 1] for the environment
        a0 = np.clip(u_opt[0] / self.mpc.max_step, -1.0, 1.0)
        a1 = np.clip(u_opt[1] / self.mpc.max_step, -1.0, 1.0)
        
        return np.array([a0, a1], dtype=np.float32)
