# Phase 21: Sophisticated Classical Baselines (MPC & Oracle)

To properly benchmark our DRL agents against State-of-the-Art classical control theory, we will implement Model Predictive Control (MPC). Unlike our greedy Extended Kalman Filter (EKF) which just chases the immediate next prediction (Pure Pursuit), an MPC minimizes tracking error over a *receding horizon*. This allows the beam to intelligently "cut corners" and intercept maneuvering targets, rather than just trailing behind them.

## Proposed Changes

### 1. Model Predictive Controller Core (`src/agents/mpc_core.py`) [NEW]
We will create a reusable MPC solver using `scipy.optimize.minimize` (SLSQP).
* **Horizon**: $H = 20$ steps (0.2 seconds).
* **State**: Current beam angles.
* **Objective**: Minimize the weighted sum of squared tracking errors over the horizon, plus a small penalty for control effort (to ensure smooth beam motion).
* **Constraints**: Bound the angular step to $[-1, 1]$ (which maps to $\pm 2.0^\circ$ per step).

### 2. Non-Privileged EKF-MPC Agent (`src/agents/ekf_mpc_agent.py`) [NEW]
This agent receives the exact same 10D observation as the SAC agent (no privileged data).
* **Estimation**: It uses an Extended Kalman Filter (EKF) to estimate the target's current angular position and velocity from noisy monopulse observations. It maintains the "Outage Filter" logic (inflating $R$ when power is low) to coast through nulls.
* **Prediction**: It projects the estimated state forward in time over the horizon $H$ assuming constant velocity.
* **Control**: It feeds this predicted trajectory into the MPC core to solve for the optimal beam steering actions, and applies the first action $u_0$.

### 3. Oracle MPC Agent (`src/agents/oracle_mpc_agent.py`) [NEW]
This agent represents the theoretical mathematical upper bound (Oracle). It receives privileged ground-truth data from the simulation engine.
* **Estimation**: It queries the environment for the exact true 9D Singer state of the target (position, velocity, acceleration).
* **Prediction**: Using the exact Singer dynamic matrices, it mathematically propagates the expected 3D trajectory of the target over the horizon $H$, perfectly anticipating maneuvers. It converts this 3D trajectory into angular targets.
* **Control**: It feeds the perfect trajectory into the MPC core to solve for the optimal beam steering actions.

### 4. Benchmark Scripts [NEW]
We will create two scripts to run these agents on the `FixedRewardDipoleEnv`:
* `scripts/benchmark_ekf_mpc.py`
* `scripts/benchmark_oracle_mpc.py`

## User Review Required

> [!IMPORTANT]  
> The MPC will use `scipy.optimize.minimize` to solve the receding horizon problem at 100Hz. This is computationally intensive but standard for classical baseline evaluations. Do you approve of building the MPC solver and benchmarking the EKF-MPC vs Oracle-MPC?
