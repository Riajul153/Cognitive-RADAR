"""Gymnasium environment for Deep Reinforcement Learning-based antenna beam tracking.

Supports two action modes configurable at construction time:

- **parametric** (recommended): The agent outputs a compact 2D action vector
  that represents commanded beam steering angles.  The environment converts
  these to element-level phase shifts using conjugate-phase beamforming
  (array physics).  Two sub-modes are available:

    * *absolute*  — the action maps directly to :math:`(\\theta_{cmd}, \\phi_{cmd})`.
    * *incremental* — the action adds angular deltas
      :math:`(\\Delta\\theta, \\Delta\\phi)` to the current steered direction.

- **raw**: The agent outputs all ``N_elements`` phase shifts directly (legacy
  64-D action space).  Requires the agent to learn array physics from scratch.

In both modes the agent is fully independent — no external tracking controller
assists.  In parametric mode the *wave physics* (how steering angles map to
element phases) is baked into the environment structure, not into a helper
controller.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any

from ..antenna.array import UniformPlanarArray
from ..antenna.beamformer import Beamformer
from ..antenna.monopulse import MonopulseProcessor
from ..antenna.steering import angular_distance, compute_optimal_phases
from ..target.dynamics import AerialTarget
from ..utils.reward import RewardComputer, RewardWeights


class BeamTrackingEnv(gym.Env):
    """Gymnasium environment for adaptive phased-array beam tracking.

    The environment simulates an M×N Uniform Planar Array (UPA) tracking a
    randomly maneuvering aerial vehicle using the Singer dynamics model.
    The agent steers the beam to maximize the normalized array gain (RCS
    illumination) at the target direction.
    """

    metadata = {"render_modes": ["human"]}

    # ──────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        n_rows: int = 8,
        n_cols: int = 8,
        frequency_hz: float = 10.0e9,
        element_spacing: float = 0.5,
        target_config: dict[str, Any] | None = None,
        reward_config: dict[str, Any] | None = None,
        success_config: dict[str, Any] | None = None,
        dt: float = 0.01,
        episode_length: int = 500,
        action_mode: str = "parametric",
        parametric_type: str = "incremental",
        max_angular_step_deg: float = 2.0,
        elevation_max_deg: float = 60.0,
        azimuth_max_deg: float = 60.0,
        rng_seed: int | None = None,
    ):
        """Initializes the beam tracking environment.

        Args:
            n_rows: Number of antenna elements along x-axis.
            n_cols: Number of antenna elements along y-axis.
            frequency_hz: Radar operating frequency in Hz.
            element_spacing: Element spacing in wavelengths.
            target_config: Settings for AerialTarget initialization.
            reward_config: Settings for RewardComputer weights.
            success_config: Thresholds for lock/success metrics and optional
                early termination.
            dt: Environment timestep in seconds (10 ms default).
            episode_length: Number of timesteps per episode.
            action_mode: ``"parametric"`` or ``"raw"``.
            parametric_type: ``"absolute"`` or ``"incremental"`` (only used
                when *action_mode* is ``"parametric"``).
            max_angular_step_deg: Maximum angular step per timestep in degrees
                (incremental mode only).
            elevation_max_deg: Maximum elevation angle from boresight in
                degrees.
            azimuth_max_deg: Maximum azimuth angle from boresight in degrees.
            rng_seed: Random number seed.
        """
        super().__init__()

        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.dt = float(dt)
        self.episode_length = int(episode_length)
        self.current_step = 0

        # ── Action mode configuration ───────────────────────────────────
        self.action_mode = str(action_mode).lower().strip()
        assert self.action_mode in ("parametric", "raw", "codebook"), (
            f"action_mode must be 'parametric', 'raw', or 'codebook', got '{self.action_mode}'"
        )
        self.parametric_type = str(parametric_type).lower().strip()
        assert self.parametric_type in ("absolute", "incremental"), (
            f"parametric_type must be 'absolute' or 'incremental', "
            f"got '{self.parametric_type}'"
        )
        self.max_angular_step_rad = np.radians(float(max_angular_step_deg))
        self.elevation_max_rad = np.radians(float(elevation_max_deg))
        self.azimuth_max_rad = np.radians(float(azimuth_max_deg))

        # Cast to float in case YAML parses scientific notation as string
        frequency_hz = float(frequency_hz)
        element_spacing = float(element_spacing)

        # ── Antenna Array & Monopulse ───────────────────────────────────
        self.array = UniformPlanarArray(
            n_rows=self.n_rows,
            n_cols=self.n_cols,
            frequency_hz=frequency_hz,
            element_spacing_wavelengths=element_spacing,
        )
        self.beamformer = Beamformer(self.array)
        self.hpbw_rad = self.array.half_power_beamwidth
        
        # New: realistic radar sensor model
        sensor_cfg = target_config.get("sensor", {}) if target_config else {}
        self.snr_db = float(sensor_cfg.get("snr_db", 20.0))
        self.monopulse = MonopulseProcessor(self.array, snr_db=self.snr_db)

        # ── Target ──────────────────────────────────────────────────────
        t_cfg = target_config or {}
        self.target = AerialTarget(
            max_speed=float(t_cfg.get("max_speed", 300.0)),
            max_acceleration=float(t_cfg.get("max_acceleration", 50.0)),
            acceleration_std=float(t_cfg.get("acceleration_std", 30.0)),
            correlation_time=float(t_cfg.get("correlation_time", 2.0)),
            initial_range_min=float(t_cfg.get("initial_range_min", 5000.0)),
            initial_range_max=float(t_cfg.get("initial_range_max", 15000.0)),
            altitude_min=float(t_cfg.get("altitude_min", 1000.0)),
            altitude_max=float(t_cfg.get("altitude_max", 10000.0)),
            azimuth_range_deg=float(t_cfg.get("azimuth_range_deg", 120.0)),
            elevation_range_deg=float(t_cfg.get("elevation_range_deg", 60.0)),
            dt=self.dt,
            rng_seed=rng_seed,
        )

        # ── Success thresholds ──────────────────────────────────────────
        s_cfg = success_config or {}
        self.min_gain = float(s_cfg.get("min_gain", 0.85))
        self.max_error_deg = float(s_cfg.get("max_error_deg", 5.0))
        self.min_locked_fraction = float(s_cfg.get("min_locked_fraction", 0.8))
        self.enable_track_loss_termination = bool(
            s_cfg.get("enable_track_loss_termination", False)
        )
        self.track_loss_error_deg = float(s_cfg.get("track_loss_error_deg", 20.0))
        self.track_loss_steps = int(s_cfg.get("track_loss_steps", 100))

        # ── Reward ──────────────────────────────────────────────────────
        r_cfg = reward_config or {}
        weights = RewardWeights(
            gain_weight=float(r_cfg.get("gain_weight", 1.0)),
            lock_bonus=float(r_cfg.get("lock_bonus", 0.55)),
            lock_streak_bonus=float(r_cfg.get("lock_streak_bonus", 0.25)),
            low_gain_penalty=float(r_cfg.get("low_gain_penalty", 0.35)),
            smoothness_penalty=float(r_cfg.get("smoothness_penalty", 0.03)),
        )
        self.reward_computer = RewardComputer(
            hpbw_rad=self.hpbw_rad,
            weights=weights,
            lock_min_power=self.min_gain,
            gain_shaping_power=float(r_cfg.get("gain_shaping_power", 0.75)),
            streak_cap_steps=int(r_cfg.get("streak_cap_steps", 50)),
            smoothness_relief_when_locked=float(
                r_cfg.get("smoothness_relief_when_locked", 0.75)
            ),
            reward_clip_min=float(r_cfg.get("reward_clip_min", -1.5)),
            reward_clip_max=float(r_cfg.get("reward_clip_max", 2.5)),
            gamma=0.99,
        )

        self._locked_steps = 0
        self._consecutive_locked = 0
        self._consecutive_loss_steps = 0

        # ── Action Space ────────────────────────────────────────────────
        if self.action_mode == "parametric":
            # 2D: (elevation command, azimuth command) in [-1, 1]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
            )
        elif self.action_mode == "raw":
            # One phase per antenna element in [-1, 1] → [-π, π]
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.array.n_elements,),
                dtype=np.float32,
            )
        else: # codebook
            # Generate codebook: 10 elevation steps, 24 azimuth steps = 240 beams
            thetas = np.linspace(0.0, self.elevation_max_rad, 10)
            phis = np.linspace(-self.azimuth_max_rad, self.azimuth_max_rad, 24)
            self.codebook = []
            for t in thetas:
                for p in phis:
                    self.codebook.append((float(t), float(p)))
            self.action_space = spaces.Discrete(len(self.codebook))

        # ── Observation Space (10D Sensor-Based) ────────────────────────
        # 10 continuous features normalised to approx [-1.0, 1.0]:
        #  0  monopulse_err_el   (noisy sensor)
        #  1  monopulse_err_az   (noisy sensor)
        #  2  received_power     (noisy sensor, 0 to 1)
        #  3  Δpower             (current - prev received_power)
        #  4  beam_θ             (agent state, mapped to [-1, 1])
        #  5  beam_φ             (agent state, mapped to [-1, 1])
        #  6  Δbeam_θ            (current - prev beam_θ)
        #  7  Δbeam_φ            (current - prev beam_φ)
        #  8  prev_action_0      (agent state)
        #  9  prev_action_1      (agent state)
        # NO GROUND TRUTH TARGET ANGLES ARE EXPOSED.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32,
        )

        # ── Internal tracking variables ─────────────────────────────────
        self.last_gain = 0.0
        self.last_error = 0.0
        if self.action_mode == "codebook":
            self.last_action = 0
        else:
            self.last_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

        # Parametric mode: commanded beam direction (maintained across steps)
        self.beam_theta_cmd = 0.0
        self.beam_phi_cmd = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Action decoder
    # ──────────────────────────────────────────────────────────────────────

    def _decode_action(self, action: np.ndarray | int) -> np.ndarray:
        """Converts the agent's action into element-level phase shifts.

        In **parametric** mode the 2D action is decoded into commanded beam
        steering angles, which are then converted to conjugate-phase weights
        using known array physics.  In **raw** mode the action maps linearly
        to ``[-π, π]``. In **codebook** mode, it selects a pre-defined beam.

        Args:
            action: Raw action from the agent.

        Returns:
            Phase array of shape ``(n_elements,)`` in radians.
        """
        if self.action_mode == "raw":
            return action * np.pi
            
        if self.action_mode == "codebook":
            self.beam_theta_cmd, self.beam_phi_cmd = self.codebook[int(action)]
            return compute_optimal_phases(
                self.array, self.beam_theta_cmd, self.beam_phi_cmd
            )

        # ── Parametric modes ────────────────────────────────────────────
        if self.parametric_type == "absolute":
            # Map [-1, 1] → [0, elevation_max_rad] for elevation
            self.beam_theta_cmd = float(
                (action[0] + 1.0) * 0.5 * self.elevation_max_rad
            )
            # Map [-1, 1] → [-azimuth_max_rad, azimuth_max_rad] for azimuth
            self.beam_phi_cmd = float(action[1] * self.azimuth_max_rad)

        else:  # incremental
            # action ∈ [-1, 1] → angular delta ∈ [-max_step, +max_step]
            d_theta = float(action[0]) * self.max_angular_step_rad
            d_phi = float(action[1]) * self.max_angular_step_rad
            self.beam_theta_cmd = float(
                np.clip(self.beam_theta_cmd + d_theta, 0.0, self.elevation_max_rad)
            )
            self.beam_phi_cmd = float(
                np.clip(
                    self.beam_phi_cmd + d_phi,
                    -self.azimuth_max_rad,
                    self.azimuth_max_rad,
                )
            )

        # Convert commanded angles to element phases via conjugate-phase physics
        return compute_optimal_phases(
            self.array, self.beam_theta_cmd, self.beam_phi_cmd
        )

    # ──────────────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Builds and normalizes the 10-D sensor observation vector."""
        # 1. Monopulse radar measurements
        # Note: target true angles are used to compute the radar's received signals,
        # but the angles themselves are NOT put into the observation.
        t_theta, t_phi = self.target.get_angular_position()
        err_el, err_az, recv_power = self.monopulse.compute_error_signals(
            self.beamformer, t_theta, t_phi, self.target.rng
        )
        
        # 2. Derived temporal features
        delta_power = recv_power - self.last_gain
        
        # 3. Agent's own beam pointing state
        if self.action_mode == "parametric":
            b_theta = self.beam_theta_cmd
            b_phi = self.beam_phi_cmd
        else:
            b_theta, b_phi = self.beamformer.estimate_beam_direction()
            
        b_phi_wrapped = (b_phi + np.pi) % (2.0 * np.pi) - np.pi
        delta_b_theta = b_theta - getattr(self, "_last_b_theta", b_theta)
        delta_b_phi = b_phi - getattr(self, "_last_b_phi", b_phi)
        
        self._last_b_theta = b_theta
        self._last_b_phi = b_phi

        # 4. Normalization
        # Monopulse errors are normalized by HPBW so they are approx [-1, 1] in mainlobe
        norm_err_el = np.clip(err_el / self.hpbw_rad, -1.0, 1.0)
        norm_err_az = np.clip(err_az / self.hpbw_rad, -1.0, 1.0)
        
        # Power is already approx [0, 1]. Map to [-1, 1]
        norm_power = np.clip(2.0 * recv_power - 1.0, -1.0, 1.0)
        norm_delta_power = np.clip(delta_power * 10.0, -1.0, 1.0)  # amplify small changes
        
        # Beam angles mapped to [-1, 1]
        norm_b_theta = (b_theta / (np.pi / 4.0)) - 1.0
        norm_b_phi = b_phi_wrapped / np.pi
        
        # Beam angular deltas
        norm_delta_b_theta = np.clip(delta_b_theta / self.max_angular_step_rad, -1.0, 1.0)
        norm_delta_b_phi = np.clip(delta_b_phi / self.max_angular_step_rad, -1.0, 1.0)
        
        # Previous action (already in [-1, 1])
        # If raw mode (64 actions), we just take first 2 to keep obs space fixed, 
        # though this env is now strongly optimized for parametric mode.
        prev_a0 = float(self.last_action[0]) if isinstance(self.last_action, np.ndarray) and len(self.last_action) > 0 else 0.0
        prev_a1 = float(self.last_action[1]) if isinstance(self.last_action, np.ndarray) and len(self.last_action) > 1 else 0.0

        obs = np.array(
            [
                norm_err_el,
                norm_err_az,
                norm_power,
                norm_delta_power,
                norm_b_theta,
                norm_b_phi,
                norm_delta_b_theta,
                norm_delta_b_phi,
                prev_a0,
                prev_a1,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the environment for a new episode."""
        super().reset(seed=seed)

        if seed is not None:
            self.target.rng = np.random.RandomState(seed)

        self.current_step = 0
        self._locked_steps = 0
        self._consecutive_locked = 0
        self._consecutive_loss_steps = 0

        # Reset sub-modules
        self.target.reset()
        self.reward_computer.reset()

        # Initialize beamformer directly towards target to start with a lock
        t_theta, t_phi = self.target.get_angular_position()
        optimal_phases = compute_optimal_phases(self.array, t_theta, t_phi)
        self.beamformer.set_weights(optimal_phases)

        # Set commanded beam direction to target (both modes start locked)
        self.beam_theta_cmd = float(t_theta)
        self.beam_phi_cmd = float(t_phi)

        # Initial metrics
        self.last_gain = self.beamformer.compute_gain_at_target(t_theta, t_phi)
        self.last_error = 0.0
        if self.action_mode == "codebook":
            self.last_action = 0
        else:
            self.last_action = np.zeros(
                self.action_space.shape[0], dtype=np.float32
            )

        obs = self._get_obs()

        locked = self.last_gain >= self.min_gain and 0.0 <= self.max_error_deg
        info = {
            "target_pos": self.target.get_position(),
            "target_angles": (t_theta, t_phi),
            "beam_angles": (t_theta, t_phi),
            "normalized_gain": self.last_gain,
            "tracking_error_deg": 0.0,
            "tracking_locked": locked,
            "consecutive_locked_steps": 1 if locked else 0,
            "episode_locked_fraction": 1.0 if locked else 0.0,
        }
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Executes one environment step.

        Decodes the action, propagates target dynamics, computes tracking
        metrics, and assembles the reward.
        """
        self.current_step += 1
        self.last_action = action.copy() if hasattr(action, "copy") else action

        # ── 1. Decode & Apply Action ────────────────────────────────────
        phases = self._decode_action(action)
        self.beamformer.set_weights(phases)

        # ── 2. Propagate Target Dynamics ────────────────────────────────
        self.target.step()
        t_pos = self.target.get_position()
        t_theta, t_phi = self.target.get_angular_position()

        # ── 3. Compute Tracking Metrics ─────────────────────────────────
        gain = self.beamformer.compute_gain_at_target(t_theta, t_phi)
        self.last_gain = gain

        if self.action_mode == "parametric":
            b_theta = self.beam_theta_cmd
            b_phi = self.beam_phi_cmd
        else:
            b_theta, b_phi = self.beamformer.estimate_beam_direction()

        error_rad = angular_distance(t_theta, t_phi, b_theta, b_phi)
        self.last_error = error_rad
        error_deg = float(np.degrees(error_rad))

        # ── 4. Compute Dense Reward (PBRS) ──────────────────────────────
        # We pass received power to the reward computer instead of angular error
        is_first_step = (self.current_step == 1)
        reward, reward_info = self.reward_computer.compute(
            gain, phases, is_first_step=is_first_step
        )

        # ── 5. Lock / success bookkeeping ───────────────────────────────
        tracking_locked = gain >= self.min_gain and error_deg <= self.max_error_deg
        if tracking_locked:
            self._locked_steps += 1
            self._consecutive_locked += 1
            self._consecutive_loss_steps = 0
        else:
            self._consecutive_locked = 0
            self._consecutive_loss_steps += 1

        locked_fraction = self._locked_steps / max(self.current_step, 1)

        # ── 6. Termination conditions ───────────────────────────────────
        terminated = False
        if (
            self.enable_track_loss_termination
            and self._consecutive_loss_steps >= self.track_loss_steps
        ):
            terminated = True
        truncated = self.current_step >= self.episode_length

        # ── 7. Assemble observation & info ──────────────────────────────
        obs = self._get_obs()
        episode_success = locked_fraction >= self.min_locked_fraction
        info = {
            "target_pos": t_pos,
            "target_angles": (t_theta, t_phi),
            "beam_angles": (b_theta, b_phi),
            "normalized_gain": gain,
            "tracking_error_deg": error_deg,
            "tracking_locked": tracking_locked,
            "consecutive_locked_steps": self._consecutive_locked,
            "episode_locked_fraction": locked_fraction,
            "episode_success": episode_success if truncated or terminated else False,
            "terminated_track_loss": terminated and self.enable_track_loss_termination,
            "reward_breakdown": {
                "gain_reward": reward_info.gain_reward,
                "lock_bonus": reward_info.lock_bonus,
                "lock_streak_bonus": reward_info.lock_streak_bonus,
                "low_gain_penalty": reward_info.low_gain_penalty,
                "improvement_bonus": reward_info.improvement_bonus,
                "smoothness_penalty": reward_info.smoothness_penalty,
            },
        }
        return obs, float(reward), terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    # Dashboard / rendering helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_dashboard_state(self) -> dict[str, Any]:
        """Serializable snapshot for live dashboard streaming (SubprocVecEnv-safe)."""
        t_theta, t_phi = self.target.get_angular_position()
        if self.action_mode == "parametric":
            b_theta = self.beam_theta_cmd
            b_phi = self.beam_phi_cmd
        else:
            b_theta, b_phi = self.beamformer.estimate_beam_direction()
        t_pos = self.target.get_position()

        theta_grid = np.linspace(0.0, np.pi / 2.0, 15)
        phi_grid = np.linspace(0.0, 2.0 * np.pi, 30)
        theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")
        power_grid = self.beamformer.compute_beam_pattern(theta_mesh, phi_mesh)
        phases_2d = np.angle(self.beamformer.get_current_weights()).reshape(
            (self.n_rows, self.n_cols)
        )

        return {
            "step": int(self.current_step),
            "episode_length": int(self.episode_length),
            "target_pos": t_pos.tolist(),
            "target_angles": [float(t_theta), float(t_phi)],
            "beam_angles": [float(b_theta), float(b_phi)],
            "gain": float(self.last_gain),
            "error_deg": float(np.degrees(self.last_error)),
            "phases": phases_2d.tolist(),
            "beam_pattern": {
                "theta": theta_grid.tolist(),
                "phi": phi_grid.tolist(),
                "power": power_grid.tolist(),
            },
        }

    def render(self) -> None:
        """Renders current tracking state to terminal console."""
        t_theta, t_phi = self.target.get_angular_position()
        if self.action_mode == "parametric":
            b_theta = self.beam_theta_cmd
            b_phi = self.beam_phi_cmd
        else:
            b_theta, b_phi = self.beamformer.estimate_beam_direction()
        print(
            f"Step: {self.current_step:3d} | "
            f"Target: θ={np.degrees(t_theta):5.1f}°, φ={np.degrees(t_phi):6.1f}° | "
            f"Beam: θ={np.degrees(b_theta):5.1f}°, φ={np.degrees(b_phi):6.1f}° | "
            f"Gain: {self.last_gain:5.3f} | "
            f"Error: {np.degrees(self.last_error):5.2f}°"
        )
