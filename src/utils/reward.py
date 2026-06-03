"""
Dense reward functions for the beam tracking RL environment.

Design goals:
    - Primary objective: lock the beam on the target (high received power)
    - Reward sustained lock (streak bonus), not one-off spikes
    - Bounded per-step reward to limit exploitation / reward hacking
    - No unconditional per-step bonuses (removed alive bonus)
    - Potential-Based Reward Shaping (PBRS) to guarantee optimal policy preservation
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class RewardWeights:
    """Weights for each reward component (all non-negative)."""

    gain_weight: float = 1.0
    lock_bonus: float = 0.55
    lock_streak_bonus: float = 0.25
    low_gain_penalty: float = 0.35
    smoothness_penalty: float = 0.03


@dataclass
class RewardInfo:
    """Detailed breakdown of reward components for logging/debugging."""

    total: float = 0.0
    gain_reward: float = 0.0
    lock_bonus: float = 0.0
    lock_streak_bonus: float = 0.0
    low_gain_penalty: float = 0.0
    improvement_bonus: float = 0.0  # PBRS shaping term
    smoothness_penalty: float = 0.0
    raw_gain: float = 0.0
    tracking_locked: bool = False
    consecutive_locked_steps: int = 0


class RewardComputer:
    """Computes dense, lock-prioritized rewards for beam tracking.

    Uses Potential-Based Reward Shaping (PBRS) following Ng et al., 1999:
        F(s, s') = γ * Φ(s') - Φ(s)
    
    where the potential function Φ(s) is proportional to the received power.
    This guarantees that the optimal policy is preserved while accelerating learning.

    Per-step base reward:
        R_base = w_power * (received_power ^ p)
               + w_lock * I(locked)
               + w_streak * min(streak / S_cap, 1)
               - w_smooth * smoothness * (1 - relief * I(locked))
    
    Total reward:
        R_total = R_base + F_shaping

    Locked condition: received_power >= lock_min_power.
    NO ground truth angular error is used in this reward function.
    """

    def __init__(
        self,
        hpbw_rad: float,
        weights: RewardWeights | None = None,
        lock_min_power: float = 0.85,
        gain_shaping_power: float = 0.75,
        streak_cap_steps: int = 50,
        smoothness_relief_when_locked: float = 0.75,
        max_phase_change: float = np.pi,
        reward_clip_min: float = -1.5,
        reward_clip_max: float = 2.5,
        gamma: float = 0.99,  # RL discount factor
        potential_alpha: float = 1.0,  # Scaling for potential function
    ):
        self.hpbw_rad = float(hpbw_rad)
        self.weights = weights or RewardWeights()
        
        # New: purely power-based lock condition
        self.lock_min_power = float(lock_min_power)
        
        self.gain_shaping_power = float(gain_shaping_power)
        self.streak_cap_steps = int(streak_cap_steps)
        self.smoothness_relief_when_locked = float(smoothness_relief_when_locked)
        self.max_phase_change = float(max_phase_change)
        self.reward_clip_min = float(reward_clip_min)
        self.reward_clip_max = float(reward_clip_max)
        
        # PBRS parameters
        self.gamma = float(gamma)
        self.potential_alpha = float(potential_alpha)

        self._previous_phases: np.ndarray | None = None
        self._previous_potential: float = 0.0
        self._temp_streak: int = 0

    def reset(self) -> None:
        """Reset internal state (phase history, potential history) for a new episode."""
        self._previous_phases = None
        self._previous_potential = 0.0
        self._temp_streak = 0

    def compute(
        self, 
        received_power: float, 
        current_phases: np.ndarray,
        is_first_step: bool = False,
    ) -> tuple[float, RewardInfo]:
        """Computes the step reward based on received power.

        Args:
            received_power: Normalised power ∈ [0, 1] received from the target.
            current_phases: Current antenna phase weights (rad).
            is_first_step: True if this is the very first step of the episode.

        Returns:
            Tuple of (total_reward, reward_info)
        """
        received_power = float(np.clip(received_power, 0.0, 1.0))
        info = RewardInfo(raw_gain=received_power)

        # 1. Base Power Reward
        info.gain_reward = self.weights.gain_weight * (received_power ** self.gain_shaping_power)
        r_base = info.gain_reward

        # 2. Lock & Streak Bonuses
        info.tracking_locked = received_power >= self.lock_min_power
        
        if info.tracking_locked:
            self._temp_streak += 1
            info.consecutive_locked_steps = self._temp_streak
            info.lock_bonus = self.weights.lock_bonus
            
            streak_ratio = min(info.consecutive_locked_steps / self.streak_cap_steps, 1.0)
            info.lock_streak_bonus = self.weights.lock_streak_bonus * streak_ratio
            
            r_base += info.lock_bonus + info.lock_streak_bonus
        else:
            self._temp_streak = 0
            info.consecutive_locked_steps = 0
            
            # Penalise very low power
            if received_power < 0.1:
                info.low_gain_penalty = self.weights.low_gain_penalty
                r_base -= info.low_gain_penalty

        # 3. Smoothness Penalty
        if self._previous_phases is not None:
            # Phase difference wrapped to [-pi, pi]
            phase_diff = (current_phases - self._previous_phases + np.pi) % (2.0 * np.pi) - np.pi
            mean_abs_diff = float(np.mean(np.abs(phase_diff)))
            
            relief = self.smoothness_relief_when_locked if info.tracking_locked else 0.0
            smoothness_factor = (mean_abs_diff / self.max_phase_change) * (1.0 - relief)
            
            info.smoothness_penalty = self.weights.smoothness_penalty * smoothness_factor
            r_base -= info.smoothness_penalty

        self._previous_phases = current_phases.copy()

        # 4. Potential-Based Reward Shaping (PBRS)
        current_potential = self.potential_alpha * received_power
        
        if is_first_step:
            # F = 0 on the first step since there's no transition
            f_shaping = 0.0
            info.improvement_bonus = 0.0
        else:
            f_shaping = self.gamma * current_potential - self._previous_potential
            info.improvement_bonus = f_shaping
            
        self._previous_potential = current_potential

        # Total reward
        r_total = r_base + f_shaping
        
        # Clip to prevent extreme values (e.g. at boundaries or due to noise)
        r_total = float(np.clip(r_total, self.reward_clip_min, self.reward_clip_max))
        info.total = r_total

        return r_total, info
