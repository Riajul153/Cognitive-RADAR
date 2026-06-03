# Phase 15: Realistic Sensor Model & Potential-Based Reward Shaping

## Problem Statement

Our current agent receives **ground-truth target angles** (θ_target, φ_target) and angular rates directly in the observation vector. This is privileged information that a real radar would never have. A reviewer will flag this as the agent "just copying 2 numbers" rather than learning a genuine tracking policy.

Additionally, our reward function uses `angular_error_rad` computed from ground-truth — while acceptable for training in simulation, combining this with oracle observations creates a system where the agent never truly learns to interpret noisy sensor feedback.

## Design Philosophy

> **"Bake in the physics we know, but without cheating or giving privileged information."**

We will embed physics at three levels:
1. **Action Space** (already done): Parametric (θ, φ) → phases via steering vector
2. **Observation Space** (NEW): Monopulse processing uses known array geometry to extract angular error from received signals — this is physics-based signal processing, not cheating
3. **Reward Shaping** (NEW): Potential-Based Reward Shaping (Ng et al., 1999) using array gain as the potential function — theoretically guaranteed to preserve optimal policy

---

## Proposed Changes

### Sensor Model: Monopulse Processor

#### [NEW] `src/antenna/monopulse.py`

A `MonopulseProcessor` class that simulates realistic monopulse angle estimation:

**How monopulse works (physics):**
- A monopulse radar forms three simultaneous beams: a **Sum beam (Σ)** and two **Difference beams (Δ_el, Δ_az)**
- The Sum beam is the standard pencil beam (our current beamformer output)
- The Difference beams have a null at boresight and opposite lobes on each side
- The ratio **Δ/Σ** produces a smooth, approximately linear error signal within the main lobe
- Outside the main lobe, this signal becomes **noisy and unreliable**

**Implementation:**
```python
class MonopulseProcessor:
    def __init__(self, array: UniformPlanarArray, snr_db: float = 20.0):
        """
        Args:
            array: The antenna array geometry (known physics).
            snr_db: Signal-to-noise ratio in dB. Controls measurement noise.
        """
    
    def compute_error_signals(
        self, 
        beamformer: Beamformer,
        target_theta: float,  # TRUE target direction (sim-internal)
        target_phi: float,
        rng: np.random.Generator,
    ) -> tuple[float, float, float]:
        """Returns (delta_el, delta_az, received_power).
        
        - delta_el, delta_az: Noisy angular error estimates (rad)
          Accurate within HPBW, degrades to noise outside mainlobe.
        - received_power: Σ-beam power + thermal noise (measurable).
        """
```

**Key physics details:**
- Δ_el pattern: Weight the upper half of the UPA with +1, lower half with -1, then compute AF
- Δ_az pattern: Weight the left half with +1, right half with -1
- Error signal = Re(Δ · Σ*) / |Σ|² (normalized, gives signed angular offset)
- Add Gaussian noise scaled by 1/SNR: `noise_std = HPBW / (2 * sqrt(SNR_linear))`
- Outside the mainlobe (|error| > HPBW), the Δ/Σ ratio saturates and noise dominates

> [!IMPORTANT]
> The monopulse processor uses the array geometry (known physics) to construct the Σ and Δ beams. This is standard radar engineering, not privileged information. The TRUE target direction is only used internally by the simulator to compute what signal the array would actually receive — the agent never sees it.

---

### Redesigned Observation Space

#### [MODIFY] `src/environment/beam_tracking_env.py`

**Old observation (9D — contains oracle info):**
```
[target_θ, target_φ, target_dθ, target_dφ, beam_θ, beam_φ, err_θ, err_φ, prev_gain]
 ^^^^^^^^   ^^^^^^^   ^^^^^^^^^   ^^^^^^^^^
 ORACLE     ORACLE    ORACLE      ORACLE
```

**New observation (10D — all realistically measurable):**
```
[monopulse_err_el, monopulse_err_az, received_power, Δpower, beam_θ, beam_φ, Δbeam_θ, Δbeam_φ, prev_action_0, prev_action_1]
 ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^  ^^^^^^  ^^^^^^  ^^^^^^^  ^^^^^^^  ^^^^^^^^^^^^   ^^^^^^^^^^^^
 SENSOR (noisy)     SENSOR (noisy)    SENSOR (noisy)  DERIVED  KNOWN   KNOWN   KNOWN    KNOWN    KNOWN          KNOWN
```

| # | Feature | Source | Description |
|---|---|---|---|
| 0 | `monopulse_err_el` | Monopulse Δ_el/Σ | Noisy elevation error estimate. Accurate within mainlobe, saturates outside. |
| 1 | `monopulse_err_az` | Monopulse Δ_az/Σ | Noisy azimuth error estimate. |
| 2 | `received_power` | Σ-beam output | Proportional to gain × RCS / R⁴. Normalized to [0, 1]. |
| 3 | `Δpower` | Current - previous | Rate of change of received power. Tells agent if it's improving. |
| 4 | `beam_θ` | Agent's own state | Where the beam is currently pointed (elevation). |
| 5 | `beam_φ` | Agent's own state | Where the beam is currently pointed (azimuth). |
| 6 | `Δbeam_θ` | Current - previous | How much the beam moved last step (elevation). |
| 7 | `Δbeam_φ` | Current - previous | How much the beam moved last step (azimuth). |
| 8 | `prev_action_0` | Agent's own action | Last action taken (for temporal context). |
| 9 | `prev_action_1` | Agent's own action | Last action taken. |

> [!NOTE]
> Every feature is either a noisy sensor measurement or the agent's own internal state. **Zero ground-truth target information leaks into the observation.**

**Why this is sufficient for tracking:**
- When the beam is ON target: monopulse errors are small and accurate → agent learns to make small corrective actions
- When the beam DRIFTS off target: monopulse errors grow, received power drops → agent learns to chase the gradient
- When the beam LOSES the target (outside mainlobe): monopulse signals saturate, power collapses → agent must learn a search/recovery strategy
- Temporal features (Δpower, Δbeam, prev_action) give the agent a sense of dynamics without oracle angular rates

---

### Potential-Based Reward Shaping (PBRS)

#### [MODIFY] `src/utils/reward.py`

Following **Ng, Harada & Russell (ICML 1999)**, we define a potential function Φ(s) and add a shaping term:

```
F(s, s') = γ · Φ(s') − Φ(s)
```

This is **guaranteed** to preserve the optimal policy while accelerating learning.

**Potential function design:**
```python
Φ(s) = α · received_power(s)
```

Where `received_power` is the normalized Σ-beam output power (proportional to array gain at the actual target direction). This is **measurable** — it's the actual signal the radar receives.

**Why this works:**
- When the agent improves its beam pointing → power increases → Φ(s') > Φ(s) → positive shaping reward
- When the agent worsens its pointing → power decreases → negative shaping
- This encodes the physics intuition "higher power = better" without revealing WHERE the target is

**New reward structure:**
```
R_total = R_base + F_shaping

R_base  = w_power · received_power^p          # Base: maximize received signal
        + w_lock · I(power > lock_threshold)   # Bonus: sustained high-power lock
        + w_streak · min(streak/S_cap, 1)      # Bonus: sustained lock streak
        - w_smooth · phase_jitter              # Penalty: erratic phase changes

F_shaping = γ · α · power(s') − α · power(s)  # PBRS: reward power improvement
```

> [!IMPORTANT]
> Notice what's **NOT** in the reward anymore:
> - `angular_error_rad` — removed (requires ground truth)
> - `error_penalty` — removed (was based on ground truth error)
> - `side_lobe_penalty` based on error — removed (was based on ground truth)
> 
> Everything is now based on **received power**, which is physically measurable. The reward tells the agent "maximize the signal you receive" — it must figure out that this means "point a focused beam at the target" on its own.

**Lock condition (also measurable):**
```python
locked = received_power >= lock_power_threshold
```
No longer uses angular error (which requires ground truth).

---

### Noise & Robustness

#### [MODIFY] `src/environment/beam_tracking_env.py`

Add configurable noise parameters:
```yaml
sensor:
  snr_db: 20.0              # Base SNR (higher = cleaner monopulse signals)
  snr_jitter_db: 3.0        # Random SNR variation per step (fading)
  monopulse_bias_std: 0.001  # Small systematic bias in monopulse cal (rad)
```

**Domain randomization** (for robustness):
- Randomize SNR at episode start within a range (e.g., 15–25 dB)
- Add small random monopulse calibration bias per episode
- This forces the agent to learn a policy robust to sensor imperfections

---

## Verification Plan

### Automated Tests

1. **Unit test `MonopulseProcessor`:**
   - Verify Δ/Σ error signal is approximately linear within HPBW
   - Verify error signal saturates/becomes noisy outside HPBW
   - Verify noise scaling with SNR

2. **Sanity check observation space:**
   - Run environment with new obs, print observations, verify no ground-truth leakage
   - Verify monopulse errors correlate with (but are noisier than) true angular errors

3. **Retrain Parametric SAC:**
   - Train with new observation space + PBRS reward for 3M steps
   - Compare learning curves: old (oracle) vs. new (sensor) observations
   - The agent should still converge but potentially slower (noisy observations are harder)
   - Benchmark against EKF baseline (which also uses monopulse measurements)

4. **PBRS Verification:**
   - Train with and without the shaping term F(s,s')
   - PBRS should accelerate convergence without changing final performance (Ng et al. guarantee)

### Manual Verification
- Inspect TensorBoard: monopulse error signals should be noisier than ground-truth errors
- Verify the agent develops a "search then track" behavior when it loses the target

## Open Questions

> [!IMPORTANT]
> 1. **SNR range**: What SNR should we train at? 20 dB is a reasonable radar scenario. Should we also test degraded conditions (10 dB) for a robustness study?
> 2. **Observation history**: Should we add a short history buffer (e.g., last 3 monopulse readings) via frame-stacking, or rely on the temporal features (Δpower, prev_action) instead? Frame stacking would give the agent implicit velocity estimation.
> 3. **Should we keep angular_error in the reward for training only?** The reward is only used during training (not deployment). Using sim-internal gain at the true target direction for the reward is standard practice in sim-to-real RL. The critical fix is the observation space. Alternatively, we can use received power (which equals gain × path effects) as the sole reward signal for maximum purity.
