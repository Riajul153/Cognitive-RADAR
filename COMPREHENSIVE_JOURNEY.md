# Comprehensive Journey: Deep Reinforcement Learning for Adaptive Antenna Array Beamforming

## Executive Summary
This document chronicles the complete experimental journey of developing a Deep Reinforcement Learning (DRL) agent for controlling phased array radar beams. We successfully transitioned from a classical mathematical control problem (which failed under real-world hardware constraints) to a learning-based solution that organically understands antenna topology. We scaled from 64 elements to 256 elements, solved partial observability with temporal frame stacking, and defeated physical hardware nulls using a custom Masked Soft Actor-Critic (SAC) Outage Filter.

---

## Experiment 1: Action Space Representation
**Goal:** Determine the optimal action space representation for the RL agent to steer the radar beam.
**Setup:** We simulated an $8 \times 8$ Uniform Planar Array (UPA) with isotropic elements. The target moved according to a highly evasive Singer maneuver model.
**Models Tested:**
1. **Raw Element Control (64D):** The agent outputs 64 continuous phase shifts directly.
2. **Physics-Informed Parametric Control (2D):** The agent outputs $(\Delta\theta, \Delta\phi)$ and the environment computes the 64 conjugate phases mathematically.

**Results:**
* The **Raw 64D** action space struggled immensely with the credit assignment problem. Normalized gain hovered around 40% after millions of steps.
* The **Parametric 2D** action space converged to near-optimal tracking (Gain $> 0.95$) in less than 500,000 steps.

**Why it worked:** By baking the underlying array wave physics into the environment, we drastically reduced the dimensionality of the exploration space. The RL agent only needed to learn kinematics (how to track a target) rather than electromagnetics (how phase shifts combine to form a beam).

---

## Experiment 2: Reward Shaping
**Goal:** Define the mathematical reward function required for successful convergence.
**Setup:** Using the 2D Parametric action space.
**Models Tested:**
1. **Sparse Reward:** $+1$ if gain $> 0.85$, else $0$.
2. **Dense Potential-Based Reward Shaping (PBRS):** A combination of absolute gain, error penalties, lock streak bonuses, and improvement bonuses.

**Results:**
* **Sparse Reward:** Failed to converge. The probability of randomly steering exactly onto a fast-moving target was near zero.
* **Dense PBRS:** Reached 100% lock rate.

**Nitty-Gritty Details:** We had to implement a `gain_shaping_power` parameter ($g^{0.75}$) to make the gradient steeper near the edges of the beam, driving the agent sharply toward the center. We also implemented `smoothness_relief_when_locked`, which removed penalties for small jitter when the agent was already locked, allowing it to maintain the track more stably.

---

## Experiment 3: Scaling to Massive Arrays
**Goal:** Test if the RL agent could control a substantially larger array without retraining the fundamental architecture.
**Setup:** Increased the array from $8 \times 8$ (64 elements) to $16 \times 16$ (256 elements).
**Results:**
* The 256-element array possesses a much narrower beamwidth (pencil beam), requiring significantly higher tracking precision.
* The exact same Parametric 2D SAC architecture converged successfully. While it took slightly longer (approx. 600k steps) due to the tighter precision requirements, it ultimately achieved a higher absolute theoretical gain because of the massive 256-element directivity.

---

## Experiment 4: Partial Observability & Memory
**Goal:** Train the agent using strictly realistic, noisy hardware sensors rather than ground-truth (Oracle) target coordinates.
**Setup:** The agent only receives a 10D observation vector: noisy monopulse error signals, received power, current beam direction, and temporal derivatives.
**Models Tested:**
1. **Oracle:** Given True 9D Singer state (pos, vel, acc).
2. **Standard SAC (No Memory):** Given 10D noisy observation for the current frame only.
3. **Frame-Stack SAC:** Given the 10D noisy observation stacked over the previous 8 frames (80 ms memory).

**Results:**
* The memoryless SAC fluctuated heavily, unable to infer target velocity or acceleration from a single noisy monopulse snapshot.
* **Frame Stacking completely solved this.** By providing an 8-frame temporal window, the SAC network implicitly calculated the derivatives (velocity, acceleration) and filtered the sensor noise, matching Oracle performance.

---

## Experiment 5: The Dipole Hardware Nulls
**Goal:** Introduce physical hardware topology. Real antennas are not isotropic. We replaced the elements with z-oriented half-wave dipoles, which have a total radiation null at boresight ($\theta=0$ and $\theta=180^\circ$).
**Setup:** 8x8 Dipole UPA. Target frequently crosses boresight.

**Results (The Catastrophe):**
* Standard SAC trained well initially but suffered **Catastrophic Forgetting** around 1.5M steps.
* **Why it failed:** When the target entered the null, the radar received pure noise. The 8-frame stack filled entirely with garbage. The Critic network received massive, uncorrelated Temporal Difference (TD) penalties. This variance completely corrupted the Q-values, which then poisoned the Actor network, permanently destroying the learned policy.

---

## Experiment 6: Hardware-Aware Solutions
We tested two separate paradigms for defeating the hardware nulls.

### Solution A: Topology Design (Crossed-Dipoles)
* **What we did:** We modified the hardware, replacing the single dipoles with a Crossed-Dipole arrangement (two orthogonal dipoles). We expanded the agent's action space to output a continuous polarization mixing coefficient $\alpha \in [0, 1]$.
* **Result:** Highly successful. The agent learned to dynamically swap polarizations as the target approached a null, maintaining 100% signal integrity.

### Solution B: Algorithmic Outage Filtering (Masked SAC)
* **What we did:** We implemented a custom subclass of SB3's SAC. We added an *Outage Filter*: if the raw received power in the observation dropped below 15%, we generated a binary mask and mathematically zeroed out the Critic's TD-error for that specific transition.
* **Result:** **100% Success Rate.** The neural network essentially "held its breath" through the blind spot, ignoring the pure noise gradients and coasting through the null using its frame stack.

---

## Final Benchmark: DRL vs Classical State-of-the-Art
To prove the superiority of the Masked SAC agent, we benchmarked it against classical Model Predictive Control (MPC) utilizing a receding horizon (20 steps) and SciPy SLSQP optimization.

| Model | Success Rate | Mean Gain | Mean Tracking Error |
| :--- | :---: | :---: | :---: |
| **EKF-MPC** (No Privileged Info) | 0.0% | 0.007 | 17.98° |
| **Oracle-MPC** (True 9D Target State) | 0.0% | 0.065 | 0.64° |
| **Masked SAC** (Proposed) | **100.0%** | **0.864** | **1.28°** |

### The Breakthrough Discovery
Why did the mathematical Oracle, which had perfect knowledge of the target's position and maneuvers, fail completely?
* Classical MPC optimizes for **kinematic alignment** (Euclidean angular distance). The Oracle achieved a phenomenal tracking error of just 0.64°. However, because it pointed perfectly at the target, and the target was located within the dipole's hardware null, the radiated power was zero!
* The DRL agent optimizes for **physics** (array gain). The SAC agent learned the topology of the hardware and learned to **intentionally misalign** the beam (increasing the error to 1.28°). By purposefully pointing slightly off-target, it caught the target on the side-lobes of the radiation pattern, maximizing the received power and maintaining the track.

**Conclusion:** Deep Reinforcement Learning structurally defeats classical optimization in physical environments because it organically internalizes the complex, non-isotropic physics of the hardware platform.
