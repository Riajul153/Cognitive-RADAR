# Deep Reinforcement Learning for Hardware-Aware Phased Array Radar Beam Tracking: Defeating Classical Oracles

**Target Journal:** IEEE Transactions on Aerospace and Electronic Systems (TAES)  
**Estimated Impact Factor:** ~5.7 (Q1)  
**Scope Match:** Perfect. TAES is the premier journal for target tracking, phased array radar systems, and aerospace control algorithms.

---

## Abstract
Traditional target tracking for phased array radars relies on classical estimators, such as the Extended Kalman Filter (EKF), coupled with optimal control theories like Model Predictive Control (MPC). These approaches are mathematically formulated to minimize kinematic tracking error (Euclidean angular distance). However, physical antenna topologies, such as half-wave dipoles, possess inherent non-isotropic radiation patterns and hardware nulls. When a target maneuvers through a physical null, minimizing angular distance fundamentally results in zero received signal power, causing catastrophic tracking failure. In this paper, we propose a Deep Reinforcement Learning (DRL) paradigm utilizing a Masked Soft Actor-Critic (Masked SAC) agent that learns the nonlinear hardware physics purely from raw noisy observations. By maintaining a temporal frame-stack and utilizing a custom critic-loss outage filter, the Masked SAC agent learns to intentionally misalign the beam to avoid physical nulls, maintaining track lock via array side-lobes. Experimental results demonstrate that while state-of-the-art Model Predictive Control equipped with Oracle state knowledge fails entirely (0.0% success rate) in the presence of nulls, our proposed hardware-aware DRL approach achieves a 100.0% tracking success rate with an average normalized array gain of 0.864.

---

## I. Introduction
The tracking of highly dynamic aerial targets has traditionally been solved using analytical models. The Extended Kalman Filter (EKF) paired with a Singer acceleration model forms the backbone of state estimation, while Proportional Navigation or Model Predictive Control (MPC) is used to steer the radar beam. 

Classical control formulates beam tracking as an error-minimization problem: the controller steers the beam to exactly intersect the target's predicted trajectory. However, this mathematical abstraction assumes an isotropic (perfect) radiation pattern. Real-world physical arrays, constructed from elements like half-wave dipoles, exhibit deep radiation nulls at specific angles (e.g., at boresight). When a target crosses these nulls, the classical controller perfectly points the beam at the target, inadvertently hitting it with zero radiated power.

In this work, we demonstrate that model-free Deep Reinforcement Learning (DRL) circumvents this failure mode. Because DRL optimizes a reward function based on *received power* (array gain) rather than *kinematic error*, it organically learns the physical topology of the antenna hardware.

## II. System Model
### A. Antenna Array and Hardware Nulls
We model an $8 \times 8$ Uniform Planar Array (UPA) operating at 10 GHz (X-band) with $\lambda/2$ spacing. To accurately reflect real-world hardware constraints, the array elements are modeled as z-oriented half-wave dipoles. The element factor imposes a total radiation null at the boresight ($\theta=0$ and $\theta=180^\circ$):
$$ E(\theta) = \frac{\cos\left(\frac{\pi}{2}\cos\theta\right)}{\sin\theta} $$

### B. Target Dynamics (Singer Model)
The target operates in 3D space, modeled using the Singer acceleration model, allowing for highly evasive random maneuvers (up to $50$ m/s$^2$ acceleration). The radar receives normalized 10D observations containing noisy monopulse error signals, received power, and temporal derivatives.

---

## III. Classical Baseline Formulations
To rigorously benchmark our proposed system against the state-of-the-art, we developed two Model Predictive Controllers (MPC) utilizing a 20-step (0.2s) receding horizon and Sequential Least Squares Programming (SLSQP).

1. **EKF-MPC (Non-Privileged)**: Uses an EKF to estimate the target state from noisy monopulse observations, feeding constant-velocity predictions into the MPC solver.
2. **Oracle-MPC (Privileged)**: Represents the absolute theoretical upper bound of classical control. The Oracle is granted perfect access to the simulation engine, retrieving the true 9D state of the target and the exact Singer dynamic matrices to project a perfect future trajectory.

---

## IV. Proposed DRL Framework (Masked SAC)
### A. Continuous Control and Frame Stacking
We formulate the beam tracking problem as a continuous Markov Decision Process (MDP). A Soft Actor-Critic (SAC) agent outputs continuous 2D actions $(\Delta\theta, \Delta\phi)$ to incrementally steer the beam. To provide temporal memory, observations are stacked over 8 frames (80 ms buffer).

### B. The Outage Filter (Critic Masking)
When the target enters a physical null, the 10D observation vector deteriorates into pure noise. This noise causes massive, uncorrelated temporal difference (TD) errors in the SAC Critic network, leading to catastrophic forgetting. We introduce an *Outage Filter*: whenever the received power drops below 15%, the TD-error for that specific transition is mathematically masked (zeroed). This prevents the destruction of the Actor's policy, allowing the neural network to "coast" through the blind spot using its frame-stack memory.

---

## V. Experimental Results
All agents were evaluated across 50 episodes (500 steps each). Tracking success is defined strictly as maintaining a normalized array gain $\ge 0.85$ and angular error $\le 5.0^\circ$ for at least 80% of the episode.

| Agent Architecture | Tracking Success Rate | Mean Tracking Error | Mean Array Gain |
| :--- | :---: | :---: | :---: |
| EKF-MPC (No Privileged Data) | 0.0 % | 17.98° | 0.007 |
| Oracle-MPC (Perfect 9D State) | 0.0 % | **0.64°** | 0.065 |
| **Masked SAC (Proposed)** | **100.0 %** | 1.28° | **0.864** |

### A. Discussion on the Failure of the Oracle
The results highlight a fundamental flaw in purely kinematic control theory when applied to physical RF systems. The Oracle-MPC achieved a spectacular mean tracking error of **0.64°**, perfectly aligning the beam with the target. However, because the target frequently maneuvered through the dipole's boresight null, perfectly pointing at the target resulted in near-zero received power (Gain = 0.065), leading to a 0% success rate.

### B. DRL Discovers Hardware Topology
Conversely, our proposed Masked SAC agent achieved a **100% success rate** with a mean gain of **0.864**. The agent actively learned to **intentionally misalign** the beam (maintaining a slightly higher tracking error of 1.28°). By purposefully pointing off-center when the target was in a hardware null, the agent caught the target on the side-lobes of the radiation pattern, successfully maximizing received power and maintaining the track lock.

---

## VI. Conclusion
We presented a Masked Soft Actor-Critic algorithm for phased array radar beam tracking. Our experiments definitively prove that while mathematical target-tracking oracles fail when faced with non-isotropic hardware topologies, Deep Reinforcement Learning naturally assimilates hardware constraints, intentionally misaligning the beam to exploit side-lobes and defeat the nulls. Future work will investigate Joint Embedding Predictive Architectures (JEPA) for planning over extended temporal horizons.
