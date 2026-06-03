# Literature Review: DRL for Adaptive Beamforming

## Novelty Assessment

> [!IMPORTANT]
> Our **Parametric SAC** architecture — where the RL agent outputs 2D steering angles (θ, φ) that are analytically expanded to N-dimensional phase shifts via the array manifold — is **not present** in the existing literature. Current approaches either use discrete codebooks (DQN) or output raw phase shifts directly (DDPG/SAC), both of which suffer from fundamental scalability limitations.

> [!WARNING]
> **Methodological Concern:** Our observation space currently provides ground-truth target angles (θ_target, φ_target) and angular rates directly to the agent. A reviewer will flag this as unrealistic — in a real radar, you would only have noisy monopulse error signals, received power, and Doppler. This should be addressed before submission.

---

## Category 1: Discrete RL (DQN) for Beam Management

These papers use DQN or bandit methods to select beams from a predefined codebook. They demonstrate the quantization limitation we proved empirically.

| Paper | Venue | Year | DOI/arXiv |
|---|---|---|---|
| Mismar et al., "Deep RL for 5G Networks: Joint Beamforming, Power Control, and Interference Coordination" | IEEE Trans. Commun. | 2020 | [10.1109/TCOMM.2019.2961332](https://doi.org/10.1109/TCOMM.2019.2961332) |
| Zhang et al., "RL of Beam Codebooks in mmWave and THz MIMO Systems" | IEEE Trans. Commun. | 2022 | [10.1109/TCOMM.2021.3126856](https://doi.org/10.1109/TCOMM.2021.3126856) |
| Va et al., "Online Learning for Position-Aided mmWave Beam Training" | IEEE Access | 2019 | [10.1109/ACCESS.2019.2902372](https://doi.org/10.1109/ACCESS.2019.2902372) |
| Klautau et al., "LIDAR Data for DL-Based mmWave Beam-Selection" | IEEE Wireless Commun. Lett. | 2019 | [10.1109/LWC.2019.2903820](https://doi.org/10.1109/LWC.2019.2903820) |
| Luong et al., "Applications of DRL in Communications: A Survey" | IEEE Commun. Surveys & Tutorials | 2019 | [10.1109/COMST.2019.2916583](https://doi.org/10.1109/COMST.2019.2916583) |

**Key takeaway:** All discrete approaches suffer from the codebook resolution vs. overhead tradeoff. Our DQN benchmark (41.1% lock, 4.93° error) validates this quantization floor.

---

## Category 2: Continuous RL (DDPG/SAC) for Phase Control

These papers output raw phase shifts or complex weights as continuous actions. They demonstrate the curse of dimensionality we proved with the 64D model.

| Paper | Venue | Year | DOI/arXiv |
|---|---|---|---|
| Huang et al., "RIS Assisted Multiuser MISO Systems Exploiting DRL" | IEEE JSAC | 2020 | [10.1109/JSAC.2020.3000708](https://doi.org/10.1109/JSAC.2020.3000708) |
| Feng et al., "DRL Based IRS Optimization for MISO Systems" | IEEE Wireless Commun. Lett. | 2020 | [10.1109/LWC.2020.2969167](https://doi.org/10.1109/LWC.2020.2969167) |
| Lillicrap et al., "Continuous Control with Deep RL" (DDPG) | ICLR | 2016 | [arXiv:1509.02971](https://arxiv.org/abs/1509.02971) |
| Mnih et al., "Human-Level Control through Deep RL" (DQN) | Nature | 2015 | [10.1038/nature14236](https://doi.org/10.1038/nature14236) |

**Key takeaway:** Huang et al. and Feng et al. are the closest to our 64D "raw" model. They use DDPG to output raw RIS phase shifts. Neither proposes a parametric reduction to steering angles. Our 64D experiment (11.2% lock after 10M steps) directly replicates and extends their approach, showing it fails at scale.

---

## Category 3: Physics-Informed / Parametric Action Spaces

These papers address the action-space dimensionality problem in RL — the theoretical foundation for our contribution.

| Paper | Venue | Year | DOI/arXiv |
|---|---|---|---|
| Masson et al., "RL with Parameterized Actions" (Q-PAMDP) | AAAI | 2016 | [10.1609/aaai.v30i1.10226](https://doi.org/10.1609/aaai.v30i1.10226) |
| Chandak et al., "Learning Action Representations for RL" | ICML | 2019 | [arXiv:1902.00183](https://arxiv.org/abs/1902.00183) |
| Dulac-Arnold et al., "Deep RL in Large Discrete Action Spaces" | arXiv | 2015 | [arXiv:1512.07679](https://arxiv.org/abs/1512.07679) |
| Dulac-Arnold et al., "Challenges of Real-World RL" | Machine Learning | 2021 | [arXiv:1904.12901](https://arxiv.org/abs/1904.12901) |

**Key takeaway:** These papers propose *general-purpose* action space reduction (learned embeddings, nearest-neighbor search). **None** of them embed domain-specific physics (e.g., array manifold equations) as a deterministic mapping layer. Our approach is the first to use electromagnetic wave physics as a non-trainable action decoder.

---

## Category 4: RL for Radar / Target Tracking

These papers apply RL to radar parameter optimization — waveform, bandwidth, scheduling — but **not** to direct beamforming/beam steering.

| Paper | Venue | Year | DOI/arXiv |
|---|---|---|---|
| Thornton et al., "DRL Control for Radar Detection and Tracking in Congested Spectral Environments" | IEEE Trans. Cogn. Commun. Netw. | 2020 | [10.1109/TCCN.2020.3019605](https://doi.org/10.1109/TCCN.2020.3019605) |
| Selvi et al., "RL for Adaptable Bandwidth Tracking Radars" | IEEE Trans. Aerosp. Electron. Syst. | 2020 | [10.1109/TAES.2020.2987443](https://doi.org/10.1109/TAES.2020.2987443) |
| Stephan et al., "Scene-Adaptive Radar Tracking with DRL" | Machine Learning with Applications | 2022 | [10.1016/j.mlwa.2022.100284](https://doi.org/10.1016/j.mlwa.2022.100284) |

**Key takeaway:** RL for radar is an active area, but existing work focuses on cognitive radar (waveform/spectrum adaptation), **not** on spatial beamforming with phased arrays. Our work fills this gap.

---

## Category 5: Reward Shaping

| Paper | Venue | Year | DOI/arXiv |
|---|---|---|---|
| Ng et al., "Policy Invariance Under Reward Transformations" | ICML | 1999 | — |
| Ibrahim et al., "Comprehensive Overview of Reward Engineering in RL" | IEEE Access | 2024 | — |

**Key takeaway:** Our reward hacking discovery (64D agent exploiting side-lobes) and the anti-hacking penalty we engineered is a practical contribution to reward design for high-dimensional continuous RL.

---

## Our Unique Contributions (Paper Positioning)

1. **Physics-Informed Action Space Decomposition**: First to embed the array steering vector as a deterministic, non-trainable mapping layer inside the RL action pipeline, reducing the effective action space from O(N) to O(1).

2. **Empirical Proof of the Curse of Dimensionality**: Head-to-head comparison of Parametric 2D vs. Raw 64D SAC over 30M+ steps, showing the 64D agent fundamentally cannot discover constructive interference.

3. **Reward Hacking Discovery**: First documented case of RL agents exploiting "side-lobe" local minima in beamforming — pointing a defocused blob at the target to minimize tracking error while avoiding the difficulty of coherent beam formation.

4. **DQN Quantization Floor**: Empirical proof that discrete codebook approaches hit an inherent resolution bound (~5° error for 240 beams in 120°×60° FOV).

5. **Gap in Radar RL Literature**: Existing radar RL focuses on waveform/spectrum adaptation. We are the first to apply DRL directly to phased-array spatial beam steering for maneuvering target tracking.
