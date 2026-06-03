# Deep Reinforcement and World-Model Planning for Adaptive Antenna Array Beamforming

## Abstract
Traditional phased array beamforming relies on analytical models and oracle target information to calculate conjugate phases. In real-world environments with hardware impairments, mutual coupling, and antenna nulls, these analytical models fall short. In this work, we demonstrate the efficacy of learning-based control for phased arrays. We benchmark two primary paradigms: model-free Deep Reinforcement Learning (SAC/TD3) and model-predictive planning via Joint Embedding Predictive Architectures (JEPA). Our experiments on 64-element Uniform Planar Arrays and Crossed-Dipole Antenna Diversity setups prove that learning-based systems can automatically learn to steer beams from noisy monopulse error signals and physical array factors without prior domain knowledge. 

## 1. Introduction
- The challenge of tracking highly dynamic aerial targets (Singer models).
- Limitations of traditional conjugate-phase analytical beamforming when hardware nulls exist.
- Proposed solution: Model-free and model-based deep learning.

## 2. Methodology
### 2.1 Antenna Array and Target Environment 
- 8x8 Uniform Planar Array (UPA) physics.
- Crossed-Dipole antennas to eliminate hardware nulls via spatial diversity.
- Observation space: Noisy monopulse error signals, received power, and temporal derivatives.
- Action space: Incremental parametric phase steering via $d\theta, d\phi$.

### 2.2 Deep Reinforcement Learning (SAC) baseline
- Continuous action space control using Soft Actor-Critic (SAC).
- Dense Potential-Based Reward Shaping (PBRS) to guide the agent toward the mainlobe.
- Frame-stacking for temporal dynamics.

### 2.3 JEPA-MPC World Model
- LeCun's cognitive architecture: Online Encoder, EMA Target Encoder, and Predictor.
- VICReg non-contrastive loss to prevent representation collapse.
- Supervised Cost Module (Energy function) targeting signal attenuation.
- Cross-Entropy Method (CEM) planner for rolling out imagined futures and executing optimal actions.

## 3. Experiments and Results
### 3.1 Hardware-Aware Null Compensation
- Discuss the catastrophic failure of analytical steering when target passes through antenna nulls (dipoles).
- How the Crossed-Dipole arrangement provides a learnable topology.
### 3.2 Baseline Performance
- SAC agent achieving < 2 degrees tracking error.
### 3.3 JEPA-MPC Performance
- Sample efficiency comparison.
- Scalability to larger arrays without redefining the action space.

## 4. Conclusion
- Learning-based controllers are highly robust for edge-hardware deployment (ONNX).
- Future work: Extending the JEPA world model to wideband multi-target tracking.
