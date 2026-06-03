# Peaceful Fermi: DRL Adaptive Beamforming Project Tasks

## Completed Phases
- [x] Phase 1: Environment Setup
- [x] Phase 2: Antenna Array Modeling
- [x] Phase 3: Target Dynamics Modeling
- [x] Phase 4: Beam Tracking Environment (Base)
- [x] Phase 5: Reward Function Design
- [x] Phase 6: Codebook Baseline
- [x] Phase 7: Real-time Dashboard
- [x] Phase 8: DRL Training Setup (PPO/SAC)
- [x] Phase 9: Parametric Action Space
- [x] Phase 10: Parametric SAC Training and Validation
- [x] Phase 11: 64D Raw SAC Baseline Experiment
- [x] Phase 12: Reward Tuning for the 64D Agent
- [x] Phase 13: 30M Step Marathon for the 64D Agent
- [x] Phase 14: Final Code Cleanup & Documentation

## Current Active Phases
- [x] Phase 15: Parametric Architecture Optimization
- [x] Refactor `BeamTrackingEnv` to use 10D `MonopulseProcessor` output instead of true targets.
- [x] Implement Potential-Based Reward Shaping in `RewardComputer` based on $P_{rx}$.
- [x] Test and validate the architecture under 20dB SNR.
- [x] Launch 3-way ablation study (Standard SAC, FrameStack SAC, RecurrentPPO)

## Phase 16: 256-Element Array Scaling Benchmark
- [x] Create 16x16 configuration for the array benchmark.
- [x] Launch 256-element DRL training.
- [x] Validate beam tracking and heatmap on the dashboard.

## Phase 17: Half-Wave Dipole Ablation Study
- [x] Create subclassed isolated environment to inject Dipole physics without touching running processes.
- [x] Launch 8x8 Dipole training run.

## Phase 18: Solving the Hardware Null
- [x] Create `src/models/jepa.py` (VICReg, Dual Encoders)
- [x] Create `src/models/cost_module.py` (Energy Function)
- [x] Create `src/models/planner.py` (CEM Planner)
- [x] Create `src/agents/jepa_agent.py`
- [x] Create `scripts/train_jepa.py`
- [x] Launch JEPA training and observe physics learning

- [x] Phase 17: Journal Submission Prep
  - [x] Complete literature review
  - [x] Draft manuscript outline/content
- [x] Phase 19: LeWorldModels / JEPA
  - [x] Justify and define cognitive architecture
  - [x] Implement Planner and Cost Module
  - [x] Execute Phase A & B dataset collection/training
- [x] Create ONNX export script for champion models

- [x] Phase 20: SAC Masked Critic (The "Outage Filter")
  - [x] Implement `src/agents/masked_sac.py`
  - [x] Create `config/sac_dipole_masked.yaml` (frame stack = 8)
  - [x] Create `scripts/train_sac_masked.py`
  - [x] Launch masked SAC training on dipole environment

- [x] Phase 21: Sophisticated Classical Baselines (MPC & Oracle)
  - [x] Implement `src/agents/mpc_core.py`
  - [x] Implement `src/agents/ekf_mpc_agent.py`
  - [x] Implement `src/agents/oracle_mpc_agent.py`
  - [x] Create benchmark scripts
  - [x] Execute benchmarks and collect metrics

- [x] Phase 22: Comprehensive Manuscript & Visualization Generation
  - [x] Write `scripts/generate_paper_plots.py` and generate plots
  - [x] Write `paper/manuscript.tex`
  - [x] Create `paper/README.md`
