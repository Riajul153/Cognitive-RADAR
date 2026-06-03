# Deep Reinforcement Learning for Hardware-Aware Phased Array Radar Beam Tracking

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Supported-green)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-SAC-orange)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

This repository contains the codebase, experimental logs, and manuscript for our research on applying Deep Reinforcement Learning (DRL) to control phased array antennas. We prove that while classical tracking algorithms (EKF, Model Predictive Control) catastrophically fail when confronted with physical hardware nulls (e.g., dipole arrays), our **Masked Soft Actor-Critic (SAC) Outage Filter** organically learns the hardware topology and achieves a 100% track lock rate.

---

## 📖 The Comprehensive Journey
If you want to read the full story of how this project evolved—from scaling a 64-element isotropic array up to a 256-element array, discovering the dipole hardware nulls, and defeating the classical MPC Oracles—please read our [**COMPREHENSIVE_JOURNEY.md**](./COMPREHENSIVE_JOURNEY.md).

## 🚀 Repository Structure

```text
.
├── COMPREHENSIVE_JOURNEY.md # Full detailed timeline of all experiments and breakthroughs
├── config/                  # YAML configuration files for the RL environments
├── dashboard/               # HTML/JS/CSS files for the 3D real-time visualization dashboard
├── docs/                    # Implementation plans and design artifacts
├── logs_*/                  # Tensorboard tfevents and CSV tracking logs for all experiments
├── paper/                   # Final IEEEtran LaTeX manuscript and plotting scripts
├── scripts/                 # Training and benchmarking runner scripts
└── src/                     # Core Python package
    ├── agents/              # EKF, MPC, Oracle, and Masked SAC agent implementations
    ├── antenna/             # Phased array wave physics engine (Dipoles, UPAs, Crossed-Dipoles)
    ├── environment/         # Gymnasium custom RL environments
    └── target/              # 3D target kinematics (Singer acceleration models)
```

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/phased-array-drl.git
cd phased-array-drl
pip install -r requirements.txt
```
*(Dependencies: `numpy`, `scipy`, `gymnasium`, `stable-baselines3`, `matplotlib`, `pyyaml`)*

## 🏃 How to Run

### 1. Train the Masked SAC Agent
To reproduce our breakthrough where the DRL agent learns to intentionally misalign the beam to avoid hardware nulls:
```bash
python scripts/train_sac_masked.py
```

### 2. Benchmark the Classical Oracles
To observe the catastrophic failure of classical Model Predictive Control in the presence of physical hardware nulls:
```bash
python scripts/benchmark_ekf_mpc.py      # Non-privileged classical baseline
python scripts/benchmark_oracle_mpc.py   # Perfect 9D state Oracle
```

### 3. Generate Paper Plots
To generate the high-resolution vector graphics used in our manuscript:
```bash
python scripts/generate_paper_plots.py
```
*(Plots will be saved to `paper/plots/`)*

## 📊 Core Benchmark Results

| Agent Architecture | Tracking Success Rate | Mean Tracking Error | Mean Array Gain |
| :--- | :---: | :---: | :---: |
| **EKF-MPC** (No Privileged Data) | 0.0 % | 17.98° | 0.007 |
| **Oracle-MPC** (Perfect 9D State) | 0.0 % | **0.64°** | 0.065 |
| **Masked SAC (Proposed)** | **100.0 %** | 1.28° | **0.864** |

Because classical control optimizes purely for kinematic alignment, pointing exactly at the target inadvertently places the target within the dipole's radiation null (Zero Gain). The Masked SAC agent actively optimizes for **physics (Array Gain)**, successfully learning to misalign the beam to exploit the side-lobes.

## 📄 Publication
The drafted IEEE manuscript can be found in the `paper/` directory. You can upload this directory directly to Overleaf for compilation. See `paper/README.md` for details.

## ⚖️ License
This project is licensed under the MIT License.
