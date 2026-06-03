# Phase 22: Comprehensive Manuscript & Visualization Generation

This plan outlines the steps to synthesize our entire experimental journey into a final, coherent IEEE-formatted scientific paper with accompanying plots.

## Proposed Changes

### 1. Data Visualization Script (`scripts/generate_paper_plots.py`) [NEW]
We will create a Python script using Matplotlib to generate publication-ready, paper-styled plots (PDF/PNG format). The plots will illustrate the chronological journey:
* **Fig 1: Action Space Comparison**: Raw 64D phase control vs Physics-Informed 2D Parametric control.
* **Fig 2: Scaling**: 64-element UPA vs 256-element UPA performance.
* **Fig 3: Observability**: Oracle (True State) vs Partial Observability (EKF/Sensors).
* **Fig 4: Reward Shaping**: The impact of Potential-Based Reward Shaping on convergence.
* **Fig 5: Hardware Nulls**: The catastrophic collapse of standard SAC on the Dipole environment vs the success of Frame-Stacking.
* **Fig 6: Final Benchmark**: Classical EKF-MPC vs Oracle-MPC vs Masked SAC Outage Filter.

### 2. LaTeX Manuscript (`paper/manuscript.tex`) [NEW]
We will write the comprehensive manuscript in standard `IEEEtran` LaTeX format. The paper will chronologically document:
* **Abstract & Introduction**
* **System Model** (Antenna Arrays, Singer Target Dynamics)
* **Action Space & Scaling** (Isotropic 64 vs 256)
* **Observability & Frame Stacking** (Handling partial observability)
* **Hardware Nulls & Classical Failures** (The Dipole dilemma, Oracle-MPC failure)
* **Proposed Solutions** (Masked SAC Outage Filter & Crossed-Dipoles)
* **Conclusion**

## User Review Required

> [!WARNING]  
> **LaTeX Compiler Missing:** I checked the local system and `pdflatex` is currently not installed. Installing a full LaTeX distribution (like TeX Live or MiKTeX) on Windows is heavily time-consuming and prone to errors. 
> 
> **My Proposal:** I will write the complete `.tex` source file and generate all the high-resolution plots into a `paper/` directory. You can then upload this folder directly to **Overleaf** to compile the PDF seamlessly. 
> 
> Does this approach work for you?
