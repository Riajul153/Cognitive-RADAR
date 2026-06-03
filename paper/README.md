# Paper Compilation Instructions

This directory contains the final manuscript and high-resolution plots for the paper:
**Deep Reinforcement Learning for Hardware-Aware Phased Array Radar Beam Tracking: Defeating Classical Oracles**

Since local LaTeX compilation (`pdflatex`) is not installed on this system, we recommend using a web-based LaTeX editor like **Overleaf** to compile this document into a beautiful PDF.

## How to Compile in Overleaf

1. Go to [Overleaf.com](https://www.overleaf.com/) and create a new **Blank Project**.
2. Delete the default `main.tex` file in the Overleaf project.
3. Upload the `manuscript.tex` file from this folder into the Overleaf project.
4. Create a folder named `plots` in the Overleaf project.
5. Upload all the `.pdf` plot files from the `paper/plots/` folder on your computer into the new `plots` folder on Overleaf.
6. Click **Recompile** (or hit Ctrl+S).

Overleaf will automatically pull the IEEEtran class (which is built-in) and generate the dual-column, publication-ready PDF document!
