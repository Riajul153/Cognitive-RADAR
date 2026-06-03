# Phase 20: SAC Critic Loss Masking (The "Outage Filter")

We will implement the "clever trick" of masking the SAC critic updates whenever the radar enters a physical antenna null. This mathematically prevents catastrophic forgetting by stopping the network from updating its weights when the input signal is pure noise.

## Proposed Changes

### 1. `src/agents/masked_sac.py` [NEW]
We will create a custom `MaskedSAC` class that inherits from Stable Baselines 3's `SAC` algorithm and overrides the core `train()` method. 

* **The Masking Logic:** 
  The observation space is a flattened 1D array of stacked frames. If `frame_stack = 8`, the observation size is $8 \times 10 = 80$. The most recent frame occupies the last 10 indices.
  Index 2 of each frame is the `normalized_power`.
  We will extract the power from the most recent frame:
  `power_norm = replay_data.observations[:, -8]`
  `power = (power_norm + 1.0) / 2.0`
  We will create a boolean mask: `valid_mask = (power > 0.15).float().unsqueeze(1)`
* **Critic Loss Update:**
  Instead of standard MSE loss, we will use `F.mse_loss(..., reduction='none')`, multiply by `valid_mask`, and then take the mean. If a batch contains noise frames, their TD-error gradients will be explicitly set to $0$.

### 2. `config/sac_dipole_masked.yaml` [NEW]
A copy of our fixed-reward dipole config, but with:
* `frame_stack: 8` (Increased from 4 to give 80ms of memory).
* `log_dir: "logs_dipole_masked/"`
* `model_dir: "models_dipole_masked/"`

### 3. `scripts/train_sac_masked.py` [NEW]
A new training and benchmarking script that instantiates the `DipoleBeamTrackingEnv` (with physical half-wave dipole nulls), wraps it in an 8-frame stack, and uses our new `MaskedSAC` algorithm.

## User Review Required

> [!IMPORTANT]  
> By masking the critic loss, we are explicitly engineering domain knowledge (the fact that low power = bad signal) directly into the RL algorithm's loss function. This violates pure "model-free" RL, but it is standard practice in applied engineering. Do you approve of extracting the power from the observation tensor to mask the loss?

> [!TIP]
> Increasing the frame stack to 8 will increase the state dimension from 40 to 80. The 2-layer MLP `[256, 256]` should still have plenty of capacity to handle this.

Please approve this plan so I can begin writing the `MaskedSAC` algorithm and start the training!
