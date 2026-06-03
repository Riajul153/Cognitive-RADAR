# Phase 19: Action-Conditioned JEPA with Model Predictive Control (No RL)

## Vision

This implements Yann LeCun's cognitive architecture from *"A Path Towards Autonomous Machine Intelligence"* (2022) for radar beamforming. **No reinforcement learning is used.** Instead, the agent:

1. Learns a **World Model** (JEPA) that predicts future radar states in latent space
2. Learns a **Cost Module** that evaluates how "good" a latent state is
3. **Plans** actions by imagining trajectories through the world model and picking the one that minimizes cost

```
┌──────────────────────────────────────────────────────────────────┐
│                    LeCun Cognitive Architecture                  │
│                                                                  │
│   Perception          World Model            Cost Module         │
│  ┌─────────┐    ┌──────────────────┐    ┌──────────────────┐    │
│  │ Encoder  │    │    Predictor     │    │  Energy/Cost     │    │
│  │  f_θ     │──►│  g_φ(z_t, a_t)   │──►│  C_ψ(z) → cost   │    │
│  │ obs→z_t  │    │    → ẑ_{t+1}    │    │                  │    │
│  └─────────┘    └──────────────────┘    └────────┬─────────┘    │
│       ▲                                          │              │
│       │              Actor / Planner             ▼              │
│       │         ┌──────────────────────┐                        │
│       │         │  CEM Optimizer       │  Minimize              │
│  obs_t│         │  Samples N action    │  Σ C_ψ(ẑ_t)           │
│       │         │  sequences, rolls    │  over horizon H        │
│       │         │  out world model,    │                        │
│       │         │  picks lowest cost   │──► a*_t (execute)      │
│       │         └──────────────────────┘                        │
└───────┼─────────────────────────────────────────────────────────┘
        │
   Environment (Radar + Target)
```

---

## Architecture Details

### 1. Encoder $f_\theta$ (Perception Module)
Maps raw radar observations to a compact latent state.

| Parameter | Value | Rationale |
|---|---|---|
| Input dim | 10 (raw obs) | Monopulse errors, power, beam angles, prev action |
| Hidden layers | [128, 64] | Sufficient for 10D input |
| Output (latent) dim | 32 | Compact but expressive |
| Activation | ELU | Smooth gradients, no dead neurons |
| LayerNorm | After each hidden layer | Stabilizes latent representations |

### 2. Predictor $g_\phi$ (World Model Core)
Predicts next latent state given current latent and action.

| Parameter | Value | Rationale |
|---|---|---|
| Input dim | 34 (32 latent + 2 action) | Concatenation of z_t and a_t |
| Hidden layers | [128, 64] | Must model antenna physics in latent space |
| Output dim | 32 | Same as latent dim |
| Activation | ELU | Matches encoder |

### 3. Target Encoder $\bar{f}_\theta$ (EMA)
Identical architecture to encoder, but weights updated via exponential moving average (EMA) of the encoder weights. Provides the "ground truth" target latent for the VICReg loss. **No gradients flow through the target encoder.**

$$\bar{\theta} \leftarrow \tau \cdot \bar{\theta} + (1 - \tau) \cdot \theta, \quad \tau = 0.996$$

### 4. Cost Module $C_\psi$ (Energy Function)
Predicts the "energy" (negative quality) of a latent state. **Trained with supervised learning**, NOT RL.

| Parameter | Value | Rationale |
|---|---|---|
| Input dim | 32 (latent z) | Evaluates quality of a latent state |
| Hidden layers | [64, 32] | Simple regression head |
| Output dim | 1 | Scalar cost value |
| Training target | $1 - \text{received\_power}$ | Low power = high cost, high power = low cost |
| Loss | MSE | Standard regression |

> [!IMPORTANT]
> The cost module uses `received_power` as a **supervised training signal**, not as an RL reward. There is no temporal credit assignment, no bootstrapping, no policy gradients. The cost module simply learns to predict the instantaneous quality of a latent state.

### 5. CEM Planner (Actor)
At each timestep, the planner optimizes an action sequence over horizon H by simulating futures through the world model.

```
For each planning step:
  1. Sample P action sequences of length H from N(μ, σ²)
  2. For each sequence, roll out the world model:
     ẑ_{t+k+1} = predictor(ẑ_{t+k}, a_{t+k})  for k = 0..H-1
  3. Evaluate total cost: C_total = Σ_{k=0}^{H} cost_module(ẑ_{t+k})
  4. Select top-K elite sequences (lowest cost)
  5. Update μ, σ from the elite set
  6. Repeat for I iterations
  7. Execute a*_0 = μ[0] (first action of the optimized sequence)
```

| Parameter | Value | Rationale |
|---|---|---|
| Planning horizon H | 15 | 150ms lookahead (15 × 10ms timestep) |
| Population size P | 256 | Sufficient for 2D action space |
| Elite fraction | 0.1 (top 25) | Standard CEM practice |
| CEM iterations I | 3 | Balance planning quality vs compute |
| Action clipping | [-1, 1] | Matches environment action space |
| Warm start | Shift previous solution left | Temporal consistency between timesteps |

---

## Training Pipeline

### Phase A: Data Collection (Exploration)
Collect diverse transitions by running mixed policies in the environment.

- **50% Oracle trajectories**: Use conjugate-phase beamforming (known target angles) to show the world model what "success" looks like
- **30% Noisy-oracle trajectories**: Oracle + Gaussian noise on actions to explore near-optimal states
- **20% Random trajectories**: Uniform random actions to explore failure modes
- Store all $(o_t, a_t, o_{t+1}, P_{rx})$ transitions in a dataset (not a replay buffer — no RL)
- Collect **500,000 transitions** (~1000 episodes × 500 steps)

### Phase B: World Model Training (Offline)
Train the JEPA world model and cost module on the collected dataset.

**JEPA Loss (VICReg):**
$$\mathcal{L}_{JEPA} = \lambda \cdot \underbrace{\frac{1}{N}\sum_i \|\hat{z}_{i} - \bar{z}_{i}\|^2}_{\text{Invariance}} + \mu \cdot \underbrace{\frac{1}{d}\sum_j \max(0, 1 - \sigma(z^j))}_{\text{Variance}} + \nu \cdot \underbrace{\frac{1}{d}\sum_{i \neq j} C_{ij}^2}_{\text{Covariance}}$$

Where:
- $\hat{z} = g_\phi(f_\theta(o_t), a_t)$ — predicted next latent
- $\bar{z} = \bar{f}_\theta(o_{t+1})$ — target next latent (stop gradient)
- $\lambda=25, \mu=25, \nu=1$ (VICReg defaults)

**Cost Module Loss:**
$$\mathcal{L}_{cost} = \frac{1}{N}\sum_i (C_\psi(f_\theta(o_i)) - (1 - P_{rx,i}))^2$$

Training jointly with a combined loss: $\mathcal{L} = \mathcal{L}_{JEPA} + \alpha \cdot \mathcal{L}_{cost}$, with $\alpha = 0.1$ to prevent the cost loss from dominating the representation.

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-5) |
| Batch size | 512 |
| Training epochs | 100 over collected dataset |
| EMA decay τ | 0.996 |
| LR scheduler | Cosine annealing |

### Phase C: Online Planning + Fine-Tuning
Deploy the CEM planner with the trained world model. Continue collecting data and fine-tuning.

1. Run the CEM planner in the environment
2. Collect new transitions into the dataset
3. Periodically re-train the world model (every 50 episodes)
4. The system continuously improves as it encounters more diverse situations

---

## Proposed Changes

### New Files

#### [NEW] `src/models/__init__.py`
Empty init for models subpackage.

#### [NEW] `src/models/jepa.py`
Core JEPA module: `JEPAEncoder`, `JEPAPredictor`, `JEPAWorldModel` with VICReg loss, EMA update.

#### [NEW] `src/models/cost_module.py`
`CostModule(nn.Module)`: MLP that maps latent z → scalar cost. Trained with MSE against `1 - received_power`.

#### [NEW] `src/models/planner.py`
`CEMPlanner`: Cross-Entropy Method optimizer that rolls out the world model over horizon H and returns the optimal first action. Includes warm-starting from the previous solution.

#### [NEW] `src/agents/jepa_agent.py`
`JEPABeamTrackingAgent`: Ties together the world model, cost module, and planner. Provides a simple `act(obs) → action` interface and handles data collection, training, and online fine-tuning.

#### [NEW] `scripts/train_jepa.py`
Main training + evaluation script with three phases:
1. Data collection (mixed oracle/noisy/random policies)
2. Offline world model training
3. Online planning with fine-tuning

#### [NEW] `config/jepa_mpc.yaml`
Configuration file with all JEPA/CEM/training hyperparameters.

> [!IMPORTANT]
> **Zero existing files are modified.** The JEPA system is entirely self-contained in `src/models/`, `src/agents/jepa_agent.py`, and `scripts/train_jepa.py`. All running SAC experiments remain completely safe.

---

## Open Questions

> [!WARNING]
> **Multi-step prediction accuracy**: World model errors compound over the planning horizon H. If the predictor is inaccurate, the CEM planner will optimize for hallucinated futures. We mitigate this with a short horizon (H=15) and warm-starting, but we should monitor prediction error during training. If the world model struggles, we can reduce H to 5-10.

> [!NOTE]
> **Comparison with SAC baseline**: For the paper, the key metrics to compare are: (1) sample efficiency (how many environment interactions to reach 90% success rate), (2) final performance (tracking error, gain, success rate), and (3) planning compute cost (CEM adds overhead per step). If JEPA-MPC matches SAC performance with 10x fewer interactions, that is a landmark result.

## Verification Plan

### Automated Tests
1. **World Model Quality**: After Phase B training, measure 1-step and 5-step prediction error in latent space. The 1-step error should be near zero; 5-step error should be reasonable.
2. **Cost Module Accuracy**: Scatter plot of predicted cost vs actual `1 - received_power` on held-out data. R² should exceed 0.9.
3. **Planning Performance**: Run 100 evaluation episodes with the CEM planner. Compare gain, error, and success rate against the SAC champion.
4. **TensorBoard**: Monitor at `http://localhost:6014/` for JEPA loss curves, cost module loss, and tracking metrics.

### Manual Verification
- Verify that the CEM planner produces smooth, physically plausible beam trajectories (no jittering)
- Check that the agent can "plan through" momentary signal drops (evidence of predictive planning)
