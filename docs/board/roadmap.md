# 🗺 Roadmap

!!! "Dictionary"
    - [ ] Open
    - [x] Closed
    - [-] Blocked
    - [~] In Progress
    
See [Phase Archive](./closed/phases.md) for the project's completed phases.

---

## Phase 6: Second-Order Cognitive Dynamics (Latent Inertia)

*Goal:* Transition the Liquid Neural Network's hidden state from a first-order kinematic model to a true second-order dynamical system. By granting the latent state "momentum" via a coupled system of Ordinary Differential Equations (ODEs), the agent can accumulate force to escape localized equilibrium traps (e.g., staring at corners) without requiring explicit exogenous input.

### 1. Configuration & Taxonomy Layer

* [ ] **ODE Configuration:** Expand the `brain` block in `app.yaml` to include an `ode_order` parameter (accepting integer values `1` or `2`).
* [ ] **State Validation:** Update `config.py` to accurately parse `ode_order` into the initialization pipelines.
* [ ] **Model Archiving Schema:** Modify `train.py` and `utils.py` to append the ODE order to the saved `.pth` weights (e.g., `<YYYY-MM-DD>.c-<depth>.w-<length>.o-<order>.<increment>.pth`). Ensure `get_latest_parameters` remains backwards-compatible with older, first-order checkpoints.

### 2. The Brain (Architecture Redesign)

* [ ] **Hamiltonian Memory Split:** Update `brain.py` to dynamically adjust the capacity of the `liquid_rnn` based on the configured `ode_order`. For second-order systems, the `working_memory` must effectively track both latent position ($x_1$) and latent momentum ($x_2$). 
* [ ] **Coupled ODE Forward Pass:** Rewrite the forward pass logic in `DoomLiquidNet` to support second-order integration. When `ode_order == 2`, mathematically decompose the second-order ODE into a system of two coupled first-order ODEs:
  $$\frac{dx_1(t)}{dt} = x_2(t)$$
  $$\frac{dx_2(t)}{dt} = f(x_1(t), x_2(t), I(t); \theta)$$

### 3. The Pipeline (State Management)

* [ ] **State Initialization:** Update `run.py` and `intervene.py` to initialize and pass the expanded/tuple-based hidden state $hx$ when the second-order architecture is active.
* [ ] **Physiological Reset (Death):** Implement a strict state-check inside the inference loops. If `game.is_player_dead()` is true, explicitly detach and zero-out the hidden state. This prevents "past life" momentum leakage where the newly respawned agent inadvertently reacts to the accumulated latent velocity of its previous death.

!!! danger "Risk Assessment"
  **Training Overhead: Moderate**
  
  The BPTT (Backpropagation Through Time) algorithm must compute gradients through a coupled system of equations:
  
  $$
  \frac{dx_1(t)}{dt} = x_2(t), \quad \frac{dx_2(t)}{dt} = f(x_1(t), x_2(t), I(t); \theta)
  $$
  
  Because the working_memory capacity must effectively double to accommodate the bipartite Hamiltonian state tuple, VRAM consumption will increase proportionally.
  
  **Runtime Overhead: Low** 
  
  Because the network relies on the Closed-form Continuous (CfC) approximation rather than a traditional numerical solver (like Runge-Kutta), you bypass the severe latency penalties of iterative ODE evaluations. The primary cost is the increased matrix multiplication dimension resulting from the doubled hidden state size.

  **Assessment**: Acceptable risks. Cleared to implement.

## Phase 7: The Parietal Binding (Cross-Attention Sensorimotor Integration) 🧠

*Goal:* Shift from naive flat tensor concatenation to a structured cross-attention bottleneck. By forcing the Thermal Cortex to query the Visual Cortex, the network learns to synthesize abstract spatial relationships (e.g., "enemy is next to the explosive barrel") rather than just reacting to isolated stimuli.


### 1. Configuration & Taxonomy Layer

* [-] **Attention Toggles:** Update `app.yaml` to include an `attention_heads` configuration integer under the `brain` block (e.g., 4).
* [-] **State Validation:** Update `config.py` to accurately parse this parameter and pass it to the model initializer.

### 2. The Brain (Architecture Redesign)

* [-] **Multi-Head Attention:** Modify `DoomLiquidNet` in `brain.py` to instantiate a `nn.MultiheadAttention` layer.
* [-] **Q-K-V Projection:** In the `forward` pass, project the flattened Thermal feature map ($T$) into the **Query** ($Q$), and the Visual/Depth feature map ($V$) into the **Keys** ($K$) and **Values** ($V$).
* [-] **Sensorimotor Fusion:** Flatten the resulting contextual output tensor and feed it as the input $I(t)$ into the Liquid Core, scaling the `working_memory` input dynamically.

!!! danger "Risk Assessment"
  **Training Overhead: High**.
  
  Attention mechanisms scale quadratically in complexity. While projecting the flattened feature maps into $Q$, $K$, and $V$ matrices adds a modest number of parameters, the actual $Q K^\top$ dot-product attention drastically increases memory bandwidth demands and FLOPs during the backward pass.
  
  **Runtime Overhead: High Risk**
  
  Multi-head attention is notoriously memory-bandwidth bound during autoregressive inference. Each attention head requires loading the respective matrices into GPU memory, which can introduce micro-latencies. Unless highly optimized (e.g., utilizing FlashAttention or grouping queries), this phase poses the highest risk of blowing past the ViZDoom engine's $35\text{Hz}$ ($\approx 28\text{ms}$) temporal limit, leading to desynchronization in the multiplayer arena.

  **Assessment**: Unacceptable risks. Do not implement.

---

## Phase 8: The Prefrontal Hierarchy (Multi-Scale Liquid Time-Constants) ⏳

*Goal:* Split the Liquid Core into two decoupled Ordinary Differential Equations (ODEs) operating at different frequencies. This allows tactical reflexes to execute in milliseconds while a strategic, long-term context is maintained over seconds, bridging the gap between instinct and abstract strategy.

### 1. Configuration Layer

* [-] **Hierarchical Timesteps:** Update `app.yaml` to include a `brain.hierarchy` block defining `fast_hz` (e.g., 35) and `slow_hz` (e.g., 2).

### 2. The Brain (Architecture Redesign)

* [-] **Bifurcated Core:** Modify `brain.py` to instantiate two separate `CfC` modules: `FastCore` (tactical motor mapping) and `SlowCore` (strategic context).
* [-] **Temporal Gating:** Implement a gating mechanism in the `forward` pass so the `SlowCore` only integrates periodically (e.g., every 17 frames). 
* [-] **Top-Down Bias:** Pass the hidden state of the `SlowCore` as a continuous, concatenating bias to the `FastCore`'s input sequence.

### 3. The Pipeline (State Management)

* [-] **Complex State Persistence:** Adjust the recurrent state tuple $hx$ across `run.py` and `intervene.py` to persist a dictionary or tuple of hierarchical memories (`hx_fast`, `hx_slow`) across the inference loop.

!!! danger "Risk Assessment"
  **Training Overhead: High**
  
  BPTT must now unroll and track gradients across two vastly different temporal scales. The parameters for the LNN essentially double, and the computational graph becomes highly complex, as the SlowCore's hidden state acts as a continuous bias injection into the FastCore.

  **Runtime Overhead: Moderate (with Jitter)** 
  
  The average FLOP count per frame only increases slightly, as the FastCore handles the bulk of the rapid inference. However, this architecture introduces latency jitter. On the frames where the SlowCore triggers its integration (approximately every 17 frames), the computational load spikes. Carefully profile this worst-case frame execution time to ensure it does not break the lockstep networking protocol.

  **Assessment**: Awaiting further assessment after Phase 6. Do not implement.

---

## Phase 9: Forward Internal Models (Predictive Coding) 🔮

*Goal:* Move beyond pure Behavioral Cloning by forcing the network to anticipate the future. Adding a self-supervised prediction head forces the latent space to encode the physics and movement dynamics of the DOOM engine, naturally inducing strategic planning.

### 1. Configuration Layer

* [ ] **Predictive Toggle:** Update `app.yaml` to toggle `training.predictive_coding` and define a temporal forecast horizon parameter $k$ (e.g., 10 frames).

### 2. The Brain (Architecture Redesign)

* [ ] **Hallucination Decoder:** Implement a transposed convolutional decoder (`nn.ConvTranspose2d`) in `brain.py` that branches off the liquid hidden state $x(t)$.
* [ ] **Future Projection:** Configure the decoder to output a hallucinated spatial prediction of the future thermal mask $\hat{T}(t+k)$.

### 3. The Pipeline (ETL & Training)

* [ ] **Temporal Offset Streaming:** Update `DoomStreamingDataset` in `dataset.py` to dynamically yield a future target tensor $T(t+k)$ alongside the standard sequence inputs and action labels.
* [ ] **Composite Loss Function:** Modify the optimization loop in `train.py` to evaluate both the action logits and the thermal hallucination via a composite loss function: $\mathcal{L}_{total} = \mathcal{L}_{BCE\_Action} + \lambda \mathcal{L}_{MSE\_ThermalFuture}$.

!!! danger "Risk Assessment"
  **Training Overhead: Severe**
  
  This is the most computationally expensive upgrade on the board. The dataset must now stream offset future tensors into memory, vastly increasing I/O pressure and RAM utilization. The addition of the Transposed CNN decoder essentially doubles the size of the network. Furthermore, optimizing the composite loss function ($\mathcal{L}_{total} = \mathcal{L}_{BCE\_Action} + \lambda \mathcal{L}_{MSE\_ThermalFuture}$) requires computing gradients for both the classification head and the dense image generation head simultaneously.
  
  **Runtime Overhead: Negligible to Zero**
  
  This is a purely structural training constraint. Because the agent only requires the output of the Linear Motor Cortex to play the game, the entire hallucination decoder can be detached and bypassed during live inference.

  **Assessment**: Acceptable, if gated behind a configuration property that is disabled by default. Cleared to implement.
  
---

## Distant Future

### The Possession (Integration) 👻

* *Goal:* Move beyond the Python wrapper and replace in-game enemy AIs.
* [-] **Engine Fork:** Compile the PyTorch model to TorchScript (`.pt`) and link `libtorch` directly into a C++ source port to bypass pixel-rendering entirely.