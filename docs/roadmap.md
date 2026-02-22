# 🗺 Roadmap

!!! "Dictionary"
    - [ ] Open
    - [x] Closed
    - [-] Blocked
    - [~] In Progress
    
## Phase 1: Setup

### The Foundation (Completed) ✅

* [x] **ETL Pipeline:** Robust recording of pixel buffers and input vectors.
* [x] **Engine Bridge:** ViZDoom integration with custom `cfg` injection.
* [x] **Data Sanitation:** Automated inspection tools.
* [x] **Architecture:** Modular, config-driven Python application.

### The Brain (Completed) ✅

* [x] **Dataset Class:** `IterableDataset` with sliding window segmentation.
* [x] **Liquid Network:** `CfC` (Closed-form Continuous) implementation.
* [x] **Training Loop:** Behavioral Cloning with `BCEWithLogitsLoss`.
* [x] **Optimization:** Auto-device selection (CUDA/MPS/CPU).

### The Body (Completed) ✅

* [x] **Inference Engine:** Real-time game loop driven by the LNN.
* [x] **Action Logic:** Thresholding and conflict suppression logic.

---

## Phase 2: Distributed Architecture (The VDAIC Arena) 🤖

*Goal:* Containerize the LNN agent and connect it to a dedicated multiplayer DOOM server. Benchmark Golem against the historical champions of the Visual Doom AI Competition (VDAIC).

### 1. The Host Server (Arena)

* [x] **Dedicated Host Script:** Initialize a central ViZDoom instance in `-host N -deathmatch` mode. 
* [x] **Multiplayer Configuration:** Update configs and select WADs with proper multiplayer spawn points.
* [x] **Host Container:** Create a lightweight `Dockerfile.host` that only installs the ViZDoom engine and the host script.

### 2. The Golem Client (Containerization)

* [x] **Headless Rendering:** Configure the ViZDoom to render the visual buffer headlessly.
* [x] **Network Synchronization:** ensure the LNN's inference tick-rate stays aligned with the network server.
* [x] **Modular Dockerfile:** The image should package Python 3.10+, PyTorch, and ViZDoom, but **omit the model weights**. 
* [x] **Volume Mounting:** Configure the container entrypoint to load the `golem.pth` brain and `app.yaml` configuration from a mounted volume directory (e.g., `-v ./data/fluid:/app/data`).

### 3. The Legacy Champions (The Opposition)

* [x] **Archive Retrieval:** Clone the legacy Dockerfiles and weights for the 2016/2017 VDAIC winners (e.g., *Arnold* by CMU, *IntelAct* by Intel Labs) from the official GitHub archives.
* [x] **Legacy Container Builds:** Build the historical images.

### 4. Orchestration (The Swarm)

* [x] **Docker Compose:** Create a `docker-compose.yml` to network the swarm. 
* [x] **The Roster:** Define the services in the compose file to simultaneously spin up:
    * 1x Host Arena Server
    * 2x Legacy Champion Bots (e.g., Arnold, IntelAct)
    * Nx Golem Agents (using the same image, but mounting different profile volumes, e.g. `basic` or `fluid`).
* [x] **Agent Parameterization:** Pass unique names and colors via environment variables.

---

## Phase 3: Multi-Modal Sensor Fusion (Phenomenology) 👁️👂

*Goal:* Expand the agent's sensory perception beyond the 2D pixel array by integrating ViZDoom's raw depth and audio buffers into the Liquid Neural Network, effectively granting stereopsis and audition without exposing underlying game-state variables.

### 1. The Configuration Layer

* [x] **Sensor Toggles:** Update `app.yaml` to include a `brain.sensors` block with boolean toggles for `depth` and `audio`.
* [x] **Dynamic Action Space:** Update `config.py` to parse these toggles and pass them to the ETL and Model initialization layers.

### 2. The ETL Pipeline (Record & Transform)

* [x] **Depth Extraction:** Modify `record.py` to capture `state.depth_buffer`. Normalize the 1D distance matrix to $[0, 1]$.
* [x] **Audio Extraction:** Modify `record.py` to capture `state.audio_buffer`. Normalize the raw stereo waveforms.
* [x] **Tensor Packaging:** Update the `.npz` saving mechanism to store `depth` and `audio` arrays only if they are enabled in the active profile, preventing massive file bloat for purely visual agents.

### 3. The Brain (Architecture Redesign)

* [x] **Stereopsis Integration:** If `depth` is enabled, modify the Visual Cortex CNN input channels from $C=3$ (RGB) to $C=4$ (RGB + Depth).
* [x] **Auditory Cortex:** If `audio` is enabled, implement a parallel 1D Convolutional Neural Network (`nn.Conv1d`) to extract features from the high-frequency audio buffer.
* [x] **Sensor Fusion:** Concatenate the flattened visual/depth feature vector with the auditory feature vector before passing the unified tensor into the Liquid `CfC` core.

---

## Phase 4: Auditory Phenomenology Refactoring (Mel Spectrograms) 🎼

*Goal:* Transition from processing raw 1D audio waveforms to 2D Mel Spectrograms. This improves LNN stability by leveraging spatial locality in convolutional networks, allowing the model to recognize the "visual" shape of audio cues (like a fireball or monster growl) while naturally compressing high-frequency acoustic noise.

### 1. The ETL Pipeline (Transformation)

* [x] **Audio Normalization:** Enforce strict zero-mean, unit-variance normalization on the raw audio buffer at extraction to prevent gradient explosion.
* [x] **Spectrogram Generation:** Integrate `torchaudio.transforms.MelSpectrogram` followed by `torchaudio.transforms.AmplitudeToDB` into the data transformation layer (`dataset.py`). This will mathematically convert the raw 1D audio arrays into dense 2D time-frequency tensors (scaled to decibels) on the fly during the `__getitem__` call.

### 2. The Configuration Layer

* [x] **DSP Hyperparameters:** Expand `app.yaml` to include a `brain.dsp` block containing parameter tunings for the Mel Spectrogram generation. Required parameters include the engine's `sample_rate`, `n_fft` (e.g., 1024), `hop_length` (e.g., 256), and `n_mels` (e.g., 64).

### 3. The Brain (Architecture Redesign)

* [x] **2D Auditory Cortex:** Replace the `nn.Conv1d` auditory cortex in `brain.py` with a standard `nn.Conv2d` architecture, mathematically aligning sound classification with the existing spatial and visual processing hierarchy.
* [x] **Sensor Fusion Re-Alignment:** Ensure the concatenation logic dynamically calculates the flattened feature size of the newly generated 2D auditory feature map before routing the unified tensor into the CfC liquid core.

---

## Phase 5: Thermal Phenomenology (Heat Vision) 🐍

*Goal:* Decouple spatial navigation from enemy detection by utilizing ViZDoom's semantic segmentation `labels_buffer`. This isolates active entities from the background into a clean, binary "thermal" mask, severely reducing the visual noise the model must parse during combat.

### 1. The Configuration Layer

* [x] **Sensor Toggle:** Expand `app.yaml` to include a `thermal: true` flag in the `brain.sensors` configuration block.
* [x] **State Validation:** Update `config.py` to accurately parse the boolean into the initialization pipelines.

### 2. The ETL Pipeline (Record & Transform)

* [x] **Engine Initialization:** Update `utils.py` to call `game.set_labels_buffer_enabled(True)` when the thermal sensor is flagged.
* [x] **Thermal Mask Extraction:** In `record.py`, capture `state.labels_buffer`, apply a binary threshold (`pixels > 0 = 1`) to drop environmental geometry, and resize the mask to 64x64.
* [x] **Tensor Packaging:** Save the resulting thermal arrays to the generated `.npz` archive.
* [x] **Dataset Streaming:** Update `dataset.py` to load the thermal arrays and feed them into the model alongside the visual input.

### 3. The Brain (Architecture Redesign)

* [x] **Parallel Thermal Cortex:** Update `brain.py` to instantiate an isolated `nn.Conv2d` branch dedicated to processing the thermal mask, allowing the network to learn independent dynamic entity-tracking filters.
* [x] **Sensor Fusion:** Concatenate the flattened thermal feature map with the visual/depth and auditory representations before routing the unified tensor into the CfC liquid core.

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

---

## Distant Future

### The Possession (Integration) 👻

* *Goal:* Move beyond the Python wrapper and replace in-game enemy AIs.
* [-] **Engine Fork:** Compile the PyTorch model to TorchScript (`.pt`) and link `libtorch` directly into a C++ source port to bypass pixel-rendering entirely.