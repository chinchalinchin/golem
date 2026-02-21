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

* [~] **Headless Rendering:** Configure the ViZDoom to render the visual buffer headlessly.
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

## Phase 3: Multi-Modal Sensor Fusion (Phenomenology) 👁️👂

*Goal:* Expand the agent's sensory perception beyond the 2D pixel array by integrating ViZDoom's raw depth and audio buffers into the Liquid Neural Network, effectively granting stereopsis and audition without exposing underlying game-state variables.

### 1. The Configuration Layer

* [ ] **Sensor Toggles:** Update `app.yaml` to include a `brain.sensors` block with boolean toggles for `depth` and `audio`.
* [ ] **Dynamic Action Space:** Update `config.py` to parse these toggles and pass them to the ETL and Model initialization layers.

### 2. The ETL Pipeline (Record & Transform)

* [ ] **Depth Extraction:** Modify `record.py` to capture `state.depth_buffer`. Normalize the 1D distance matrix to $[0, 1]$.
* [ ] **Audio Extraction:** Modify `record.py` to capture `state.audio_buffer`. Normalize the raw stereo waveforms.
* [ ] **Tensor Packaging:** Update the `.npz` saving mechanism to store `depth` and `audio` arrays only if they are enabled in the active profile, preventing massive file bloat for purely visual agents.

### 3. The Brain (Architecture Redesign)

* [ ] **Stereopsis Integration:** If `depth` is enabled, modify the Visual Cortex CNN input channels from $C=3$ (RGB) to $C=4$ (RGB + Depth).
* [ ] **Auditory Cortex:** If `audio` is enabled, implement a parallel 1D Convolutional Neural Network (`nn.Conv1d`) to extract features from the high-frequency audio buffer.
* [ ] **Sensor Fusion:** Concatenate the flattened visual/depth feature vector with the auditory feature vector before passing the unified tensor into the Liquid `CfC` core.

---

## Distant Future

### The Possession (Integration) 👻
* *Goal:* Move beyond the Python wrapper and replace in-game enemy AIs.
* [ ] **Server-Side Agent:** Run Golem as a "Ghost Client" on a dedicated Zandronum/Odamex server.
* [ ] **Engine Fork:** Compile the PyTorch model to TorchScript (`.pt`) and link `libtorch` directly into a C++ source port to bypass pixel-rendering entirely.