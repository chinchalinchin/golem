# 🗺 Roadmap

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

## Phase 3: Distributed Architecture (The VDAIC Arena) 🤖

*Goal:* Containerize the LNN agent and connect it to a dedicated multiplayer DOOM server. Benchmark Golem against the historical champions of the Visual Doom AI Competition (VDAIC).

### 1. The Host Server (Arena)

* [ ] **Dedicated Host Script:** Create `app/handlers/host.py` to initialize a central ViZDoom instance in `-host N -deathmatch` mode. This node manages physics and state but does not require a neural network.
* [ ] **Multiplayer Configuration:** Update configs and select WADs with proper multiplayer spawn points (e.g., the official `cig.wad` used in the tournaments).
* [ ] **Host Container:** Create a lightweight `Dockerfile.host` that only installs the ViZDoom engine and the host script.

### 2. The Golem Client (Containerization)

* [ ] **Headless Rendering:** Configure the ViZDoom engine inside the client script (`run_client.py`) to render the visual buffer entirely off-screen, bypassing X11/display requirements.
* [ ] **Network Synchronization:** Utilize ViZDoom's `Mode.PLAYER` (Sync Mode) to ensure the LNN's inference tick-rate stays perfectly aligned with the network server.
* [ ] **Modular Dockerfile:** Create `Dockerfile.client`. The image should package Python 3.10+, PyTorch, and ViZDoom, but **omit the model weights**. 
* [ ] **Volume Mounting:** Configure the container entrypoint to load the `golem.pth` brain and `app.yaml` configuration from a mounted volume directory (e.g., `-v ./data/fluid:/app/data`).

### 3. The Legacy Champions (The Opposition)

* [ ] **Archive Retrieval:** Clone the legacy Dockerfiles and weights for the 2016/2017 VDAIC winners (e.g., *Arnold* by CMU, *IntelAct* by Intel Labs) from the official GitHub archives.
* [ ] **Legacy Container Builds:** Build the historical images. (Note: These will likely require older base images like Ubuntu 16.04 and deprecated versions of PyTorch/TensorFlow).

### 4. Orchestration (The Swarm)

* [ ] **Docker Compose:** Create a `docker-compose.yml` to effortlessly network the swarm. 
* [ ] **The Roster:** Define the services in the compose file to simultaneously spin up:
    * 1x Host Arena Server
    * 2x Legacy Champion Bots (e.g., Arnold, IntelAct)
    * Nx Golem Agents (using the same image, but mounting different profile volumes like `basic` or `fluid` to test action spaces against each other).
* [ ] **Agent Parameterization:** Pass unique names and colors via environment variables so the agents can be easily identified in the deathmatch logs.

---

## Distant Future

### The Possession (Integration) 👻
* *Goal:* Move beyond the Python wrapper and replace in-game enemy AIs.
* [ ] **Server-Side Agent:** Run Golem as a "Ghost Client" on a dedicated Zandronum/Odamex server.
* [ ] **Engine Fork:** Compile the PyTorch model to TorchScript (`.pt`) and link `libtorch` directly into a C++ source port to bypass pixel-rendering entirely.