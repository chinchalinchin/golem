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

## Phase 2: Distributed Architecture (The Cyber-Marine) 🤖

*Goal:* Containerize the LNN agent and connect it to a dedicated multiplayer DOOM server as an external player client. This scales the training and inference pipeline without requiring engine-level C++ modifications.

### 1. The Host Server (Arena)

* [ ] **Dedicated Host Script:** Create a `host.py` script to initialize a central ViZDoom instance in `-host N -deathmatch` mode. This node manages physics and state but does not require a neural network.
* [ ] **Multiplayer Configuration:** Update `custom.cfg` and select WADs with proper multiplayer spawn points (e.g., `cig.wad`).

### 2. The Container (Ghost in the Shell)

* [ ] **Dockerfile:** Package the trained Golem Brain (`.pth`), Python 3.10+, PyTorch, and the ViZDoom engine dependencies (SDL2, OpenAL) into a lightweight Linux base image.
* [ ] **Headless Rendering:** Configure the container to render the 64x64 ViZDoom visual buffer completely off-screen, bypassing the need for an X11 window display.

### 3. The Client Interface (Network Bridge)

* [ ] **Multiplayer Inference Script:** Create `run_client.py`. Instead of launching a local game, this script uses ViZDoom's `-join <IP>` launch parameter to connect to the Host.
* [ ] **State Synchronization:** Utilize ViZDoom's `Mode.PLAYER` (Sync Mode) to ensure the LNN's inference tick-rate stays perfectly aligned with the network server, preventing action drift or network drops.

### 4. Orchestration (The Swarm)

* [ ] **Docker Compose:** Create a `docker-compose.yml` to effortlessly spin up 1 Host and $N$ Agent containers simultaneously on a single machine or cluster.
* [ ] **Agent Parameterization:** Pass unique names, colors, and hyperparameters to individual containers via environment variables so the agents can be easily identified in the deathmatch logs.

---

## Distant Future

### The Possession (Integration) 👻
* *Goal:* Move beyond the Python wrapper and replace in-game enemy AIs.
* [ ] **Server-Side Agent:** Run Golem as a "Ghost Client" on a dedicated Zandronum/Odamex server.
* [ ] **Engine Fork:** Compile the PyTorch model to TorchScript (`.pt`) and link `libtorch` directly into a C++ source port to bypass pixel-rendering entirely.