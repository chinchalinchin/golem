
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

## Distant Future

### The Possession (Integration) 👻

* *Goal:* Move beyond the Python wrapper.
* [ ] **Server-Side Agent:** Run Golem as a "Ghost Client" on a dedicated Zandronum/Odamex server.
* [ ] **ACS/ZScript Bridge:** Expose decision vectors to internal engine scripting.
