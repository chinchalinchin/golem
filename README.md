# Golem: The DOOM LNN Project

**Golem** is an open-source initiative to develop autonomous, adaptive agents for *DOOM* using **Liquid Neural Networks (LNNs)**.

Current AI in *DOOM* relies on finite state machines (FSMs) written in the 90s. While functional, they are predictable and stateless. Golem aims to replace these static heuristics with **Neural Circuit Policies (NCPs)**—biologically inspired neural networks that model time as a continuous flow rather than discrete ticks.



Unlike Large Language Models (LLMs) which hallucinate state, or traditional Reinforcement Learning (RL) which requires millions of training steps, LNNs are:
* **Causal:** They learn cause-and-effect relationships in noisy environments.
* **Compact:** Runnable on consumer hardware with minimal latency (<20ms).
* **Continuous:** They handle the variable time-steps of a game engine natively.

---

## 🏗 Architecture

The project follows a strict ETL (Extract, Transform, Load) pipeline pattern.

```text
/golem
├── conf/               # Centralized Configuration
│   ├── app.yaml        # App settings (hyperparameters, paths)
│   └── custom.cfg      # ViZDoom engine constraints
├── data/               # Data Storage
│   ├── *.npz           # Training Tensors
│   └── golem_brain.pth # Trained Model Weights
├── app/                # Source Code
│   ├── record.py       # ETL: Capture gameplay -> Tensor
│   ├── dataset.py      # Stream: Sliding window time-series loader
│   ├── brain.py        # Model: CNN + Liquid CfC Architecture
│   ├── train.py        # Spark: Behavioral Cloning Loop
│   └── run.py          # Body: Live Inference Engine
└── main.py             # CLI Entrypoint

```

## 🚀 Setup

**Prerequisites:** Python 3.10+ (ViZDoom requires a modern C++ compiler if building from source).

```bash
# 1. Create Environment
python -m venv .venv
source ./.venv/bin/activate

# 2. Install Dependencies
pip install -r requirements.txt

```

## 🛠 Usage

### 1. Configure

Edit `conf/app.yaml` to adjust hyperparameters or resolution.
Edit `conf/custom.cfg` to modify the available button definitions.

### 2. Record (The Eyes)

Launch the engine in Spectator Mode to capture training data.

```bash
python main.py record

```

*Controls:* `W` (Attack), `A` (Left), `D` (Right), `Space` (Attack).

### 3. Inspect (QA)

Verify your dataset is balanced and normalized.

```bash
python main.py inspect

```

### 4. Train (The Spark)

Run the training loop to create a `.pth` model file.

```bash
python main.py train

```

*Note: On Apple Silicon (M1/M2/M3/M4), this automatically uses Metal Performance Shaders (MPS).*

### 5. Run (The Body)

Watch the LNN play the game live.

```bash
python main.py run

```

---

## 🗺 Roadmap

### Phase 1: The Foundation (Completed) ✅

* [x] **ETL Pipeline:** Robust recording of pixel buffers and input vectors.
* [x] **Engine Bridge:** ViZDoom integration with custom `cfg` injection.
* [x] **Data Sanitation:** Automated inspection tools.
* [x] **Architecture:** Modular, config-driven Python application.

### Phase 2: The Brain (Completed) ✅

* [x] **Dataset Class:** `IterableDataset` with sliding window segmentation.
* [x] **Liquid Network:** `CfC` (Closed-form Continuous) implementation.
* [x] **Training Loop:** Behavioral Cloning with `BCEWithLogitsLoss`.
* [x] **Optimization:** Auto-device selection (CUDA/MPS/CPU).

### Phase 3: The Body (Completed) ✅

* [x] **Inference Engine:** Real-time game loop driven by the LNN.
* [x] **Action Logic:** Thresholding and conflict suppression logic.

### Phase 4: The Possession (Integration) 👻

* *Goal:* Move beyond the Python wrapper.
* [ ] **Server-Side Agent:** Run Golem as a "Ghost Client" on a dedicated Zandronum/Odamex server.
* [ ] **ACS/ZScript Bridge:** Expose decision vectors to internal engine scripting.

---

## 📜 License

MIT License.
*DOOM is a registered trademark of id Software.*