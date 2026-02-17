# Golem: The DOOM LNN Project

**Golem** is an open-source initiative to develop autonomous, adaptive agents for *DOOM* using **Liquid Neural Networks (LNNs)**.

Current AI in *DOOM* relies on finite state machines (FSMs) written in the 90s. While functional, they are predictable and stateless. Golem aims to replace these static heuristics with **Neural Circuit Policies (NCPs)**—biologically inspired neural networks that model time as a continuous flow rather than discrete ticks.

Unlike Large Language Models (LLMs) which hallucinate state, or traditional Reinforcement Learning (RL) which requires millions of training steps, LNNs are:

* **Causal:** They learn cause-and-effect relationships in noisy environments.
* **Compact:** Runnable on consumer hardware with minimal latency (<20ms).
* **Continuous:** They handle the variable time-steps of a game engine natively.

---

## 🏗 Architecture

The project follows a strict ETL (Extract, Transform, Load) pipeline pattern to ensure data integrity and reproducibility.

```text
/golem
├── conf/               # Centralized Configuration (YAML + CFG)
│   ├── app.yaml        # Application settings (logging, paths, hyperparameters)
│   └── custom.cfg      # ViZDoom engine constraints
├── data/               # Artifact Storage
│   └── *.npz           # Normalized training tensors (frames + actions)
├── app/                # Source Code
│   ├── record.py       # ETL: Capture human gameplay -> Tensor
│   ├── inspect.py      # QA: Analyze class balance and tensor shapes
│   ├── brain.py        # Model: CNN + Liquid CfC Architecture
│   └── config.py       # Pydantic schema for type-safe config
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

All interactions are handled via the `main.py` CLI.

### 1. Configure

Edit `conf/app.yaml` to adjust resolution, logging levels, or storage paths.

Edit `conf/custom.cfg` to modify the DOOM engine parameters (rendering flags, available buttons).

### 2. Extract Data (Record)

Launch the engine in **Spectator Mode**. The agent records your inputs and the raw pixel buffer, transforming them into normalized tensors.

```bash
python main.py record
```

* **Controls:** `W` (Attack), `A` (Left), `D` (Right), `Space` (Attack).
* **Output:** Saved to `data/doom_training_data_X.npz`.

### 3. Quality Assurance (Inspect)

Before training, verify your dataset isn't biased toward inaction (Idling).

```bash
python main.py inspect
```

* **Checks:** Tensor normalization (0-1), Action distribution, Idle percentage.

---

## 🗺 Roadmap

This is a hallowed tradition. We do not ship spaghetti code.

### Phase 1: The Foundation (Completed) ✅

* [x] **ETL Pipeline:** Robust recording of pixel buffers and input vectors.
* [x] **Engine Bridge:** ViZDoom integration with custom `cfg` injection.
* [x] **Data Sanitation:** Automated inspection tools to detect class imbalance.
* [x] **Architecture:** Modular, config-driven Python application.

### Phase 2: The Brain (In Progress) 🧠

* [ ] **Dataset Class:** Implement a PyTorch `IterableDataset` to handle time-series windowing (e.g., sequence length of 32 frames).
* [ ] **Training Loop:** Implement `train.py` using Behavioral Cloning (BC).
* *Loss Function:* CrossEntropy (for discrete actions) or MSE (for continuous).
* [ ] **Model Evaluation:** Visualizing loss convergence and validation accuracy.

### Phase 3: The Body (Inference) 🤖

* [ ] **Inference Engine:** Create `run.py` to load trained weights and drive the `DoomGame` instance directly.
* [ ] **Metrics:** Automated measuring of "Time to Death" and "Frags per Minute."
* [ ] **Visualizer:** Real-time overlay of the LNN's hidden states (neuron firing rates) during gameplay.

### Phase 4: The Possession (Integration) 👻

* *Goal:* Move beyond the Python wrapper.
* [ ] **Server-Side Agent:** Run Golem as a "Ghost Client" on a dedicated Zandronum/Odamex server to act as a dynamic bot.
* [ ] **ACS/ZScript Bridge:** Investigate exposing the model's decision vector directly to DOOM's internal scripting via named pipes or shared memory, allowing level designers to control monsters with LNN brains.

---

## 📜 License

MIT License.
*DOOM is a registered trademark of id Software.*