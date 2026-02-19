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
├── docs/               # Documentation
│   ├── brain.md        # Model docs
│   ├── environment.md  # Environment docs
│   ├── index.md        # Docs Index
│   ├── roadmap.md      # Project roadmap
│   └── training.md     # Training docs
├── data/               # Data Storage
│   ├── *.npz           # Training Tensors
│   └── golem_brain.pth # Trained Model Weights
├── app/                # Source Code
│   ├── audit.py        # Analyis: Post-training Model Analysis
│   ├── brain.py        # Model: CNN + Liquid CfC Architecture
│   ├── config.py       # Configuration: Application models.
│   ├── dataset.py      # Stream: Sliding window time-series loader
│   ├── inspect.py      # Analysis: Data analysis handler
│   ├── record.py       # ETL: Capture gameplay -> Tensor
│   └── run.py          # Run: Live Inference Engine
│   ├── train.py        # Training: Behavioral Cloning Loop
│   └── utils.py        # Utilties: Functions
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

### 2. Record

Launch the engine in Spectator Mode to capture training data.

```bash
python main.py record
```

*Controls:* `W` (Attack), `A` (Left), `D` (Right), `Space` (Attack).

### 3. Inspect

Verify your dataset is balanced and normalized.

```bash
python main.py inspect
```

### 4. Train

Run the training loop to create a `.pth` model file.

```bash
python main.py train
```

*Note: On Apple Silicon (M1/M2/M3/M4), this automatically uses Metal Performance Shaders (MPS).*

### 5. Audit

Verify the training created a well balanced model.

```bash
python main.py audit
```

### 6. Run

Watch the LNN play the game live.

```bash
python main.py run
```

## Current Model

The current model is available in `data/golem_brain.pth`.

---

## References 

- [vizdoom](https://vizdoom.farama.org/)