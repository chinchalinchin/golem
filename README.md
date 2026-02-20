# Golem: The DOOM LNN Project

**Golem** is an open-source initiative to develop autonomous, adaptive agents for *DOOM* using **Liquid Neural Networks (LNNs)**.

Current AI in *DOOM* relies on finite state machines (FSMs) written in the 90s. While functional, they are predictable and stateless. Golem aims to replace these static heuristics with **Neural Circuit Policies (NCPs)**—biologically inspired neural networks that model time as a continuous flow rather than discrete ticks.

Unlike Large Language Models (LLMs) which hallucinate state, or traditional Reinforcement Learning (RL) which requires millions of training steps, LNNs are:

* **Causal:** They learn cause-and-effect relationships in noisy environments.
* **Compact:** Runnable on consumer hardware with minimal latency (<20ms).
* **Continuous:** They handle the variable time-steps of a game engine natively natively via Ordinary Differential Equations (ODEs).

---

## 🏗 Architecture

The project follows a strict ETL (Extract, Transform, Load) pipeline pattern, utilizing map-style datasets loaded via pointer indices to prevent memory overflow during sequence shuffling.

```text
/golem
├── conf/               # Centralized Configuration
│   ├── app.yaml        # App settings (hyperparameters, bindings, architecture)
│   ├── basic.cfg       # 8-dim superset (Movement + Shoot/Use)
│   ├── classic.cfg     # 10-dim superset (Explicit weapon slots)
│   └── fluid.cfg       # 9-dim superset (Sequential weapon toggles)
├── docs/               # Documentation
│   └── ...
├── data/               # Data Storage
│   ├── *.npz           # Training Tensors (Images + Action Vectors)
│   └── golem_brain.pth # Trained Model Weights
├── app/                # Source Code
│   ├── audit.py        # Analysis: Precision/Recall matrix generation
│   ├── brain.py        # Model: Dynamic CNN + Liquid CfC Architecture
│   ├── config.py       # Configuration: Pydantic state models
│   ├── dataset.py      # Stream: Sliding window sequence mapping & permutation
│   ├── inspect.py      # Analysis: Dataset balance and normalization checks
│   ├── intervene.py    # DAgger: Human-in-the-loop dataset aggregation
│   ├── record.py       # ETL: Capture gameplay -> Tensor
│   ├── run.py          # Run: Live Inference Engine
│   ├── templates/      # Views: Jinja2 templates for CLI reporting
│   ├── train.py        # Training: Behavioral Cloning Loop
│   └── utils.py        # Utilities: Path resolution
├── tests/              # Unit Tests
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

Edit `conf/app.yaml` to adjust hyperparameters, dynamically scale the brain architecture (`cortical_depth`, `working_memory`), and set the active environment profile (`basic`, `classic`, or `fluid`). Keybindings are injected dynamically.

### 2. Record

Launch the engine in Spectator Mode to capture training data.

```bash
python main.py record --module combat
```

### 3. Intervene (DAgger)

Run the agent autonomously, but hold **Left Shift** to instantly override the LNN logits with manual keyboard input. This generates a `_recovery` dataset to cure Covariate Shift.

```bash
python main.py intervene --module combat
```

### 4. Inspect

Verify your dataset is balanced and normalized (checks for high idle time).

```bash
python main.py inspect
```

### 5. Train

Run the Behavioral Cloning loop. Uses dynamic tensor permutation for spatial Mirror Augmentation.

```bash
python main.py train --module all
```

*Note: On Apple Silicon (M1/M2/M3/M4), this automatically uses Metal Performance Shaders (MPS).*

### 6. Audit

Run a diagnostic Brain Scan to check for class-imbalance failures against a strict threshold.

```bash
python main.py audit --module all
```

### 7. Run

Watch the LNN play the game live. The agent manages a persistent hidden state (`hx`) continuously.

```bash
python main.py run --module combat
```

## Continuous Integration

The Github Actions defined in `.github/workflows/ci.yml` run the unit tests in `tests/*` and compile the docs in `docs/*`. These actions are run inside of a container built with the `Dockerfile.ci` image. This image pre-packages all of the runtime dependencies.

```bash
docker buildx build \
    -f Dockerfile.ci \
    --platform linux/amd64 \
    -t chinchalinchin/golem-ci:latest \
    . \
    --push
```

---

## References

* [ViZDoom Official Documentation](https://vizdoom.farama.org/)
* [Liquid Neural Networks (Hasani et al.)](https://arxiv.org/abs/2006.04439)
