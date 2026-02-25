# Golem: The DOOM LNN Project

**Golem** is an open-source initiative to develop autonomous, adaptive agents for *DOOM* using **Liquid Neural Networks (LNNs)**.

Current AI in *DOOM* relies on finite state machines (FSMs) written in the 90s. While functional, they are predictable and stateless. Golem aims to replace these static heuristics with **Neural Circuit Policies (NCPs)**—biologically inspired neural networks that model time as a continuous flow rather than discrete ticks.


Unlike Large Language Models (LLMs) which hallucinate state, or traditional Reinforcement Learning (RL) which requires millions of training steps, LNNs are:

* **Causal:** They learn cause-and-effect relationships in noisy environments.
* **Compact:** Runnable on consumer hardware with minimal latency (<20ms).
* **Continuous:** They handle the variable time-steps of a game engine natively via Ordinary Differential Equations (ODEs).

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
│   └── training/       # Training Data
│       └── <mode>/     # Recorded Training Sessions (.npz)
│   └── model/          # Model Archive
│       └── <mode>/     # Previous Trained Model Weights (.pth)
├── app/                # Source Code
│   ├── client/         # Gameplay Modules (agent, host)
│   ├── metrics/        # Analysis Modules (audit, examine, inspect, summary)
│   ├── models/         # Data Models (brain, config, dataset)
│   ├── pipeline/       # ML Modules (analyze, intervene, record, run, train)
│   ├── templates/      # Jinja2 report templates
│   └── utils           # Application utilities
├── tests/              # Unit Tests
└── main.py             # CLI Entrypoint
```

The `./data/model/<mode>/` directory archives models using the naming schema,

`{YYYY-MM-DD}.c-{c}.w-{w}.v-{v}.d-{d}.a-{a}.t-{t}.{sr-{sr}.nf-{nf}.hl-{hl}.nm-{nm}}.{increment}.{loss}-{params}`

Where,

* `YYYY-MM-DD`: The date the model was trained.
* `c`: The cortical depth (`brain.cortical_depth`) of the model.
* `w`: The working memory (`brain.working_memory`) of the model.
* `v`: The visual field (`brain.sensors.visual`) of the model.
* `d`: The depth field (`brain.sensors.depth`) of the model.
* `a`: The audio field (`brain.sensors.audio`) of the model.
* `t`: The thermal field (`brain.sensors.thermal`) of the model.
* The fields `{sr-{sr}.nf-{nf}.hl-{hl}.nm-{nm}}.` are conditional on the `brain.sensors.audio` being enabled. 
    * `sr`: The sample rate of the DSP (`brain.dsp.sample_rate`) of the model.
    * `nf`: The length of the STFT signal (`brain.dsp.nfft`) for the model.
    * `hl`: The hop length (`brain.dsp.hop_length`) of the model.
* `increment`: Auto-incrementing integer to prevent overwrites.
* `loss`: Loss function used to train the model.
* `params`: Loss function parameters used to train the model.

The active model for any given profile is always saved to `./data/<mode>/golem.pth` to isolate action-space dimensions and prevent PyTorch tensor mismatch errors during inference.

## 🚀 Setup

**Prerequisites:** Python 3.10+ (ViZDoom requires a modern C++ compiler if building from source).

```bash
# 1. Create Environment
python -m venv .venv
source ./.venv/bin/activate

# 2. Install Dependencies
pip install -r requirements.txt
```

### 🗺️ Procedural Generation (Oblige)

To prevent spatial overfitting and Covariate Shift, Golem utilizes Oblige 7.70 to procedurally generate training maps on the fly. Because the official binaries are outdated, macOS/Apple Silicon users must build the engine from source.

Download the source from [this link](http://sourceforge.net/projects/oblige/files/Oblige/7.70/oblige-770-source.zip) and extract it to a directory adjacent to Golem (e.g., ../oblige).

Install Dependencies,

```bash
brew install fltk zlib
```

Oblige 7.70 expects FLTK 1.3. Bypass this check to compile with modern 1.4+ versions,

```bash
cd ../oblige
sed -i '' 's/#error "Require FLTK version 1.3.0 or later"/\/\/ Bypassed FLTK 1.4.x version check/g' gui/ui_window.cc
```

Link Homebrew Libraries and Compile,

```bash
export LIBRARY_PATH="/opt/homebrew/lib:/opt/homebrew/opt/zlib/lib:$LIBRARY_PATH"
export CPATH="/opt/homebrew/include:/opt/homebrew/opt/zlib/include:$CPATH"
make -f Makefile.macos
```

Update `conf/app.yaml` to point to the compiled executable,

```yaml
randomizer:
  executable: "/absolute/path/to/oblige/Oblige"
  output: "data/wads/"
```

## 🛠 Usage

### 1. Configure

Edit `conf/app.yaml` to adjust hyperparameters, dynamically scale the brain architecture (`cortical_depth`, `working_memory`), and set the active environment profile via `brain.mode` (`basic`, `classic`, or `fluid`). Keybindings are injected dynamically based on this mode.

### 2. Record

Launch the engine in Spectator Mode to capture training data.

```bash
python main.py record --module combat
```

### 3. Generate

Compile a randomized BSP map using Oblige,

```bash
python main.py generate
```

The generated wad can be wired into the training sessions through the `app.yaml`'s `modules` property. In addition, to simplify the process, the application can run through a series of randomized maps and record the results,

```bash
python main.py randomize
```

### 3. Intervene

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

Find available models,

```bash
python.py main.py list
```

Run a diagnostic Brain Scan to check for class-imbalance failures against a strict threshold.

```bash
python main.py audit --module all
```

### 7. Run

Watch the LNN play the game live. The agent manages a persistent hidden state (`hx`) continuously.

```bash
python main.py run --module combat
```

## ⚔️ Multiplayer Benchmarking (The VDAIC Arena)

Golem features a containerized, deterministic lockstep networking mode to benchmark its Liquid Neural Network against historical champions from the Visual Doom AI Competition (VDAIC).

### 1. Build the Golem Agent Image

Build the unified Docker image that contains the ViZDoom engine, Python 3.10, and the Golem source code.

```bash
docker buildx build \
    -f Dockerfile.agent \
    --platform linux/amd64 \
    -t golem-agent:latest .
```

### 2. Build Legacy Champions (IntelAct)

To pit Golem against the 2017 VDAIC champion, you must clone the historical repository and build its legacy Python 2.7 environment. The legacy Intelact Dockerfile requires patching to use an `ubuntu:16.04` base image and a CPU-only TensorFlow 0.9.0 wheel to run on modern hardware. The historical repository has been forked and these updates have been committed there.

```bash
# Clone the repository
git clone https://github.com/chinchalinchin/VDAIC2017
cd VDAIC2017
cp cig2017.wad intelact/
cp _vizdoom.cfg intelact/
docker buildx build \
    -f intelact/Dockerfile \
    --platform linux/amd64 \
    -t intelact:local intelact/
```

!!! "Docker Image"
    The image can also be retrieved from [DockerHub](https://hub.docker.com/repository/docker/chinchalinchin/intelact/general)

### 3. Enter the Arena

From the Golem project root, use Docker Compose to orchestrate the swarm. This spins up the headless Host server, mounts your local data/ directory so Golem can read its trained weights, and boots the legacy IntelAct adversary.

```bash
docker-compose up
```

## ⚙️ Continuous Integration & Deployment

Golem uses GitHub Actions to automate testing, documentation deployment, and container image releases. 

### CI/CD Workflows

* **Integration (`ci.yml`):** Triggered on pushes to the `master` branch. This workflow runs the unit tests in `tests/*` and compiles the MkDocs documentation. These steps execute inside the pre-packaged `chinchalinchin/golem-ci` container to ensure environment consistency.
* **Release (`release.yml`):** Triggered on pushes to the `release` branch. This workflow automatically builds and pushes the latest Docker images to Docker Hub. 
    * [chinchalinchin/golem-ci:latest](https://hub.docker.com/repository/docker/chinchalinchin/golem-ci) - The base image containing all heavy system dependencies for testing and building.
    * [chinchalinchin/golem-agent:latest](https://hub.docker.com/repository/docker/chinchalinchin/golem-agent/) - The containerized ViZDoom agent used for the VDAIC Arena swarm.

### Manual Builds

If you need to manually build these images locally (e.g., for testing architecture changes), you can use Docker Buildx to build platform agnostic images:

```bash
# Build and push the CI image
docker buildx build \
    -f Dockerfile.ci \
    --platform linux/amd64 \
    -t chinchalinchin/golem-ci:latest \
    . \
    --load

# Build and push the Agent image
docker buildx build \
    -f Dockerfile.agent \
    --platform linux/amd64 \
    -t chinchalinchin/golem-agent:latest \
    . \
    --load
```

---

## References

* [VDAIC 2017 Github](https://github.com/mihahauke/VDAIC2017)
* [ViZDoom Github](https://github.com/Farama-Foundation/ViZDoom/tree/master)
* [ViZDoom Official Documentation](https://vizdoom.farama.org/)
* [Oblige Level Maker](https://oblige.sourceforge.net/)
* [Pynput Documentation](https://pynput.readthedocs.io/en/latest/)

### Papers

* [Asymmetric Loss (Ben-Baruch et al.)](https://arxiv.org/abs/2009.14119)
* [Liquid Neural Networks (Hasani et al.)](https://arxiv.org/abs/2006.04439)