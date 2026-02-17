# Golem: The DOOM LNN

**Golem** is an experiment in applying **Liquid Neural Networks (LNNs)** to autonomous agents in *DOOM*.

Unlike Large Language Models (LLMs) which struggle with object permanence and reaction time, or traditional Reinforcement Learning which requires millions of episodes, LNNs (specifically Neural Circuit Policies) are designed to be:

* **Continuous:** Modeling time as a flow, not just discrete steps.
* **Causal:** Understanding cause-and-effect relationships.
* **Efficient:** Runnable on consumer hardware (or even edge devices).

## Project Structure

* `app/record.py`: Records human gameplay into a training dataset (`.npz`).
* `app/brain.py`: Defines the LNN architecture (CNN + CfC).
* `app/custom.cfg`: ViZDoom configuration for the "Basic" scenario.

## Setup

```bash
python -m venv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Record Training Data

Play the game to teach the Golem how to move and shoot.

```bash
python app/record.py

```

* **Controls:** `W` (Shoot), `A` (Left), `D` (Right), `Space` (Shoot).
* **Output:** Saves to `app/doom_training_data.npz` (auto-iterates filenames).

## Roadmap

* [x] **Phase 1: The Eyes & Ears**
* [x] Configure ViZDoom environment.
* [x] Create recording script to capture human input and pixel data.
* [x] Verify data integrity (Tensor shapes and Action mapping).


* [ ] **Phase 2: The Brain (Current Focus)**
* [ ] Create `train.py`.
* [ ] Implement a `Dataset` class to slice the recording into time-series windows (e.g., 32-frame sequences).
* [ ] Train the `CfC` (Closed-form Continuous) network using Behavioral Cloning.


* [ ] **Phase 3: The Body**
* [ ] Create `run.py` to let the LNN play the game live.
* [ ] Measure survival time and kill count.
