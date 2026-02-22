# Issue Archive

## Issue: DRY Violation in Sensory Extraction (Pipeline Refactoring)

**Status:** Closed | **Priority:** Low | | **Opened**: 2026/02/21 | **Closed**: 2026/02/21

**Description:**

The ETL logic that extracts, resizes, and normalizes the visual, depth, audio, and thermal buffers is heavily duplicated across `record.py`, `intervene.py`, and `run.py`. This violates the DRY (Don't Repeat Yourself) principle. If a new phenomenological sensor is added in the future, the extraction logic must be manually updated in three separate pipeline modules.

**Proposed Solution:**

Abstract the game-state processing logic into a centralized `SensoryExtractor` utility class or function in `utils.py`. The pipelines should simply call `tensors = SensoryExtractor.process(game.get_state(), cfg.brain.sensors)` to receive a standardized dictionary of normalized inputs.

## Issue: Stateful "Past Life" Memory Leakage (Death & Respawn)

**Status:** Closed | **Priority:** High | **Opened**: 2026/02/21 | **Closed**: 2026/02/21

**Description:**

The LNN relies on accumulating evidence in its hidden state `hx`. During `run.py` and `intervene.py`, `hx` is persisted across the entire episode loop. However, in Deathmatch or custom WADs where the agent respawns after death without resetting the episode, the agent retains the `hx` state from its previous life. This causes "phantom" action potentials where the newly spawned agent reacts to stimuli that killed it seconds ago.

**Proposed Solution:**

Implement a physiological state-check inside the inference loop. If `game.is_player_dead()` is true, explicitly detach and zero-out the hidden state (`hx = None`).

## Issue: The "Hold W" Convergence Trap (Class Imbalance)

**Status:** Closed | **Priority:** Medium | **Opened**: 2026/02/19 | **Closed**: 2026/02/22

**Description:**

DOOM datasets—especially navigation-focused modules like `my_way_home.wad`—are inherently unbalanced. An agent will spend 90% of an episode holding `MOVE_FORWARD`, which causes standard Binary Cross-Entropy (`BCEWithLogitsLoss`) to converge to a local minimum. The agent learns that perpetually predicting `MOVE_FORWARD` and ignoring visual stimuli yields the lowest overall loss. 

**Proposed Solution:**

Replace standard BCE with a **Focal Loss** function, or apply **Dynamic Sample Weighting** within the `DoomStreamingDataset`. 

Focal loss adds a modulating factor $(1 - p_t)^\gamma$ to the standard cross-entropy criterion, dynamically scaling down the gradient of easily classified, high-frequency actions (like walking forward), and heavily penalizing the network when it misses rare, high-value actions (like `ATTACK` or `USE`).

**Implementation Notes:**

* Calculate dataset action distributions during the `transform` or `dataset` loading phase.
* Pass the resulting weights tensor to the loss function in `train.py`.

## Issue: Stateful Backpropagation Through Time (BPTT) Amnesia

**Status:** Closed | **Priority:** High | **Opened**: 2026/02/21 | **Closed**: 2026/02/22

**Description:**

The LNN's Closed-form Continuous (CfC) cells require a continuous flow of time to accurately accumulate evidence and trigger action potentials. Currently, the training loop implicitly initializes the hidden state `hx` as `None` (a zero-tensor) for every 32-frame sequence batch. This forces computational amnesia at the boundary of every sequence, breaking the mathematical continuity of the ODEs representing the agent's memory.

**Proposed Solution:**

Implement Stateful BPTT in `train.py`:

1. Disable `shuffle=True` across sequences of the same trajectory to ensure chronological streaming.
2. Retain the hidden state output `hx` from the previous batch.
3. Detach the state from the computational graph (`hx = hx.detach()`) to prevent backpropagating into infinite history.
4. Pass the detached state as the prior for the subsequent batch.
