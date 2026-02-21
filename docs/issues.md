# Issue Board: Open Issues & Enhancements

## The "Hold W" Convergence Trap (Class Imbalance)

**Status:** Open | **Priority:** Medium

**Description:**

DOOM datasets—especially navigation-focused modules like `my_way_home.wad`—are inherently unbalanced. An agent will spend 90% of an episode holding `MOVE_FORWARD`, which causes standard Binary Cross-Entropy (`BCEWithLogitsLoss`) to converge to a local minimum. The agent learns that perpetually predicting `MOVE_FORWARD` and ignoring visual stimuli yields the lowest overall loss. 

**Proposed Solution:**

Replace standard BCE with a **Focal Loss** function, or apply **Dynamic Sample Weighting** within the `DoomStreamingDataset`. 

Focal loss adds a modulating factor $(1 - p_t)^\gamma$ to the standard cross-entropy criterion, dynamically scaling down the gradient of easily classified, high-frequency actions (like walking forward), and heavily penalizing the network when it misses rare, high-value actions (like `ATTACK` or `USE`).

**Implementation Notes:**

* Calculate dataset action distributions during the `transform` or `dataset` loading phase.
* Pass the resulting weights tensor to the loss function in `train.py`.