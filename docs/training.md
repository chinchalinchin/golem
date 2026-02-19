# Training Methodology

Golem is trained via **Behavioral Cloning (BC)**, a form of Imitation Learning (IL). We treat the problem as a supervised multi-label classification task over continuous time-series data.

## Sliding Window Data Loading

LNNs require temporal context. We cannot shuffle individual frames. The `DoomStreamingDataset` implements a sliding window strategy. Given a recording of length $T$ and a sequence length $L$ (e.g., 32 frames):

$$
X_i = \{ s_t \}_{t=i}^{i+L}, \quad Y_i = \{ a_t \}_{t=i}^{i+L}
$$

## Class Imbalance & Mirror Augmentation

Human gameplay data is notoriously unbalanced. Players naturally favor turning one direction (e.g., left) over another in mazes, and spend 95% of their time walking rather than shooting. Without intervention, models suffer from the "Zoolander Problem" (inability to turn right) or become "Pacifists" (refusing to shoot to minimize statistical error).

Golem solves spatial biases using **Mirror Augmentation**. During data streaming, the dataset dynamically yields a horizontally flipped version of the video tensor and explicitly swaps the corresponding action labels:

* Move Left $\leftrightarrow$ Move Right
* Turn Left $\leftrightarrow$ Turn Right

This effectively doubles the dataset size for free and forces perfect symmetry in the agent's spatial reasoning.

## Diagnostic Auditing

Because standard Loss metrics obscure class-imbalance failures, Golem includes an `audit` module. This runs a static "Brain Scan" on a validation slice, generating a Precision/Recall matrix for every individual button in the 8-button Superset. 

* **Precision:** When the agent presses a button, how often was it correct?
* **Recall:** When a situation demanded a button press, how often did the agent react?