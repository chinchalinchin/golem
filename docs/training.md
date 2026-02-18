
# Training Methodology

Golem is trained via **Behavioral Cloning (BC)**, a form of Imitation Learning (IL). We treat the problem as a supervised multi-label classification task.

## The Objective

Given a dataset of expert trajectories $\mathcal{D} = \{ (s_t, a_t) \}_{t=1}^T$, we aim to minimize the divergence between the policy $\pi_\theta(s_t)$ and the expert action $a_t$.

## Loss Function

Since the action space is multi-label (actions are not mutually exclusive), we cannot use `CrossEntropyLoss` (Softmax). Instead, we model each action channel as an independent Bernoulli trial using **Binary Cross Entropy with Logits (BCEWithLogitsLoss)**.

$$
L(\theta) = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C \left[ y_{i,c} \cdot \log(\sigma(\hat{y}_{i,c})) + (1 - y_{i,c}) \cdot \log(1 - \sigma(\hat{y}_{i,c})) \right]
$$

Where:
* $N$: Batch size.
* $C$: Number of action channels (3: Left, Right, Attack).
* $\sigma$: Sigmoid function $\frac{1}{1+e^{-x}}$.

## Sliding Window Data Loading

LNNs require temporal context. We cannot shuffle individual frames. The `DoomStreamingDataset` implements a sliding window strategy.

Given a recording of length $T$ and a sequence length $L$ (e.g., 32 frames):

$$
X_i = \{ s_t \}_{t=i}^{i+L}, \quad Y_i = \{ a_t \}_{t=i}^{i+L}
$$

The batch tensor shapes are:
* Input: $(B, L, C, H, W)$
* Target: $(B, L, A_{dim})$

This ensures the CfC cell receives a coherent stream of $L$ timesteps to establish its internal derivative state.
