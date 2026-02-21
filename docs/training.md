# Training Methodology

Golem is trained via **Behavioral Cloning (BC)**, a foundational paradigm of Imitation Learning (IL). By treating the expert's gameplay traces as the optimal policy $\pi^*$, the training regime is formulated as a supervised, multi-label sequence classification task over continuous time-series data.

## 1. Sliding Window Temporal Loading

Because Liquid Neural Networks (LNNs) and their Closed-form Continuous (CfC) approximations model a continuous hidden state $x(t)$, individual frames cannot be uniformly shuffled during training. The dataset pipeline enforces temporal causality via a sliding window extraction protocol, dynamically loading tensors from isolated profile directories (e.g., `data/fluid/`).

Given an expert trajectory of length $T$, defined as $\tau = \{ (o_1, y_1), (o_2, y_2), \dots, (o_T, y_T) \}$, and a fixed temporal sequence length $L$ (e.g., $L=32$), we extract sequence batches. The input tensor sequence $\mathbf{X}_i$ and target action sequence $\mathbf{Y}_i$ starting at index $i$ are:

$$
\mathbf{X}_i = \{ o_t \}_{t=i}^{i+L-1}, \quad \mathbf{X}_i \in \mathbb{R}^{L \times 3 \times 64 \times 64}
$$

$$
\mathbf{Y}_i = \{ y_t \}_{t=i}^{i+L-1}, \quad \mathbf{Y}_i \in \{0, 1\}^{L \times n_\rho}
$$

Where $n_\rho$ is the dimensionality of the action space dictated by the active environment profile $\rho$ (e.g., Basic, Classic, Fluid).

## 2. The Objective Function (BCE Loss)

At each time step $t$, the network outputs a vector of raw logits $\mathbf{z}_t \in \mathbb{R}^{n_\rho}$. Because the action space allows for simultaneous key presses (e.g., strafing right while firing), the objective is evaluated using **Binary Cross-Entropy** with Logits Loss.

The loss $\mathcal{L}$ for a single sequence of length $L$ over $n_\rho$ independent binary action channels is computed as:

$$
\mathcal{L}(\theta) = - \frac{1}{L \cdot n_\rho} \sum_{t=1}^{L} \sum_{j=1}^{n_\rho} \left[ y_{t,j} \log(\sigma(z_{t,j})) + (1 - y_{t,j}) \log(1 - \sigma(z_{t,j})) \right]
$$

Where $\sigma(\cdot)$ is the Sigmoid activation function, $y_{t,j}$ is the ground truth label, and $z_{t,j}$ is the network's prediction. The network parameters $\theta$ are updated via Backpropagation Through Time (BPTT).

## 3. Class Imbalance & Mirror Augmentation

Human gameplay datasets exhibit severe topological and behavioral biases. For example, a dataset derived from a specific maze may contain an 80/20 ratio of left turns to right turns. Furthermore, the expert spends a vast majority of frames moving rather than firing weapons. Unmitigated, this sparsity causes the network to collapse into localized minima, such as the "Zoolander Problem" (inability to turn right) or the "Hold W Trap" (convergence to a permanent state of forward movement due to high dataset idle times).

Golem counteracts spatial bias dynamically via **Mirror Augmentation**. During data streaming, the dataset yields a reflected observation tensor $o'_t$ across the vertical axis (width):

$$
o'_{t, c, h, w} = o_{t, c, h, W - w - 1}
$$

To maintain ground-truth causality, the corresponding target vector $y'_t$ must undergo a specific permutation. Let $P_\rho$ be an $n_\rho \times n_\rho$ permutation matrix defined by the active profile $\rho$, which swaps the indices corresponding to strictly spatial actions:

- $idx_{\text{MoveLeft}} \leftrightarrow idx_{\text{MoveRight}}$
- $idx_{\text{TurnLeft}} \leftrightarrow idx_{\text{TurnRight}}$

All state-invariant actions (e.g., Attack, Use, NextWeapon) map to the identity matrix within $P_\rho$. The augmented target vector is thus:

$$
y'_t = P_\rho y_t
$$

This geometric inversion enforces perfect spatial symmetry in the agent's spatial reasoning, effectively doubling the dataset's topological variance without requiring additional recording sessions.

## 4. Covariate Shift & DAgger Intervention

A fundamental flaw of pure Behavioral Cloning is **Covariate Shift** (the "Perfect Play" trap). If the network is trained exclusively on flawless expert demonstrations, it never learns how to recover from mistakes. During live inference, a microscopic mathematical error will push the agent slightly off the optimal trajectory. Because this sub-optimal state $s_{err}$ exists outside the training distribution, the agent's predictions become chaotic, and the errors rapidly compound until the agent is completely stuck.

To cure this, Golem employs **DAgger (Dataset Aggregation)**. During live inference, the human expert monitors the autonomous agent. If the agent enters an equilibrium state (e.g., staring into a corner), the human holds a hotkey to instantly suspend the LNN's logits and hijack the controls. This intervention steers the agent back to the optimal path, automatically appending a `_recovery` dataset trace to the active profile's training directory. This explicitly teaches the network how to correct trajectory deviations.

## 5. Diagnostic Auditing

Because the aggregate BCE loss scalar $\mathcal{L}(\theta)$ fundamentally obscures multi-label class imbalances (a model that never shoots will still achieve 95% accuracy if the "Attack" label is sparse), Golem utilizes a dedicated static `audit` module.

The audit evaluates the trained weights over a validation slice by generating a strictly thresholded ($\sigma(\mathbf{z}) > 0.5$) Confusion Matrix for every individual channel $j$ in the $n_\rho$ action space. It evaluates the network based on:

**Precision ($P_j$)**

The probability that the agent's decision to act was correct.
    
$$
P_j = \frac{TP_j}{TP_j + FP_j}
$$

**Recall ($R_j$)**: 

The probability that the agent successfully reacted to an environmental stimulus requiring action $j$.

$$
R_j = \frac{TP_j}{TP_j + FN_j}
$$

Where $TP$, $FP$, and $FN$ represent True Positives, False Positives, and False Negatives, respectively.

---

## API Reference

The data extraction, LNN optimization, and evaluation mechanics are orchestrated by the handlers below.

### The Training Loop

::: app.pipeline.train.train

### DAgger Intervention

::: app.pipeline.intervene.intervene_agent

### Data Inspection & Auditing

::: app.pipeline.analyze.inspect_data

::: app.pipeline.analyze.audit_agent