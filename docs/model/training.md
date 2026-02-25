# Training Methodology

Golem is trained via **Behavioral Cloning (BC)**, a foundational paradigm of Imitation Learning (IL). By treating the expert's gameplay traces as the optimal policy $\pi^*$, the training regime is formulated as a supervised, multi-label sequence classification task over continuous time-series data.

## 1. Contiguous Temporal Loading

Because Liquid Neural Networks (LNNs) and their Closed-form Continuous (CfC) approximations model a continuous hidden state $x(t)$, individual frames cannot be uniformly shuffled during training. The dataset pipeline enforces temporal causality via a contiguous sequence extraction protocol, dynamically loading tensors from isolated profile directories (e.g., `data/fluid/`).

To prevent Out-Of-Memory (OOM) crashes when processing hours of high-dimensional multi-modal gameplay, Golem does not duplicate flat arrays in memory. Instead, it constructs a lightweight pointer map consisting of tuples `(file_idx, start_idx, is_mirrored, is_first)`. During training, continuous arrays are lazily sliced on-the-fly into strictly contiguous, non-overlapping sequences with a stride equal to $L$.



To prevent Stateful BPTT Continuity Collapse (where a naive dataloader would effectively teleport the agent's memory hundreds of frames into the future between batches), PyTorch's default `DataLoader` iteration is replaced with a custom `StatefulStratifiedBatchSampler`. This sampler abandons a flat list structure and instead manages $B$ independent parallel streams. It guarantees that row $b$ in batch step $k+1$ is the exact chronological continuation of row $b$ from step $k$, seamlessly preserving the ODE time constants across the entire epoch.

Given an expert trajectory of length $T$, defined as $\tau=\{(o_1,y_1),(o_2,y_2),\dots,(o_T,y_T)\}$, and a fixed temporal sequence length $L$ (e.g., $L=32$), we extract sequence batches. With the introduction of multi-modal sensor fusion, the observation $o_t$ is a composite of visual, auditory, and thermal inputs. The input tensor sequences $\mathbf{X}^{(vis)}_i$, $\mathbf{X}^{(aud)}_i$, and $\mathbf{X}^{(thm)}_i$, and the target action sequence $\mathbf{Y}_i$ starting at index $i$ are:

$$
\mathbf{X}^{(vis)}_i=\{o^{(vis)}_t\}_{t=i}^{i+L-1},\quad\mathbf{X}^{(vis)}_i\in\mathbb{R}^{L\times C\times64\times64}
$$

$$
\mathbf{X}^{(aud)}_i=\{o^{(aud)}_t\}_{t=i}^{i+L-1},\quad\mathbf{X}^{(aud)}_i\in\mathbb{R}^{L\times2\times H_{mels}\times W_{time}}
$$

$$
\mathbf{X}^{(thm)}_i=\{o^{(thm)}_t\}_{t=i}^{i+L-1},\quad\mathbf{X}^{(thm)}_i\in\{0,1\}^{L\times1\times64\times64}
$$

$$
\mathbf{Y}_i=\{y_t\}_{t=i}^{i+L-1},\quad\mathbf{Y}_i\in\{0,1\}^{L\times n_\rho}
$$

Where $C \in \{3,4\}$ depends on whether the depth buffer is enabled, and $n_\rho$ is the dimensionality of the action space dictated by the active environment profile $\rho$ (e.g., Basic, Classic, Fluid).

---

## 2. The Objective Function (BCE & Focal Loss)

At each time step $t$, the network outputs a vector of raw logits $\mathbf{z}_t\in\mathbb{R}^{n_\rho}$. Because the action space allows for simultaneous key presses (e.g., strafing right while firing), the foundation of the objective is evaluated using **Binary Cross-Entropy (BCE)** with Logits Loss.

The baseline BCE loss $\mathcal{L}_{BCE}$ for a single sequence of length $L$ over $n_\rho$ independent binary action channels is computed as:

$$
\mathcal{L}_{BCE}(\theta)=-\frac{1}{L\cdot n_\rho}\sum_{t=1}^{L}\sum_{j=1}^{n_\rho}\left[y_{t,j}\log(\sigma(z_{t,j}))+(1-y_{t,j})\log(1-\sigma(z_{t,j}))\right]
$$

Where $\sigma(\cdot)$ is the Sigmoid activation function, $y_{t,j}$ is the ground truth label, and $z_{t,j}$ is the network's prediction. 

However, pure BCE treats all errors equally. Because human expert demonstrations consist overwhelmingly of simple navigation frames (the "Hold W Trap"), the cumulative gradient of these easily classified background actions overwhelms the sparse, high-value gradients of rare actions like combat. 



To cure this convergence trap, Golem implements **Focal Loss**, which extends the BCE formulation by introducing a dynamically scaled modulating factor. Let $p_{t,j}=\sigma(z_{t,j})$. The Focal Loss $\mathcal{L}_{focal}$ is computed as:

$$
\mathcal{L}_{focal}(\theta)=-\frac{1}{L\cdot n_\rho}\sum_{t=1}^{L}\sum_{j=1}^{n_\rho}\left[\alpha y_{t,j}(1-p_{t,j})^\gamma\log(p_{t,j})+(1-\alpha)(1-y_{t,j})p_{t,j}^\gamma\log(1-p_{t,j})\right]
$$

* **The Focusing Parameter ($\gamma$):** As the model's confidence in a correct prediction increases ($p_{t,j}\to1$ for positive classes, or $p_{t,j}\to0$ for negative classes), the modulating factor $(1-p_{t,j})^\gamma$ decays to zero. This exponentially suppresses the gradient contribution of easily classified navigation frames, forcing the optimizer to focus strictly on hard, misclassified instances. Standard BCE is recovered when $\gamma=0$.
* **The Weighting Factor ($\alpha$):** A static scalar (e.g., $\alpha=0.25$) that balances the intrinsic priority of positive targets versus negative targets, mitigating the sheer volume of `0`s (keys not pressed) in the multi-label distribution.

The network parameters $\theta$ are subsequently updated via Backpropagation Through Time (BPTT).

### Stateful Backpropagation Through Time (BPTT)

Because the LNN's Closed-form Continuous (CfC) cells require a continuous temporal flow to accurately accumulate evidence and trigger action potentials, the training loop utilizes Stateful BPTT. The hidden state output $hx$ from a batch is retained, detached from the computational graph ($hx = hx.detach()$), and passed as the prior state for the subsequent batch. To prevent mathematical amnesia while respecting independent trajectory boundaries, the sampler streams maintain sequence chronologies, while a dynamic boolean mask zeros out the hidden state exclusively for sequences mapped to the start of a new `.npz` file, preventing "past life" momentum leakage.

---

## 3. Class Imbalance & Mirror Augmentation

While Focal Loss successfully mitigates action-frequency bias, human gameplay datasets also exhibit severe topological and spatial biases. For example, a dataset derived from a specific maze may contain an 80/20 ratio of left turns to right turns. Unmitigated, this spatial sparsity causes the network to collapse into localized minima, such as the "Zoolander Problem" (inability to turn right).

Golem counteracts spatial bias dynamically via **Mirror Augmentation**. During data streaming, the dataset yields reflected visual and thermal observation tensors $o'^{(vis)}_t$ and $o'^{(thm)}_t$ across the vertical axis (width):

$$
o'^{(vis)}_{t,c,h,w}=o^{(vis)}_{t,c,h,W-w-1}
$$

$$
o'^{(thm)}_{t,c,h,w}=o^{(thm)}_{t,c,h,W-w-1}
$$

If the auditory sensor is enabled, perfect spatial symmetry must also be maintained across the agent's "hearing." This is achieved by physically swapping the left and right stereo channels (channel index 0 and 1) across the 2D Mel Spectrogram:

$$
o'^{(aud)}_{t,c_{flip},h_{mel},w_{time}}=o^{(aud)}_{t,1-c_{flip},h_{mel},w_{time}}
$$

To maintain ground-truth causality, the corresponding target vector $y'_t$ must undergo a specific permutation. Let $P_\rho$ be an $n_\rho\times n_\rho$ permutation matrix defined by the active profile $\rho$, which swaps the indices corresponding to strictly spatial actions:

- $idx_{\text{MoveLeft}}\leftrightarrow idx_{\text{MoveRight}}$
- $idx_{\text{TurnLeft}}\leftrightarrow idx_{\text{TurnRight}}$

All state-invariant actions (e.g., Attack, Use, NextWeapon) map to the identity matrix within $P_\rho$. The augmented target vector is thus:

$$
y'_t=P_\rho y_t
$$

This geometric inversion enforces perfect spatial symmetry in the agent's spatial reasoning, effectively doubling the dataset's topological variance without requiring additional recording sessions.

---

## 4. Covariate Shift & DAgger Intervention



A fundamental flaw of pure Behavioral Cloning is **Covariate Shift** (the "Perfect Play" trap). If the network is trained exclusively on flawless expert demonstrations, it never learns how to recover from mistakes. During live inference, a microscopic mathematical error will push the agent slightly off the optimal trajectory. Because this sub-optimal state $s_{err}$ exists outside the training distribution, the agent's predictions become chaotic, and the errors rapidly compound until the agent is completely stuck.

To cure this, Golem employs **DAgger (Dataset Aggregation)**. During live inference, the human expert monitors the autonomous agent. If the agent enters an equilibrium state (e.g., staring into a corner), the human holds a hotkey to instantly suspend the LNN's logits and hijack the controls. 

To ensure the network understands the causal sequence that led to the error, the intervention pipeline utilizes a rolling `collections.deque` buffer. This temporarily stores the $L$ autonomous frames immediately preceding the override. When the human operator takes control, this historical context is flushed into a `_recovery` trace alongside the corrective actions. This explicitly teaches the network not only the correction, but the specific sub-optimal visual precursors that demand it, actively teaching the network how to correct trajectory deviations.

### Catastrophic Forgetting & Stratified Sampling



When applying the DAgger interventions, pure sequential fine-tuning on the `_recovery` traces would cause the model to suffer from **catastrophic forgetting**. The continuous differential equations governing the LNN's hidden state would overfit to the highly localized corrective vectors, effectively collapsing the broader phenomenological heuristics previously learned for normal navigation.

To maintain structural integrity of the dynamical system, Golem utilizes deterministic **Stratified Sampling**. The custom `StatefulStratifiedBatchSampler` explicitly allocates a percentage of the parallel batch streams (e.g., 25%) strictly to recovery sequences, and the remaining 75% to base expert play. This mathematically guarantees that every backpropagation step contains a balanced gradient representing both the optimal base policy $\pi^*$ and the localized recovery vectors, perfectly preserving general topological reasoning while routing the optimizer's computational effort toward out-of-distribution corrections.

---

## 5. Diagnostic Auditing & Validation

Because the aggregate loss scalar $\mathcal{L}(\theta)$ fundamentally obscures multi-label class imbalances (a model that never shoots will still achieve 95% accuracy if the "Attack" label is sparse), Golem utilizes a dedicated static `audit` module.

The audit evaluates the trained weights over a validation slice by generating a strictly thresholded ($\sigma(\mathbf{z})>0.5$) Confusion Matrix for every individual channel $j$ in the $n_\rho$ action space. It evaluates the network based on:

**Precision ($P_j$)**

The probability that the agent's decision to act was correct.
    
$$
P_j=\frac{TP_j}{TP_j+FP_j}
$$

**Recall ($R_j$)**: 

The probability that the agent successfully reacted to an environmental stimulus requiring action $j$.

$$
R_j=\frac{TP_j}{TP_j+FN_j}
$$

Where $TP$, $FP$, and $FN$ represent True Positives, False Positives, and False Negatives, respectively.

### Addressing Redundancy (Stride)

Historically, during normal training, the sliding window overlapped by shifting exactly one frame per sequence step. However, during auditing, evaluating identical frames repeatedly artificially inflates the diagnostic support counts and yields inaccurate exact-match metrics. To resolve this, the `audit` dataloader enforces a sequence stride equal to $L$, meaning non-overlapping segments are tested to precisely evaluate every frame only once. This non-overlapping stride is now mirrored in the training dataloader to satisfy the strict chronological requirements of Stateful BPTT.

---

## API Reference

The data extraction, LNN optimization, and evaluation mechanics are orchestrated by the handlers below.

### The Training Loop

::: app.pipeline.train.train

### DAgger Intervention

::: app.pipeline.intervene.intervene

### Data Inspection & Auditing

::: app.metrics.audit

::: app.metrics.examine

::: app.metrics.inspect

::: app.metrics.summary