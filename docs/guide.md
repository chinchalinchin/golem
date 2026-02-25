# Field Guide: Tuning Golem

Training a continuous-time Liquid Neural Network (LNN) in a multi-modal Partially Observable Markov Decision Process (POMDP) is not a static process. The hyperparameters governing the model's architecture (`brain`), its optimization dynamics (`training`), and its penalty landscape (`loss`) are mathematically and practically entangled.

This guide provides practical heuristics for adjusting these parameters in tandem to achieve convergence and prevent behavioral collapse.

## 1. Memory and Horizons (Brain vs. Sequence)

The fundamental advantage of the Closed-form Continuous (CfC) liquid core is its ability to maintain a temporal hidden state . However, the capacity of this state and the temporal horizon over which it is optimized must be balanced.

* **`brain.working_memory`** ($W_m$): Dictates the sheer capacity of the agent's temporal state tensor.
* **`training.sequence_length`** ($L$): Dictates the temporal window (number of contiguous frames) passed into the network during a single Backpropagation Through Time (BPTT) step.

### Tuning Heuristics

* **The Amnesia Trap:**  If $L$ is too short (e.g., 8 frames), the network cannot reliably link cause and effect (e.g., observing a projectile being fired and the subsequent impact). The gradients are truncated before the ODE can establish a meaningful time constant. Always maintain $L \ge 32$ for combat scenarios.
* **The Capacity Bottleneck:** If you increase $L$ to capture longer causal chains (e.g., $L=64$), you must ensure $W_m$ is large enough to encode that much history. Pushing 64 frames of dense combat data into a tiny $W_m=16$ core will result in catastrophic state overwriting. As a rule of thumb, if you double $L$, consider increasing $W_m$ to prevent bottlenecking.

## 2. Capacity vs. Convergence (Cortical Depth vs. Learning Dynamics)

Scaling the sensory cortices provides the model with higher-resolution spatial reasoning but drastically alters the optimization landscape.

* **`brain.cortical_depth`**:  ($D$): Determines the number of convolutional layers. Higher values aggressively pool the tensor into a denser latent representation $V(t)$.
* **`training.batch_size`** and **`training.learning_rate`&**: Govern how the Adam optimizer navigates the gradient manifold.

### Tuning Heuristics

* **Scaling the Visual Cortex (The Resolution Trade-off):** Adjusting $D$ forces a trade-off between **spatial resolution** and **semantic abstraction (receptive field)**. 
    * A shallow network ($D=2$) preserves a relatively high-resolution $16 \times 16$ spatial grid before flattening. This is excellent for tactical combat, as it allows the agent to resolve and aim at distant, low-pixel enemies. However, its small receptive field means it struggles to comprehend macro-geometry, like navigating complex maze layouts.
    * Increasing to $D=4$ drastically expands the receptive field, allowing the network to abstract and understand complex room geometry. However, it aggressively pools the spatial grid down to $4 \times 4$. Distant enemies will be mathematically obliterated by the convolutions, rendering the agent near-sighted. Furthermore, deeper networks increase the convolutional parameter count, requiring more `epochs` to converge.
* **The Linear Scaling Rule:** Hardware VRAM is your primary constraint. If increasing $D$ or $L$ causes an Out-Of-Memory (OOM) error, you must reduce the `batch_size`. If you halve your `batch_size` (e.g., from 32 to 16), the gradient estimates become noisier. To prevent the optimizer from over-correcting on this noise, you should generally halve your `learning_rate` (e.g., from $0.0002$ to $0.0001$).

## 3. Multi-Modal Hardware Constraints

Enabling sensor fusion geometrically expands the network's flat size  ($W_f$), drastically altering memory requirements.

* **Audio Impact (`brain.dsp`)**: Enabling `brain.sensors.audio` converts 1D waveforms to 2D Mel Spectrograms. This is the most computationally expensive modality. A small `dsp.hop_length` creates a very wide temporal tensor. If VRAM is exhausted, double the `hop_length` (e.g., 256 to 512) before compromising on `batch_size` or `sequence_length`.
* **Thermal Impact (`brain.sensors.thermal`)**: The thermal mask adds an isolated, parallel CNN. Because it is a binary mask  ($1\times64\times64$),, it is computationally lightweight compared to visual or auditory tensors. You can generally enable this without drastically reducing your batch size.

## 4. Escaping Convergence Traps (Loss vs. Behavior)

Standard Behavioral Cloning frequently results in behavioral collapse due to the overwhelming volume of "boring" navigation frames in human datasets compared to sparse, critical combat frames.

### Scenario A: The Pacifist Agent

**Symptom:** The agent navigates beautifully but refuses to fire its weapon or switch guns, even when directly facing an enemy.
**Diagnosis:** The gradients from the sheer volume of "Move Forward" frames have drowned out the "Attack" gradients.

**Cure:** 

1.  **Switch to Asymmetric Loss (`loss: asymmetric`)**.
2.  Decrease `loss.asymmetric.gamma_pos`  (e.g., to $0.0$) to ensure the network receives the full, unmitigated gradient whenever a rare "Attack" label appears in the dataset.
3.  Increase `loss.asymmetric.gamma_neg` (e.g., to $4.0$)  to aggressively silence the gradients of easily predicted background frames.

### Scenario B: The Twitchy Agent

**Symptom:** The agent constantly fires at walls or rapidly toggles weapons even when no enemies are present.
**Diagnosis:** The model is overfitting to absolute certainty and lacks a confidence threshold, likely resulting from a high learning rate or strict Binary Cross-Entropy (BCE) over-penalizing slight hesitancy.

**Cure:**

1. **Switch to Label Smoothing (`loss: smooth`)**.
2. Set `loss.smooth.epsilon` to $0.1$ or $0.15$. This injects a uniform noise prior into the targets, acknowledging that human demonstrators are noisy and reactive. It prevents the logits from being pushed to extreme asymptotes, resulting in smoother, more deliberate decision-making.

### Scenario C: The "Zoolander" Trap

**Symptom:** The agent navigates well but explicitly favors turning in one direction and gets stuck in corners requiring the opposite turn.
**Diagnosis:** Spatial bias in the human demonstration data (e.g., clearing a map by hugging the left wall).
**Cure:** Ensure `training.augmentation.mirror` is set to `true`. This mathematically doubles the topological variance by flipping the visual/thermal tensors and swapping the corresponding left/right action indices, guaranteeing perfect spatial symmetry in the gradient updates.

Here is the markdown table summarizing the optimal training settings and hyperparameters for each loss function, based on the architectural constraints and behavioral heuristics we've established for Golem.

## 5. Loss Function Hyperparameter Guide

| Loss Function | Primary Use Case | Epochs | Batch Size | Learning Rate | Sequence Length | Specific Hyperparameters |
| --- | --- | --- | --- | --- | --- | --- |
| **Focal** (`focal`) | General baseline. Mitigates the "Hold W" convergence trap during standard navigation/combat datasets. | 150 | 16 | 0.0001 | 32+ | `alpha`: 0.25, `gamma`: 2.0 |
| **Asymmetric** (`asymmetric`) | Curing the "Pacifist Agent." Highly imbalanced datasets where critical actions (shooting) are exceptionally rare. | 150+ | 16 | 0.0001 | 32+ | `gamma_pos`: 0.0 - 1.0, `gamma_neg`: 4.0<br>, `clip`: 0.05 |
| **Smooth** (`smooth`) | Curing the "Twitchy Agent." Datasets with high demonstrator noise, reaction lag, or over-correction. | 100 - 150 | 16 | 0.0001 | 32+ | `epsilon`: 0.1 - 0.15 |
| **BCE** (`bce`) | Benchmarking and debugging. Requires a perfectly balanced dataset to prevent convergence failure. | 100 | 16 | 0.0001 | 32+ | None |

### Dynamic Scaling Rules

When adjusting the baseline values in the table, keep these mathematical constraints in mind:

* **The Linear Scaling Rule:** The `batch_size` and `learning_rate` are coupled. If you run out of VRAM and need to halve the batch size (e.g., from 16 to 8), you should halve the learning rate (e.g., to 0.00005) to prevent the optimizer from wildly over-correcting on the noisier gradient estimates. Conversely, if you have the VRAM to bump the batch size to 32, increase the learning rate to 0.0002.
* **The Temporal Constant:** Sequence length (`sequence_length`) should rarely drop below 32 for combat modules, regardless of the loss function. The LNN's Closed-form Continuous (CfC) cells need a sufficiently long temporal runway to establish the causal link between an action (firing a weapon) and its delayed effect (a projectile hitting a target).
* **Epoch Scaling:** Deeper cortical depths (`cortical_depth` 3 or 4) increase the convolutional parameter count significantly. You will likely need to push your epochs beyond 150 to achieve convergence on deep networks, particularly when using Asymmetric Loss which aggressively throttles gradient flow for background frames.