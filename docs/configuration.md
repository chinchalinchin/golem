# Application Configuration (`app.yaml`)

The `app.yaml` file acts as the central nervous system for Golem. It defines the active brain architecture, training hyperparameters, dataset routing, and environment multi-modality. This single source of truth ensures that the ETL pipelines and the PyTorch models remain perfectly synchronized without requiring hardcoded magic numbers.

## Configuration Blocks

### 1. `app`

General metadata and logging settings for the application.

* **`name`**: The application identifier (e.g., "Golem").
* **`version`**: Current software version.
* **`log_level`**: Standard Python logging level (e.g., `INFO`, `DEBUG`, `WARNING`).

### 2. `config`

Maps the high-level brain modes to the specific underlying ViZDoom `.cfg` files. These files dictate the engine's available buttons and game variables.

* **`basic`**: Maps to `conf/basic.cfg` (8 dimensions).
* **`classic`**: Maps to `conf/classic.cfg` (10 dimensions).
* **`fluid`**: Maps to `conf/fluid.cfg` (9 dimensions).

### 3. `training`

Defines the Behavioral Cloning optimization loop dynamics.

| Property | Description |
| :--- | :--- |
| **`epochs`** | Total number of complete passes through the training dataset. |
| **`sequence_length`** | The temporal window size ($L$) for Backpropagation Through Time (e.g., 32 frames). |
| **`augmentation.mirror`** | Boolean toggle to enable dynamic horizontal mirror augmentation, doubling topological variance and curing turning bias. |
| **`alpha`** | The weighting factor used in the Focal Loss function to balance positive and negative classes (e.g., 0.25). |
| **`gamma`** | The focusing parameter used in the Focal Loss function to dynamically scale down the gradient of easily classified examples (e.g., 2.0). |

#### Hyperparameter Dynamics: `learning_rate` and `batch_size`

The **`learning_rate`** (e.g., 0.0001) controls the step size the Adam optimizer takes when updating the LNN's weights against the gradient of the loss function. 

* **Too High:** The model will overshoot the optimal minima, leading to erratic loss oscillation or complete divergence. 
* **Too Low:** The model will converge too slowly, wasting computational time, or become trapped in a suboptimal local minimum.

The **`batch_size`** (e.g., 16) determines how many temporal sequences are processed concurrently before a weight update occurs. 

* **Small Batch Size:** Results in "noisy" gradient estimates. This noise acts as a natural regularizer, often helping the network escape sharp, suboptimal local minima and generalize better to unseen environments. However, it trains slower sequentially.
* **Large Batch Size:** Provides a highly accurate gradient estimate and allows for massive hardware parallelization (faster wall-clock time per epoch). However, if the batch is too large, the model tends to settle into "sharp" minima, severely degrading generalization.

**The Interplay:** These two parameters are mathematically coupled. A common deep learning heuristic is the *Linear Scaling Rule*: if you double your `batch_size` (smoothing the gradient), you should generally double your `learning_rate` to maintain the same training dynamics and convergence speed.

#### Focal Loss Dynamics: `alpha` and `gamma`

To counteract severe class imbalance in human demonstrations (the "Hold W" convergence trap), Golem utilizes a Focal Loss function instead of standard Binary Cross-Entropy.

The **`gamma`** parameter acts as a dynamic focusing mechanism. In standard classification, frequent and "easy" actions (like walking forward) accumulate massive loss gradients simply by volume, effectively drowning out the gradients of rare, critical actions (like firing a weapon). By setting `gamma > 0`, the loss function exponentially scales down the contribution of predictions the model is already confident about. If the network successfully predicts a navigation frame, its gradient contribution approaches zero, forcing the optimizer to focus computational effort on the sparse, difficult combat sequences it is currently failing. 

The **`alpha`** parameter acts as a static weighting factor. It directly balances the importance of positive targets versus negative targets across the multi-label distribution, ensuring that the sheer volume of `0`s (keys not pressed) does not overwhelm the rare `1`s (keys pressed).

### 4. `brain`

Defines the active architecture of the Neural Circuit Policy (NCP).

| Property | Description |
| :--- | :--- |
| **`mode`** | The active profile (`basic`, `classic`, or `fluid`). This dictates which configuration dictionary is loaded across the pipeline. |
| **`cortical_depth`** | The number of CNN layers in the visual cortex. Higher depths aggressively pool spatial features into denser representations. |
| **`working_memory`** | The number of hidden units in the CfC liquid core, defining the capacity of the agent's continuous temporal state. |
| **`sensors`** | Boolean toggles (`visual`, `depth`, `audio`, `thermal`) that dynamically scale the input channels and parallel network branches (e.g., activating the parallel 2D Auditory and Thermal Cortices for multi-modal sensor fusion). |

#### Signal Processing Dynamics: `dsp`

When the `audio` sensor is enabled, the `dsp` block governs how raw 1D audio waveforms are mathematically converted into 2D Mel Spectrograms. 

* **`sample_rate`**: The temporal resolution of the engine's audio buffer (e.g., 44100 Hz).
* **`n_fft`**: The length of the windowed signal used for the Short-Time Fourier Transform (STFT). Higher values increase frequency resolution but decrease temporal resolution.
* **`n_mels`**: The number of Mel filterbanks applied. This defines the final height ($H_{mels}$) of the generated spectrogram tensor.

**The Impact of `hop_length`:**

The `hop_length` (e.g., 256) defines the number of audio samples between successive STFT windows. It is the fundamental parameter dictating the temporal width ($W_{time}$) of the resulting 2D audio tensor.

* **Small `hop_length`:** The STFT windows overlap heavily, yielding a highly granular, wide spectrogram matrix. The model gains exceptional temporal resolution (able to pinpoint the exact millisecond a monster growls), but memory consumption scales linearly, heavily bottlenecking VRAM during training.
* **Large `hop_length`:** The windows are spaced further apart, creating a narrow, compressed matrix. Training executes significantly faster with a smaller memory footprint, but the LNN may lose the ability to detect transient, high-frequency acoustic events (like a brief weapon click).

### 5. `data`

* **`prefix`**: The string prefix for generated tensor arrays (e.g., "golem_").
* **`dirs.training` / `dirs.model`**: Relative paths dictating where `.npz` datasets and `.pth` weight archives are saved.

### 6. `modules`

A dictionary mapping human-readable task names (e.g., `combat`, `navigation`) to their specific `.wad` scenario files and the default number of episodes to record during extraction.

### 7. `keybindings`

A dictionary mapping the agent's action space profiles (`basic`, `classic`, `fluid`) to physical keyboard inputs. These are injected dynamically into the ViZDoom engine during `record` and mapped to `pynput` listeners during `intervene`.