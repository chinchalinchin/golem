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

* **`simple`**: Maps to `conf/simple.cfg` (7 dimensions: Movement + Turn + Attack).
* **`basic`**: Maps to `conf/basic.cfg` (8 dimensions: Super-set adding Use).
* **`classic`**: Maps to `conf/classic.cfg` (10 dimensions: Super-set adding Explicit weapon slots).
* **`fluid`**: Maps to `conf/fluid.cfg` (9 dimensions: Super-set adding Sequential weapon toggles).

### 3. `data`

Defines the file routing and prefix naming conventions for the ETL pipeline.

* **`prefix`**: The string prefix for generated tensor arrays (e.g., "golem_").
* **`dirs.training` / `dirs.model`**: Relative paths dictating where `.npz` datasets and `.pth` weight archives are saved.

### 4. `brain`

Defines the active architecture of the Neural Circuit Policy (NCP).

| Property | Description |
| :--- | :--- |
| **`mode`** | The active profile (`simple`, `basic`, `classic`, or `fluid`). This dictates which configuration dictionary is loaded across the pipeline. |
| **`cortical_depth`** | The number of CNN layers in the visual cortex. Higher depths aggressively pool spatial features into denser representations. |
| **`working_memory`** | The number of hidden units in the CfC liquid core, defining the capacity of the agent's continuous temporal state. |
| **`activation`** | The probability threshold (e.g., `0.5`) applied to the LNN's sigmoid logits during live inference to determine if a multi-label action should be triggered. |
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

### 5. `loss`

Defines the hyperparameters for the various objective functions available to the optimizer. The active loss function is selected via `training.loss`.

| Property | Description |
| :--- | :--- |
| **`focal.alpha`** | The static weighting factor used to balance the intrinsic priority of positive vs. negative classes (e.g., 0.25). |
| **`focal.gamma`** | The focusing parameter used to dynamically down-weight the gradient of easily classified examples. |
| **`asymmetric.gamma_pos`** | The focusing parameter strictly for the positive class. Kept low to preserve gradients for rare actions. |
| **`asymmetric.gamma_neg`** | The focusing parameter strictly for the negative class. Kept high to aggressively decay background frame gradients. |
| **`asymmetric.clip`** | The probability margin (e.g., 0.05) under which easy negative predictions are completely discarded from the loss calculation. |
| **`smooth.epsilon`** | The uniform noise prior injected into the target distribution for Label Smoothing BCE (e.g., 0.1). |

#### Objective Function Dynamics

To counteract severe class imbalance in human demonstrations (the "Hold W" convergence trap), Golem provides alternatives to standard Binary Cross-Entropy:

* **Focal Loss:** The `gamma` parameter acts as a dynamic focusing mechanism. By setting $\gamma > 0$, the loss function exponentially scales down the contribution of predictions the model is already confident about. If the network successfully predicts a basic navigation frame, its gradient contribution approaches zero, forcing the optimizer to focus strictly on sparse, difficult combat sequences.
* **Asymmetric Loss (ASL):** Decouples the focusing parameters. Because video game inputs are heavily skewed toward negatives (keys not pressed), ASL aggressively penalizes easy negatives (high `gamma_neg`) while retaining robust gradients for rare positive actions (low `gamma_pos`).
* **Label Smoothing BCE:** Injects an $\epsilon$ noise prior into the target labels. This mathematically acknowledges human demonstrator noise (e.g., reaction time lag) and prevents the model from overfitting to absolute certainty, softening the confidence bounds.

### 6. `training`

Defines the Behavioral Cloning optimization loop dynamics.

| Property | Description |
| :--- | :--- |
| **`epochs`** | Total number of complete passes through the training dataset. |
| **`batch_size`** | The number of sequences processed concurrently before a weight update. |
| **`learning_rate`** | The step size the Adam optimizer takes against the gradient of the loss function. |
| **`sequence_length`** | The temporal window size ($L$) for Backpropagation Through Time (e.g., 32 frames). |
| **`loss`** | The active objective function (`focal`, `bce`, `smooth`, or `asymmetric`). |
| **`augmentation.mirror`** | Boolean toggle to enable dynamic horizontal mirror augmentation, doubling topological variance and curing turning bias. |

#### Hyperparameter Dynamics: `learning_rate` and `batch_size`

The **`learning_rate`** (e.g., 0.0001) controls convergence stability. 

* **Too High:** The model will overshoot the optimal minima, leading to erratic loss oscillation or complete divergence. 
* **Too Low:** The model will converge too slowly, wasting computational time, or become trapped in a suboptimal local minimum.

The **`batch_size`** (e.g., 16) regulates gradient noise. 

* **Small Batch Size:** Results in "noisy" gradient estimates. This noise acts as a natural regularizer, often helping the network escape sharp, suboptimal local minima and generalize better to unseen environments. However, it trains slower sequentially.
* **Large Batch Size:** Provides a highly accurate gradient estimate and allows for massive hardware parallelization. However, if the batch is too large, the model tends to settle into "sharp" minima, severely degrading generalization.

**The Interplay:** These two parameters are mathematically coupled. A common deep learning heuristic is the *Linear Scaling Rule*: if you double your `batch_size` (smoothing the gradient), you should generally double your `learning_rate` to maintain the same training dynamics and convergence speed.

### 7. `randomizer`

Configures the external procedural generation engine used to prevent spatial overfitting and Covariate Shift. The `randomize` pipeline utilizes this block to inject massive geographic variance into the training corpus.

* **`executable`**: The absolute path to the compiled Oblige 7.70 binary.
* **`output`**: The directory where procedurally generated `.wad` files are stored before being loaded by the pipeline.
* **`iterations`**: The number of continuous maps to generate, record, and save during a single run of the `randomize` pipeline.
* **`duration`**: The maximum lifespan (in seconds) of a recorded episode on a generated map before the pipeline truncates it and moves to the next iteration.
* **`oblige`**: Defines the specific topological rules and dimensions for the generator.

=== "Engine Settings"

  | Parameter | Description | Possible Values |
  | --- | --- | --- |
  | **`game`** | The base target game and asset roster. | `doom1`, `doom2`, `tnt`, `plutonia`, `heretic` |
  | **`engine`** | The source port format, dictating engine limits and features (like ZDoom slopes). | `vanilla`, `limit_removing`, `boom`, `zdoom` |
  | **`length`** | The number of maps compiled into the WAD. | `single`, `episode`, `game` |
  | **`theme`** | The architectural style, texture sets, and skyboxes. | `original`, `tech`, `tech_ish`, `urban`, `urban_ish`, `hell`, `hell_ish`, `jumbled`, `mixed` |

=== "Topology Settings"

  | Parameter | Description | Possible Values |
  | --- | --- | --- |
  | **`size`** | The map's geographic footprint and total room count. | `micro`, `small`, `regular`, `large`, `huge`, `epic`, `progressive` |
  | **`outdoors`** | Frequency of sky-exposed, open-air environments. | `none`, `mixed`, `plenty` |
  | **`caves`** | Presence of cavernous, natural rock formations. | `none`, `mixed`, `plenty` |
  | **`liquids`** | Amount of liquid hazards (nukage, lava, water, slime). | `none`, `mixed`, `plenty` |
  | **`hallways`** | Frequency of narrow corridors connecting main rooms. | `none`, `mixed`, `plenty` |
  | **`teleporters`** | Inclusion of teleportation pads for traversal. | `none`, `mixed`, `plenty` |
  | **`steepness`** | Degree of verticality, ledges, stairs, and height variation. | `none`, `mixed`, `plenty` |
  | **`doors`** | Ratio of physical doors to open archways. | `none`, `some`, `lots` |
  | **`secrets`** | Number of hidden rooms or illusory walls containing extra resources. | `none`, `mixed`, `plenty` |

=== "Combat Settings"

  | Parameter | Description | Possible Values |
  | --- | --- | --- |
  | **`mons`** | The overall density and quantity of monster spawns. | `none`, `sparse`, `normal`, `lots`, `swarms` |
  | **`strength`** | The toughness and tier-scaling of the spawned enemies. | `easier`, `normal`, `harder`, `tougher` |
  | **`ramp_up`** | How quickly monster toughness and numbers scale up across an episode or game. | `slow`, `normal`, `fast` |
  | **`bosses`** | Inclusion and frequency of boss-tier monsters (Cyberdemon, Spider Mastermind). | `none`, `normal`, `lots` |
  | **`traps`** | Frequency of monster closets that open when picking up items or crossing lines. | `none`, `mixed`, `plenty` |
  | **`cages`** | Frequency of monsters placed in inaccessible elevated cages or windows. | `none`, `mixed`, `plenty` |

=== "Item Settings"

  | Parameter | Description | Possible Values |
  | --- | --- | --- |
  | **`health`** | The abundance of medkits, stimpacks, and health potions. | `starved`, `scarce`, `normal`, `plenty`, `heaps` |
  | **`ammo`** | The abundance of ammunition pickups and backpacks. | `starved`, `scarce`, `normal`, `plenty`, `heaps` |
  | **`weapons`** | How early high-tier weapons (SSG, Plasma, BFG) are introduced into the map progression. | `later`, `normal`, `sooner` |
  | **`powerups`** | Frequency of high-tier powerups (Soulspheres, Megaspheres, Invulnerability, Berserk). | `none`, `scarce`, `normal`, `plenty`, `heaps` |
  | **`barrels`** | Density of explosive environmental barrels. | `none`, `some`, `lots` |

### 8. `modules`

A dictionary mapping human-readable task names (e.g., `combat`, `navigation`) to their specific `.wad` scenario files and the default number of episodes to record during extraction.

### 9. `keybindings`

A dictionary mapping the agent's action space profiles (`simple`, `basic`, `classic`, `fluid`) to physical keyboard inputs. These are injected dynamically into the ViZDoom engine during `record` and mapped to `pynput` listeners during `intervene`.