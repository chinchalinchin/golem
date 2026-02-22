# The Environment: ViZDoom

Golem operates within a Partially Observable Markov Decision Process (POMDP) defined by the DOOM engine. We utilize ViZDoom as the API bridge to extract observations and inject actions. Formally, this POMDP is defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma)$.

## Observation Space

While the true underlying engine state $s_t \in \mathcal{S}$ contains exact entity coordinates and internal variables, the agent's observation $o_t \in \Omega$ at time $t$ is strictly constrained to its egocentric sensory field. With the introduction of multi-modal sensor fusion, this observation space scales dynamically based on the active configuration.

The primary visual and spatial tensor $o_{vis}$ is defined as:

$$
o_{vis} \in \mathbb{R}^{C \times 64 \times 64}
$$

- **Channels** ($C$): 3 (RGB) by default, expanding to 4 if the stereoscopic depth buffer is enabled.
- **Resolution**: 64x64 pixels (processed via bilinear interpolation).
- **Normalization**: $o_{i,j,k} \in [0, 1]$.

If the auditory sensor is enabled, the agent also receives an audio tensor $o_{aud}$. While initially extracted from the engine as raw, high-frequency stereo waveforms, the ETL pipeline mathematically transforms these 1D arrays into dense 2D time-frequency representations (Mel Spectrograms) to leverage spatial locality within the convolutional network.

The Digital Signal Processing (DSP) transformation is mathematically defined by the active `dsp` configuration block:

- **Strict Normalization**: The raw buffer is scaled to zero-mean and unit-variance to stabilize gradients.
- **Mel Scale Transformation**: The normalized waveform is processed via a Short-Time Fourier Transform (STFT) mapped to the Mel scale, governed by the `sample_rate`, `n_fft`, `hop_length`, and `n_mels` hyperparameters.
- **Decibel Scaling**: The resulting magnitudes are compressed logarithmically using an Amplitude-to-DB conversion.

The resulting multi-modal audio tensor is defined as:

$$
o_{aud} \in \mathbb{R}^{C \times H_{mels} \times W_{time}}
$$

Where $C=2$ (stereo channels), $H_{mels}$ represents the frequency bins dictated by `n_mels`, and $W_{time}$ is the temporal width calculated dynamically from the engine's audio buffer capacity and the STFT `hop_length`.

If the thermal sensor is enabled, the agent also receives a discrete thermal tensor $o_{thm}$. Extracted via ViZDoom's semantic segmentation labels_buffer, this modality isolates active, dynamic entities (e.g., monsters, projectiles, and interactive items) from the static environmental background plane.  The transformation pipeline applies a strict binary threshold operation ($o_{i,j} = 1 \text{ if } \text{label}_{i,j} > 0 \text{ else } 0$) to the raw buffer and subsequently downsamples the mask to $64 \times 64$ utilizing nearest-neighbor interpolation to prevent edge anti-aliasing artifacts.

The resulting thermal mask tensor is defined as:

$$
o_{thm} \in \{0, 1\}^{1 \times 64 \times 64}
$$

We explicitly discard latent game variables (e.g., Health, Ammo, Coordinates) from the observation vector to force the model to learn multi-modal heuristics (e.g., a "red screen tint" implies damage, a specific visual spectrogram pattern implies a nearby threat, or a binary thermal cluster denotes a dynamic entity), encouraging robust topological generalization across unseen levels.

## Action Space

Unlike standard architectures that utilize a rigid output structure, Golem's action space is a discrete, multi-label domain dynamically conditioned on the active environment configuration profile, denoted as $\rho$.

Let $\rho \in \{\text{basic}, \text{classic}, \text{fluid}\}$. The dimensionality of the action space $n_\rho$ expands or contracts based on the superset defined by $\rho$:

- **Basic Profile** ($n_{\text{basic}} = 8$): $\mathcal{A}_{\text{basic}} = \{ \text{Fwd}, \text{Back}, \text{MoveL}, \text{MoveR}, \text{TurnL}, \text{TurnR}, \text{Attack}, \text{Use} \}$
- **Fluid Profile** ($n_{\text{fluid}} = 9$): $\mathcal{A}_{\text{fluid}} = \mathcal{A}_{\text{basic}} \cup \{ \text{NextWeapon} \}$
- **Classic Profile** ($n_{\text{classic}} = 10$): $\mathcal{A}_{\text{classic}} = \mathcal{A}_{\text{basic}} \cup \{ \text{Weapon2}, \text{Weapon3} \}$

At any time step $t$, the output vector $y_t$ is drawn from a Multivariate Bernoulli distribution over these actions. Assuming conditional independence between individual key presses given the latent state representation, the network predicts the probability vector $\mathbf{p}_t$, yielding:

$$
y_t \in \{0, 1\}^{n_\rho}
$$

During inference, this distribution is thresholded at $0.5$ to produce the deterministic binary vector fed back into the ViZDoom engine. This dynamic scaling prevents data sparsity and gradient dilution that would occur if unused weapon keys were permanently mapped to the output layer during non-combat tasks.

## Temporal Dynamics & Network Synchronization

The environment runs at a fixed tic rate of 35 Hz ($\Delta t \approx 28.5\text{ms}$). Because standard RNNs are discrete, they struggle with the asynchronous loop of live gameplay. Golem utilizes Liquid Time-Constant (LTC) networks to model the hidden state $x(t)$ as an Ordinary Differential Equation (ODE).

To maintain strict temporal consistency between the continuous differential solver and the discrete game clock across a distributed architecture, Golem relies on Deterministic Lockstep networking via ViZDoom's `Mode.PLAYER` (Sync Mode).

Instead of relying on manual sleep heuristics (which are prone to drift and desynchronization), the central Host Server dictates the flow of time. The Host collects asynchronous ticcmds (action vectors) from all connected clients. The local ViZDoom engine inside the Golem container explicitly blocks execution until it receives the synchronized broadcast back from the Host. This guarantees that the numerical integration steps within the LNN perfectly align with the simulated passage of time within the multiplayer POMDP, regardless of hardware inference speeds.

---

## API Reference

The extraction and temporal windowing of the observation space is handled dynamically by the streaming dataset module.

::: app.models.dataset.DoomStreamingDataset