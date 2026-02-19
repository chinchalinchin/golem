# The Environment: ViZDoom

Golem operates within a Partially Observable Markov Decision Process (POMDP) defined by the *DOOM* engine. We utilize **ViZDoom** as the API bridge.

## State Space ($S$)

The state $s_t$ at time $t$ is a tensor representing the visual field of the agent.

$$
s_t \in \mathbb{R}^{3 \times 64 \times 64}
$$

* **Channels:** 3 (RGB).
* **Resolution:** 64x64 pixels.
* **Normalization:** $s_{i,j,k} \in [0, 1]$.

We explicitly discard game variables (Health, Ammo) from the input vector to force the model to learn visual cues (e.g., "screen flashing red" implies damage), encouraging robust generalization.

## Action Space ($A$)

The action space is discrete and multi-label. To ensure the Liquid Neural Network (LNN) can seamlessly generalize across different maps (Navigation vs. Combat), Golem utilizes a fixed **8-Button Superset**.

$$
A = \{ \text{Fwd}, \text{Back}, \text{MoveL}, \text{MoveR}, \text{TurnL}, \text{TurnR}, \text{Attack}, \text{Use} \}
$$

The output vector $y_t$ is a Bernoulli distribution over these actions:

$$
y_t \in \{0, 1\}^8
$$

Even if a specific curriculum module (like a maze) does not require the "Attack" button, the dimensionality of the brain remains rigidly fixed to 8 to prevent tensor dimension mismatches during Continual Learning.

## Temporal Dynamics

The environment runs at a fixed tic rate of 35 Hz. However, the agent operates in an asynchronous loop. To maintain temporal consistency, we employ a **Frame Skip** or sleep-cycle to align the inference latency ($\tau_{inf}$) with the game's clock.

$$
\tau_{total} = \tau_{inf} + \tau_{sleep} \approx 28ms
$$