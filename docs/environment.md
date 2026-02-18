
# The Environment: ViZDoom

Golem operates within a Partially Observable Markov Decision Process (POMDP) defined by the *DOOM* engine. We utilize **ViZDoom** as the API bridge.

## State Space ($S$)

The state $s_t$ at time $t$ is a tensor representing the visual field of the agent.

$$
s_t \in \mathbb{R}^{3 \times 64 \times 64}
$$

* **Channels:** 3 (RGB).
* **Resolution:** $64 \times 64$ pixels.
* **Normalization:** $s_{i,j,k} \in [0, 1]$.

We explicitly discard game variables (Health, Ammo) from the input vector to force the model to learn visual cues (e.g., "screen flashing red" implies damage).

## Action Space ($A$)

The action space is discrete and multi-label. Unlike standard RL environments (like Gym's `Discrete`), *DOOM* allows simultaneous inputs (e.g., strafing left while shooting).

$$
A = \{ \text{Left}, \text{Right}, \text{Attack} \}
$$

The output vector $y_t$ is a Bernoulli distribution over these actions:

$$
y_t \in \{0, 1\}^3
$$

* $y_{t,0}$: Move Left
* $y_{t,1}$: Move Right
* $y_{t,2}$: Attack

## Temporal Dynamics

The environment runs at a fixed tic rate of 35 Hz. However, the agent operates in an asynchronous loop. To maintain temporal consistency, we employ a **Frame Skip** or sleep-cycle to align the inference latency ($\tau_{inf}$) with the game's clock.

$$
\tau_{total} = \tau_{inf} + \tau_{sleep} \approx 28ms
$$