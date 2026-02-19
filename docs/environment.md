# The Environment: ViZDoom



Golem operates within a Partially Observable Markov Decision Process (POMDP) defined by the *DOOM* engine. We utilize **ViZDoom** as the API bridge to extract observations and inject actions. Formally, this POMDP is defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma)$. 

## Observation Space ($\Omega$)

While the true underlying engine state $s_t \in \mathcal{S}$ contains exact entity coordinates and internal variables, the agent's observation $o_t \in \Omega$ at time $t$ is strictly constrained to a downsampled tensor representing its egocentric visual field.

$$
o_t \in \mathbb{R}^{3 \times 64 \times 64}
$$

* **Channels:** 3 (RGB).
* **Resolution:** 64x64 pixels (processed via bilinear interpolation).
* **Normalization:** $o_{i,j,k} \in [0, 1]$.

We explicitly discard latent game variables (e.g., Health, Ammo, Coordinates) from the observation vector to force the model to learn visual heuristics (e.g., a "red screen tint" implies damage), encouraging robust topological generalization across unseen levels.

## Action Space ($\mathcal{A}_\rho$)

Unlike standard architectures that utilize a rigid output structure, Golem's action space is a discrete, multi-label domain dynamically conditioned on the active environment configuration profile, denoted as $\rho$. 

Let $\rho \in \{\text{basic}, \text{classic}, \text{fluid}\}$. The dimensionality of the action space $n_\rho$ expands or contracts based on the superset defined by $\rho$:

1. **Basic Profile** ($n_{\text{basic}} = 8$): $\mathcal{A}_{\text{basic}} = \{ \text{Fwd}, \text{Back}, \text{MoveL}, \text{MoveR}, \text{TurnL}, \text{TurnR}, \text{Attack}, \text{Use} \}$
2. **Fluid Profile** ($n_{\text{fluid}} = 9$): $\mathcal{A}_{\text{fluid}} = \mathcal{A}_{\text{basic}} \cup \{ \text{NextWeapon} \}$
3. **Classic Profile** ($n_{\text{classic}} = 10$): $\mathcal{A}_{\text{classic}} = \mathcal{A}_{\text{basic}} \cup \{ \text{Weapon2}, \text{Weapon3} \}$

At any time step $t$, the output vector $y_t$ is drawn from a Multivariate Bernoulli distribution over these actions. Assuming conditional independence between individual key presses given the latent state representation, the network predicts the probability vector $\mathbf{p}_t$, yielding:

$$
y_t \in \{0, 1\}^{n_\rho}
$$

During inference, this distribution is thresholded at $0.5$ to produce the deterministic binary vector fed back into the ViZDoom engine. This dynamic scaling prevents data sparsity and gradient dilution that would occur if unused weapon keys were permanently mapped to the output layer during non-combat tasks.

## Temporal Dynamics

The environment runs at a fixed tic rate of 35 Hz ($\Delta t \approx 28.5\text{ms}$). Because standard RNNs are discrete, they struggle with the asynchronous loop of live gameplay. Golem utilizes Liquid Time-Constant (LTC) networks to model the hidden state $x(t)$ as an Ordinary Differential Equation (ODE). 

To maintain strict temporal consistency between the continuous differential solver and the discrete game clock, we employ a **Frame Skip** (or sleep-cycle) heuristic to lock the inference latency ($\tau_{inf}$) to the engine's tic rate:

$$
\tau_{total} = \tau_{inf} + \tau_{sleep} \approx 28\text{ms}
$$

This ensures the numerical integration steps within the LNN correctly correspond to the simulated passage of time within the POMDP.