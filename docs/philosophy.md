# Philosophy of Golem

The architecture of Golem represents a fundamental departure from both classical algorithmic game AI (Finite State Machines) and modern Large Language Models (LLMs). Rather than hard-coding conditional logic or predicting the next token in a sequence, Golem is built on principles of **Embodied Cognition** and **Biologically Inspired Computing**. 

This document outlines the philosophical assumptions and epistemological constraints that dictate how the agent is trained and how it interacts with its environment.

## 1. Phenomenological Purity (Epistemology)

In classical game AI, bots are typically "omniscient." They read the engine's memory addresses to know their exact $(X, Y, Z)$ coordinates, the precise integer value of their health, and the locations of enemies through walls. This creates a brittle, mechanical intelligence that operates outside the physics of the world it inhabits.

Golem is constrained to a strict **Phenomenological Boundary**. The agent only knows what it can "experience" through its sensory inputs. 

* **The Rejection of Latent State:** We explicitly discard the underlying game state variables $s_t \in \mathcal{S}$ provided by the ViZDoom API. The agent's reality is entirely constructed from the raw pixel buffer $o_t \in \Omega$ and its active sensory extensions. 

* **The "Doomguy" Heuristic (Physiological Awareness):** To grant the agent a sense of mortality, we preserve the classic DOOM status bar (HUD) within its visual field. Rather than reading a health integer, the agent must learn to interpret its own physical deterioration by recognizing the pixel distortion of the marine's bloody face. 

* **Auditory Phenomenology (Acoustic Spatialization):**  We do not feed the agent discrete alerts for nearby enemies. Instead, raw audio waveforms are extracted and mathematically transformed into Mel Spectrograms. The agent processes sound as a 2D spatial map through its Auditory Cortex, learning the "visual shape" of a fireball's hiss or a demon's growl, naturally compressing high-frequency acoustic noise without violating its phenomenological constraints.

* **Thermal Phenomenology (Biomimicry):**  The texture palette of DOOM is highly monochromatic, dominated by heavily dithered browns, greys, and blacks. We hypothesize that a pure RGB visual model clusters too narrowly, forcing the agent to struggle when distinguishing camouflaged threats from static geometric backgrounds. Taking inspiration from the animal kingdom—specifically predators that utilize heat vision—Golem utilizes semantic segmentation buffers to experience a binary "thermal" qualia. This projects dynamic entities as distinct hotspots, effectively decoupling the neural circuitry responsible for background spatial navigation (Visual Cortex) from the circuitry dedicated to active threat detection (Thermal Cortex).

* **Derivatives over Statics:** While static UI elements normally cause standard Convolutional Neural Networks (CNNs) to overfit to spurious correlations, Golem's liquid core relies on differential equations. It naturally ignores static plastic bezels ($\frac{dx}{dt} = 0$) and hyper-fixates on sudden flashes of damage, ammunition changes, or sudden thermal spikes ($\frac{dx}{dt} > 0$).

By restricting the agent to phenomenological variables, we force the neural network to develop robust, generalized multi-modal heuristics (e.g., "red screen tint implies danger," "this audio shape means a plasma rifle is firing") rather than memorizing a specific map's coordinate grid.

## 2. Time as a Continuous Flow (Ontology)

Standard Recurrent Neural Networks (RNNs), Transformers, and Deep Q-Networks (DQNs) treat time as a sequence of discrete, uniform snapshots. They are inherently stateless between ticks. 

Biological organisms do not experience time as a slideshow; they experience it as a continuous flow. Golem utilizes **Liquid Time-Constant (LTC)** networks, which are mathematically modeled as a system of Ordinary Differential Equations (ODEs):

$$
\frac{dx(t)}{dt} = -\left[w_\tau + f(x(t), I(t); \theta)\right] \odot x(t) + A \odot f(x(t), I(t); \theta)
$$

* **Inherent Causality:** Because the hidden state $x(t)$ represents an integrated accumulation of past evidence, the Liquid Neural Network (LNN) acts as a highly sensitive change-detector. It does not just analyze a frame; it analyzes the *delta* between frames. This allows the network to naturally deduce cause-and-effect relationships (e.g., "Pressing `ATTACK` causes the pixels in the center of the screen to explode").
* **Temporal Elasticity:** The environment operates at a fixed 35 Hz, but network latency and rendering variations introduce micro-fluctuations. Standard discrete networks break down when the temporal gap between frames is inconsistent. Because the LNN is a continuous function solved via the Closed-form Continuous (CfC) approximation, it can evaluate its state at any arbitrary temporal slice without catastrophic failure.

## 3. The Fallibility of the Expert (Behavioral Cloning)

Golem is trained via **Behavioral Cloning (BC)**. The philosophical premise of BC is that the human demonstrator's trajectory represents the optimal policy $\pi^*$. The network attempts to minimize the Binary Cross-Entropy between its predicted logits and the expert's keystrokes.

However, pure imitation relies on a flawed assumption: that the expert is perfect, and the agent will execute the imitation perfectly. 

* **The Covariate Shift Problem:** During live inference, microscopic mathematical variations will eventually push the agent into a sub-optimal state outside the bounds of the training data (e.g., walking into a corner). Because the expert was never recorded staring at a corner, the agent has no phenomenological frame of reference for how to escape. Errors compound until the agent collapses into a permanent equilibrium.
* **DAgger (Dataset Aggregation) as the Epistemological Cure:** To resolve this, Golem employs an active intervention pipeline. The human operator monitors the autonomous agent and forcibly overrides its neural pathway when it makes a mistake. This generates `_recovery` traces, formally teaching the agent not just how to succeed, but how to recognize its own failures and return to the optimal path. We introduce "sin" (error) into the dataset so the agent can learn the concept of "redemption" (correction).

## 4. Architectural Scaling (Biological Plasticity)

The architecture of the brain is not hardcoded. Just as biological neural circuits scale based on the complexity of the organism, Golem's `DoomLiquidNet` scales dynamically via configuration (`app.yaml`).

The depth of the visual cortex (`cortical_depth`) and the capacity of the temporal recurrence (`working_memory`) expand to match the dimensionality of the active action superset $\rho$. This ensures computational resources are not wasted on simple navigation tasks, while preserving the capacity required for complex, multi-modal deathmatches incorporating stereopsis, audition, and thermal mapping.