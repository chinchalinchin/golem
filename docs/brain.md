# The Brain: Liquid Neural Networks

The core of Golem is a Neural Circuit Policy (NCP) utilizing Closed-form Continuous (CfC) cells. With the introduction of multi-modal sensor fusion, the brain can dynamically scale its perception across visual, spatial (depth), and auditory domains.

## 1. Visual Cortex (CNN)

The input observation $o_t$ is first processed by a Convolutional Neural Network (CNN) to extract spatial features. This hierarchy reduces the high-dimensional pixel space into a flattened, latent feature vector $V(t)$. 

The architecture scales dynamically based on the configured `cortical_depth` ($D$) and active `sensors`. Given an input tensor of $C \times 64 \times 64$ (where $C=3$ for standard RGB, or $C=4$ if the stereoscopic depth buffer is enabled), the sequential convolutions (kernel size 4, stride 2) and ReLU activations compress the spatial manifold. For example, a depth of $D=2$ yields a highly dimensional feature map, while deeper configurations (e.g., $D=4$) aggressively pool the feature maps to a much smaller dense representation.

## 2. Auditory Cortex (1D CNN)

If the `audio` sensor is enabled, Golem expands its phenomenology by routing the raw, high-frequency stereo audio buffer through a parallel 1D Convolutional Neural Network (`nn.Conv1d`). 

This pathway consists of sequential 1D convolutions (e.g., kernel size 8, stride 4) and an `AdaptiveAvgPool1d` layer to extract auditory features, outputting a latent audio vector $A(t)$. 

If multiple modalities are active, their respective feature vectors are concatenated:

$$
I(t) = V(t) \oplus A(t)
$$

Where $I(t) \in \mathbb{R}^{W_f}$ is the final, unified multi-modal representation fed into the liquid core, and $W_f$ is the dynamically calculated flat size.

## 3. Liquid Core (CfC) & State Persistence

Standard Recurrent Neural Networks (RNNs) update their hidden state via discrete, uniform steps. In contrast, **Liquid Time-Constant (LTC)** networks model the hidden state $x(t)$ as a system of Ordinary Differential Equations (ODEs) responding to a continuous flow of time:

$$
\frac{dx(t)}{dt} = -\left[w_\tau + f(x(t), I(t); \theta)\right] \odot x(t) + A \odot f(x(t), I(t); \theta)
$$

However, solving this ODE numerically during live gameplay introduces severe computational latency. To achieve the $\approx 28\text{ms}$ inference target required by the ViZDoom engine, Golem utilizes the Closed-form Continuous (CfC) approximation (Hasani et al., 2022). This mathematically bypasses the numerical solver entirely by approximating the integral with a tight, closed-form gating mechanism:

$$
x(t) = \sigma(-f(x, I; \theta_f) t) \odot g(x, I; \theta_g) + \left[1 - \sigma(-f(x, I; \theta_f) t)\right] \odot h(x, I; \theta_h)
$$

Where $f$, $g$, and $h$ represent distinct neural network branches parameterizing the state flow, and $\odot$ denotes the Hadamard product.

### The "Amnesia" Constraint (Stateful Inference)

Because the underlying differential mathematics assume a continuous temporal flow, the network must accumulate evidence to build action potential. During asynchronous live gameplay (inference), the engine feeds the visual and auditory cortices discrete buffers. The hidden state $hx$ must be explicitly captured and recursively fed back into the network on the subsequent frame. Failing to persist this state across the deployment loop lobotomizes the network 35 times a second, preventing the CfC activation threshold from ever being reached.

## 4. Motor Cortex (Linear Head)

The liquid hidden state $x(t) \in \mathbb{R}^{W_m}$ (where $W_m$ is the dynamically configured `working_memory`, e.g., 64 or 128) is projected to the dynamic action space via a final linear transformation. To accommodate the variable supersets defined by the active profile $\rho$, the output weight matrix dynamically scales its dimensionality $n_\rho \in \{8, 9, 10\}$:

$$
\mathbf{z}_t = W_{out} x(t) + b_{out}
$$

This produces raw logits $\mathbf{z}_t$, which are subsequently passed through a continuous Sigmoid activation function to yield the final predicted probabilities for the multi-label Bernoulli distribution:

$$
\hat{\mathbf{y}}_t = \sigma(\mathbf{z}_t)
$$

---

## API Reference

Because the architecture is fully dynamic, the `DoomLiquidNet` class constructs its layers on-the-fly based on the active `app.yaml` configuration profile and the selected sensor fusion modalities.

::: app.models.brain.DoomLiquidNet