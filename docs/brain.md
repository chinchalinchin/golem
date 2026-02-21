# The Brain: Liquid Neural Networks

The core of Golem is a Neural Circuit Policy (NCP) utilizing Closed-form Continuous (CfC) cells.

## 1. Visual Cortex (CNN)

The input observation $o_t$ is first processed by a Convolutional Neural Network (CNN) to extract spatial features. This hierarchy reduces the high-dimensional pixel space into a flattened, latent feature vector $I(t)$. 

The architecture scales dynamically based on the configured `cortical_depth` ($D$). Given an input tensor of $3 \times 64 \times 64$, the sequential convolutions (kernel size 4, stride 2) and ReLU activations compress the spatial manifold. For example, a depth of $D=2$ yields $I(t) \in \mathbb{R}^{12544}$, while deeper configurations (e.g., $D=4$) aggressively pool the feature maps to a much smaller dense representation:

$$
I(t) \in \mathbb{R}^{W_f}
$$

Where $W_f$ is the dynamically calculated flat size fed into the liquid core.

## 2. Liquid Core (CfC) & State Persistence

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

Because the underlying differential mathematics assume a continuous temporal flow, the network must accumulate evidence to build action potential. During asynchronous live gameplay (inference), the engine feeds the visual cortex discrete frames. The hidden state $hx$ must be explicitly captured and recursively fed back into the network on the subsequent frame. Failing to persist this state across the deployment loop lobotomizes the network 35 times a second, preventing the CfC activation threshold from ever being reached.

## 3. Motor Cortex (Linear Head)

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

Because the architecture is fully dynamic, the `DoomLiquidNet` class constructs its layers on-the-fly based on the active `app.yaml` configuration profile.

::: app.models.brain.DoomLiquidNet