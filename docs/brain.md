
# The Brain: Liquid Neural Networks

The core of Golem is a **Neural Circuit Policy (NCP)** utilizing **Closed-form Continuous (CfC)** cells. This architecture is chosen over LSTMs or Transformers due to its superior handling of causality and temporal dynamics in continuous-time environments.

## Visual Cortex (CNN)

The input $s_t$ is first processed by a standard Convolutional Neural Network to extract spatial features.

$$
h_{visual} = \text{ReLU}(\text{Conv2d}(s_t))
$$

This reduces the high-dimensional pixel space into a latent feature vector $I(t)$ which serves as the sensory input to the liquid network.

## Liquid Core (CfC)

Standard Recurrent Neural Networks (RNNs) update their hidden state via discrete steps: $h_{t+1} = f(h_t, x_t)$.

**Liquid Time-Constant (LTC)** networks, conversely, model the hidden state $x(t)$ as a system of Ordinary Differential Equations (ODEs):

$$
\frac{dx(t)}{dt} = - \left[ w_\tau + f(x(t), I(t), \theta) \right] \cdot x(t) + A \cdot f(x(t), I(t), \theta)
$$

Where:

* $x(t)$: The hidden state (membrane potential).
* $I(t)$: The input signal (from the CNN).
* $w_\tau$: The time-constant (decay rate).
* $f(\cdot)$: A non-linear nonlinearity (sigmoid/tanh).

### The Closed-Form Solution

Solving ODEs numerically step-by-step is slow and unstable (vanishing gradients). Golem uses the **Closed-form Continuous (CfC)** variant (Hasani et al., 2022), which approximates the integral of the liquid ODE:

$$
x(t) \approx \sigma( -f(x, I, w_\tau) t ) \odot g(x, I, A)
$$

This allows the network to:

1.  **Adaptive Causality:** Adjust its effective time-constant based on the input. If the game is chaotic, the network reacts faster. If the game is static, it retains memory longer.
2.  **Stable Gradients:** The closed-form solution permits standard backpropagation through time (BPTT) without numerical solvers.

## Motor Cortex (Linear Head)

The liquid state $x(t)$ is projected to the action space via a linear layer:

$$
\hat{y}_t = W_{out} x(t) + b_{out}
$$

These logits are then passed through a Sigmoid function during inference to obtain probabilities.
