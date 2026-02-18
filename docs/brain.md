# The Brain: Liquid Neural Networks

The core of Golem is a **Neural Circuit Policy (NCP)** utilizing **Closed-form Continuous (CfC)** cells. This architecture is chosen over LSTMs or Transformers due to its superior handling of causality and temporal dynamics in continuous-time environments.



---

## 1. Visual Cortex (CNN)

The input state $s_t$ is first processed by a standard Convolutional Neural Network (CNN) to extract spatial features.

$$
h_{visual} = \text{ReLU}(\text{Conv2d}(s_t))
$$

This reduces the high-dimensional pixel space ($3 \times 64 \times 64$) into a latent feature vector $I(t)$ of size 12,544. This vector serves as the sensory input to the liquid network.

---

## 2. Liquid Core (CfC)

Standard Recurrent Neural Networks (RNNs) update their hidden state via discrete steps: 

$$
h_{t+1} = f(h_t, x_t)
$$

**Liquid Time-Constant (LTC)** networks, conversely, model the hidden state $x(t)$ as a system of Ordinary Differential Equations (ODEs). This allows the network to represent time as a continuous flow.

### The Liquid ODE

$$
\frac{dx(t)}{dt} = - \left[ w_\tau + f(x(t), I(t), \theta) \right] \cdot x(t) + A \cdot f(x(t), I(t), \theta)
$$

Where:

* $x(t)$: The hidden state (membrane potential).
* $I(t)$: The input signal (from the CNN).
* $w_\tau$: The **Time-Constant** (decay rate).
* $f(\cdot)$: A non-linear nonlinearity (sigmoid/tanh).

!!! info "Adaptive Causality"
    The term $w_\tau + f(x(t), I(t))$ acts as a **dynamic time-constant**. 
    
    * If the input $I(t)$ is strong/novel, the time-constant shrinks, and the network reacts **fast**.
    * If the input is weak/static, the time-constant grows, and the network retains memory **longer**.

### The Closed-Form Solution

Solving ODEs numerically step-by-step is slow and unstable (vanishing gradients). Golem uses the **Closed-form Continuous (CfC)** variant (Hasani et al., 2022), which approximates the integral of the liquid ODE:

$$
x(t) \approx \underbrace{\sigma( -f(x, I, w_\tau) t )}_{\text{Decay}} \odot \underbrace{g(x, I, A)}_{\text{Input}}
$$

This allows the network to enjoy the benefits of ODEs (causality, continuous time) with the speed and stability of standard RNNs.

---

## 3. Motor Cortex (Linear Head)

The liquid state $x(t)$ is projected to the action space via a linear layer.

$$
\hat{y}_t = W_{out} x(t) + b_{out}
$$

These logits are then passed through a Sigmoid function during inference to obtain probabilities.

```mermaid
graph TD
    Input[Input Tensor (3x64x64)] --> CNN[Visual Cortex (CNN)]
    CNN --> Latent[Latent Vector (12544)]
    Latent --> CfC{Liquid Core (CfC)}
    CfC --> Linear[Linear Projection]
    Linear --> Logits[Action Logits (3)]
    Logits --> Sigmoid((Sigmoid))
    Sigmoid --> Output[Probabilities]

    style CfC fill:#f9f,stroke:#333,stroke-width:4px

```