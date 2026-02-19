# The Brain: Liquid Neural Networks

The core of Golem is a **Neural Circuit Policy (NCP)** utilizing **Closed-form Continuous (CfC)** cells. 

## 1. Visual Cortex (CNN)

The input state $s_t$ is first processed by a standard Convolutional Neural Network (CNN) to extract spatial features, reducing the pixel space into a latent feature vector $I(t)$ of size 12,544.

## 2. Liquid Core (CfC) & State Persistence

Standard Recurrent Neural Networks (RNNs) update their hidden state via discrete steps. **Liquid Time-Constant (LTC)** networks model the hidden state $x(t)$ as a system of Ordinary Differential Equations (ODEs).

$$
\frac{dx(t)}{dt} = - \left[ w_\tau + f(x(t), I(t), \theta) \right] \cdot x(t) + A \cdot f(x(t), I(t), \theta)
$$

### The "Amnesia" Constraint (Stateful Inference)

Because the network relies on differential equations, it requires a continuous flow of time to accumulate evidence and build potential. 

During live gameplay (Inference), the engine feeds the brain one frame at a time. **The hidden state $hx$ must be explicitly captured and fed back into the network on the next frame.** Failing to persist the hidden state lobotomizes the network 35 times a second, preventing the activation threshold from ever being reached.

## 3. Motor Cortex (Linear Head)

The liquid state $x(t)$ is projected to the 8-dimensional action space via a linear layer.

$$
\hat{y}_t = W_{out} x(t) + b_{out}
$$