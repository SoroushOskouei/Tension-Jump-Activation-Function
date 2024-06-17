# Tension-Jump-Activation-Function
A trainable physics-based activation layer. For citation please check [![DOI](https://zenodo.org/badge/816137721.svg)](https://zenodo.org/doi/10.5281/zenodo.11949670)

![image](https://github.com/SoroushOskouei/Tension-Jump-Activation-Function/assets/57323986/99674c1c-3c06-43bf-975d-88a9735f9156)


# Tension Jump Activation Layer

The `Tension_Jump_Activation` class defines a custom activation layer where the response of the activation function changes based on a set threshold. This layer is designed for scenarios where a linear response is sufficient up to a certain point (threshold), beyond which a more aggressive response (exponential) is required.

## Components

This activation layer includes three primary components, each represented as trainable weights with constraints:

- **Threshold (T):** The point at which the activation behavior changes from a stable (linear) response to a breaking (exponential) response.
- **Resistance (K):** Controls the slope of the linear part of the activation function, representing how much input resistance affects the output before the threshold.
- **Baseline (B):** Influences the starting point of the exponential response, simulating a baseline or offset value in the output beyond the threshold.

## Mathematical Formulation

Given an input \( x \), the activation function \( f(x) \) is defined as follows:

### For inputs \( x \) less than or equal to the threshold \( T \):

f(x) = K * x

### For inputs \( x \) greater than the threshold \( T \):

f(x) = B * T + exp(K * (x - T))

Where:
- \( x \) is the input to the neuron,
- \( T \) is the threshold,
- \( K \) is the resistance factor,
- \( B \) is the baseline value.

### Behavior

- **Below or at the Threshold ( \( x \leq T \) ):** The output is directly proportional to the input, scaled by the resistance factor \( K \).
- **Above the Threshold ( \( x > T \) ):** The output starts from \( B \cdot T \) and grows exponentially with the increase in input beyond the threshold, modulated by \( K \).

## Constraints

Each weight (T, K, B) is constrained within specified minimum and maximum values to ensure that the parameters do not diverge during training. This is managed using a custom constraint class `MinMaxValueConstraint`, which clips the weights to stay within the provided bounds.

## Implementation Details

The layer is implemented using TensorFlow's `Layer` class, and weights are defined with initial random values within their respective ranges. This allows the layer to learn the most suitable values for \( T \), \( K \), and \( B \) based on the training data.

