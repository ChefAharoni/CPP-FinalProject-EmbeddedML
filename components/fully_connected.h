// components/fully_connected.h
// Fully Connected (Dense) layer component
// Pure C++ implementation for embedded systems
// Supports fused activation functions (matching TensorFlow/TFLite behavior)

#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <cstddef>
#include <algorithm>

namespace embedded_ml {

// Activation function types (matching TFLite)
enum class ActivationType {
    NONE,
    RELU
};

// Fully Connected layer: output = activation(input * weights^T + bias)
// input: input vector of size input_size
// weights: weight matrix of size [output_size x input_size] (row-major)
// bias: bias vector of size output_size
// output: output vector of size output_size
// activation: activation function to apply (NONE or RELU)
template<typename T>
void FullyConnected(
    const T* input,
    const T* weights,
    const T* bias,
    T* output,
    size_t input_size,
    size_t output_size,
    ActivationType activation = ActivationType::NONE
) {
    // Initialize output with bias
    for (size_t i = 0; i < output_size; ++i) {
        output[i] = bias[i];
    }
    
    // Matrix-vector multiplication: output += input * weights^T
    // For each output neuron
    for (size_t i = 0; i < output_size; ++i) {
        // For each input feature
        for (size_t j = 0; j < input_size; ++j) {
            // weights[i * input_size + j] is weight from input j to output i
            output[i] += input[j] * weights[i * input_size + j];
        }
    }
    
    // Apply fused activation function (matching TensorFlow behavior)
    if (activation == ActivationType::RELU) {
        for (size_t i = 0; i < output_size; ++i) {
            output[i] = std::max(static_cast<T>(0), output[i]);
        }
    }
}

} // namespace embedded_ml

#endif // FULLY_CONNECTED_H

// components/fully_connected.h
// Fully Connected (Dense) layer component
// Pure C++ implementation for embedded systems
// Supports fused activation functions (matching TensorFlow/TFLite behavior)

#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <cstddef>
#include <algorithm>

namespace embedded_ml {

// Activation function types (matching TFLite)
enum class ActivationType {
    NONE,
    RELU
};

// Fully Connected layer: output = activation(input * weights^T + bias)
// input: input vector of size input_size
// weights: weight matrix of size [output_size x input_size] (row-major)
// bias: bias vector of size output_size
// output: output vector of size output_size
// activation: activation function to apply (NONE or RELU)
template<typename T>
void FullyConnected(
    const T* input,
    const T* weights,
    const T* bias,
    T* output,
    size_t input_size,
    size_t output_size,
    ActivationType activation = ActivationType::NONE
) {
    // Initialize output with bias
    for (size_t i = 0; i < output_size; ++i) {
        output[i] = bias[i];
    }
    
    // Matrix-vector multiplication: output += input * weights^T
    // For each output neuron
    for (size_t i = 0; i < output_size; ++i) {
        // For each input feature
        for (size_t j = 0; j < input_size; ++j) {
            // weights[i * input_size + j] is weight from input j to output i
            output[i] += input[j] * weights[i * input_size + j];
        }
    }
    
    // Apply fused activation function (matching TensorFlow behavior)
    if (activation == ActivationType::RELU) {
        for (size_t i = 0; i < output_size; ++i) {
            output[i] = std::max(static_cast<T>(0), output[i]);
        }
    }
}

} // namespace embedded_ml

#endif // FULLY_CONNECTED_H

