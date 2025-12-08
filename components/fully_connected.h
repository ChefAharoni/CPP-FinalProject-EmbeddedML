// components/fully_connected.h
// Fully Connected (Dense) layer component
// Pure C++ implementation for embedded systems
// Supports fused activation functions (matching TensorFlow/TFLite behavior)
// Optimized with loop unrolling and restrict pointers for better performance

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
// Optimized with loop unrolling (4 elements at a time) and restrict pointers
template<typename T>
void FullyConnected(
    const T* __restrict input,
    const T* __restrict weights,
    const T* __restrict bias,
    T* __restrict output,
    size_t input_size,
    size_t output_size,
    ActivationType activation = ActivationType::NONE
) {
    // For each output neuron, compute: output[i] = activation(bias[i] + input * weights[i])
    // Fused computation: bias initialization, matrix multiplication, and activation in one pass
    for (size_t i = 0; i < output_size; ++i) {
        const T* w_row = weights + i * input_size;
        T acc = bias[i];
        
        // Unrolled inner product (4 elements at a time) for better performance
        size_t j = 0;
        for (; j + 3 < input_size; j += 4) {
            acc += input[j]     * w_row[j];
            acc += input[j + 1] * w_row[j + 1];
            acc += input[j + 2] * w_row[j + 2];
            acc += input[j + 3] * w_row[j + 3];
        }
        
        // Handle remaining elements
        for (; j < input_size; ++j) {
            acc += input[j] * w_row[j];
        }
        
        // Apply fused activation function (matching TensorFlow behavior)
        if (activation == ActivationType::RELU) {
            acc = acc > static_cast<T>(0) ? acc : static_cast<T>(0);
        }
        
        output[i] = acc;
    }
}

} // namespace embedded_ml

#endif // FULLY_CONNECTED_H
