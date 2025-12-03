// components/add.h
// Add (element-wise addition) component
// Pure C++ implementation for embedded systems
// Supports fused activation functions (matching TensorFlow/TFLite behavior)

#ifndef ADD_H
#define ADD_H

#include <cstddef>
#include <algorithm>
#include "fully_connected.h"  // For ActivationType

namespace embedded_ml {

// Helper function to apply activation
template<typename T>
inline T ApplyActivation(T value, ActivationType activation) {
    switch (activation) {
        case ActivationType::RELU:
            return std::max(static_cast<T>(0), value);
        case ActivationType::NONE:
        default:
            return value;
    }
}

// Element-wise addition: output = activation(input1 + input2)
// For same-shape tensors (no broadcasting)
// input1: first input tensor of size size
// input2: second input tensor of size size
// output: output tensor of size size
// size: number of elements in each tensor
// activation: activation function to apply (NONE or RELU)
template<typename T>
void Add(
    const T* input1,
    const T* input2,
    T* output,
    size_t size,
    ActivationType activation = ActivationType::NONE
) {
    for (size_t i = 0; i < size; ++i) {
        T sum = input1[i] + input2[i];
        output[i] = ApplyActivation(sum, activation);
    }
}

// In-place addition: input1 = activation(input1 + input2)
// Modifies input1 array
template<typename T>
void AddInPlace(
    T* input1,
    const T* input2,
    size_t size,
    ActivationType activation = ActivationType::NONE
) {
    for (size_t i = 0; i < size; ++i) {
        input1[i] = ApplyActivation(input1[i] + input2[i], activation);
    }
}

} // namespace embedded_ml

#endif // ADD_H

