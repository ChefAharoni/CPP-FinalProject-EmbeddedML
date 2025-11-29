// components/relu.h
// ReLU (Rectified Linear Unit) activation function component
// Pure C++ implementation for embedded systems

#ifndef RELU_H
#define RELU_H

#include <algorithm>
#include <cstddef>

namespace embedded_ml {

// ReLU activation: output = max(0, input)
// In-place operation: modifies the input array
template<typename T>
void ReLU(T* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::max(static_cast<T>(0), data[i]);
    }
}

// ReLU with separate input/output arrays
template<typename T>
void ReLU(const T* input, T* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::max(static_cast<T>(0), input[i]);
    }
}

} // namespace embedded_ml

#endif // RELU_H

