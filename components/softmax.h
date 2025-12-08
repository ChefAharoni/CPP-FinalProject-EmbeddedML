// components/softmax.h
// Softmax activation function component
// Pure C++ implementation for embedded systems

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cmath>
#include <algorithm>
#include <cstddef>

namespace embedded_ml {

// Softmax activation: output[i] = exp(input[i] - max) / sum(exp(input[j] - max))
// The subtraction of max is for numerical stability
// In-place operation: modifies the input array
template<typename T>
void Softmax(T* data, size_t size) {
    if (size == 0) return;
    
    // Find maximum value for numerical stability
    T max_val = data[0];
    for (size_t i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    
    // Compute exponentials and sum
    T sum = static_cast<T>(0);
    for (size_t i = 0; i < size; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    
    // Normalize
    if (sum > static_cast<T>(0)) {
        for (size_t i = 0; i < size; ++i) {
            data[i] /= sum;
        }
    }
}

// Softmax with separate input/output arrays
template<typename T>
void Softmax(const T* input, T* output, size_t size) {
    if (size == 0) return;
    
    // Find maximum value for numerical stability
    T max_val = input[0];
    for (size_t i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    // Compute exponentials and sum
    T sum = static_cast<T>(0);
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    if (sum > static_cast<T>(0)) {
        for (size_t i = 0; i < size; ++i) {
            output[i] /= sum;
        }
    }
}

} // namespace embedded_ml

#endif // SOFTMAX_H

