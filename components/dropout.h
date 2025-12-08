// components/dropout.h
// Dropout operation component
// Pure C++ implementation for embedded systems
// During inference, dropout is a no-op (passes through input unchanged)
// Dropout only affects values during training

#ifndef DROPOUT_H
#define DROPOUT_H

#include <cstddef>

namespace embedded_ml {

// Dropout operation: output = input (during inference)
// input: input tensor data
// output: output tensor data (must be pre-allocated)
// size: number of elements
// rate: dropout rate (ignored during inference, kept for API compatibility)
// Note: During inference, dropout is a no-op - it just copies input to output
//       The dropout rate parameter is kept for API compatibility but is not used
template<typename T>
void Dropout(
    const T* input,
    T* output,
    size_t size,
    float rate = 0.0f
) {
    // During inference, dropout is a no-op - just copy input to output
    // The rate parameter is ignored during inference
    for (size_t i = 0; i < size; ++i) {
        output[i] = input[i];
    }
}

// In-place version: modifies the input array
template<typename T>
void Dropout(T* data, size_t size, float rate = 0.0f) {
    // During inference, dropout is a no-op - do nothing
    // The rate parameter is ignored during inference
    (void)data;  // Suppress unused parameter warning
    (void)size;
    (void)rate;
}

} // namespace embedded_ml

#endif // DROPOUT_H


