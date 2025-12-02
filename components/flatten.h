// components/flatten.h
// Flatten operation component
// Pure C++ implementation for embedded systems
// Flattens a multi-dimensional tensor to 1D (except batch dimension)

#ifndef FLATTEN_H
#define FLATTEN_H

#include <cstddef>

namespace embedded_ml {

// Flatten operation: output = flatten(input)
// Flattens all dimensions except the first (batch) dimension
// input: input tensor data
// input_size: total number of elements in input
// output: output tensor data (must be pre-allocated)
// output_size: total number of elements in output (must equal input_size)
// Note: This is essentially a reshape operation that flattens dimensions
//       For contiguous memory layouts, this is just a copy operation
template<typename T>
void Flatten(
    const T* input,
    size_t input_size,
    T* output,
    size_t output_size
) {
    // Flatten is just a copy operation for contiguous memory
    // The shape information is only used for indexing, but since we're
    // working with flat arrays, we just copy the data
    if (input_size != output_size) {
        return;  // Invalid flatten - sizes don't match
    }
    
    for (size_t i = 0; i < input_size; ++i) {
        output[i] = input[i];
    }
}

// Flatten with explicit dimensions (for clarity/documentation)
// batch_size: number of batches (first dimension)
// feature_size: total size of all feature dimensions (product of remaining dims)
template<typename T>
void Flatten(
    const T* input,
    size_t batch_size,
    size_t feature_size,
    T* output
) {
    size_t total_size = batch_size * feature_size;
    for (size_t i = 0; i < total_size; ++i) {
        output[i] = input[i];
    }
}

} // namespace embedded_ml

#endif // FLATTEN_H


