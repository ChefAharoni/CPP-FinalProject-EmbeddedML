// components/reshape.h
// Reshape operation component
// Pure C++ implementation for embedded systems
// Reshapes a tensor to a new shape without changing the data

#ifndef RESHAPE_H
#define RESHAPE_H

#include <cstddef>

namespace embedded_ml {

// Reshape operation: output = reshape(input, new_shape)
// input: input tensor data
// input_size: total number of elements in input
// output: output tensor data (must be pre-allocated)
// output_size: total number of elements in output (must equal input_size)
// Note: This is a simple copy operation since reshape doesn't change data layout
//       in memory (both are stored in row-major order)
template<typename T>
void Reshape(
    const T* input,
    size_t input_size,
    T* output,
    size_t output_size
) {
    // Reshape is just a copy operation for contiguous memory
    // The shape information is only used for indexing, but since we're
    // working with flat arrays, we just copy the data
    if (input_size != output_size) {
        return;  // Invalid reshape - sizes don't match
    }
    
    for (size_t i = 0; i < input_size; ++i) {
        output[i] = input[i];
    }
}

} // namespace embedded_ml

#endif // RESHAPE_H

