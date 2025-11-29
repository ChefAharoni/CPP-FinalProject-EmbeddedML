// components/pack.h
// Pack operation component
// Pure C++ implementation for embedded systems
// Packs multiple tensors along a specified axis

#ifndef PACK_H
#define PACK_H

#include <cstddef>
#include "strided_slice.h"  // For ComputeIndex

namespace embedded_ml {

// Pack operation: output = pack(inputs[0], inputs[1], ..., inputs[N-1], axis)
// inputs: array of input tensor pointers
// num_inputs: number of input tensors
// input_dims: array of input dimension sizes (same for all inputs) [dim0, dim1, ...]
// num_dims: number of dimensions (before packing)
// axis: axis along which to pack (0-based)
// output: output tensor data (must be pre-allocated)
// output_dims: array of output dimension sizes
// output_num_dims: number of output dimensions (num_dims + 1)
template<typename T>
void Pack(
    const T* const* inputs,
    size_t num_inputs,
    const int* input_dims,
    size_t num_dims,
    int axis,
    T* output,
    const int* output_dims,
    size_t output_num_dims
) {
    // Compute output size
    size_t output_size = 1;
    for (size_t i = 0; i < output_num_dims; ++i) {
        output_size *= static_cast<size_t>(output_dims[i]);
    }
    
    // Normalize axis to be in range [0, num_dims]
    if (axis < 0) {
        axis += static_cast<int>(num_dims) + 1;
    }
    
    // Iterate over output positions
    int* output_coords = new int[output_num_dims];
    
    for (size_t out_idx = 0; out_idx < output_size; ++out_idx) {
        // Convert output index to coordinates
        size_t temp = out_idx;
        for (int i = static_cast<int>(output_num_dims) - 1; i >= 0; --i) {
            output_coords[i] = static_cast<int>(temp % static_cast<size_t>(output_dims[i]));
            temp /= static_cast<size_t>(output_dims[i]);
        }
        
        // Determine which input tensor this position corresponds to
        int input_idx = output_coords[axis];
        if (input_idx < 0 || input_idx >= static_cast<int>(num_inputs)) {
            continue;  // Skip invalid input index
        }
        
        // Map output coordinates to input coordinates
        int* input_coords = new int[num_dims];
        size_t input_coord_idx = 0;
        for (size_t i = 0; i < output_num_dims; ++i) {
            if (i == static_cast<size_t>(axis)) {
                continue;  // Skip the pack axis
            }
            input_coords[input_coord_idx++] = output_coords[i];
        }
        
        // Copy value from input
        size_t in_idx = ComputeIndex(input_dims, input_coords, num_dims);
        output[out_idx] = inputs[input_idx][in_idx];
        
        delete[] input_coords;
    }
    
    delete[] output_coords;
}

} // namespace embedded_ml

#endif // PACK_H

