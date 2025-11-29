// components/strided_slice.h
// Strided Slice operation component
// Pure C++ implementation for embedded systems
// Extracts a slice from a tensor using begin, end, and stride indices

#ifndef STRIDED_SLICE_H
#define STRIDED_SLICE_H

#include <cstddef>
#include <algorithm>
#include <limits>

namespace embedded_ml {

// Helper function to clamp a value between min and max
template<typename T>
inline T Clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(max_val, value));
}

// Helper function to compute start index for an axis
// Handles negative indices, begin_mask, and clamping
inline int ComputeStartIndex(int start, int axis_size, int stride, bool begin_mask) {
    if (begin_mask) {
        return (stride > 0) ? 0 : axis_size - 1;
    }
    
    // Handle negative indices
    if (start < 0) {
        start += axis_size;
    }
    
    // Clamp based on stride direction
    if (stride > 0) {
        return Clamp(start, 0, axis_size);
    } else {
        return Clamp(start, -1, axis_size - 1);
    }
}

// Helper function to compute end index for an axis
// Handles negative indices, end_mask, shrink_axis_mask, and clamping
inline int ComputeEndIndex(int end, int axis_size, int stride, bool end_mask, 
                          bool shrink_axis, int start) {
    if (shrink_axis) {
        return start + 1;
    }
    
    if (end_mask) {
        return (stride > 0) ? axis_size : -1;
    }
    
    // Handle negative indices
    if (end < 0) {
        end += axis_size;
    }
    
    // Clamp based on stride direction
    if (stride > 0) {
        return Clamp(end, 0, axis_size);
    } else {
        return Clamp(end, -1, axis_size - 1);
    }
}

// Helper function to compute linear index in multi-dimensional tensor
// dims: array of dimension sizes
// indices: array of indices for each dimension
// num_dims: number of dimensions
inline size_t ComputeIndex(const int* dims, const int* indices, size_t num_dims) {
    size_t index = 0;
    size_t stride = 1;
    for (int i = static_cast<int>(num_dims) - 1; i >= 0; --i) {
        index += static_cast<size_t>(indices[i]) * stride;
        stride *= static_cast<size_t>(dims[i]);
    }
    return index;
}

// Helper function to compute linear index in input tensor
inline size_t ComputeInputIndex(const int* input_dims, const int* coords, size_t num_dims) {
    return ComputeIndex(input_dims, coords, num_dims);
}

// Strided Slice: output = input[begin:end:stride]
// input: input tensor data
// input_dims: array of input dimension sizes [dim0, dim1, ...]
// num_dims: number of dimensions
// begin: array of start indices for each dimension
// end: array of end indices for each dimension (exclusive)
// stride: array of stride values for each dimension
// begin_mask: bitmask indicating which dimensions should use begin=0
// end_mask: bitmask indicating which dimensions should use end=dim_size
// shrink_axis_mask: bitmask indicating which dimensions should be removed
// output: output tensor data (must be pre-allocated)
// output_dims: array of output dimension sizes
// output_num_dims: number of output dimensions
template<typename T>
void StridedSlice(
    const T* input,
    const int* input_dims,
    size_t num_dims,
    const int* begin,
    const int* end,
    const int* stride,
    int begin_mask,
    int end_mask,
    int shrink_axis_mask,
    T* output,
    const int* output_dims,
    size_t output_num_dims
) {
    // Compute start and end indices for each axis
    int* starts = new int[num_dims];
    int* ends = new int[num_dims];
    
    for (size_t i = 0; i < num_dims; ++i) {
        bool begin_mask_bit = (begin_mask & (1 << i)) != 0;
        bool end_mask_bit = (end_mask & (1 << i)) != 0;
        bool shrink_axis = (shrink_axis_mask & (1 << i)) != 0;
        
        starts[i] = ComputeStartIndex(begin[i], input_dims[i], stride[i], begin_mask_bit);
        ends[i] = ComputeEndIndex(end[i], input_dims[i], stride[i], end_mask_bit, 
                                  shrink_axis, starts[i]);
    }
    
    // Compute output size
    size_t output_size = 1;
    for (size_t i = 0; i < output_num_dims; ++i) {
        output_size *= static_cast<size_t>(output_dims[i]);
    }
    
    // Iterate over output positions
    int* output_coords = new int[output_num_dims];
    int* input_coords = new int[num_dims];
    
    for (size_t out_idx = 0; out_idx < output_size; ++out_idx) {
        // Convert output index to coordinates
        size_t temp = out_idx;
        for (int i = static_cast<int>(output_num_dims) - 1; i >= 0; --i) {
            output_coords[i] = static_cast<int>(temp % static_cast<size_t>(output_dims[i]));
            temp /= static_cast<size_t>(output_dims[i]);
        }
        
        // Map output coordinates to input coordinates
        size_t input_dim_idx = 0;
        for (size_t i = 0; i < num_dims; ++i) {
            bool shrink_axis = (shrink_axis_mask & (1 << i)) != 0;
            if (shrink_axis) {
                input_coords[i] = starts[i];
            } else {
                if (input_dim_idx < output_num_dims) {
                    input_coords[i] = starts[i] + output_coords[input_dim_idx] * stride[i];
                    input_dim_idx++;
                } else {
                    input_coords[i] = starts[i];
                }
            }
        }
        
        // Copy value
        size_t input_idx = ComputeInputIndex(input_dims, input_coords, num_dims);
        output[out_idx] = input[input_idx];
    }
    
    delete[] starts;
    delete[] ends;
    delete[] output_coords;
    delete[] input_coords;
}

} // namespace embedded_ml

#endif // STRIDED_SLICE_H

