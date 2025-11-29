// components/shape.h
// Shape operation component
// Pure C++ implementation for embedded systems
// Extracts the shape of a tensor as a 1D integer array

#ifndef SHAPE_H
#define SHAPE_H

#include <cstddef>

namespace embedded_ml {

// Shape operation: output = shape(input)
// input_shape: array of dimension sizes [dim0, dim1, dim2, ...]
// num_dims: number of dimensions
// output: output array of size num_dims (will be filled with shape values)
// Note: This is a simple copy operation for static shapes
template<typename T>
void Shape(
    const T* input_shape,
    size_t num_dims,
    int32_t* output
) {
    for (size_t i = 0; i < num_dims; ++i) {
        output[i] = static_cast<int32_t>(input_shape[i]);
    }
}

// Alternative version that takes dimension sizes directly
inline void Shape(
    const int32_t* dims,
    size_t num_dims,
    int32_t* output
) {
    for (size_t i = 0; i < num_dims; ++i) {
        output[i] = dims[i];
    }
}

} // namespace embedded_ml

#endif // SHAPE_H

