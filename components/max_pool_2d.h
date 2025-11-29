// components/max_pool_2d.h
// 2D Max Pooling layer component
// Pure C++ implementation for embedded systems
// Supports padding, stride, and fused activation functions

#ifndef MAX_POOL_2D_H
#define MAX_POOL_2D_H

#include <cstddef>
#include <algorithm>
#include <limits>
#include "conv_2d.h"  // For PaddingType, Offset4D, ActivationType, and ApplyActivation

namespace embedded_ml {

// 2D Max Pooling: output = activation(max_pool(input))
// input: input tensor of shape [batch, input_height, input_width, channels] (NHWC format)
// output: output tensor of shape [batch, output_height, output_width, channels]
// batch_size: number of batches
// input_height, input_width: input spatial dimensions
// channels: number of channels
// filter_height, filter_width: pooling window size
// stride_height, stride_width: pooling strides
// padding: padding type (VALID or SAME)
// activation: activation function to apply (NONE or RELU)
template<typename T>
void MaxPool2D(
    const T* input,
    T* output,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t channels,
    size_t filter_height,
    size_t filter_width,
    size_t stride_height,
    size_t stride_width,
    PaddingType padding,
    ActivationType activation = ActivationType::NONE
) {
    // Compute padding values
    int pad_height_before, pad_height_after;
    int pad_width_before, pad_width_after;
    ComputePadding(input_height, filter_height, stride_height, padding,
                   &pad_height_before, &pad_height_after);
    ComputePadding(input_width, filter_width, stride_width, padding,
                   &pad_width_before, &pad_width_after);

    // Compute output dimensions
    size_t output_height = ComputeOutputSize(input_height, filter_height, stride_height, padding);
    size_t output_width = ComputeOutputSize(input_width, filter_width, stride_width, padding);

    // Process each batch
    for (size_t batch = 0; batch < batch_size; ++batch) {
        // Process each output position
        for (size_t out_y = 0; out_y < output_height; ++out_y) {
            for (size_t out_x = 0; out_x < output_width; ++out_x) {
                // Process each channel
                for (size_t channel = 0; channel < channels; ++channel) {
                    T max_val = std::numeric_limits<T>::lowest();

                    // Find maximum in pooling window
                    for (size_t filter_y = 0; filter_y < filter_height; ++filter_y) {
                        for (size_t filter_x = 0; filter_x < filter_width; ++filter_x) {
                            // Compute input position
                            int in_y = static_cast<int>(out_y * stride_height) - pad_height_before +
                                      static_cast<int>(filter_y);
                            int in_x = static_cast<int>(out_x * stride_width) - pad_width_before +
                                      static_cast<int>(filter_x);

                            // Zero padding: skip if outside input bounds
                            if (in_y < 0 || in_y >= static_cast<int>(input_height) ||
                                in_x < 0 || in_x >= static_cast<int>(input_width)) {
                                continue;
                            }

                            size_t input_idx = Offset4D(batch, in_y, in_x, channel,
                                                        batch_size, input_height, input_width, channels);
                            max_val = std::max(max_val, input[input_idx]);
                        }
                    }

                    // Apply activation
                    size_t output_idx = Offset4D(batch, out_y, out_x, channel,
                                                batch_size, output_height, output_width, channels);
                    output[output_idx] = ApplyActivation(max_val, activation);
                }
            }
        }
    }
}

} // namespace embedded_ml

#endif // MAX_POOL_2D_H

