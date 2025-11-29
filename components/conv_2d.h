// components/conv_2d.h
// 2D Convolution layer component
// Pure C++ implementation for embedded systems
// Supports padding, stride, dilation, and fused activation functions

#ifndef CONV_2D_H
#define CONV_2D_H

#include <cstddef>
#include <algorithm>
#include <cmath>
#include "fully_connected.h"  // For ActivationType

namespace embedded_ml {

// Padding type (matching TFLite)
enum class PaddingType {
    VALID,   // No padding
    SAME     // Padding to maintain output size
};

// Helper function to compute output size with padding
inline int ComputeOutputSize(int input_size, int filter_size, int stride, PaddingType padding) {
    if (padding == PaddingType::SAME) {
        return (input_size + stride - 1) / stride;
    } else {  // VALID
        return (input_size - filter_size + stride) / stride;
    }
}

// Helper function to compute padding values
inline void ComputePadding(int input_size, int filter_size, int stride, PaddingType padding_type,
                          int* pad_before, int* pad_after) {
    if (padding_type == PaddingType::SAME) {
        int output_size = ComputeOutputSize(input_size, filter_size, stride, PaddingType::SAME);
        int total_padding = (output_size - 1) * stride + filter_size - input_size;
        *pad_before = total_padding / 2;
        *pad_after = total_padding - *pad_before;
    } else {  // VALID
        *pad_before = 0;
        *pad_after = 0;
    }
}

// Helper function to compute linear index in 4D tensor [batch, height, width, channel]
inline size_t Offset4D(size_t batch, size_t height, size_t width, size_t channel,
                       size_t batch_size, size_t height_size, size_t width_size, size_t channel_size) {
    return ((batch * height_size + height) * width_size + width) * channel_size + channel;
}

// Apply activation function
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

// 2D Convolution: output = activation(conv(input, filter) + bias)
// input: input tensor of shape [batch, input_height, input_width, input_channels] (NHWC format)
// filter: filter tensor of shape [output_channels, filter_height, filter_width, input_channels]
// bias: bias vector of size output_channels (can be nullptr)
// output: output tensor of shape [batch, output_height, output_width, output_channels]
// batch_size: number of batches
// input_height, input_width: input spatial dimensions
// input_channels: number of input channels
// filter_height, filter_width: filter spatial dimensions
// output_channels: number of output channels
// stride_height, stride_width: convolution strides
// padding: padding type (VALID or SAME)
// dilation_height, dilation_width: dilation factors (default 1)
// activation: activation function to apply (NONE or RELU)
template<typename T>
void Conv2D(
    const T* input,
    const T* filter,
    const T* bias,
    T* output,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    size_t input_channels,
    size_t filter_height,
    size_t filter_width,
    size_t output_channels,
    size_t stride_height,
    size_t stride_width,
    PaddingType padding,
    ActivationType activation = ActivationType::NONE,
    size_t dilation_height = 1,
    size_t dilation_width = 1
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
                // Process each output channel
                for (size_t out_channel = 0; out_channel < output_channels; ++out_channel) {
                    T total = static_cast<T>(0);

                    // Convolve filter over input
                    for (size_t filter_y = 0; filter_y < filter_height; ++filter_y) {
                        for (size_t filter_x = 0; filter_x < filter_width; ++filter_x) {
                            // Compute input position with dilation
                            int in_y = static_cast<int>(out_y * stride_height) - pad_height_before +
                                      static_cast<int>(filter_y * dilation_height);
                            int in_x = static_cast<int>(out_x * stride_width) - pad_width_before +
                                      static_cast<int>(filter_x * dilation_width);

                            // Zero padding: skip if outside input bounds
                            if (in_y < 0 || in_y >= static_cast<int>(input_height) ||
                                in_x < 0 || in_x >= static_cast<int>(input_width)) {
                                continue;
                            }

                            // Accumulate over input channels
                            for (size_t in_channel = 0; in_channel < input_channels; ++in_channel) {
                                size_t input_idx = Offset4D(batch, in_y, in_x, in_channel,
                                                           batch_size, input_height, input_width, input_channels);
                                size_t filter_idx = Offset4D(out_channel, filter_y, filter_x, in_channel,
                                                             output_channels, filter_height, filter_width, input_channels);
                                total += input[input_idx] * filter[filter_idx];
                            }
                        }
                    }

                    // Add bias
                    if (bias) {
                        total += bias[out_channel];
                    }

                    // Apply activation
                    size_t output_idx = Offset4D(batch, out_y, out_x, out_channel,
                                                batch_size, output_height, output_width, output_channels);
                    output[output_idx] = ApplyActivation(total, activation);
                }
            }
        }
    }
}

} // namespace embedded_ml

#endif // CONV_2D_H

