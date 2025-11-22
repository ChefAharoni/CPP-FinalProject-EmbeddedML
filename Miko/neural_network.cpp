/**
 * Custom Neural Network Implementation
 * Simple, readable implementation of feedforward neural network operations
 */

#include "neural_network.h"
#include <cmath>
#include <algorithm>

namespace CustomNN {

// ============================================================================
// Activation Functions
// ============================================================================

void Activation::relu(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        // ReLU: max(0, x)
        data[i] = (data[i] > 0.0f) ? data[i] : 0.0f;
    }
}

void Activation::softmax(float* data, size_t size) {
    // Find max value for numerical stability
    float max_val = data[0];
    for (size_t i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        data[i] = expf(data[i] - max_val);  // Subtract max for stability
        sum += data[i];
    }

    // Normalize by sum
    for (size_t i = 0; i < size; ++i) {
        data[i] /= sum;
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

void MatrixOps::matvec_multiply(
    const float* weights,  // weights[rows][cols]
    const float* input,     // input[rows]
    float* output,          // output[cols]
    size_t rows,
    size_t cols
) {
    // Initialize output to zero
    for (size_t j = 0; j < cols; ++j) {
        output[j] = 0.0f;
    }

    // Compute: output[j] = sum_i(weights[i][j] * input[i])
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // Access weights as 1D array: weights[i][j] = weights[i * cols + j]
            output[j] += weights[i * cols + j] * input[i];
        }
    }
}

void MatrixOps::vector_add(
    const float* a,
    const float* b,
    float* output,
    size_t size
) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = a[i] + b[i];
    }
}

void MatrixOps::dense_forward(
    const float* input,
    const float* weights,  // weights[input_size][output_size]
    const float* bias,
    float* output,
    size_t input_size,
    size_t output_size
) {
    // Step 1: Matrix-vector multiply: output = weights^T * input
    matvec_multiply(weights, input, output, input_size, output_size);

    // Step 2: Add bias: output = output + bias
    for (size_t i = 0; i < output_size; ++i) {
        output[i] += bias[i];
    }
}

// ============================================================================
// Neural Network
// ============================================================================

NeuralNetwork::NeuralNetwork(
    const float (*l1_weights)[18],
    const float* l1_bias,
    size_t l1_in,
    size_t l1_out,
    const float (*l2_weights)[3],
    const float* l2_bias,
    size_t l2_in,
    size_t l2_out
) : layer1_weights_(l1_weights),
    layer1_bias_(l1_bias),
    layer1_input_size_(l1_in),
    layer1_output_size_(l1_out),
    layer2_weights_(l2_weights),
    layer2_bias_(l2_bias),
    layer2_input_size_(l2_in),
    layer2_output_size_(l2_out)
{
    // Initialize intermediate buffers to zero
    for (size_t i = 0; i < 18; ++i) {
        layer1_output_[i] = 0.0f;
    }
}

void NeuralNetwork::predict(const float* input, float* output) {
    // Layer 1: Dense(18) + ReLU
    // input[2] -> layer1_output[18]
    MatrixOps::dense_forward(
        input,
        reinterpret_cast<const float*>(layer1_weights_),
        layer1_bias_,
        layer1_output_,
        layer1_input_size_,
        layer1_output_size_
    );

    // Apply ReLU activation
    Activation::relu(layer1_output_, layer1_output_size_);

    // Layer 2: Dense(3) + Softmax
    // layer1_output[18] -> output[3]
    MatrixOps::dense_forward(
        layer1_output_,
        reinterpret_cast<const float*>(layer2_weights_),
        layer2_bias_,
        output,
        layer2_input_size_,
        layer2_output_size_
    );

    // Apply Softmax activation
    Activation::softmax(output, layer2_output_size_);
}

int NeuralNetwork::predict_class(const float* input) {
    float output[3];
    predict(input, output);

    // Find argmax
    int predicted_class = 0;
    float max_prob = output[0];

    for (int i = 1; i < 3; ++i) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted_class = i;
        }
    }

    return predicted_class;
}

} // namespace CustomNN
