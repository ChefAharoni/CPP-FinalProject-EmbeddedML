/**
 * Custom Neural Network Inference Engine
 * Replaces TensorFlow Lite for simple feedforward networks
 *
 * Architecture: Input(2) -> Dense(18, ReLU) -> Dense(3, Softmax)
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <cstddef>
#include <cmath>

namespace CustomNN {

/**
 * Activation Functions
 */
class Activation {
public:
    // ReLU: max(0, x)
    static void relu(float* data, size_t size);

    // Softmax: exp(x_i) / sum(exp(x_j))
    static void softmax(float* data, size_t size);
};

/**
 * Matrix Operations
 */
class MatrixOps {
public:
    /**
     * Dense layer forward pass: output = input * weights + bias
     *
     * @param input Input vector [input_size]
     * @param weights Weight matrix [input_size][output_size]
     * @param bias Bias vector [output_size]
     * @param output Output vector [output_size]
     * @param input_size Size of input vector
     * @param output_size Size of output vector
     */
    static void dense_forward(
        const float* input,
        const float* weights,  // Stored as weights[input_size][output_size]
        const float* bias,
        float* output,
        size_t input_size,
        size_t output_size
    );

    /**
     * Matrix-vector multiplication: output = weights^T * input
     * For a weight matrix stored as weights[rows][cols],
     * this computes: output[j] = sum_i(weights[i][j] * input[i])
     */
    static void matvec_multiply(
        const float* weights,  // weights[rows][cols]
        const float* input,     // input[rows]
        float* output,          // output[cols]
        size_t rows,
        size_t cols
    );

    /**
     * Vector addition: output = a + b
     */
    static void vector_add(
        const float* a,
        const float* b,
        float* output,
        size_t size
    );
};

/**
 * Neural Network Model
 * Simple 2-layer feedforward network with configurable weights
 */
class NeuralNetwork {
private:
    // Layer 1: Dense layer with ReLU
    const float* layer1_weights_;  // Stored as 1D array: [input_size * output_size]
    const float* layer1_bias_;     // [output_size]
    size_t layer1_input_size_;
    size_t layer1_output_size_;

    // Layer 2: Dense layer with Softmax
    const float* layer2_weights_;  // Stored as 1D array: [input_size * output_size]
    const float* layer2_bias_;     // [output_size]
    size_t layer2_input_size_;
    size_t layer2_output_size_;

    // Intermediate buffers for layer outputs (max size: 18 for compatibility)
    float layer1_output_[18];  // Output of layer 1 (after ReLU)

public:
    /**
     * Constructor
     * @param l1_weights Layer 1 weights as 1D array [input_size * hidden_size]
     * @param l1_bias Layer 1 bias [hidden_size]
     * @param l1_in Layer 1 input size
     * @param l1_out Layer 1 output size
     * @param l2_weights Layer 2 weights as 1D array [hidden_size * output_size]
     * @param l2_bias Layer 2 bias [output_size]
     * @param l2_in Layer 2 input size
     * @param l2_out Layer 2 output size
     */
    NeuralNetwork(
        const float* l1_weights,
        const float* l1_bias,
        size_t l1_in,
        size_t l1_out,
        const float* l2_weights,
        const float* l2_bias,
        size_t l2_in,
        size_t l2_out
    );

    /**
     * Run inference on input data
     * @param input Input vector [2]
     * @param output Output probabilities [3] (after softmax)
     */
    void predict(const float* input, float* output);

    /**
     * Get the predicted class (argmax of output)
     * @param input Input vector [2]
     * @return Predicted class index (0, 1, or 2)
     */
    int predict_class(const float* input);
};

} // namespace CustomNN

#endif // NEURAL_NETWORK_H
