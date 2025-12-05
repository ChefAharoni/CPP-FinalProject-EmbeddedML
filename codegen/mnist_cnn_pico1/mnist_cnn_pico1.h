// mnist_cnn_pico1.h
// Auto-generated inference model header
// Pure C++ implementation for embedded systems

#ifndef MNIST_CNN_PICO1_MODEL_H
#define MNIST_CNN_PICO1_MODEL_H

#include <cstddef>

namespace embedded_ml {

class mnist_cnn_pico1Model {
public:
    static constexpr size_t kInputSize = 784;
    static constexpr size_t kOutputSize = 10;


private:
    // Intermediate buffers for layer outputs

    static float buffer_16[21632];

    static float buffer_17[18432];

    static float buffer_18[4608];

    static float buffer_19[6400];

    static float buffer_20[4096];

    static float buffer_21[1024];

    static float buffer_22[4];

    static float buffer_23[1];

    static float buffer_24[2];

    static float buffer_25[1024];

    static float buffer_26[256];

    static float buffer_27[10];

    static float buffer_11[1];

    static float buffer_12[1];

    static float buffer_13[1];



public:
    // Run inference on input data
    // input: array of size kInputSize
    // output: array of size kOutputSize (will be filled with results)
    static void Inference(const float* input, float* output);
};

} // namespace embedded_ml

#endif // MNIST_CNN_PICO1_MODEL_H

