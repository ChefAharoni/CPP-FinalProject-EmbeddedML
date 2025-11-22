// Temperature Anomaly Detection Model Weights
// Architecture: Input(10) -> Dense(8, ReLU) -> Dense(2, Softmax)
//
// NOTE: These are PLACEHOLDER weights for initial testing.
// To get a working model, you need to:
// 1. Collect temperature data (normal vs touched)
// 2. Train a model using the training script
// 3. Replace these weights with trained weights

#ifndef TEMP_MODEL_WEIGHTS_H
#define TEMP_MODEL_WEIGHTS_H

#include <cstddef>

// Layer 1: Dense(8, ReLU) - Input size: 10, Output size: 8
constexpr size_t TEMP_LAYER1_INPUT_SIZE = 10;
constexpr size_t TEMP_LAYER1_OUTPUT_SIZE = 8;

// Placeholder weights - randomly initialized
// In a real model, these would be learned from data
constexpr float TEMP_LAYER1_WEIGHTS[10][8] = {
    {0.15f, -0.22f, 0.18f, -0.11f, 0.25f, -0.19f, 0.14f, -0.23f},
    {-0.17f, 0.21f, -0.13f, 0.24f, -0.16f, 0.20f, -0.12f, 0.19f},
    {0.19f, -0.15f, 0.22f, -0.18f, 0.14f, -0.21f, 0.17f, -0.13f},
    {-0.16f, 0.23f, -0.14f, 0.19f, -0.20f, 0.15f, -0.18f, 0.22f},
    {0.21f, -0.18f, 0.16f, -0.22f, 0.13f, -0.17f, 0.20f, -0.15f},
    {-0.14f, 0.19f, -0.21f, 0.16f, -0.13f, 0.24f, -0.17f, 0.18f},
    {0.17f, -0.20f, 0.14f, -0.19f, 0.22f, -0.16f, 0.13f, -0.21f},
    {-0.19f, 0.16f, -0.18f, 0.21f, -0.15f, 0.14f, -0.22f, 0.17f},
    {0.22f, -0.14f, 0.19f, -0.17f, 0.16f, -0.20f, 0.15f, -0.18f},
    {-0.15f, 0.18f, -0.16f, 0.20f, -0.19f, 0.13f, -0.17f, 0.21f}
};

constexpr float TEMP_LAYER1_BIAS[8] = {
    0.05f, -0.03f, 0.07f, -0.04f, 0.06f, -0.02f, 0.04f, -0.05f
};

// Layer 2: Dense(2, Softmax) - Input size: 8, Output size: 2
constexpr size_t TEMP_LAYER2_INPUT_SIZE = 8;
constexpr size_t TEMP_LAYER2_OUTPUT_SIZE = 2;

// Placeholder weights for binary classification (Normal vs Touched)
constexpr float TEMP_LAYER2_WEIGHTS[8][2] = {
    {0.35f, -0.35f},
    {-0.42f, 0.42f},
    {0.38f, -0.38f},
    {-0.40f, 0.40f},
    {0.37f, -0.37f},
    {-0.39f, 0.39f},
    {0.41f, -0.41f},
    {-0.36f, 0.36f}
};

constexpr float TEMP_LAYER2_BIAS[2] = {
    0.10f, -0.10f
};

#endif // TEMP_MODEL_WEIGHTS_H
