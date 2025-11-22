/**
 * Miko - Custom Neural Network Inference Demo
 * Uses custom C++ implementation instead of TensorFlow Lite
 *
 * Model: Gaussian Blob Classification
 * Architecture: Input(2) -> Dense(18, ReLU) -> Dense(3, Softmax)
 */

#include <stdio.h>
#include "pico/stdlib.h"
#include "neural_network.h"
#include "model_weights.h"

using namespace CustomNN;

// Global neural network instance
NeuralNetwork* model = nullptr;

void setup_model() {
    printf("Initializing custom neural network...\n");

    // Create neural network with extracted weights
    model = new NeuralNetwork(
        LAYER1_WEIGHTS,  // Layer 1 weights [2][18]
        LAYER1_BIAS,     // Layer 1 bias [18]
        LAYER1_INPUT_SIZE,   // 2
        LAYER1_OUTPUT_SIZE,  // 18
        LAYER2_WEIGHTS,  // Layer 2 weights [18][3]
        LAYER2_BIAS,     // Layer 2 bias [3]
        LAYER2_INPUT_SIZE,   // 18
        LAYER2_OUTPUT_SIZE   // 3
    );

    printf("✓ Custom neural network initialized!\n");
    printf("  Input size: %zu\n", LAYER1_INPUT_SIZE);
    printf("  Hidden layer: %zu neurons (ReLU)\n", LAYER1_OUTPUT_SIZE);
    printf("  Output size: %zu classes (Softmax)\n", LAYER2_OUTPUT_SIZE);
    sleep_ms(500);
    printf("This is the custom Miko, zero dependent code!");
    sleep_ms(500);
}

void run_inference(float x, float y) {
    if (!model) {
        printf("Error: Model not initialized!\n");
        return;
    }

    // Prepare input
    float input[2] = {x, y};
    float output[3];

    // Run inference
    model->predict(input, output);

    // Find predicted class
    int predicted_class = 0;
    float max_prob = output[0];

    for (int i = 1; i < 3; ++i) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted_class = i;
        }
    }

    // Print results
    printf("\n--- Custom NN Inference ---\n");
    printf("Input: (%.2f, %.2f)\n", x, y);
    printf("Class 0 probability: %.4f\n", output[0]);
    printf("Class 1 probability: %.4f\n", output[1]);
    printf("Class 2 probability: %.4f\n", output[2]);
    printf("Predicted class: %d (confidence: %.4f)\n", predicted_class, max_prob);
    printf("---------------------------\n");
}

int main() {
    stdio_init_all();

    // Initialize LED
    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    printf("\n=== Miko Custom NN Demo ===\n");
    printf("Model: Gaussian Blob Classifier\n");
    printf("Implementation: Custom C++ (no TFLite)\n\n");

    // Setup the custom neural network
    setup_model();

    // Example test inputs
    // These are sample coordinates that the model can classify
    float test_inputs[][2] = {
        {0.5f, 0.5f},
        {-2.0f, 3.0f},
        {4.0f, -1.0f},
        {1.0f, 1.0f},
        {-1.0f, -1.0f}
    };

    int test_idx = 0;
    while (true) {
        printf("=========================================================\n");
        printf("Welcome to the Miko Microcontroller!\n");
        printf("=========================================================\n");
        sleep_ms(500);
        printf("\n\nRunning demo tests for inferences: \n\n");
        // Blink LED to show activity
        gpio_put(LED_PIN, 1);

        // Run inference on current test input
        printf("\nTest %d:\n", test_idx + 1);
        run_inference(test_inputs[test_idx][0], test_inputs[test_idx][1]);

        sleep_ms(2000);
        gpio_put(LED_PIN, 0);
        sleep_ms(1000);

        // Cycle through test inputs
        test_idx = (test_idx + 1) % 5;
    }

    return 0;
}
