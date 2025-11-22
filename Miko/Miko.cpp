/**
 * Miko - Thermal Anomaly Detector
 * Uses custom C++ neural network to detect finger presence via temperature
 *
 * Model: Thermal Anomaly Classification
 * Architecture: Input(10) -> Dense(8, ReLU) -> Dense(2, Softmax)
 * Classes: 0=Normal, 1=Touched/Heating Event
 */

#include <stdio.h>
#include "pico/stdlib.h"
#include "neural_network.h"
#include "temp_model_weights.h"
#include "temp_sensor.h"

using namespace CustomNN;

// Configuration
#define DATA_COLLECTION_MODE true  // Set to true to collect training data
#define WINDOW_SIZE 10              // Number of temperature readings in sliding window
#define SAMPLE_INTERVAL_MS 100      // Time between temperature readings
#define DETECTION_THRESHOLD 0.7f    // Confidence threshold for "Touched" detection

// Global neural network instance
NeuralNetwork* model = nullptr;

// Sliding window buffer for temperature readings
float temp_window[WINDOW_SIZE] = {0};

void setup_model() {
    printf("Initializing Thermal Anomaly Detection Model...\n");

    // Create neural network with temperature model weights
    // Cast 2D arrays to 1D pointers for compatibility
    model = new NeuralNetwork(
        &TEMP_LAYER1_WEIGHTS[0][0],  // Layer 1 weights [10][8] -> 1D array
        TEMP_LAYER1_BIAS,             // Layer 1 bias [8]
        TEMP_LAYER1_INPUT_SIZE,       // 10
        TEMP_LAYER1_OUTPUT_SIZE,      // 8
        &TEMP_LAYER2_WEIGHTS[0][0],  // Layer 2 weights [8][2] -> 1D array
        TEMP_LAYER2_BIAS,             // Layer 2 bias [2]
        TEMP_LAYER2_INPUT_SIZE,       // 8
        TEMP_LAYER2_OUTPUT_SIZE       // 2
    );

    printf("✓ Model initialized!\n");
    printf("  Input: %zu temperature readings (sliding window)\n", TEMP_LAYER1_INPUT_SIZE);
    printf("  Hidden layer: %zu neurons (ReLU)\n", TEMP_LAYER1_OUTPUT_SIZE);
    printf("  Output: %zu classes (Normal, Touched)\n", TEMP_LAYER2_OUTPUT_SIZE);
    printf("  Sample interval: %d ms\n", SAMPLE_INTERVAL_MS);
    sleep_ms(500);
}

void add_temperature_to_window(float new_temp) {
    // Shift window left (discard oldest reading)
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        temp_window[i] = temp_window[i + 1];
    }
    // Add new reading at the end
    temp_window[WINDOW_SIZE - 1] = new_temp;
}

void run_inference() {
    if (!model) {
        printf("Error: Model not initialized!\n");
        return;
    }

    // Prepare output buffer
    float output[2];

    // Run inference on the temperature window
    model->predict(temp_window, output);

    // Extract probabilities
    float normal_prob = output[0];
    float touched_prob = output[1];

    // Determine if finger/heat detected
    bool detected = (touched_prob > DETECTION_THRESHOLD);

    // Print results
    printf("Temp: %.2f°C | Normal: %.2f | Touched: %.2f | %s\n",
           temp_window[WINDOW_SIZE - 1],
           normal_prob,
           touched_prob,
           detected ? "🔥 DETECTED!" : "Normal");

    // Control LED based on detection
    gpio_put(PICO_DEFAULT_LED_PIN, detected ? 1 : 0);
}

void data_collection_mode() {
    printf("\n========================================\n");
    printf("DATA COLLECTION MODE\n");
    printf("========================================\n");
    printf("Instructions:\n");
    printf("1. Let the Pico sit idle for 1 minute (Normal data)\n");
    printf("2. Touch the RP2040 chip for 10 seconds\n");
    printf("3. Wait 20 seconds\n");
    printf("4. Repeat touch 3-5 times\n");
    printf("5. Copy this log to train your model\n");
    printf("========================================\n\n");

    // CRITICAL: Wait for serial connection to be ready
    // This prevents USB buffer overflow and printf() blocking
    printf("Waiting for serial connection...\n");
    printf("Please start the data collection script now!\n");
    printf("Starting in: ");
    for (int i = 5; i > 0; i--) {
        printf("%d... ", i);
        gpio_put(PICO_DEFAULT_LED_PIN, 1);  // LED on
        sleep_ms(500);
        gpio_put(PICO_DEFAULT_LED_PIN, 0);  // LED off
        sleep_ms(500);
    }
    printf("GO!\n\n");

    printf("temperature\n");  // CSV header

    uint32_t sample_count = 0;
    while (true) {
        float temp = read_temperature();
        printf("%.2f\n", temp);

        // Blink LED to show it's alive
        if (sample_count % 10 == 0) {
            gpio_put(PICO_DEFAULT_LED_PIN, !gpio_get(PICO_DEFAULT_LED_PIN));
        }

        sample_count++;
        sleep_ms(SAMPLE_INTERVAL_MS);
    }
}

int main() {
    stdio_init_all();

    // Initialize LED
    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    gpio_put(LED_PIN, 0);

    // Initialize temperature sensor
    init_temp_sensor();

    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  MIKO - Thermal Anomaly Detector        ║\n");
    printf("║  Finger Presence Detection via ML       ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("\n");

    // Check if we're in data collection mode
    if (DATA_COLLECTION_MODE) {
        data_collection_mode();
        return 0;  // Never reached
    }

    // Normal inference mode
    setup_model();

    printf("\n");
    printf("Warming up temperature sensor...\n");
    printf("Filling initial window with readings...\n");

    // Fill the window with initial readings
    for (int i = 0; i < WINDOW_SIZE; i++) {
        float temp = read_temperature();
        add_temperature_to_window(temp);
        printf("  [%d/%d] %.2f°C\n", i + 1, WINDOW_SIZE, temp);
        sleep_ms(SAMPLE_INTERVAL_MS);
    }

    printf("\n✓ Ready! Monitoring for thermal anomalies...\n");
    printf("(Touch the RP2040 chip to trigger detection)\n\n");

    // Main inference loop
    uint32_t sample_count = 0;
    while (true) {
        // Read new temperature
        float temp = read_temperature();

        // Add to sliding window
        add_temperature_to_window(temp);

        // Run inference every reading
        if (sample_count % 1 == 0) {  // Can adjust to run inference less frequently
            run_inference();
        }

        sample_count++;
        sleep_ms(SAMPLE_INTERVAL_MS);
    }

    return 0;
}
