#include <stdio.h>
#include "pico/stdlib.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

namespace {
    // TFLite globals
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;

    // Tensor arena for allocations
    // Adjust this size based on your model requirements
    constexpr int kTensorArenaSize = 10 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];
}

void setup_model() {
    // Load the model
    model = tflite::GetModel(scripts_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema version %d doesn't match supported version %d!\n",
               model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    printf("Model loaded successfully!\n");

    // Create the operator resolver
    // Add only the operations your model needs to reduce memory usage
    static tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    // Build the interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Allocate memory for the model's tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("AllocateTensors() failed!\n");
        return;
    }
    printf("Tensors allocated successfully!\n");

    // Get pointers to the model's input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Print input/output info
    printf("Input shape: [%d, %d]\n", input->dims->data[0], input->dims->data[1]);
    printf("Output shape: [%d, %d]\n", output->dims->data[0], output->dims->data[1]);
}

void run_inference(float x, float y) {
    if (!interpreter || !input || !output) {
        printf("Model not initialized!\n");
        return;
    }

    // Set input values
    input->data.f[0] = x;
    input->data.f[1] = y;

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        printf("Invoke failed!\n");
        return;
    }

    // Read output
    float class_0_prob = output->data.f[0];
    float class_1_prob = output->data.f[1];
    float class_2_prob = output->data.f[2];

    // Find predicted class
    int predicted_class = 0;
    float max_prob = class_0_prob;

    if (class_1_prob > max_prob) {
        predicted_class = 1;
        max_prob = class_1_prob;
    }
    if (class_2_prob > max_prob) {
        predicted_class = 2;
        max_prob = class_2_prob;
    }

    // Print results
    printf("\n--- Inference Results ---\n");
    printf("Input: (%.2f, %.2f)\n", x, y);
    printf("Class 0 probability: %.4f\n", class_0_prob);
    printf("Class 1 probability: %.4f\n", class_1_prob);
    printf("Class 2 probability: %.4f\n", class_2_prob);
    printf("Predicted class: %d (confidence: %.4f)\n", predicted_class, max_prob);
    printf("------------------------\n");
}

int main() {
    stdio_init_all();
    tflite::InitializeTarget();

    // Initialize LED
    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    printf("\n=== Miko TFLite Demo ===\n");
    printf("Model size: %u bytes\n", scripts_model_tflite_len);

    // Setup the TFLite model
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
