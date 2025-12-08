// mnist_cnn_pico1_inference.cpp
// Pico-compatible inference code - reads PNG from filesystem and runs inference
// No TensorFlow dependencies, optimized for Raspberry Pi Pico

#include "mnist_cnn_pico1.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pico/stdlib.h"

using namespace embedded_ml;

// Fixed path to PNG file on Pico filesystem
#define PNG_FILE_PATH "/test_image.png"

// Note: Full PNG decoding on Pico is complex and requires zlib/deflate.
// For embedded systems, it's recommended to preprocess PNGs to raw format.
// This function is a placeholder - use LoadRawImage28x28 instead.
static bool LoadPngAsInput28x28(const char* path, float* out, size_t expected_size) {
    (void)path;  // Unused
    (void)out;
    (void)expected_size;
    printf("Error: PNG decoding not implemented for Pico.\n");
    printf("Please use raw binary format (.raw file) instead.\n");
    printf("To convert: convert input.png -depth 8 -colorspace Gray test_image.raw\n");
    return false;
}

// Alternative: Read raw binary image data (28x28 bytes, grayscale)
// This is simpler and more efficient for embedded systems
static bool LoadRawImage28x28(const char* path, float* out, size_t expected_size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        printf("Error: Failed to open image file: %s\n", path);
        return false;
    }

    // Read 28x28 = 784 bytes of raw grayscale data
    unsigned char* image_data = (unsigned char*)malloc(784);
    if (!image_data) {
        fclose(fp);
        printf("Error: Failed to allocate memory for image\n");
        return false;
    }

    size_t bytes_read = fread(image_data, 1, 784, fp);
    fclose(fp);

    if (bytes_read != 784) {
        free(image_data);
        printf("Error: Expected 784 bytes, got %zu\n", bytes_read);
        return false;
    }

    // Normalize to [0,1] and fill output
    for (size_t i = 0; i < expected_size; ++i) {
        out[i] = static_cast<float>(image_data[i]) / 255.0f;
    }

    free(image_data);
    return true;
}

int main() {
    // Initialize Pico SDK
    stdio_init_all();
    
    // Wait for serial connection (optional, but helpful for debugging)
    sleep_ms(2000);
    
    printf("\n");
    printf("========================================\n");
    printf("MNIST Inference on Raspberry Pi Pico\n");
    printf("========================================\n");
    printf("\n");

    // Allocate input buffer dynamically (not on stack to avoid stack overflow)
    float* input = (float*)malloc(mnist_cnn_pico1Model::kInputSize * sizeof(float));
    if (!input) {
        printf("Error: Failed to allocate input buffer\n");
        return 1;
    }

    // Allocate output buffer
    float* output = (float*)malloc(mnist_cnn_pico1Model::kOutputSize * sizeof(float));
    if (!output) {
        printf("Error: Failed to allocate output buffer\n");
        free(input);
        return 1;
    }

    // Try to load raw image first (simpler and more efficient)
    const char* image_path = "/test_image.raw";  // Raw binary format
    bool loaded = LoadRawImage28x28(image_path, input, mnist_cnn_pico1Model::kInputSize);
    
    // If raw format not found, try PNG (with limited support)
    if (!loaded) {
        printf("Raw image not found, trying PNG...\n");
        loaded = LoadPngAsInput28x28(PNG_FILE_PATH, input, mnist_cnn_pico1Model::kInputSize);
    }

    if (!loaded) {
        printf("\nError: Could not load image file.\n");
        printf("Please ensure one of these files exists on the Pico filesystem:\n");
        printf("  - /test_image.raw (784 bytes, raw grayscale)\n");
        printf("  - /test_image.png (28x28 grayscale PNG)\n");
        printf("\nTo create a raw file from PNG:\n");
        printf("  convert input.png -depth 8 -colorspace Gray test_image.raw\n");
        free(input);
        free(output);
        return 1;
    }

    printf("Image loaded successfully!\n");
    printf("Running inference...\n");

    // Run inference
    mnist_cnn_pico1Model::Inference(input, output);

    // Find predicted class
    size_t predicted_class = 0;
    float max_prob = output[0];
    for (size_t i = 1; i < mnist_cnn_pico1Model::kOutputSize; ++i) {
        if (output[i] > max_prob) {
            max_prob = output[i];
            predicted_class = i;
        }
    }

    // Print results
    printf("\n=== Inference Results ===\n");
    printf("Predicted digit: %zu (confidence: %.2f%%)\n", predicted_class, max_prob * 100.0f);
    printf("\nAll class probabilities:\n");
    for (size_t i = 0; i < mnist_cnn_pico1Model::kOutputSize; ++i) {
        printf("  Class %zu: %.4f", i, output[i]);
        if (i == predicted_class) {
            printf(" <-- PREDICTED");
        }
        printf("\n");
    }

    // Cleanup
    free(input);
    free(output);

    printf("\nInference complete!\n");
    return 0;
}
