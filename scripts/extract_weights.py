#!/usr/bin/env python3
"""
Extract weights and biases from TFLite model and generate C++ header file.
"""

import numpy as np
import tensorflow as tf
import argparse


def extract_weights(model_path, output_path):
    """Extract weights from TFLite model and save as C++ header."""

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get tensor details
    tensor_details = interpreter.get_tensor_details()

    print(f"Model has {len(tensor_details)} tensors")
    print("\nTensor details:")
    for i, detail in enumerate(tensor_details):
        print(f"{i}: {detail['name']} - shape: {detail['shape']}, dtype: {detail['dtype']}")

    # Extract weights and biases
    # For a Sequential model with 2 Dense layers, we expect:
    # - Layer 1: weights [2, 18], bias [18]
    # - Layer 2: weights [18, 3], bias [3]

    weights_found = {}

    for detail in tensor_details:
        name = detail['name']

        # Try to get tensor data - skip if it's an input/output tensor
        try:
            # For constant tensors, we can access them directly via quantization params
            if detail['quantization_parameters']['scales'].size > 0 or 'serving_default' in name or 'StatefulPartitionedCall' in name:
                continue
            tensor = interpreter.tensor(detail['index'])()
        except (ValueError, KeyError):
            continue

        # Dense layer 1 (hidden layer)
        if 'hidden' in name.lower() and 'kernel' in name.lower():
            weights_found['layer1_weights'] = tensor
            print(f"\nFound Layer 1 weights: {tensor.shape}")
        elif 'hidden' in name.lower() and 'bias' in name.lower():
            weights_found['layer1_bias'] = tensor
            print(f"Found Layer 1 bias: {tensor.shape}")

        # Dense layer 2 (output layer)
        elif 'output' in name.lower() and 'kernel' in name.lower():
            weights_found['layer2_weights'] = tensor
            print(f"Found Layer 2 weights: {tensor.shape}")
        elif 'output' in name.lower() and 'bias' in name.lower():
            weights_found['layer2_bias'] = tensor
            print(f"Found Layer 2 bias: {tensor.shape}")

    # If not found by name, try by shape and pattern (fallback)
    if len(weights_found) < 4:
        print("\nWarning: Could not find all weights by name. Trying by shape...")
        # Try to extract all tensors again by shape
        for detail in tensor_details:
            name = detail['name']
            shape = detail['shape']

            # Skip input/output tensors
            if 'serving_default' in name or 'StatefulPartitionedCall' in name:
                continue

            try:
                tensor = interpreter.tensor(detail['index'])()

                # Layer 1: weights [18, 2] (might be transposed) or [2, 18]
                if len(shape) == 2 and (shape[0] == 18 and shape[1] == 2):
                    if 'layer1_weights' not in weights_found:
                        weights_found['layer1_weights'] = tensor.T  # Transpose to [2, 18]
                        print(f"Found Layer 1 weights (transposed): {tensor.shape} → {tensor.T.shape}")
                elif len(shape) == 2 and (shape[0] == 2 and shape[1] == 18):
                    if 'layer1_weights' not in weights_found:
                        weights_found['layer1_weights'] = tensor
                        print(f"Found Layer 1 weights: {tensor.shape}")

                # Layer 1 bias [18]
                elif len(shape) == 1 and shape[0] == 18:
                    if 'layer1_bias' not in weights_found:
                        weights_found['layer1_bias'] = tensor
                        print(f"Found Layer 1 bias: {tensor.shape}")

                # Layer 2: weights [3, 18] (might be transposed) or [18, 3]
                elif len(shape) == 2 and (shape[0] == 3 and shape[1] == 18):
                    if 'layer2_weights' not in weights_found:
                        weights_found['layer2_weights'] = tensor.T  # Transpose to [18, 3]
                        print(f"Found Layer 2 weights (transposed): {tensor.shape} → {tensor.T.shape}")
                elif len(shape) == 2 and (shape[0] == 18 and shape[1] == 3):
                    if 'layer2_weights' not in weights_found:
                        weights_found['layer2_weights'] = tensor
                        print(f"Found Layer 2 weights: {tensor.shape}")

                # Layer 2 bias [3]
                elif len(shape) == 1 and shape[0] == 3:
                    if 'layer2_bias' not in weights_found:
                        weights_found['layer2_bias'] = tensor
                        print(f"Found Layer 2 bias: {tensor.shape}")

            except (ValueError, KeyError, Exception) as e:
                continue

    if len(weights_found) != 4:
        print(f"\nError: Expected 4 weight tensors, found {len(weights_found)}")
        print("Found:", list(weights_found.keys()))
        return

    # Generate C++ header file
    with open(output_path, 'w') as f:
        f.write("// Auto-generated model weights\n")
        f.write("// DO NOT EDIT MANUALLY\n\n")
        f.write("#ifndef MODEL_WEIGHTS_H\n")
        f.write("#define MODEL_WEIGHTS_H\n\n")
        f.write("#include <cstddef>\n\n")

        # Layer 1 weights [2, 18]
        w1 = weights_found['layer1_weights']
        f.write(f"// Layer 1: Dense(18, ReLU) - Weights [{w1.shape[0]}, {w1.shape[1]}]\n")
        f.write(f"constexpr size_t LAYER1_INPUT_SIZE = {w1.shape[0]};\n")
        f.write(f"constexpr size_t LAYER1_OUTPUT_SIZE = {w1.shape[1]};\n")
        f.write(f"constexpr float LAYER1_WEIGHTS[{w1.shape[0]}][{w1.shape[1]}] = {{\n")
        for i in range(w1.shape[0]):
            f.write("    {")
            f.write(", ".join([f"{w1[i, j]:.8f}f" for j in range(w1.shape[1])]))
            f.write("},\n")
        f.write("};\n\n")

        # Layer 1 bias [18]
        b1 = weights_found['layer1_bias']
        f.write(f"// Layer 1 Bias [{b1.shape[0]}]\n")
        f.write(f"constexpr float LAYER1_BIAS[{b1.shape[0]}] = {{\n    ")
        f.write(", ".join([f"{b1[i]:.8f}f" for i in range(b1.shape[0])]))
        f.write("\n};\n\n")

        # Layer 2 weights [18, 3]
        w2 = weights_found['layer2_weights']
        f.write(f"// Layer 2: Dense(3, Softmax) - Weights [{w2.shape[0]}, {w2.shape[1]}]\n")
        f.write(f"constexpr size_t LAYER2_INPUT_SIZE = {w2.shape[0]};\n")
        f.write(f"constexpr size_t LAYER2_OUTPUT_SIZE = {w2.shape[1]};\n")
        f.write(f"constexpr float LAYER2_WEIGHTS[{w2.shape[0]}][{w2.shape[1]}] = {{\n")
        for i in range(w2.shape[0]):
            f.write("    {")
            f.write(", ".join([f"{w2[i, j]:.8f}f" for j in range(w2.shape[1])]))
            f.write("},\n")
        f.write("};\n\n")

        # Layer 2 bias [3]
        b2 = weights_found['layer2_bias']
        f.write(f"// Layer 2 Bias [{b2.shape[0]}]\n")
        f.write(f"constexpr float LAYER2_BIAS[{b2.shape[0]}] = {{\n    ")
        f.write(", ".join([f"{b2[i]:.8f}f" for i in range(b2.shape[0])]))
        f.write("\n};\n\n")

        f.write("#endif // MODEL_WEIGHTS_H\n")

    print(f"\n✓ Weights extracted successfully to {output_path}")
    print(f"  Layer 1: {w1.shape} weights + {b1.shape} bias")
    print(f"  Layer 2: {w2.shape} weights + {b2.shape} bias")


def main():
    parser = argparse.ArgumentParser(description='Extract weights from TFLite model')
    parser.add_argument('--model', type=str, default='scripts/model.tflite',
                        help='Path to TFLite model file')
    parser.add_argument('--output', type=str, default='Miko/model_weights.h',
                        help='Output C++ header file path')

    args = parser.parse_args()
    extract_weights(args.model, args.output)


if __name__ == '__main__':
    main()
