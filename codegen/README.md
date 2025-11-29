# Code Generator for Embedded ML Models

This code generator reads TFLite models and generates pure C++ code suitable for embedded systems with no external ML library dependencies.

## Overview

The code generator creates a directory `{base_name}/` containing:

1. **`{base_name}_weights.cpp`**: Contains all model weights as C++ arrays
2. **`{base_name}.cpp`**: Contains the inference model class that uses the weights and components
3. **`{base_name}_inference.cpp`**: Example inference code that demonstrates usage
4. **`Makefile`**: Build file for easy compilation

## Components

The generated code uses reusable components located in `../../components/` (relative to the generated directory):
- `fully_connected.h`: Dense/fully connected layer implementation
- `relu.h`: ReLU activation function
- `softmax.h`: Softmax activation function

## Usage

### Building the Code Generator

The code generator itself requires TFLite libraries to read the model:

```bash
cd codegen
# Compile using your existing build system
# The generator uses TFLite to parse models, but generated code does not
```

### Generating Code

```bash
./codegen <path_to_model.tflite> <base_name>
```

Example:
```bash
./codegen ../scripts/model.tflite my_model
```

This creates a directory `my_model/` containing:
- `my_model_weights.cpp`
- `my_model.cpp`
- `my_model_inference.cpp`
- `Makefile`

### Using Generated Code

The generated code can be compiled using the included Makefile:

```bash
cd my_model
make
./my_model_inference
```

The generated code has no TFLite dependencies. You only need:
- Standard C++ library
- The component headers in `../../components/` (relative to generated directory)

## Model Support

**V1.0** supports models with:
- FULLY_CONNECTED layers
- RELU activation
- SOFTMAX activation
- Single input and single output
- FLOAT32 weights and tensors

## Generated Code Structure

### Weights File
Contains static const arrays with all model weights, organized by tensor.

### Model File
Contains a class `{base_name}Model` with:
- `kInputSize`: Input tensor size
- `kOutputSize`: Output tensor size
- `Inference(input, output)`: Static method to run inference

### Inference File
Example code showing:
- How to prepare input data
- How to call the inference method
- How to process output results

Edit this file to match your specific inference needs.

### Makefile
Automatically generated build file that:
- Compiles all source files
- Links them into an executable named `{base_name}_inference`
- Includes a `clean` target for removing build artifacts

## Notes

- The code generator requires TFLite to parse models, but the generated code has zero external ML dependencies
- Generated code is optimized for embedded systems (no dynamic allocation, minimal dependencies)
- Intermediate buffers are statically allocated within the model class
- All operations are in-place or use pre-allocated buffers

