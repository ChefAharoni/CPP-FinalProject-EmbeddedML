# Miko - A light-weight system for ML on embedded systems.

Using Miko, you can quickly, robustly and easily generate near-zero-overhead C++ code for ML on embedded systems, all while being faster and smaller than leading competitors.

## Installation

- Make sure your system has a functioning C++ build pipeline installed and $CXX set.
- Run `make`
- Done!
- Note that if you want to move ./miko_codegen elsewhere on your system you are free to, as long as you remember to also move /templates and /components to that same directory, or alternatively specify --template-path and --component-path

## Overview

The code generator creates a directory `{base_name}/` containing:

1. **`{base_name}_weights.cpp`**: Contains all model weights as C++ arrays
2. **`{base_name}.cpp`**: Contains the inference model class that uses the weights and components
3. **`{base_name}_inference.cpp`**: Example inference code that demonstrates usage
4. **`Makefile`**: (optional) Build file for easy compilation

## Usage

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

Note that this example only covers the simplest use case - if you want to learn more, either run ./miko_codegen -h or consult the manual.

### Using Generated Code

The generated code can be compiled using the included Makefile:

```bash
cd my_model
make
./my_model_inference
```

If you choose to run your code in a different build system, as you are likely to for embedded systems, follow the steps specific to your build system instead.

The generated code has no TFLite dependencies. You only need:
- Standard C++ library (also tested on the subset of std available on the Raspberry Pi Pico)

## Running on embedded

Since we tested in VS Code on a Raspberry Pi pico, these steps will be specific to that setup. However, they should be similar for other systems as well.

- Install the official Raspberry Pi Pico VSCode extension
- Navigate to the "Raspberry Pi Pico Project" screen, create new C/C++ project. Make sure to select your board (for me, the regular Pico), and tick the box to use C++ code. Also, I recommend checking the box for Serial over USB.
- Use Miko:
```bash
./miko_codegen models/mnist_small_model.tflite /path/to/your/project --replace --no-makefile --inf=pico-img-bench
```
- If compilation does not work at this point, try finding `add_executable(ProjectName ProjectName.cpp )` and changing ProjectName.cpp to the actual name of your generated inference code.
- Compile and flash to your device

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

- The code generator requires (some) TFLite to parse models, but the generated code has zero external ML dependencies
- Generated code is optimized for embedded systems (no dynamic allocation, minimal dependencies)
- Intermediate buffers are statically allocated within the model class
- All operations are in-place or use pre-allocated buffers

## Sources

MNIST Digit recognition TFLite model by Mirosław Stanek, via https://github.com/frogermcs/MNIST-TFLite
MNIST Dataset by Yann LeCunn https://www.kaggle.com/datasets/alexanderyyy/mnist-png
Simple MNIST Convolutional Neural Network trained by Kunal Jain https://github.com/kj7kunal/MNIST-Keras/tree/master
Speech Command Recognition model by Coimbra de Andrade et al https://arxiv.org/pdf/1808.08929 https://github.com/douglas125/SpeechCmdRecognition

