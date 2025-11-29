# Embedded ML Code Generation System

This system generates pure C++ inference code from TFLite models, suitable for embedded systems with no external ML library dependencies.

## Architecture

### Components (`components/`)
Reusable, header-only C++ components for neural network operations:
- **`fully_connected.h`**: Dense/fully connected layer
- **`relu.h`**: ReLU activation function
- **`softmax.h`**: Softmax activation function

All components are template-based, work with float types, and have zero external dependencies (only std library).

### Code Generator (`codegen/`)
A C++ program that:
1. Reads TFLite models using TFLite libraries
2. Extracts weights and model structure
3. Generates three files:
   - `{base_name}_weights.cpp`: Weight arrays
   - `{base_name}.cpp`: Inference model class
   - `{base_name}_inference.cpp`: Example usage code

## Usage

### Step 1: Build the Code Generator

```bash
cd codegen
make
```

This requires TFLite libraries (only for code generation, not for generated code).

### Step 2: Generate Code

```bash
./codegen ../scripts/model.tflite my_model
```

This creates a directory `my_model/` containing:
- `my_model_weights.cpp`
- `my_model.cpp`
- `my_model_inference.cpp`
- `Makefile`

### Step 3: Build and Run

The generated code can be compiled using the included Makefile:

```bash
cd my_model
make
./my_model_inference
```

No TFLite or external ML libraries required! The Makefile handles all compilation.

## Generated Code Structure

### `{base_name}_weights.cpp`
- Contains all model weights as `static const float` arrays
- Organized by tensor with comments showing shapes
- Namespace: `embedded_ml`

### `{base_name}.cpp`
- Contains `{base_name}Model` class
- Public constants: `kInputSize`, `kOutputSize`
- Static method: `Inference(input, output)`
- Includes only necessary components
- Uses intermediate buffers for layer outputs

### `{base_name}_inference.cpp`
- Example "hello world" inference code
- Shows input preparation
- Demonstrates inference call
- Includes commented examples for result processing
- **User-editable** for specific use cases

## Model Support (V1.0)

Currently supports:
- ✅ FULLY_CONNECTED layers
- ✅ RELU activation
- ✅ SOFTMAX activation
- ✅ Single input tensor
- ✅ Single output tensor
- ✅ FLOAT32 weights and tensors

## Key Features

1. **Zero External Dependencies**: Generated code only uses std library and components
2. **Embedded-Friendly**: No dynamic allocation, minimal memory footprint
3. **Static Allocation**: All buffers allocated at compile time
4. **Template-Based**: Components work with different float types
5. **Well-Commented**: Generated code includes helpful comments

## File Organization

```
.
├── components/          # Reusable operation components
│   ├── fully_connected.h
│   ├── relu.h
│   └── softmax.h
├── codegen/            # Code generator (requires TFLite)
│   ├── codegen.cpp
│   ├── Makefile
│   └── README.md
└── {name}/             # Generated directory (created by codegen)
    ├── {name}_weights.cpp
    ├── {name}.cpp
    ├── {name}_inference.cpp
    └── Makefile
```

## Example Workflow

```bash
# 1. Build code generator
cd codegen && make

# 2. Generate code (creates iris_classifier/ directory)
./codegen ../scripts/model.tflite iris_classifier

# 3. Edit inference file (optional)
cd iris_classifier
# Edit iris_classifier_inference.cpp with your input data

# 4. Build and run (using generated Makefile)
make
./iris_classifier_inference
```

## Notes

- The code generator itself requires TFLite to parse models
- Generated code has **zero** TFLite dependencies
- All operations are statically allocated
- Components are header-only for easy integration
- Generated code is optimized for embedded systems

