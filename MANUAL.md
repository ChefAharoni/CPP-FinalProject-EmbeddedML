# Technical Manual: Embedded ML Code Generator

## Table of Contents

1. [Introduction][1]
2. [Installation and Building][2]
3. [Command-Line Interface Reference][3]
4. [Component Library Reference][4]
5. [Supported Operations and Layers][5]
6. [Generated Code Structure][6]
7. [Usage Examples][7]
8. [Technical Specifications][8]
9. [Limitations and Constraints][9]
10. [Troubleshooting][10]
11. [Advanced Topics][11]
12. [Appendix][12]

---

## 1. Introduction

### 1.1 Purpose and Goals

Miko is a tool that transforms TensorFlow Lite (TFLite) models into pure, dependency-free C++ inference code suitable for embedded systems. The generated code is optimized for low-power, resource-constrained devices such as the Raspberry Pi Pico.

### 1.2 Target Platforms

- **Embedded Systems**: Microcontrollers, IoT devices
- **Raspberry Pi Pico**: Specifically supported with `--inf=pico-img-bench` flag
- **Any C++17-compatible platform**: Desktop, server, or embedded

### 1.3 Key Features

- **Zero External Dependencies**: Generated code uses only the C++ standard library and custom header-only components
- **Static Allocation**: All memory allocated at compile time, no dynamic allocation
- **Header-Only Components**: Reusable neural network operation components
- **Template-Based**: Type-flexible components using C++ templates
- **Portable**: Self-contained generated code with components copied to output directory
- **Optimized**: Loop unrolling, restrict pointers, fused operations

### 1.4 Architecture Overview

The system consists of:

1. **Code Generator** (`src/`): C++ program that parses TFLite models and generates code
2. **Component Library** (`components/`): Header-only neural network operation implementations
3. **Templates** (`templates/`): Inja template files defining code structure
4. **Generated Code**: Self-contained inference code with all necessary components

### 1.5 Prerequisites and System Requirements

**Note**: The code repository includes the required `/include` directory with all necessary headers (TensorFlow Lite schema headers, Inja templating library, etc.). No additional setup or dependency installation is needed beyond downloading the code.

**Requirements**:
- C++17 compatible compiler (g++, clang++, etc.)
- Standard build tools (make)
- TFLite model file (`.tflite` format)

**For Code Generator Build**:
- All dependencies are included in `/include` directory
- No external package installation required

**For Generated Code**:
- C++17 compiler
- Standard C++ library only
- No external ML libraries required

---

## 2. Installation and Building

### 2.1 Building the Code Generator

The code generator is located in the `src/` directory. To build:

```bash
cd /path/to/CPP-FinalProject-EmbeddedML
make
```

This will create the `miko_codegen` executable (or similar name depending on your Makefile configuration).

**Dependencies**: All required headers are included in the `/include` directory:
- TensorFlow Lite schema headers (`tensorflow/lite/schema/schema_generated.h`)
- Inja templating library (`inja/`)
- JSON library (`nlohmann/json.hpp`)

No external package installation is required.

### 2.2 Verifying Installation

After building, verify the installation:

```bash
./miko_codegen -h
```

This should display a help message. If you see the help text, the installation is successful.

### 2.3 Directory Structure

The repository structure:

```
.
├── src/                    # Code generator source files
│   ├── codegen.cpp        # Main entry point
│   ├── code_generator.cpp # Code generation logic
│   ├── model_utils.cpp    # Model parsing utilities
│   ├── model_validator.cpp # Model validation
│   └── filesystem_utils.cpp # File operations
├── components/            # Neural network component library
│   ├── fully_connected.h
│   ├── conv_2d.h
│   ├── max_pool_2d.h
│   ├── relu.h
│   ├── softmax.h
│   └── ... (other components)
├── templates/            # Code generation templates
│   ├── model.h.inja
│   ├── model.cpp.inja
│   ├── weights.cpp.inja
│   ├── inference.cpp.inja
│   ├── inference_pico.cpp.inja
│   └── Makefile.inja
├── include/              # Included dependencies
│   ├── ...              # Dependencies for the generator
└── Makefile             # Build configuration
```

---

## 3. Command-Line Interface Reference

### 3.1 Basic Usage

```bash
./miko_codegen <path_to_model.tflite> <base_name> [OPTIONS]
```

### 3.2 Required Arguments

#### `<path_to_model.tflite>`
- **Description**: Path to the input TFLite model file
- **Format**: Must be a valid TFLite FlatBuffer file (`.tflite` extension)
- **Validation**: File must exist and be readable
- **Example**: `models/mnist_cnn.tflite`

#### `<base_name>`
- **Description**: Base name for generated files and output directory
- **Format**: Can be a simple name or a path (e.g., `my_model` or `codegen/my_model`)
- **Behavior**: 
  - If path is provided (e.g., `codegen/testnew84`), creates that directory structure
  - File base name is extracted from the last path component
  - Example: `codegen/testnew84` → directory: `codegen/testnew84/`, files: `testnew84_weights.cpp`, etc.
- **Example**: `my_model` or `output/my_model`

### 3.3 Optional Flags

#### `--output-index=N`
- **Description**: Select output tensor index for multi-output models
- **Default**: `0`
- **When to Use**: Models with multiple output tensors
- **Values**: Non-negative integer (0-based index)
- **Validation**: Must be less than the number of outputs in the model
- **Error Handling**: Fails with clear error message if index is invalid
- **Example**: `--output-index=1`

**Note**: If a model has multiple outputs and this flag is not specified, the generator will prompt you to use it.

#### `--template-path=PATH`
- **Description**: Specify custom templates directory
- **Default**: `./templates`
- **Validation**: 
  - Directory must exist
  - Must be a directory (not a file)
  - Must contain all required template files:
	- `weights.cpp.inja`
	- `model.h.inja`
	- `model.cpp.inja`
	- `inference.cpp.inja`
	- `Makefile.inja`
- **Error Handling**: 
  - If not found and flag not used: Suggests using the flag
  - Provides example command with flag
- **Example**: `--template-path=/custom/path/to/templates`

#### `--component-path=PATH`
- **Description**: Specify custom components directory
- **Default**: `./components`
- **Validation**:
  - Directory must exist
  - Must be a directory (not a file)
- **Error Handling**:
  - If not found and flag not used: Suggests using the flag
  - Provides example command with flag
- **Example**: `--component-path=/custom/path/to/components`

#### `--inf=TYPE`
- **Description**: Control inference script generation type
- **Default**: `standard`
- **Options**:
  - `none`: Skip inference file generation entirely
  - `standard`: Generate standard C++ inference file (uses `iostream`, suitable for desktop/server)
  - `pico-img-bench`: Generate Raspberry Pi Pico benchmark code (uses `printf`, Pico SDK, suitable for embedded)
- **Validation**: 
  - Enum-based validation (case-insensitive)
  - Fails with error message if invalid value provided
  - Valid values: `none`, `standard`, `pico-img-bench`
- **Error Handling**: Returns error with list of valid options if invalid value provided
- **Examples**: 
  - `--inf=none`
  - `--inf=standard`
  - `--inf=pico-img-bench`

#### `--no-makefile`
- **Description**: Skip Makefile generation
- **Default**: Makefile is generated
- **When to Use**: When integrating into existing build systems (CMake, custom Makefiles, etc.)
- **Example**: `--no-makefile`

#### `--replace`
- **Description**: Allow overwriting files in existing output directory
- **Default**: Fails if output directory exists
- **Behavior**: 
  - Does NOT delete the directory
  - Only overwrites existing files
  - Directory must exist (fails if it doesn't)
- **Error Handling**: 
  - If directory doesn't exist: Fails with error message
  - If directory exists but flag not used: Fails with suggestion to use `--replace`
- **Example**: `--replace`

#### `-h, --help`
- **Description**: Display detailed help message
- **Behavior**: Prints comprehensive help and exits
- **Example**: `./miko_codegen -h` or `./miko_codegen --help`

### 3.4 Short Usage Message

When incorrect arguments are provided (without `-h`), a short usage message is displayed:

```
Usage: ./miko_codegen <path_to_model.tflite> <base_name> [OPTIONS]
Use -h or --help for detailed help
```

### 3.5 Exit Codes

- `0`: Success
- `1`: Error (invalid arguments, model validation failure, code generation failure, etc.)

### 3.6 Error Messages

The code generator provides clear, actionable error messages:

- **Template/Component Not Found**: Suggests using `--template-path` or `--component-path` flags
- **Unsupported Operations**: Lists supported operations and indicates which operations in the model are unsupported
- **Invalid Model Structure**: Describes what's wrong with the model
- **Directory Already Exists**: Suggests using `--replace` flag
- **Invalid Flag Values**: Lists valid options

---

## 4. Component Library Reference

All components are located in the `components/` directory and are header-only implementations in the `embedded_ml` namespace. Components are template-based, allowing different numeric types (typically `float`).

### 4.1 Common Characteristics

All components share these characteristics:

- **Namespace**: `embedded_ml`
- **Template Parameter**: `<T>` (typically `float`)
- **Memory Model**: Pointer-based with `__restrict` annotations for compiler optimization
- **Thread Safety**: Not thread-safe (designed for single-threaded execution)
- **Dependencies**: Only C++ standard library
- **Header-Only**: No separate implementation files

### 4.2 fully\_connected.h

**Purpose**: Dense/fully connected layer (matrix multiplication with bias and optional activation)

**Function Signature**:
```cpp
template<typename T>
void FullyConnected(
    const T* __restrict input,        // Input vector [input_size]
    const T* __restrict weights,      // Weight matrix [output_size × input_size] (row-major)
    const T* __restrict bias,         // Bias vector [output_size]
    T* __restrict output,             // Output vector [output_size]
    size_t input_size,                // Size of input vector
    size_t output_size,               // Size of output vector
    ActivationType activation = ActivationType::NONE  // Activation function
);
```

**Parameters**:
- `input`: Input feature vector of size `input_size`
- `weights`: Weight matrix stored in row-major format. For output neuron `i`, weights are at `weights + i * input_size`
- `bias`: Bias values, one per output neuron
- `output`: Pre-allocated output array of size `output_size`
- `input_size`: Number of input features
- `output_size`: Number of output neurons
- `activation`: Activation function to apply (`ActivationType::NONE` or `ActivationType::RELU`)

**Mathematical Operation**:
```
output[i] = activation(bias[i] + sum(input[j] * weights[i][j] for j in [0, input_size)))
```

**Optimizations**:
- Loop unrolling: Processes 4 elements at a time in inner loop
- Fused activation: Activation applied inline to avoid extra memory pass
- Restrict pointers: Enables compiler optimizations (vectorization, etc.)

**Activation Types**:
- `ActivationType::NONE`: No activation (linear)
- `ActivationType::RELU`: ReLU activation: `max(0, x)`

**Example Usage**:
```cpp
float input[784];
float weights[784 * 128];  // 128 output neurons, 784 input features
float bias[128];
float output[128];

embedded_ml::FullyConnected(input, weights, bias, output, 784, 128, 
                             embedded_ml::ActivationType::RELU);
```

### 4.3 conv\_2d.h

**Purpose**: 2D convolution layer with padding, stride, dilation, and fused activation

**Function Signature**:
```cpp
template<typename T>
void Conv2D(
    const T* input,                   // Input tensor [batch, height, width, channels]
    const T* filter,                  // Filter tensor [output_channels, filter_h, filter_w, input_channels]
    const T* bias,                    // Bias vector [output_channels] (can be nullptr)
    T* output,                        // Output tensor [batch, out_h, out_w, output_channels]
    size_t batch_size,                // Number of batches
    size_t input_height,              // Input height
    size_t input_width,               // Input width
    size_t input_channels,            // Number of input channels
    size_t filter_height,             // Filter height
    size_t filter_width,              // Filter width
    size_t output_channels,           // Number of output channels
    size_t stride_height,             // Vertical stride
    size_t stride_width,              // Horizontal stride
    PaddingType padding,              // Padding type (VALID or SAME)
    ActivationType activation = ActivationType::NONE,  // Activation function
    size_t dilation_height = 1,       // Vertical dilation factor
    size_t dilation_width = 1         // Horizontal dilation factor
);
```

**Parameters**:
- `input`: Input tensor in NHWC format (batch, height, width, channels)
- `filter`: Filter tensor shape `[output_channels, filter_height, filter_width, input_channels]`
- `bias`: Bias vector of size `output_channels` (can be `nullptr` for no bias)
- `output`: Pre-allocated output tensor in NHWC format
- `batch_size`: Number of samples in batch
- `input_height`, `input_width`: Spatial dimensions of input
- `input_channels`: Number of input feature maps
- `filter_height`, `filter_width`: Spatial dimensions of convolution kernel
- `output_channels`: Number of output feature maps
- `stride_height`, `stride_width`: Convolution stride in each dimension
- `padding`: `PaddingType::VALID` (no padding) or `PaddingType::SAME` (maintain output size)
- `activation`: `ActivationType::NONE` or `ActivationType::RELU`
- `dilation_height`, `dilation_width`: Dilation factors (default 1, meaning no dilation)

**Padding Types**:
- `PaddingType::VALID`: No padding. Output size = `(input_size - filter_size + stride) / stride`
- `PaddingType::SAME`: Padding to maintain output size. Output size = `ceil(input_size / stride)`

**Memory Layout**: NHWC format (batch, height, width, channels) - row-major order

**Example Usage**:
```cpp
// 1 batch, 28x28 image, 1 channel input
// 32 output channels, 3x3 filter, stride 1, SAME padding, ReLU activation
float input[1 * 28 * 28 * 1];
float filter[32 * 3 * 3 * 1];
float bias[32];
float output[1 * 28 * 28 * 32];

embedded_ml::Conv2D(input, filter, bias, output,
                    1, 28, 28, 1,      // batch, height, width, input_channels
                    3, 3,              // filter_height, filter_width
                    32,                // output_channels
                    1, 1,              // stride_height, stride_width
                    embedded_ml::PaddingType::SAME,
                    embedded_ml::ActivationType::RELU);
```

### 4.4 max\_pool\_2d.h

**Purpose**: 2D max pooling operation with padding and fused activation

**Function Signature**:
```cpp
template<typename T>
void MaxPool2D(
    const T* input,                   // Input tensor [batch, height, width, channels]
    T* output,                       // Output tensor [batch, out_h, out_w, channels]
    size_t batch_size,               // Number of batches
    size_t input_height,             // Input height
    size_t input_width,              // Input width
    size_t channels,                 // Number of channels
    size_t filter_height,            // Pooling window height
    size_t filter_width,             // Pooling window width
    size_t stride_height,            // Vertical stride
    size_t stride_width,             // Horizontal stride
    PaddingType padding,             // Padding type (VALID or SAME)
    ActivationType activation = ActivationType::NONE  // Activation function
);
```

**Parameters**:
- `input`: Input tensor in NHWC format
- `output`: Pre-allocated output tensor in NHWC format
- `batch_size`: Number of samples
- `input_height`, `input_width`: Spatial dimensions of input
- `channels`: Number of channels (same for input and output)
- `filter_height`, `filter_width`: Pooling window size
- `stride_height`, `stride_width`: Stride in each dimension
- `padding`: `PaddingType::VALID` or `PaddingType::SAME`
- `activation`: Optional activation function

**Operation**: For each pooling window, outputs the maximum value within that window.

**Example Usage**:
```cpp
float input[1 * 28 * 28 * 32];
float output[1 * 14 * 14 * 32];

embedded_ml::MaxPool2D(input, output,
                       1, 28, 28, 32,  // batch, height, width, channels
                       2, 2,           // filter_height, filter_width
                       2, 2,           // stride_height, stride_width
                       embedded_ml::PaddingType::VALID,
                       embedded_ml::ActivationType::NONE);
```

### 4.5 relu.h

**Purpose**: ReLU (Rectified Linear Unit) activation function

**Function Signatures**:
```cpp
// In-place version
template<typename T>
void ReLU(T* data, size_t size);

// Separate input/output version
template<typename T>
void ReLU(const T* input, T* output, size_t size);
```

**Parameters**:
- `data` (in-place): Array to modify in-place
- `input`, `output` (separate): Input and output arrays
- `size`: Number of elements

**Operation**: `output[i] = max(0, input[i])`

**Example Usage**:
```cpp
float data[128];
embedded_ml::ReLU(data, 128);  // In-place

// Or with separate arrays
float input[128];
float output[128];
embedded_ml::ReLU(input, output, 128);
```

### 4.6 softmax.h

**Purpose**: Softmax activation function (normalizes to probability distribution)

**Function Signatures**:
```cpp
// In-place version
template<typename T>
void Softmax(T* data, size_t size);

// Separate input/output version
template<typename T>
void Softmax(const T* input, T* output, size_t size);
```

**Parameters**:
- `data` (in-place): Array to normalize in-place
- `input`, `output` (separate): Input and output arrays
- `size`: Number of elements

**Mathematical Operation**:
```
max_val = max(input)
exp_sum = sum(exp(input[i] - max_val) for all i)
output[i] = exp(input[i] - max_val) / exp_sum
```

**Numerical Stability**: Subtracts maximum value before exponentiation to prevent overflow.

**Example Usage**:
```cpp
float logits[10];
embedded_ml::Softmax(logits, 10);  // Normalizes to probabilities
```

### 4.7 add.h

**Purpose**: Element-wise addition with optional fused activation

**Function Signatures**:
```cpp
// Standard version
template<typename T>
void Add(
    const T* input1,
    const T* input2,
    T* output,
    size_t size,
    ActivationType activation = ActivationType::NONE
);

// In-place version
template<typename T>
void AddInPlace(
    T* input1,        // Modified in-place
    const T* input2,
    size_t size,
    ActivationType activation = ActivationType::NONE
);
```

**Parameters**:
- `input1`, `input2`: Input arrays of same size
- `output`: Pre-allocated output array
- `size`: Number of elements (must be same for all arrays)
- `activation`: Optional activation function

**Operation**: `output[i] = activation(input1[i] + input2[i])`

**Example Usage**:
```cpp
float a[128];
float b[128];
float output[128];
embedded_ml::Add(a, b, output, 128, embedded_ml::ActivationType::RELU);
```

### 4.8 reshape.h

**Purpose**: Reshape tensor to new shape (data-preserving operation)

**Function Signature**:
```cpp
template<typename T>
void Reshape(
    const T* input,
    size_t input_size,
    T* output,
    size_t output_size
);
```

**Parameters**:
- `input`: Input tensor data
- `input_size`: Total number of elements in input
- `output`: Pre-allocated output array
- `output_size`: Total number of elements in output (must equal `input_size`)

**Operation**: For contiguous memory layouts (row-major), this is a simple copy operation. The shape information affects indexing but not memory layout.

**Note**: `input_size` must equal `output_size`. The function returns early if sizes don't match.

**Example Usage**:
```cpp
float input[784];      // 28×28 image flattened
float output[784];    // Same data, different shape interpretation
embedded_ml::Reshape(input, 784, output, 784);
```

### 4.9 flatten.h

**Purpose**: Flatten multi-dimensional tensor to 1D (except batch dimension)

**Function Signatures**:
```cpp
// Simple version
template<typename T>
void Flatten(
    const T* input,
    size_t input_size,
    T* output,
    size_t output_size
);

// With explicit dimensions
template<typename T>
void Flatten(
    const T* input,
    size_t batch_size,
    size_t feature_size,
    T* output
);
```

**Parameters**:
- `input`: Input tensor data
- `input_size`: Total number of elements
- `output`: Pre-allocated output array
- `output_size`: Total number of elements (must equal `input_size`)
- `batch_size`: Number of batches (first dimension)
- `feature_size`: Total size of feature dimensions

**Operation**: For contiguous memory, this is equivalent to a copy operation.

**Example Usage**:
```cpp
float input[1 * 7 * 7 * 64];  // 1 batch, 7×7×64 features
float output[1 * 3136];       // Flattened
embedded_ml::Flatten(input, 3136, output, 3136);
```

### 4.10 dropout.h

**Purpose**: Dropout operation (no-op during inference)

**Function Signatures**:
```cpp
// Standard version
template<typename T>
void Dropout(
    const T* input,
    T* output,
    size_t size,
    float rate = 0.0f
);

// In-place version
template<typename T>
void Dropout(T* data, size_t size, float rate = 0.0f);
```

**Parameters**:
- `input`: Input array
- `output`: Output array (copy of input)
- `data`: Array (unchanged in in-place version)
- `size`: Number of elements
- `rate`: Dropout rate (ignored during inference, kept for API compatibility)

**Operation**: During inference, dropout is a no-op - it simply copies input to output. The `rate` parameter is ignored but kept for API compatibility with training code.

**Example Usage**:
```cpp
float input[128];
float output[128];
embedded_ml::Dropout(input, output, 128, 0.5f);  // rate ignored
```

### 4.11 shape.h

**Purpose**: Extract tensor shape as integer array

**Function Signatures**:
```cpp
template<typename T>
void Shape(
    const T* input_shape,
    size_t num_dims,
    int32_t* output
);

// Alternative version with int32_t input
inline void Shape(
    const int32_t* dims,
    size_t num_dims,
    int32_t* output
);
```

**Parameters**:
- `input_shape`: Array of dimension sizes
- `num_dims`: Number of dimensions
- `output`: Pre-allocated output array of size `num_dims`
- `dims`: Dimension sizes (int32\_t version)

**Operation**: Copies dimension sizes to output array.

**Example Usage**:
```cpp
int32_t dims[4] = {1, 28, 28, 1};
int32_t shape[4];
embedded_ml::Shape(dims, 4, shape);
```

### 4.12 strided\_slice.h

**Purpose**: Extract slice from tensor using begin, end, and stride indices

**Function Signature**:
```cpp
template<typename T>
void StridedSlice(
    const T* input,
    const int* input_dims,
    size_t num_dims,
    const int* begin,
    const int* end,
    const int* stride,
    int begin_mask,
    int end_mask,
    int shrink_axis_mask,
    T* output,
    const int* output_dims,
    size_t output_num_dims
);
```

**Parameters**:
- `input`: Input tensor data
- `input_dims`: Array of input dimension sizes
- `num_dims`: Number of input dimensions
- `begin`: Start indices for each dimension
- `end`: End indices for each dimension (exclusive)
- `stride`: Stride values for each dimension
- `begin_mask`: Bitmask for dimensions using begin=0
- `end_mask`: Bitmask for dimensions using end=dim\_size
- `shrink_axis_mask`: Bitmask for dimensions to remove
- `output`: Pre-allocated output array
- `output_dims`: Array of output dimension sizes
- `output_num_dims`: Number of output dimensions

**Operation**: Extracts a slice `input[begin:end:stride]` with support for negative indices, masks, and axis shrinking.

**Note**: This is a complex operation. See TensorFlow documentation for detailed behavior.

### 4.13 pack.h

**Purpose**: Pack multiple tensors along a specified axis

**Function Signature**:
```cpp
template<typename T>
void Pack(
    const T* const* inputs,      // Array of input tensor pointers
    size_t num_inputs,           // Number of input tensors
    const int* input_dims,       // Input dimension sizes (same for all)
    size_t num_dims,             // Number of dimensions (before packing)
    int axis,                    // Axis along which to pack
    T* output,                   // Pre-allocated output array
    const int* output_dims,      // Output dimension sizes
    size_t output_num_dims       // Number of output dimensions (num_dims + 1)
);
```

**Parameters**:
- `inputs`: Array of pointers to input tensors (all same shape)
- `num_inputs`: Number of input tensors
- `input_dims`: Dimension sizes for each input
- `num_dims`: Number of dimensions in each input
- `axis`: Axis along which to pack (0-based, can be negative)
- `output`: Pre-allocated output array
- `output_dims`: Output dimension sizes
- `output_num_dims`: Number of output dimensions

**Operation**: Concatenates multiple tensors along a new axis, increasing dimensionality by 1.

---

## 5. Supported Operations and Layers

### 5.1 Complete List of Supported TFLite Operations

The code generator supports the following TFLite builtin operators:

1. **FULLY\_CONNECTED**
   2. Activation: `NONE` or `RELU` (fused)
   3. Description: Dense/fully connected layer
   4. Component: `fully_connected.h`

2. **CONV\_2D**
   2. Activation: `NONE` or `RELU` (fused)
   3. Padding: `VALID` or `SAME`
   4. Stride: Supported
   5. Dilation: Supported
   6. Description: 2D convolution
   7. Component: `conv_2d.h`

3. **MAX\_POOL\_2D**
   2. Activation: `NONE` or `RELU` (fused)
   3. Padding: `VALID` or `SAME`
   4. Stride: Supported
   5. Description: 2D max pooling
   6. Component: `max_pool_2d.h`

4. **RELU**
   2. Description: Standalone ReLU activation
   3. Component: `relu.h`

5. **SOFTMAX**
   2. Description: Softmax activation
   3. Component: `softmax.h`

6. **ADD**
   2. Activation: `NONE` or `RELU` (fused)
   3. Description: Element-wise addition
   4. Component: `add.h`

7. **RESHAPE**
   2. Description: Tensor reshaping
   3. Component: `reshape.h`

8. **SHAPE**
   2. Description: Extract tensor shape
   3. Component: `shape.h`

9. **STRIDED\_SLICE**
   2. Description: Extract tensor slice
   3. Component: `strided_slice.h`

10. **PACK**
	- Description: Pack tensors along axis
	- Component: `pack.h`

11. **DROPOUT** (Custom operator)
	- Description: Dropout (no-op during inference)
	- Component: `dropout.h`
	- Note: Recognized by name matching "DROPOUT" or "Dropout"

12. **FLATTEN** (Custom operator)
	- Description: Flatten tensor (converted to RESHAPE)
	- Component: `flatten.h`
	- Note: Recognized by name matching "FLATTEN" or "Flatten"

### 5.2 Activation Function Support

**Supported Activations**:
- `NONE`: No activation (linear/pass-through)
- `RELU`: Rectified Linear Unit: `max(0, x)`
- `SOFTMAX`: Softmax normalization (standalone operation)

**Fused Activations** (applied inline with operations):
- `RELU` can be fused with: `FULLY_CONNECTED`, `CONV_2D`, `MAX_POOL_2D`, `ADD`
- `SOFTMAX` is always a standalone operation

### 5.3 Padding Support

**Padding Types**:
- `VALID`: No padding. Output size = `(input_size - filter_size + stride) / stride`
- `SAME`: Padding to maintain output size. Output size = `ceil(input_size / stride)`

**Supported Operations**: `CONV_2D`, `MAX_POOL_2D`

### 5.4 Model Constraints

**Input/Output**:
- **Single Input Tensor**: Only models with exactly one input tensor are supported
- **Single Output Tensor**: Models with one output are fully supported
- **Multi-Output Models**: Supported with `--output-index=N` flag to select which output to use

**Data Types**:
- **FLOAT32 Only**: All weights and tensors must be FLOAT32
- **Quantization**: Not supported (INT8, INT16, etc.)

**Shape Constraints**:
- **Fixed Shapes**: Input and output shapes must be known at compile time
- **Dynamic Shapes**: Not supported (shapes must be constant)

**Other Constraints**:
- **Batch Size**: Typically 1 for inference (batch dimension supported but usually 1)
- **Memory**: All buffers statically allocated (no dynamic allocation)

### 5.5 Unsupported Operations

The following TFLite operations are **not** currently supported:

- Quantized operations (INT8, INT16, etc.)
- Operations requiring dynamic shapes
- Multi-input operations (beyond ADD)
- Complex operations: LSTM, GRU, RNN
- Custom operations (except DROPOUT and FLATTEN by name matching)
- Operations with variable input/output counts

**How to Check if a Model is Supported**:

Run the code generator on your model. It will validate the model and report any unsupported operations:

```bash
./miko_codegen model.tflite test_model
```

If the model contains unsupported operations, you'll see an error message listing:
- Which operations are unsupported
- Which operations are supported
- The operator indices where unsupported operations occur

---

## 6. Generated Code Structure

### 6.1 `{base_name}_weights.cpp`

**Purpose**: Stores all model weights as compile-time constants

**Structure**:
```cpp
namespace embedded_ml {

// Weight tensor 0: conv2d/kernel
// Shape: [32, 3, 3, 1]
// Elements: 288
static const float weights_0[288] = {
    0.123456f, 0.234567f, ...
};

// Weight tensor 1: conv2d/bias
// Shape: [32]
// Elements: 32
static const float bias_0[32] = {
    0.001f, 0.002f, ...
};

} // namespace embedded_ml
```

**Characteristics**:
- All weights are `static const float` arrays
- Organized by tensor with comments showing:
  - Tensor index
  - Tensor name (if available)
  - Shape
  - Number of elements
- Values formatted with 9 decimal places precision
- Arrays are compile-time constants (stored in read-only memory)

**Memory**: Weights are stored in the program's data segment (not stack or heap).

### 6.2 `{base_name}.h`

**Purpose**: Model class header file

**Structure**:
```cpp
#ifndef MY_MODEL_MODEL_H
#define MY_MODEL_MODEL_H

#include <cstddef>

namespace embedded_ml {

class MyModelModel {
public:
    static constexpr size_t kInputSize = 784;
    static constexpr size_t kOutputSize = 10;

private:
    // Intermediate buffers for layer outputs
    static float buffer_2[6272];
    static float buffer_3[1568];

public:
    // Run inference on input data
    // input: array of size kInputSize
    // output: array of size kOutputSize (will be filled with results)
    static void Inference(const float* input, float* output);
};

} // namespace embedded_ml

#endif // MY_MODEL_MODEL_H
```

**Public Members**:
- `kInputSize`: Input tensor size (constexpr compile-time constant)
- `kOutputSize`: Output tensor size (constexpr compile-time constant)
- `Inference(input, output)`: Static method to run inference

**Private Members**:
- Intermediate buffers: Static arrays for layer outputs (only if needed)

**Include Guards**: Automatically generated based on base name (uppercase, with underscores).

### 6.3 `{base_name}.cpp`

**Purpose**: Model implementation with inference logic

**Structure**:
```cpp
#include "my_model.h"
#include "my_model_weights.cpp"
#include "components/fully_connected.h"
#include "components/conv_2d.h"
#include "components/max_pool_2d.h"
#include "components/softmax.h"
#include <cstddef>

namespace embedded_ml {

void MyModelModel::Inference(const float* input, float* output) {
    // Layer 0: CONV_2D
    Conv2D(input, weights_0, bias_0, buffer_2, 1, 28, 28, 1, 3, 3, 32, 1, 1, 
           PaddingType::SAME, ActivationType::RELU, 1, 1);
    
    // Layer 1: MAX_POOL_2D
    MaxPool2D(buffer_2, buffer_3, 1, 28, 28, 32, 2, 2, 2, 2, 
              PaddingType::VALID, ActivationType::NONE);
    
    // Layer 2: FULLY_CONNECTED
    FullyConnected(buffer_3, weights_1, bias_1, output, 1568, 10, 
                   ActivationType::NONE);
    
    // Layer 3: SOFTMAX
    Softmax(output, 10);
}

// Intermediate buffer definitions
float MyModelModel::buffer_2[6272];
float MyModelModel::buffer_3[1568];

} // namespace embedded_ml
```

**Includes**:
- Model header: `{base_name}.h`
- Weights file: `{base_name}_weights.cpp`
- Components: Only includes components actually used in the model (conditional includes)

**Inference Method**:
- Executes layers in order
- Uses intermediate buffers for layer outputs
- Input and output are provided by caller
- All operations are static (no instance required)

**Intermediate Buffers**:
- Defined as static class members
- One buffer per intermediate tensor (excluding input, output, and weight tensors)
- Sizes calculated from tensor shapes

### 6.4 `{base_name}_inference.cpp` (Conditional)

Generated only if `--inf` is not set to `none`.

#### 6.4.1 Standard Version (`--inf=standard` or default)

**Purpose**: Example inference code for desktop/server platforms

**Structure**:
```cpp
#include "my_model.h"
#include <iostream>
#include <iomanip>
#include <cstddef>

using namespace embedded_ml;
using namespace std;

int main() {
    // Example input data
    float input[MyModelModel::kInputSize] = {
        0.0f, 0.0f, ...  // TODO: Replace with actual input data
    };

    // Output array
    float output[MyModelModel::kOutputSize];

    // Run inference
    MyModelModel::Inference(input, output);

    // Print results
    cout << "Inference Results:" << endl;
    cout << fixed << setprecision(6);
    for (size_t i = 0; i < MyModelModel::kOutputSize; ++i) {
        cout << "  Output[" << i << "] = " << output[i] << endl;
    }

    return 0;
}
```

**Characteristics**:
- Uses `iostream` for output
- Includes example input preparation
- Shows inference call
- Includes commented examples for result processing
- User-editable for specific use cases

#### 6.4.2 Pico Version (`--inf=pico-img-bench`)

**Purpose**: Benchmark inference code for Raspberry Pi Pico

**Structure**:
```cpp
#include "my_model.cpp"
#include <stdio.h>
#include <cmath>
#include <cstddef>
#include "pico/stdlib.h"

using namespace embedded_ml;

// Simple random number generator (LCG)
static uint32_t lcg_state = 1;
static float random_float() {
    lcg_state = lcg_state * 1103515245 + 12345;
    return static_cast<float>(lcg_state & 0x7FFFFFFF) / 2147483648.0f;
}

int main() {
    stdio_init_all();
    sleep_ms(2000);  // Wait for USB serial
    
    const int num_tests = 10;
    float* input = (float*)malloc(MyModelModel::kInputSize * sizeof(float));
    float* output = (float*)malloc(MyModelModel::kOutputSize * sizeof(float));
    
    // LED blinking
    const uint LED_PIN = 25;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    
    for (int test = 0; test < num_tests; ++test) {
        // Generate random input
        for (size_t i = 0; i < MyModelModel::kInputSize; ++i) {
            input[i] = random_float();
        }
        
        // Run inference
        MyModelModel::Inference(input, output);
        
        // Find predicted class
        size_t predicted_class = 0;
        float max_prob = output[0];
        for (size_t i = 1; i < MyModelModel::kOutputSize; ++i) {
            if (output[i] > max_prob) {
                max_prob = output[i];
                predicted_class = i;
            }
        }
        
        printf("Test %2d: Predicted class=%zu (prob=%.4f)\n", 
               test + 1, predicted_class, max_prob);
    }
    
    free(input);
    free(output);
    return 0;
}
```

**Characteristics**:
- Uses Pico SDK (`pico/stdlib.h`)
- Uses `printf` instead of `cout` (Pico doesn't support iostream)
- Includes LCG random number generator (no `std::random`)
- Initializes USB serial with `stdio_init_all()`
- Uses `malloc`/`free` for dynamic allocation (Pico-compatible)
- Includes LED blinking code (GPIO 25)
- Runs multiple benchmark iterations
- Prints results in compact format

**Pico-Specific Considerations**:
- No iostream support
- No exceptions
- Limited C++ standard library features
- Uses Pico SDK functions for GPIO and timing

### 6.5 `Makefile` (Conditional)

Generated only if `--no-makefile` is not set.

**Structure**:
```makefile
# Makefile for my_model inference
# Auto-generated - compiles the inference executable

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall

# Source files
SOURCES = my_model_weights.cpp \
          my_model.cpp \
          my_model_inference.cpp

# Header files
HEADERS = my_model.h

OBJECTS = $(SOURCES:.cpp=.o)
TARGET = my_model_inference

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean
```

**Characteristics**:
- Compiler: `g++` (can be overridden)
- Flags: `-std=c++17 -O2 -Wall`
- Source files: Automatically includes all generated `.cpp` files
- Target: `{base_name}_inference` executable
- Clean target: Removes object files and executable

**Building**:
```bash
cd my_model
make
./my_model_inference
```

### 6.6 `components/` Directory

**Purpose**: Local copy of component files needed by the model

**Structure**:
```
my_model/
├── components/
│   ├── fully_connected.h
│   ├── conv_2d.h
│   ├── max_pool_2d.h
│   └── softmax.h
├── my_model_weights.cpp
├── my_model.h
├── my_model.cpp
├── my_model_inference.cpp
└── Makefile
```

**Characteristics**:
- Only components actually used in the model are copied
- Components are referenced with local path: `#include "components/fully_connected.h"`
- Self-contained: Generated code doesn't depend on external component directories
- Portable: Can move the generated directory anywhere

**Which Components are Copied**:
- Always: `fully_connected.h`, `softmax.h` (always needed)
- Conditionally: Other components based on operations in the model:
  - `relu.h` if standalone RELU operations exist
  - `conv_2d.h` if CONV\_2D operations exist
  - `max_pool_2d.h` if MAX\_POOL\_2D operations exist
  - `add.h` if ADD operations exist
  - `reshape.h` if RESHAPE operations exist
  - `flatten.h` if FLATTEN operations exist
  - `dropout.h` if DROPOUT operations exist
  - `shape.h` if SHAPE operations exist
  - `strided_slice.h` if STRIDED\_SLICE operations exist
  - `pack.h` if PACK operations exist

---

## 7. Usage Examples

### 7.1 Basic Usage

**Simple Model Generation**:
```bash
./miko_codegen models/mnist_cnn.tflite mnist_model
```

This creates `mnist_model/` directory with:
- `mnist_model_weights.cpp`
- `mnist_model.h`
- `mnist_model.cpp`
- `mnist_model_inference.cpp`
- `Makefile`
- `components/` directory

**Build and Run**:
```bash
cd mnist_model
make
./mnist_model_inference
```

### 7.2 Multi-Output Model

**Model with Multiple Outputs**:
```bash
./miko_codegen models/multi_output.tflite my_model --output-index=1
```

The generator will:
- Show available outputs if `--output-index` is not specified
- Use the specified output index (1 in this case)
- Generate code for that specific output

### 7.3 Custom Paths

**Custom Templates and Components**:
```bash
./miko_codegen model.tflite my_model \
    --template-path=/custom/templates \
    --component-path=/custom/components
```

Useful when:
- Templates/components are in non-standard locations
- Using custom templates/components
- Organizing multiple projects

### 7.4 Pico Deployment

**Generate Pico-Compatible Code**:
```bash
./miko_codegen models/mnist_cnn.tflite pico_model --inf=pico-img-bench
```

This generates:
- Pico-compatible inference file (uses `printf`, Pico SDK)
- Includes LED blinking code
- Benchmark loop with random inputs
- Uses `malloc`/`free` for allocation

**Integration with Pico Project**:
1. Copy generated files to your Pico project
2. Include in CMakeLists.txt or your build system
3. Link against Pico SDK
4. Flash to Pico and run

### 7.5 Skip Inference File

**Generate Only Model Files**:
```bash
./miko_codegen model.tflite my_model --inf=none
```

Useful when:
- Writing custom inference code
- Integrating into existing projects
- Only need model implementation

### 7.6 Overwrite Existing Directory

**Regenerate in Existing Directory**:
```bash
./miko_codegen model.tflite existing_dir --replace
```

**Behavior**:
- Does NOT delete the directory
- Overwrites existing files
- Preserves any custom files you've added
- Fails if directory doesn't exist (must use without `--replace` to create)

### 7.7 Complete Workflow Example

**End-to-End Workflow**:
```bash
# 1. Generate code
./miko_codegen models/my_model.tflite my_inference

# 2. (Optional) Edit inference file
cd my_inference
# Edit my_inference_inference.cpp with your input data

# 3. Build
make

# 4. Run
./my_inference_inference

# 5. (Optional) Integrate into larger project
# Copy my_inference.h, my_inference.cpp, my_inference_weights.cpp
# to your project and include components/ directory
```

### 7.8 Integration Examples

**Using Generated Code in Existing Project**:

```cpp
// your_project.cpp
#include "my_model.h"
#include <vector>

int main() {
    // Prepare input from your data source
    std::vector<float> input(MyModelModel::kInputSize);
    // ... fill input with your data ...
    
    // Allocate output
    std::vector<float> output(MyModelModel::kOutputSize);
    
    // Run inference
    MyModelModel::Inference(input.data(), output.data());
    
    // Process results
    // ...
    
    return 0;
}
```

**CMake Integration**:
```cmake
# CMakeLists.txt
add_executable(my_app
    my_app.cpp
    my_model/my_model.cpp
    my_model/my_model_weights.cpp
)

target_include_directories(my_app PRIVATE
    my_model
    my_model/components
)
```

---

## 8. Technical Specifications

### 8.1 Memory Model

**Static Allocation**:
- All buffers allocated at compile time
- No dynamic memory allocation in generated code
- Weights stored as `static const` arrays (read-only data segment)
- Intermediate buffers as static class members

**Memory Layout**:
- **Weights**: Stored in program's data segment (read-only)
- **Intermediate Buffers**: Static class members (data segment)
- **Input/Output**: Provided by caller (stack or heap, caller's choice)

**Memory Efficiency**:
- Predictable memory usage (known at compile time)
- No heap fragmentation
- Better cache locality (static allocation)
- No allocation failures

**Example Memory Usage**:
```cpp
class MyModelModel {
    static float buffer_2[6272];  // ~25 KB (static)
    static float buffer_3[1568];  // ~6 KB (static)
    // Weights: ~50 KB (read-only, static const)
};
```

### 8.2 Performance Characteristics

**Optimizations Applied**:

1. **Loop Unrolling**:
   2. Fully connected layer: 4 elements at a time
   3. Reduces loop overhead
   4. Better instruction-level parallelism

2. **Restrict Pointers**:
   2. `__restrict` annotations inform compiler of no aliasing
   3. Enables vectorization
   4. More aggressive optimizations

3. **Fused Operations**:
   2. Activation functions applied inline
   3. Reduces memory bandwidth
   4. Fewer passes over data

4. **Inlining Opportunities**:
   2. Header-only components
   3. Compiler can inline entire call chains
   4. Eliminates function call overhead

5. **Static Allocation**:
   2. Predictable memory layout
   3. Better cache behavior
   4. Compiler can optimize memory access patterns

**Performance Considerations**:
- Single-threaded execution
- No SIMD intrinsics (relies on compiler auto-vectorization)
- Cache-friendly memory access patterns
- Minimal branching in hot loops

### 8.3 Code Size Considerations

**Factors Affecting Code Size**:
- Number of layers
- Model complexity
- Components included (only used components are included)
- Compiler optimization level

**Optimization Tips**:
- Use `-Os` for size optimization (trades some speed for size)
- Use `-O2` or `-O3` for speed optimization
- Link-time optimization (`-flto`) can reduce size further

### 8.4 Execution Model

**Single-Threaded**:
- All operations execute sequentially
- No thread synchronization needed
- Deterministic execution

**No Exceptions**:
- Generated code doesn't throw exceptions
- Suitable for `-fno-exceptions` builds
- Embedded-friendly

**Deterministic**:
- Same input always produces same output
- No random number generation (except in Pico benchmark code)
- Reproducible results

### 8.5 Template Design

**Type Flexibility**:
- Components are templates: `template<typename T>`
- Typically instantiated with `float`
- Could be extended to fixed-point types

**Zero-Cost Abstractions**:
- No runtime polymorphism
- No virtual function overhead
- Compile-time type checking

**Compile-Time Optimization**:
- Compiler can specialize for specific types
- Constant propagation
- Dead code elimination

---

## 9. Limitations and Constraints

### 9.1 Model Format

**Supported**: TensorFlow Lite (`.tflite`) only

**Not Supported**:
- ONNX models
- TensorFlow SavedModel
- PyTorch models
- Other formats

### 9.2 Data Types

**Supported**: FLOAT32 only

**Not Supported**:
- INT8 quantization
- INT16 quantization
- UINT8
- Other numeric types

### 9.3 Input/Output Constraints

**Input**:
- Single input tensor only
- Fixed shape (known at compile time)
- FLOAT32 data type

**Output**:
- Single output tensor (or use `--output-index` for multi-output models)
- Fixed shape
- FLOAT32 data type

### 9.4 Dynamic Shapes

**Not Supported**:
- Variable input sizes
- Dynamic batch sizes
- Runtime shape determination

**Reason**: All buffers are statically allocated with compile-time known sizes.

### 9.5 Quantization

**Not Supported**:
- INT8 quantization
- INT16 quantization
- Per-channel quantization
- Hybrid quantization

**Future Consideration**: Template design allows for adding quantized types.

### 9.6 Operator Coverage

**Supported Operators**: See Section 5.1

**Unsupported Operators** (examples):
- LSTM, GRU, RNN
- CONCATENATE
- SPLIT
- TRANSPOSE
- GATHER
- Many others

**Extensibility**: Architecture allows adding new operators by:
1. Creating component in `components/`
2. Adding operator handling in code generator
3. Updating template

### 9.7 Memory Constraints

**Static Allocation**:
- All memory requirements must be known at compile time
- Large models may exceed available memory
- No dynamic resizing

**Buffer Sizes**:
- Intermediate buffers can be large for high-resolution images
- Consider model architecture for memory-constrained systems

### 9.8 Platform Constraints

**C++17 Required**:
- Generated code requires C++17
- Code generator requires C++17
- `constexpr`, `std::filesystem`, etc.

**Standard Library**:
- Requires standard C++ library
- Some embedded systems may have limited standard library support
- Pico version uses minimal standard library features

---

## 10. Troubleshooting

### 10.1 Common Errors and Solutions

#### Template Directory Not Found

**Error**:
```
Templates directory does not exist or is not a directory: templates
Tip: If your templates are in a different location, use --template-path=PATH to specify it.
```

**Solution**:
- Ensure `templates/` directory exists in current directory, OR
- Use `--template-path=/path/to/templates` to specify location

#### Component Directory Not Found

**Error**:
```
Components directory does not exist or is not a directory: components
Tip: If your components are in a different location, use --component-path=PATH to specify it.
```

**Solution**:
- Ensure `components/` directory exists in current directory, OR
- Use `--component-path=/path/to/components` to specify location

#### Unsupported Operations

**Error**:
```
ERROR: Model contains unsupported operations!
This code generator only supports the following operations:
  - FULLY_CONNECTED (with NONE or RELU activation)
  - SOFTMAX
  ...
```

**Solution**:
- Check the list of supported operations (Section 5.1)
- Modify your model to use only supported operations
- Or extend the code generator to support the needed operation

#### Invalid Model Structure

**Error**:
```
Error: Invalid model structure - missing inputs/outputs
```

**Solution**:
- Verify the TFLite file is valid
- Check that the model has at least one input and one output
- Ensure the file is not corrupted

#### Directory Already Exists

**Error**:
```
Output directory already exists: my_model
Use --replace to overwrite files in it.
```

**Solution**:
- Use `--replace` flag to overwrite files: `./miko_codegen model.tflite my_model --replace`
- Or choose a different base name
- Note: `--replace` does NOT delete the directory, only overwrites files

#### Invalid Flag Values

**Error**:
```
Error: Invalid inference type: invalid_value
Valid inference types are: none, standard, pico-img-bench
```

**Solution**:
- Check the valid values for the flag
- Use case-insensitive matching (e.g., `STANDARD`, `standard`, `Standard` all work)

#### Multi-Output Model Without Flag

**Error**:
```
Error: This model has 3 output(s).
You must specify which output to use with --output-index=N
```

**Solution**:
- Use `--output-index=N` where N is the desired output index (0-based)
- Check the list of available outputs shown in the error message

### 10.2 Model Validation Errors

**Unsupported Activation**:
- Only NONE and RELU are supported for fused activations
- SOFTMAX must be a standalone operation

**Invalid Tensor Shapes**:
- All shapes must be known and constant
- Dynamic shapes are not supported

### 10.3 Code Generation Errors

**Template Rendering Failed**:
- Check that template files are valid Inja templates
- Verify template syntax
- Check for missing template variables

**File Creation Failed**:
- Check write permissions in output directory
- Ensure sufficient disk space
- Verify output path is valid

### 10.4 Build Errors with Generated Code

**Missing Components**:
- Ensure `components/` directory is in the include path
- Check that all needed components were copied

**Linker Errors**:
- Ensure all source files are included in build
- Check that weights file is compiled and linked

**Compilation Errors**:
- Verify C++17 support (`-std=c++17`)
- Check for missing includes
- Verify component compatibility

### 10.5 Runtime Issues

**Incorrect Results**:
- Verify input data format matches model expectations
- Check input normalization (if model was trained with normalized data)
- Ensure input shape matches `kInputSize`

**Memory Issues**:
- Check that intermediate buffers fit in available memory
- Consider reducing model size for memory-constrained systems

**Performance Issues**:
- Use compiler optimizations (`-O2`, `-O3`)
- Consider using `-march=native` for target-specific optimizations
- Profile to identify bottlenecks

---

## 11. Advanced Topics

### 11.1 Extending the Code Generator

#### Adding a New Component

1. **Create Component File** (`components/new_op.h`):
```cpp
namespace embedded_ml {
    template<typename T>
    void NewOp(const T* input, T* output, size_t size) {
        // Implementation
    }
}
```

2. **Update Code Generator** (`src/code_generator.cpp`):
   2. Add detection logic in `GenerateModelFile()`
   3. Set flag (e.g., `has_new_op = true`)
   4. Add to JSON data structure

3. **Update Template** (`templates/model.cpp.inja`):
   2. Add conditional include: `{% if has_new_op %}#include "components/new_op.h"{% endif %}`
   3. Add operation call in layer loop

4. **Update Validator** (`src/model_validator.cpp`):
   2. Add operator to `supported_ops` set

#### Adding a New Operation

Similar to adding a component, but also:
- Update model validator to recognize the TFLite operator
- Handle operator-specific parameters
- Generate appropriate code in template

#### Customizing Templates

Templates use Inja syntax. Key features:
- Variables: `{{ variable_name }}`
- Conditionals: `{% if condition %}...{% endif %}`
- Loops: `{% for item in list %}...{% endfor %}`
- Comments: `{# comment #}`

Modify templates in `templates/` directory to change generated code structure.

### 11.2 Integration Strategies

#### Embedding in Larger Projects

**Option 1: Copy Generated Files**:
```bash
# Generate code
./miko_codegen model.tflite my_model

# Copy to your project
cp my_model/my_model.* your_project/src/
cp -r my_model/components your_project/include/
```

**Option 2: Include as Subdirectory**:
- Add generated directory to your build system
- Include paths appropriately

#### Custom Build Systems

**CMake Example**:
```cmake
add_library(my_model STATIC
    my_model/my_model.cpp
    my_model/my_model_weights.cpp
)

target_include_directories(my_model PUBLIC
    my_model
    my_model/components
)
```

**Custom Makefile**:
```makefile
MODEL_SRCS = my_model/my_model.cpp my_model/my_model_weights.cpp
MODEL_INCLUDES = -Imy_model -Imy_model/components
```

#### Building Without Makefile

If using `--no-makefile`, manually compile:
```bash
g++ -std=c++17 -O2 -I. -Icomponents \
    my_model_weights.cpp \
    my_model.cpp \
    my_model_inference.cpp \
    -o my_model_inference
```

### 11.3 Performance Tuning

#### Compiler Flags

**Optimization Levels**:
- `-O0`: No optimization (debugging)
- `-O1`: Basic optimizations
- `-O2`: Recommended (good balance)
- `-O3`: Aggressive optimizations (may increase code size)
- `-Os`: Optimize for size

**Target-Specific**:
- `-march=native`: Use instructions for current CPU
- `-march=armv7-a`: For ARM targets
- `-mfloat-abi=hard`: For ARM with hardware FPU

**Link-Time Optimization**:
- `-flto`: Enable link-time optimization
- Can reduce code size and improve performance
- Requires LTO support in linker

#### Memory Layout Considerations

- Static allocation provides predictable layout
- Consider cache line alignment for hot data
- Group frequently accessed data together

#### Profiling

Use profiling tools to identify bottlenecks:
- `perf` (Linux)
- `gprof`
- Platform-specific profilers

---

## 12. Appendix

### 12.1 File Structure Reference

**Repository Structure**:
```
.
├── src/                    # Code generator source
│   ├── codegen.cpp        # Main entry point
│   ├── code_generator.cpp # Code generation logic
│   ├── code_generator.h   # Code generator interface
│   ├── model_utils.cpp    # Model parsing utilities
│   ├── model_utils.h
│   ├── model_validator.cpp # Model validation
│   ├── model_validator.h
│   ├── filesystem_utils.cpp # File operations
│   ├── filesystem_utils.h
│   └── exceptions.h       # Exception classes
├── components/            # Component library
│   ├── fully_connected.h
│   ├── conv_2d.h
│   ├── max_pool_2d.h
│   ├── relu.h
│   ├── softmax.h
│   ├── add.h
│   ├── reshape.h
│   ├── flatten.h
│   ├── dropout.h
│   ├── shape.h
│   ├── strided_slice.h
│   └── pack.h
├── templates/            # Code generation templates
│   ├── model.h.inja
│   ├── model.cpp.inja
│   ├── weights.cpp.inja
│   ├── inference.cpp.inja
│   ├── inference_pico.cpp.inja
│   └── Makefile.inja
├── include/              # Included dependencies
│   ├── tensorflow/      # TFLite schema
│   ├── inja/            # Templating library
│   └── nlohmann/        # JSON library
└── Makefile             # Build configuration
```

**Generated Directory Structure**:
```
{base_name}/
├── components/           # Copied component files
│   ├── fully_connected.h
│   └── ... (only used components)
├── {base_name}_weights.cpp
├── {base_name}.h
├── {base_name}.cpp
├── {base_name}_inference.cpp  # (if --inf != none)
└── Makefile             # (if --no-makefile not used)
```

### 12.2 Template Syntax (Inja) Reference

**Variables**:
```
{{ variable_name }}
```

**Conditionals**:
```
{% if condition %}
  ...
{% endif %}

{% if condition %}
  ...
{% else %}
  ...
{% endif %}
```

**Loops**:
```
{% for item in list %}
  {{ item.property }}
{% endfor %}
```

**Comments**:
```
{# This is a comment #}
```

**Filters** (Inja built-in):
- `upper`: `{{ name | upper }}`
- `lower`: `{{ name | lower }}`
- `length`: `{{ list | length }}`

### 12.3 TFLite Model Format Overview

TFLite models use FlatBuffer format:
- Binary format for efficient serialization
- Contains: operators, tensors, weights, metadata
- Schema defined in `tensorflow/lite/schema/schema_generated.h`

**Key Concepts**:
- **Subgraph**: Main computation graph
- **Operator**: A neural network operation (CONV\_2D, FULLY\_CONNECTED, etc.)
- **Tensor**: Multi-dimensional array (input, output, or intermediate)
- **Buffer**: Weight data storage

### 12.4 Component Dependency Graph

```
fully_connected.h (defines ActivationType)
    ↑
    ├── conv_2d.h (uses ActivationType, defines PaddingType)
    │   ↑
    │   └── max_pool_2d.h (uses PaddingType, ActivationType)
    │
    └── add.h (uses ActivationType)

strided_slice.h
    ↑
    └── pack.h (uses ComputeIndex from strided_slice.h)

(Independent components: relu.h, softmax.h, reshape.h, flatten.h, dropout.h, shape.h)
```

### 12.5 Glossary of Terms

- **TFLite**: TensorFlow Lite, a format for deploying ML models on mobile/embedded devices
- **FlatBuffer**: Efficient cross-platform serialization format used by TFLite
- **Subgraph**: A computation graph in a TFLite model (models typically have one main subgraph)
- **Operator**: A neural network operation (layer) in the model
- **Tensor**: A multi-dimensional array in the model (input, output, or intermediate)
- **Buffer**: Storage for weight data in TFLite format
- **Fused Activation**: Activation function applied inline with an operation (e.g., ReLU after convolution)
- **NHWC**: Tensor layout format: Number of batches, Height, Width, Channels
- **Row-major**: Memory layout where elements in the same row are stored consecutively
- **Static Allocation**: Memory allocated at compile time, not runtime
- **Header-only**: Library implementation entirely in header files (no separate .cpp files)
- **Template**: C++ feature allowing type-parameterized code
- **Inja**: Template engine for C++ (similar to Jinja2)

### 12.6 References and Resources

**TensorFlow Lite**:
- Official documentation: https://www.tensorflow.org/lite
- Model format specification
- Operator reference

**C++ Standards**:
- C++17 standard specification
- Compiler documentation (GCC, Clang)

**Embedded Systems**:
- Raspberry Pi Pico SDK documentation
- Embedded C++ best practices

**Related Tools**:
- TensorFlow Lite Converter (for creating .tflite files)
- FlatBuffer documentation

[1]:	#1-introduction
[2]:	#2-installation-and-building
[3]:	#3-command-line-interface-reference
[4]:	#4-component-library-reference
[5]:	#5-supported-operations-and-layers
[6]:	#6-generated-code-structure
[7]:	#7-usage-examples
[8]:	#8-technical-specifications
[9]:	#9-limitations-and-constraints
[10]:	#10-troubleshooting
[11]:	#11-advanced-topics
[12]:	#12-appendix