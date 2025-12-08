# Design Document: Embedded ML Code Generation System

## Overview

This project implements a code generation system that transforms TensorFlow Lite (TFLite) models into pure, dependency-free C++ inference code suitable for embedded systems, particularly low-powered devices like the Raspberry Pi Pico. The design prioritizes clean modern C++, minimal runtime overhead, and zero external dependencies in the generated code.

## Core Design Principles

### 1. Zero External Dependencies (Generated Code)
- **Rationale**: Embedded systems often have strict constraints on library availability and binary size
- **Implementation**: Generated code uses only the C++ standard library and custom header-only components
- **Trade-off**: Code generator itself requires TFLite libraries, but this is a build-time dependency only

### 2. Static Allocation
- **Rationale**: Dynamic memory allocation introduces unpredictability, fragmentation risks, and runtime overhead
- **Implementation**: 
  - All buffers allocated as static class members or stack arrays
  - Weight arrays stored as `static const` arrays
  - Intermediate layer buffers pre-allocated at compile time
- **Benefits**: Predictable memory usage, no allocation failures, better cache locality

### 3. Header-Only Component Architecture
- **Rationale**: Simplifies integration, enables aggressive inlining, reduces linking complexity
- **Implementation**: All neural network operations (`fully_connected.h`, `conv_2d.h`, etc.) are template-based header-only libraries
- **Benefits**: 
  - Zero linking overhead for unused components
  - Compiler can inline entire call chains
  - Easy to integrate into existing projects

### 4. Template-Based Design
- **Rationale**: Type flexibility without runtime polymorphism overhead
- **Implementation**: Components use templates (`template<typename T>`) allowing different numeric types (float, fixed-point, etc.)
- **Benefits**: 
  - Zero abstraction penalty
  - Compile-time type checking
  - Potential for fixed-point arithmetic on systems without FPU

### 5. Inja Templating for Code Generation
- **Rationale**: Separates code generation logic from output formatting, improving maintainability
- **Implementation**: 
  - Model structure parsed into JSON data structures
  - Inja templates (`.inja` files) define code structure
  - Clean separation between "what to generate" (logic) and "how it looks" (templates)
- **Benefits**:
  - Easy to modify generated code format without touching C++ logic
  - Template syntax is readable and maintainable
  - Supports conditional includes and iteration naturally

## Architecture

### Component Layer (`components/`)

The component layer provides reusable, header-only implementations of neural network operations:

- **Design Pattern**: Free functions in `embedded_ml` namespace
- **Template Parameters**: All components are templated on numeric type
- **Memory Model**: Pointer-based with `__restrict` annotations for compiler optimization
- **Activation Functions**: Fused into operations (e.g., `FullyConnected` with `ActivationType::RELU`) to reduce memory passes

**Example Component Structure**:
```cpp
namespace embedded_ml {
    template<typename T>
    void FullyConnected(
        const T* __restrict input,
        const T* __restrict weights,
        const T* __restrict bias,
        T* __restrict output,
        size_t input_size,
        size_t output_size,
        ActivationType activation = ActivationType::NONE
    );
}
```

**Key Design Choices**:
- Loop unrolling: Manual unrolling (4 elements at a time) for common cases
- Fused operations: Activation functions applied inline to avoid extra memory passes

### Code Generator (`codegen/`)

The code generator is a C++ program that:

1. **Parses TFLite Models**: Uses FlatBuffer API to extract model structure and weights
2. **Builds JSON Data**: Converts model structure into JSON for template rendering
3. **Renders Templates**: Uses Inja to generate C++ source files

**Design Choices**:
- Separation of Concerns: Model parsing logic separate from code generation templates
- Error Handling: Validates model structure before generation, provides clear error messages
- Conditional Generation: Only includes components actually used in the model

### Generated Code Structure

Each generated model consists of three files:

#### 1. `{model}_weights.cpp`
- Stores all model weights as compile-time constants
- stores as `static const float` arrays in `embedded_ml` namespace
- Weights organized by tensor with shape comments for readability

#### 2. `{model}.h` and `{model}.cpp`
- **Purpose**: Model class with inference method
- **Design**:
  - Static class with `constexpr` size constants
  - Static `Inference()` method (no instance required)
  - Intermediate buffers as static class members
  - Only includes components actually used

#### 3. `{model}_inference.cpp`
- **Purpose**: Example usage code
- **Design**: User-editable template showing input preparation and result processing

## Modern C++ Features Used

### 1. `constexpr` for Compile-Time Constants
```cpp
static constexpr size_t kInputSize = 784;
static constexpr size_t kOutputSize = 10;
```
- **Rationale**: Enables compile-time optimization and type safety
- **Benefit**: Compiler can optimize based on known sizes

### 2. Namespaces
- **Rationale**: Prevents symbol pollution, enables clean component organization
- **Implementation**: All components and generated code in `embedded_ml` namespace

### 3. Strong Typing
- **Rationale**: Catch errors at compile time
- **Implementation**: Enum classes for activation/padding types instead of magic numbers

### 4. Template Metaprogramming
- **Rationale**: Zero-cost abstractions
- **Implementation**: Components are templates, allowing type flexibility without runtime overhead

### 5. `static` Methods
- **Rationale**: No instance required, reduces memory footprint
- **Implementation**: `Inference()` is static, operating on provided buffers

## Performance Optimizations

### 1. Restrict Pointers
- **Purpose**: Inform compiler that pointers don't alias
- **Benefit**: Enables vectorization and more aggressive optimizations

### 2. Loop Unrolling
- **Purpose**: Reduce loop overhead for small, hot loops
- **Implementation**: Manual unrolling in fully connected layer (4 elements at a time)

### 3. Fused Operations
- **Purpose**: Reduce memory bandwidth
- **Implementation**: Activation functions applied inline during computation

### 4. Static Allocation
- **Purpose**: Predictable memory layout, better cache behavior
- **Benefit**: Compiler can optimize memory access patterns

### 5. Header-Only Components
- **Purpose**: Enable inlining across translation units
- **Benefit**: Eliminates function call overhead

## Low-Power System Considerations

### Memory Efficiency
- **Static Allocation**: All memory requirements known at compile time
- **No Dynamic Allocation**: Eliminates heap fragmentation and allocation overhead
- **Minimal Stack Usage**: Large buffers are static class members, not stack-allocated

### CPU Efficiency
- **No Virtual Functions**: Zero vtable overhead
- **Inline Functions**: Compiler can inline entire inference path
- **Template Specialization**: Potential for fixed-point specializations on systems without FPU

### Code Size
- **Conditional Includes**: Only includes components actually used
- **No RTTI**: Reduces binary size
- **Minimal Dependencies**: Only standard library in generated code

## Inja Templating Design

### Template Organization
- **Separation**: Logic in C++ (`codegen.cpp`), presentation in templates (`.inja` files)
- **Data Structure**: Model structure converted to JSON for template consumption
- **Conditional Logic**: Templates use Inja conditionals to include only needed components

### Template Files
1. **`model.h.inja`**: Generates model header with class definition
2. **`model.cpp.inja`**: Generates model implementation with inference logic
3. **`inference.cpp.inja`**: Generates example usage code
4. **`Makefile.inja`**: Generates build configuration

### Benefits of Inja
- **Readability**: Templates are close to final output format
- **Maintainability**: Code structure changes don't require C++ changes
- **Flexibility**: Easy to add new output formats (e.g., CMake, different code styles)

## Error Handling Strategy

### Code Generator
- **Validation**: Checks model structure before generation
- **Error Messages**: Clear, actionable error messages
- **Graceful Failure**: Stops generation on errors, doesn't produce partial files

### Generated Code
- **No Exceptions**: Embedded systems often disable exceptions
- **No Error Returns**: Inference is deterministic, failures are programming errors
- **Assertions**: Could add `assert()` for bounds checking in debug builds

## Extensibility

### Adding New Operations
1. **Create Component**: Add header-only component in `components/`
2. **Update Parser**: Add operator handling in `codegen.cpp`
3. **Update Template**: Add conditional include and operation call in `model.cpp.inja`

### Supporting New Model Formats
- **Abstraction**: Model parsing logic separated from code generation
- **Extensibility**: Could add parsers for other formats (ONNX, etc.) while reusing templates

## Trade-offs and Limitations

### Accepted Trade-offs

1. **Code Generator Dependency**: Generator requires TFLite, but this is acceptable as it's a build-time tool
2. **Static Memory**: Fixed buffer sizes limit model flexibility, but provide predictability
3. **Template Complexity**: Inja templates can be complex, but provide better separation than string concatenation
4. **No Dynamic Shapes**: Models must have fixed input/output sizes (common in embedded ML)

### Current Limitations

1. **Operator Support**: Not all TFLite operators supported (extensible design allows adding more)
2. **Quantization**: Currently supports FLOAT32 only (template design allows adding quantized types)
3. **Multi-Input/Output**: Currently supports single input/output (architecture allows extension)

## Future Design Considerations

### Potential Enhancements
1. **Quantization Support**: Add template specializations for int8/int16 operations
2. **Fixed-Point Arithmetic**: Template specializations for systems without FPU
3. **Operator Fusion**: Combine adjacent operations to reduce memory passes
4. **SIMD Optimizations**: Platform-specific vectorization hints
5. **Memory Pool**: Optional memory pool for systems that can tolerate dynamic allocation

## Conclusion

This design prioritizes:
- **Clean Modern C++**: Leverages templates, constexpr, namespaces, and strong typing
- **Inja Templating**: Separates logic from presentation for maintainability
- **Low Overhead**: Static allocation, header-only components, zero dependencies
- **Embedded-Friendly**: Predictable memory, minimal runtime, no dynamic allocation

The architecture is extensible, allowing new operations and optimizations while maintaining the core principles of zero dependencies and minimal overhead.

