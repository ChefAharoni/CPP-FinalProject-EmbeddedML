// codegen/codegen.cpp
// Code generator that reads TFLite models and generates pure C++ inference code
// Direct FlatBuffer inspection without requiring operator implementations

#include <iostream>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <ranges>
#include <sstream>
#include <map>
#include <set>
#include <format>
#include <filesystem>
#include <stdexcept>
#include <exception>
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

// Inja (template engine) + JSON (required by Inja)
#include <inja/inja.hpp>
#include <nlohmann/json.hpp>

//TensorFlow includes - we only need these headers to parse the .tflite model, NOT the entire TensorFlow library! They are included in /include.
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/builtin_op_data.h"

// using json = nlohmann::json;

namespace {

// Custom exception types for better error handling
class ModelValidationError : public std::runtime_error {
public:
    explicit ModelValidationError(const std::string& msg) : std::runtime_error(msg) {}
};

class CodeGenerationError : public std::runtime_error {
public:
    explicit CodeGenerationError(const std::string& msg) : std::runtime_error(msg) {}
};

class FileSystemError : public std::runtime_error {
public:
    explicit FileSystemError(const std::string& msg) : std::runtime_error(msg) {}
};

// Helper to get the actual builtin code (handles schema v3 compatibility)
// For v3 models, builtin_code defaults to 0, so we need to check deprecated_builtin_code
// For newer models, builtin_code contains the actual value
tflite::BuiltinOperator GetActualBuiltinCode(const tflite::OperatorCode* op_code) {
    if (!op_code) {
        return tflite::BuiltinOperator_CUSTOM;
    }
    // Since TFLite
    return static_cast<tflite::BuiltinOperator>(
        std::max(static_cast<int>(op_code->builtin_code()),
                 static_cast<int>(op_code->deprecated_builtin_code())));
}

// Helper to get operator name from FlatBuffer
std::string GetOperatorName(const tflite::OperatorCode* op_code) {
    if (!op_code) {
        return "UNKNOWN";
    }
    
    tflite::BuiltinOperator builtin_code = GetActualBuiltinCode(op_code);
    
    if (builtin_code != tflite::BuiltinOperator_CUSTOM) {
        const char* name = tflite::EnumNameBuiltinOperator(builtin_code);
        return name ? name : "UNKNOWN";
    } else {
        std::string result = "CUSTOM:";
        if (op_code->custom_code()) {
            result += op_code->custom_code()->c_str();
        }
        return result;
    }
}

// Calculate tensor size from FlatBuffer shape
std::size_t CalculateTensorSize(const flatbuffers::Vector<int32_t>* shape) {
    if (!shape || shape->size() == 0) {
        return 1;
    }
    
    // Use std::accumulate
    auto shape_view = std::ranges::views::iota(0u, shape->size())
        | std::ranges::views::transform([shape](std::size_t i) {
            return static_cast<std::size_t>(shape->Get(i));
        });
    
    return std::accumulate(shape_view.begin(), shape_view.end(), 
                          std::size_t{1}, std::multiplies<>{});
}

// Get shape as string from FlatBuffer
std::string GetShapeString(const flatbuffers::Vector<int32_t>* shape) {
    if (!shape || shape->size() == 0) {
        return "1";
    }
    
    // Use ranges to transform shape values to strings, then join with ", "
    auto shape_strs = std::ranges::views::iota(0u, shape->size())
        | std::ranges::views::transform([shape](std::size_t i) {
            return std::to_string(shape->Get(i));
        });
    
    // Join strings with ", " separator
    std::string result;
    bool first = true;
    for (const auto& str : shape_strs) {
        if (!first) {
            result += ", ";
        }
        result += str;
        first = false;
    }
    return result;
}

// Get bytes per element based on tensor type
constexpr std::size_t GetBytesPerElement(tflite::TensorType type) {
    switch (type) {
        case tflite::TensorType_FLOAT32: return 4;
        case tflite::TensorType_INT32: return 4;
        case tflite::TensorType_UINT8: return 1;
        case tflite::TensorType_INT64: return 8;
        case tflite::TensorType_INT16: return 2;
        case tflite::TensorType_INT8: return 1;
        case tflite::TensorType_FLOAT16: return 2;
        case tflite::TensorType_FLOAT64: return 8;
        case tflite::TensorType_UINT64: return 8;
        case tflite::TensorType_UINT32: return 4;
        case tflite::TensorType_UINT16: return 2;
        case tflite::TensorType_BOOL: return 1;
        default: return 0;
    }
}

// Validate operators from model schema
// This provides clear error messages for unsupported operations
void ValidateModelSchema(const tflite::Model* model) {
    if (!model || !model->subgraphs() || model->subgraphs()->size() == 0) {
        throw ModelValidationError("Invalid model structure - no subgraphs found.");
    }

    // Get the main subgraph (usually index 0)
    const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
    if (!subgraph || !subgraph->operators()) {
        throw ModelValidationError("Invalid model structure - no operators found.");
    }

    // Get operator codes
    const flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>* op_codes = 
        model->operator_codes();
    if (!op_codes) {
        throw ModelValidationError("Invalid model structure - no operator codes found.");
    }

    bool ok = true;
    std::vector<std::pair<int, std::string>> unsupported_ops; // (operator_index, operator_name)
    std::vector<std::pair<int, int>> unsupported_activations; // (operator_index, activation_code)
    const auto* tensors = subgraph->tensors();

    // Set of supported operators
    std::set<tflite::BuiltinOperator> supported_ops = {
        tflite::BuiltinOperator_FULLY_CONNECTED,
        tflite::BuiltinOperator_SOFTMAX,
        tflite::BuiltinOperator_RELU,
        tflite::BuiltinOperator_CONV_2D,
        tflite::BuiltinOperator_MAX_POOL_2D,
        tflite::BuiltinOperator_SHAPE,
        tflite::BuiltinOperator_STRIDED_SLICE,
        tflite::BuiltinOperator_PACK,
        tflite::BuiltinOperator_RESHAPE,
        tflite::BuiltinOperator_ADD
    };

    // Check each operator in the model
    for (std::size_t i = 0; i < subgraph->operators()->size(); ++i) {
        const tflite::Operator* op = subgraph->operators()->Get(i);
        if (!op) continue;

        // Get operator code index
        int op_code_index = op->opcode_index();
        if (op_code_index < 0 || op_code_index >= static_cast<int>(op_codes->size())) {
            throw ModelValidationError(
                std::format("Invalid operator code index {} at operator {}.", op_code_index, i)
            );
        }

        const tflite::OperatorCode* op_code = op_codes->Get(op_code_index);
        if (!op_code) {
            throw ModelValidationError(
                std::format("Operator {} has null operator code", i)
            );
        }

        // Get actual builtin code (handles schema v3 compatibility)
        // For v3 models, builtin_code defaults to 0, so we need to check deprecated_builtin_code
        // For newer models, builtin_code contains the actual value
        tflite::BuiltinOperator builtin_code = static_cast<tflite::BuiltinOperator>(
            std::max(static_cast<int>(op_code->builtin_code()),
                     static_cast<int>(op_code->deprecated_builtin_code())));

        // Check if operator is supported
        if (!supported_ops.contains(builtin_code)) {
            std::string op_name;
            if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
                op_name = std::string("CUSTOM:") + 
                    (op_code->custom_code() ? op_code->custom_code()->c_str() : "");
                // Check if it's a supported custom operator (DROPOUT or FLATTEN)
                std::string custom_name = op_code->custom_code() ? op_code->custom_code()->c_str() : "";
                if (custom_name.find("DROPOUT") != std::string::npos || 
                    custom_name.find("Dropout") != std::string::npos ||
                    custom_name.find("FLATTEN") != std::string::npos ||
                    custom_name.find("Flatten") != std::string::npos) {
                    // Supported custom operator, skip the error
                    continue;
                }
            } else {
                const char* op_name_ptr = tflite::EnumNameBuiltinOperator(builtin_code);
                op_name = op_name_ptr ? op_name_ptr : "UNKNOWN";
            }
            
            // Collect detailed information about this unsupported operator
            std::string op_details = op_name;
            if (tensors && op->inputs() && op->inputs()->size() > 0) {
                op_details += " (inputs: ";
                for (std::size_t j = 0; j < op->inputs()->size(); ++j) {
                    int input_idx = op->inputs()->Get(j);
                    if (input_idx >= 0 && input_idx < static_cast<int>(tensors->size())) {
                        const tflite::Tensor* input_tensor = tensors->Get(input_idx);
                        if (input_tensor && input_tensor->shape()) {
                            op_details += "tensor[" + std::to_string(input_idx) + ":" + 
                                         GetShapeString(input_tensor->shape()) + "]";
                        } else {
                            op_details += "tensor[" + std::to_string(input_idx) + "]";
                        }
                    } else {
                        op_details += "tensor[" + std::to_string(input_idx) + "]";
                    }
                    if (j < op->inputs()->size() - 1) op_details += ", ";
                }
                op_details += ")";
            }
            
            unsupported_ops.push_back({static_cast<int>(i), op_details});
            ok = false;
        }

        // For FULLY_CONNECTED, CONV_2D, and ADD, check fused activation
        if (builtin_code == tflite::BuiltinOperator_FULLY_CONNECTED) {
            const tflite::FullyConnectedOptions* options = 
                op->builtin_options_as_FullyConnectedOptions();
            if (options) {
                tflite::ActivationFunctionType activation = options->fused_activation_function();
                if (activation != tflite::ActivationFunctionType_NONE &&
                    activation != tflite::ActivationFunctionType_RELU) {
                    unsupported_activations.push_back({static_cast<int>(i), static_cast<int>(activation)});
                    ok = false;
                }
            }
        } else if (builtin_code == tflite::BuiltinOperator_CONV_2D) {
            const tflite::Conv2DOptions* options = 
                op->builtin_options_as_Conv2DOptions();
            if (options) {
                tflite::ActivationFunctionType activation = options->fused_activation_function();
                if (activation != tflite::ActivationFunctionType_NONE &&
                    activation != tflite::ActivationFunctionType_RELU) {
                    unsupported_activations.push_back({static_cast<int>(i), static_cast<int>(activation)});
                    ok = false;
                }
            }
        } else if (builtin_code == tflite::BuiltinOperator_ADD) {
            const tflite::AddOptions* options = 
                op->builtin_options_as_AddOptions();
            if (options) {
                tflite::ActivationFunctionType activation = options->fused_activation_function();
                if (activation != tflite::ActivationFunctionType_NONE &&
                    activation != tflite::ActivationFunctionType_RELU) {
                    unsupported_activations.push_back({static_cast<int>(i), static_cast<int>(activation)});
                    ok = false;
                }
            }
        }
    }

    // Build error message if validation failed
    if (!ok) {
        std::string error_msg = "\n================================================\n";
        error_msg += "ERROR: Model contains unsupported operations!\n";
        error_msg += "================================================\n\n";
        error_msg += "This code generator only supports the following operations:\n";
        error_msg += "  - FULLY_CONNECTED (with NONE or RELU activation)\n";
        error_msg += "  - SOFTMAX\n";
        error_msg += "  - RELU\n";
        error_msg += "  - CONV_2D (with NONE or RELU activation)\n";
        error_msg += "  - MAX_POOL_2D\n";
        error_msg += "  - SHAPE\n";
        error_msg += "  - STRIDED_SLICE\n";
        error_msg += "  - PACK\n";
        error_msg += "  - RESHAPE\n";
        error_msg += "  - ADD (with NONE or RELU activation)\n";
        error_msg += "  - DROPOUT (custom operator, no-op during inference)\n";
        error_msg += "  - FLATTEN (custom operator, converts to reshape)\n";
        
        if (!unsupported_ops.empty()) {
            error_msg += std::format("\n{} unsupported operator(s) found in model:\n", unsupported_ops.size());
            error_msg += "  (Operator indices are 0-based, in execution order)\n";
            for (const auto& [op_idx, op_details] : unsupported_ops) {
                error_msg += std::format("  [Operator {}] {}\n", op_idx, op_details);
            }
            error_msg += "\n  Note: These operators appear early in the model graph.\n";
            error_msg += "        The code generator cannot skip them as they may transform\n";
            error_msg += "        tensor shapes or data that subsequent operations depend on.\n";
        }
        
        if (!unsupported_activations.empty()) {
            error_msg += "\nUnsupported fused activations found:\n";
            for (const auto& [op_idx, act_code] : unsupported_activations) {
                error_msg += std::format("  - Operator {} has unsupported activation code: {} (only NONE=0 and RELU=1 are supported)\n", 
                    op_idx, act_code);
            }
        }
        
        error_msg += "\nCode generation aborted.\n";
        error_msg += "\nTo fix this issue:\n";
        error_msg += "  1. Use a model that only contains supported operations, OR\n";
        error_msg += "  2. Request support for the missing operators to be added to the code generator\n";
        error_msg += "\nThe model structure cannot be partially generated - all operations\n";
        error_msg += "must be supported for correct code generation.\n";
        error_msg += "================================================\n";
        
        throw ModelValidationError(error_msg);
    }
}

// Escape identifier for C++
std::string EscapeIdentifier(std::string_view name) {
    std::string result;
    result.reserve(name.size());  // Reserve space for efficiency
    
    // Use ranges::transform to map characters
    std::ranges::transform(name, std::back_inserter(result),
        [](char c) -> char {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || 
                (c >= '0' && c <= '9') || c == '_') {
                return c;
            } else {
                return '_';
            }
        });
    
    return result;
}

// Create mapping from tensor index to weight variable name
std::map<int, std::string> CreateWeightMapping(
    const tflite::Model* model,
    const tflite::SubGraph* subgraph
) {
    std::map<int, std::string> tensor_to_weight;
    const auto* tensors = subgraph->tensors();
    const auto* buffers = model->buffers();
    
    if (!tensors || !buffers) {
        return tensor_to_weight;
    }
    
    int weight_index = 0;
    
    for (std::size_t i = 0; i < tensors->size(); ++i) {
        const tflite::Tensor* tensor = tensors->Get(i);
        if (!tensor) continue;
        
        // Weight tensors have a buffer index > 0 (0 is typically empty/unused)
        uint32_t buffer_index = tensor->buffer();
        if (buffer_index == 0 || buffer_index >= buffers->size()) {
            continue;
        }
        
        const tflite::Buffer* buffer = buffers->Get(buffer_index);
        if (!buffer || !buffer->data()) {
            continue;
        }
        
        // Only process FLOAT32 weights
        if (tensor->type() == tflite::TensorType_FLOAT32) {
            std::string tensor_name = tensor->name() ? tensor->name()->c_str() : std::format("tensor_{}", i);
            tensor_to_weight[static_cast<int>(i)] = std::format("weight_{}_{}", weight_index, EscapeIdentifier(tensor_name));
            weight_index++;
        }
    }
    
    return tensor_to_weight;
}

// Generate model_weights.cpp
void GenerateWeightsFile(
    const std::string& output_path,
    const std::string& base_name,
    const tflite::Model* model,
    const tflite::SubGraph* subgraph,
    const std::map<int, std::string>& tensor_to_weight
) {
    inja::Environment env;
    
    const auto* tensors = subgraph->tensors();
    const auto* buffers = model->buffers();
    
    if (!tensors || !buffers) {
        // Create empty file
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        out << "// " << base_name << "_weights.cpp\n";
        out << "// Auto-generated weight data for embedded inference\n\n";
        out << "#include <cstddef>\n\n";
        out << "namespace embedded_ml {\n\n";
        out << "} // namespace embedded_ml\n";
        std::cout << std::format("Generated: {}\n", output_path);
        return;
    }
    
    nlohmann::json data;
    data["base_name"] = base_name;
    data["weights"] = nlohmann::json::array();
    
    for (std::size_t i = 0; i < tensors->size(); ++i) {
        if (!tensor_to_weight.contains(static_cast<int>(i))) {
            continue;
        }
        
        const tflite::Tensor* tensor = tensors->Get(i);
        if (!tensor) {
            continue;
        }
        
        uint32_t buffer_index = tensor->buffer();
        if (buffer_index == 0 || buffer_index >= buffers->size()) {
            continue;
        }
        
        const tflite::Buffer* buffer = buffers->Get(buffer_index);
        if (!buffer || !buffer->data()) {
            continue;
        }
        
        const std::size_t num_elements = CalculateTensorSize(tensor->shape());
        std::string tensor_name = tensor->name() ? tensor->name()->c_str() : std::format("tensor_{}", i);
        std::string var_name = tensor_to_weight.at(static_cast<int>(i));
        
        // Get buffer data
        const flatbuffers::Vector<uint8_t>* data_vec = buffer->data();
        if (!data_vec || data_vec->size() < num_elements * sizeof(float)) {
            std::cerr << std::format("Warning: Buffer size mismatch for tensor {}\n", i);
            continue;
        }
        
        const float* float_data = reinterpret_cast<const float*>(data_vec->data());
        
        nlohmann::json weight;
        weight["tensor_index"] = static_cast<int>(i);
        weight["tensor_name"] = tensor_name;
        weight["var_name"] = var_name;
        weight["shape"] = GetShapeString(tensor->shape());
        weight["num_elements"] = num_elements;
        
        // Format float values with precision and track newline positions
        std::ostringstream value_stream;
        value_stream << std::fixed << std::setprecision(9);
        weight["values"] = nlohmann::json::array();
        for (std::size_t j = 0; j < num_elements; ++j) {
            nlohmann::json value_obj;
            value_stream.str("");
            value_stream << float_data[j];
            value_obj["value"] = value_stream.str();
            value_obj["needs_newline"] = ((j + 1) % 8 == 0);
            value_obj["is_last"] = (j == num_elements - 1);
            weight["values"].push_back(value_obj);
        }
        
        data["weights"].push_back(weight);
    }
    
    // Load and render template
    std::ifstream template_file("templates/weights.cpp.inja");
    if (!template_file) {
        throw CodeGenerationError("Cannot open template file templates/weights.cpp.inja");
    }
    
    std::string template_content((std::istreambuf_iterator<char>(template_file)),
                           std::istreambuf_iterator<char>());
    
    try {
        inja::Template temp = env.parse(template_content);
        std::string result = env.render(temp, data);
        
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        
        out << result;
        std::cout << std::format("Generated: {}\n", output_path);
    } catch (const std::exception& e) {
        throw CodeGenerationError(
            std::format("Template rendering failed: {}", e.what())
        );
    }
}

// Generate model.h (header file)
void GenerateModelHeader(
    const std::string& output_path,
    const std::string& base_name,
    std::size_t input_size,
    std::size_t output_size,
    const std::vector<std::pair<int, std::size_t>>& intermediate_buffers
) {
    inja::Environment env;
    
    // Create include guard name (uppercase, with underscores)
    std::string guard_name = base_name;
    std::transform(guard_name.begin(), guard_name.end(), guard_name.begin(), ::toupper);
    for (std::size_t i = 0; i < guard_name.length(); ++i) {
        if (guard_name[i] == '-' || guard_name[i] == '.') {
            guard_name[i] = '_';
        }
    }
    guard_name += "_MODEL_H";
    
    nlohmann::json data;
    data["base_name"] = base_name;
    data["guard_name"] = guard_name;
    data["input_size"] = input_size;
    data["output_size"] = output_size;
    data["intermediate_buffers"] = nlohmann::json::array();
    
    for (const auto& [idx, size] : intermediate_buffers) {
        nlohmann::json buffer;
        buffer["index"] = idx;
        buffer["size"] = size;
        data["intermediate_buffers"].push_back(buffer);
    }
    
    // Load and render template
    std::ifstream template_file("templates/model.h.inja");
    if (!template_file) {
        throw CodeGenerationError("Cannot open template file templates/model.h.inja");
    }
    
    std::string template_content((std::istreambuf_iterator<char>(template_file)),
                           std::istreambuf_iterator<char>());
    
    try {
        inja::Template temp = env.parse(template_content);
        std::string result = env.render(temp, data);
        
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        
        out << result;
        std::cout << std::format("Generated: {}\n", output_path);
    } catch (const std::exception& e) {
        throw CodeGenerationError(
            std::format("Template rendering failed: {}", e.what())
        );
    }
}

// Convert FlatBuffer activation to TfLite activation enum
constexpr TfLiteFusedActivation ConvertActivation(tflite::ActivationFunctionType activation) {
    switch (activation) {
        case tflite::ActivationFunctionType_NONE: return kTfLiteActNone;
        case tflite::ActivationFunctionType_RELU: return kTfLiteActRelu;
        case tflite::ActivationFunctionType_RELU_N1_TO_1: return kTfLiteActReluN1To1;
        case tflite::ActivationFunctionType_RELU6: return kTfLiteActRelu6;
        case tflite::ActivationFunctionType_TANH: return kTfLiteActTanh;
        case tflite::ActivationFunctionType_SIGN_BIT: return kTfLiteActSignBit;
        default: return kTfLiteActNone;
    }
}

// Convert FlatBuffer padding to TfLite padding enum
constexpr TfLitePadding ConvertPadding(tflite::Padding padding) {
    switch (padding) {
        case tflite::Padding_SAME: return kTfLitePaddingSame;
        case tflite::Padding_VALID: return kTfLitePaddingValid;
        default: return kTfLitePaddingSame;
    }
}

// Generate model.cpp
void GenerateModelFile(
    const std::string& output_path,
    const std::string& base_name,
    const tflite::Model* model,
    const tflite::SubGraph* subgraph,
    const std::map<int, std::string>& tensor_to_weight,
    const std::vector<std::pair<int, std::size_t>>& intermediate_buffers,
    const std::map<int, std::size_t>& tensor_sizes,
    int32_t input_tensor_idx,
    int32_t output_tensor_idx
) {
    inja::Environment env;
    
    std::vector<std::string> generation_errors;
    
    const auto* operators = subgraph->operators();
    const auto* operator_codes = model->operator_codes();
    const auto* tensors = subgraph->tensors();
    
    if (!operators || !operator_codes || !tensors) {
        throw CodeGenerationError("Invalid model structure");
    }
    
    // Check which components are needed
    bool has_standalone_relu = false;
    bool has_conv2d = false;
    bool has_max_pool2d = false;
    bool has_shape = false;
    bool has_strided_slice = false;
    bool has_pack = false;
    bool has_reshape = false;
    bool has_add = false;
    bool has_dropout = false;
    bool has_flatten = false;
    
    // Build JSON data structure
    nlohmann::json data;
    data["base_name"] = base_name;
    data["has_standalone_relu"] = false;
    data["has_conv2d"] = false;
    data["has_max_pool2d"] = false;
    data["has_shape"] = false;
    data["has_strided_slice"] = false;
    data["has_pack"] = false;
    data["has_reshape"] = false;
    data["has_add"] = false;
    data["has_dropout"] = false;
    data["has_flatten"] = false;
    data["layers"] = nlohmann::json::array();
    data["intermediate_buffers"] = nlohmann::json::array();
    
    // Process each operator in order (this is the execution plan)
    for (std::size_t i = 0; i < operators->size(); ++i) {
        const tflite::Operator* op = operators->Get(i);
        if (!op) continue;
        
        int op_code_index = op->opcode_index();
        if (op_code_index < 0 || op_code_index >= static_cast<int>(operator_codes->size())) {
            continue;
        }
        
        const tflite::OperatorCode* op_code = operator_codes->Get(op_code_index);
        if (!op_code) continue;
        
        std::string op_name = GetOperatorName(op_code);
        
        // Track which components are needed
        if (op_name == "RELU") {
            has_standalone_relu = true;
            data["has_standalone_relu"] = true;
        } else if (op_name == "CONV_2D") {
            has_conv2d = true;
            data["has_conv2d"] = true;
        } else if (op_name == "MAX_POOL_2D") {
            has_max_pool2d = true;
            data["has_max_pool2d"] = true;
        } else if (op_name == "SHAPE") {
            has_shape = true;
            data["has_shape"] = true;
        } else if (op_name == "STRIDED_SLICE") {
            has_strided_slice = true;
            data["has_strided_slice"] = true;
        } else if (op_name == "PACK") {
            has_pack = true;
            data["has_pack"] = true;
        } else if (op_name == "RESHAPE") {
            has_reshape = true;
            data["has_reshape"] = true;
        } else if (op_name == "ADD") {
            has_add = true;
            data["has_add"] = true;
        } else if (op_name.find("DROPOUT") != std::string::npos || 
                   op_name.find("Dropout") != std::string::npos) {
            has_dropout = true;
            data["has_dropout"] = true;
            op_name = "DROPOUT";  // Normalize name
        } else if (op_name.find("FLATTEN") != std::string::npos ||
                   op_name.find("Flatten") != std::string::npos) {
            has_flatten = true;
            data["has_flatten"] = true;
            op_name = "FLATTEN";  // Normalize name
        }
        
        // Get input/output tensor indices
        const auto* op_inputs = op->inputs();
        const auto* op_outputs = op->outputs();
        
        int op_input_tensor_idx = -1;
        int op_output_tensor_idx = -1;
        
        if (op_inputs && op_inputs->size() > 0) {
            op_input_tensor_idx = op_inputs->Get(0);
        }
        if (op_outputs && op_outputs->size() > 0) {
            op_output_tensor_idx = op_outputs->Get(0);
        }
        
        if (op_input_tensor_idx < 0 || op_output_tensor_idx < 0) {
            continue;
        }
        
        // Determine input/output pointers
        std::string input_ptr, output_ptr;
        if (op_input_tensor_idx == input_tensor_idx) {
            input_ptr = "input";
        } else {
            input_ptr = std::format("buffer_{}", op_input_tensor_idx);
        }
        
        if (op_output_tensor_idx == output_tensor_idx) {
            output_ptr = "output";
        } else {
            output_ptr = std::format("buffer_{}", op_output_tensor_idx);
        }
        
        std::size_t input_size_layer = tensor_sizes.count(op_input_tensor_idx) ? tensor_sizes.at(op_input_tensor_idx) : 0;
        std::size_t output_size_layer = tensor_sizes.count(op_output_tensor_idx) ? tensor_sizes.at(op_output_tensor_idx) : 0;
        
        // Build layer JSON based on operation type
        nlohmann::json layer;
        layer["index"] = static_cast<int>(i);
        layer["op_name"] = op_name;
        layer["input_ptr"] = input_ptr;
        layer["output_ptr"] = output_ptr;
        layer["input_size"] = input_size_layer;
        layer["output_size"] = output_size_layer;
        
        // Generate code based on operation
        if (op_name == "FULLY_CONNECTED") {
            // Find weights and bias
            int weights_tensor_idx = -1;
            int bias_tensor_idx = -1;
            
            if (op_inputs && op_inputs->size() >= 3) {
                weights_tensor_idx = op_inputs->Get(1);
                bias_tensor_idx = op_inputs->Get(2);
            }
            
            if (weights_tensor_idx >= 0 && bias_tensor_idx >= 0) {
                if (!tensor_to_weight.contains(weights_tensor_idx) ||
                    !tensor_to_weight.contains(bias_tensor_idx)) {
                    std::string error_msg = std::format(
                        "FULLY_CONNECTED layer {} missing required weight/bias tensors (weights_idx={}, bias_idx={})",
                        i, weights_tensor_idx, bias_tensor_idx
                    );
                    generation_errors.push_back(error_msg);
                    continue;
                }
                
                std::string weights_var = tensor_to_weight.at(weights_tensor_idx);
                std::string bias_var = tensor_to_weight.at(bias_tensor_idx);
                
                // Determine actual input size from weights shape
                std::size_t fc_input_size = input_size_layer;
                if (weights_tensor_idx >= 0 && weights_tensor_idx < static_cast<int>(tensors->size())) {
                    const tflite::Tensor* weights_tensor = tensors->Get(weights_tensor_idx);
                    if (weights_tensor && weights_tensor->shape() && weights_tensor->shape()->size() >= 2) {
                        fc_input_size = weights_tensor->shape()->Get(1); // weights shape: [output, input]
                    }
                }
                
                // Check for fused activation function
                TfLiteFusedActivation fused_activation = kTfLiteActNone;
                const tflite::FullyConnectedOptions* options = 
                    op->builtin_options_as_FullyConnectedOptions();
                if (options) {
                    fused_activation = ConvertActivation(options->fused_activation_function());
                }
                
                // Determine activation parameter
                std::string activation_param = "ActivationType::NONE";
                if (fused_activation == kTfLiteActRelu) {
                    activation_param = "ActivationType::RELU";
                } else if (fused_activation != kTfLiteActNone) {
                    std::cerr << std::format("Warning: Unsupported fused activation {} in FULLY_CONNECTED layer {} (only ReLU supported, using NONE)\n",
                        static_cast<int>(fused_activation), i);
                }
                
                layer["weights_var"] = weights_var;
                layer["bias_var"] = bias_var;
                layer["fc_input_size"] = fc_input_size;
                layer["activation_param"] = activation_param;
            } else {
                std::string error_msg = std::format(
                    "FULLY_CONNECTED layer {} has insufficient inputs (expected at least 3: input, weights, bias)",
                    i
                );
                generation_errors.push_back(error_msg);
                continue;
            }
        } else if (op_name == "RELU" || op_name == "SOFTMAX") {
            // Simple operations, already have all needed fields
        } else if (op_name == "CONV_2D") {
            // CONV_2D has inputs: [input, filter, bias]
            int filter_tensor_idx = -1;
            int bias_tensor_idx = -1;
            
            if (op_inputs && op_inputs->size() >= 3) {
                filter_tensor_idx = op_inputs->Get(1);
                bias_tensor_idx = op_inputs->Get(2);
            }
            
            if (filter_tensor_idx >= 0 && bias_tensor_idx >= 0) {
                if (!tensor_to_weight.contains(filter_tensor_idx) ||
                    !tensor_to_weight.contains(bias_tensor_idx)) {
                    std::cerr << std::format("Warning: Cannot find filter/bias tensors for CONV_2D layer {}\n", i);
                    continue;
                }
                
                std::string filter_var = tensor_to_weight.at(filter_tensor_idx);
                std::string bias_var = tensor_to_weight.at(bias_tensor_idx);
                
                // Get tensor shapes
                const tflite::Tensor* input_tensor = (op_input_tensor_idx >= 0 && op_input_tensor_idx < static_cast<int>(tensors->size())) 
                    ? tensors->Get(op_input_tensor_idx) : nullptr;
                const tflite::Tensor* filter_tensor = (filter_tensor_idx >= 0 && filter_tensor_idx < static_cast<int>(tensors->size()))
                    ? tensors->Get(filter_tensor_idx) : nullptr;
                const tflite::Tensor* output_tensor = (op_output_tensor_idx >= 0 && op_output_tensor_idx < static_cast<int>(tensors->size()))
                    ? tensors->Get(op_output_tensor_idx) : nullptr;
                
                if (!input_tensor || !filter_tensor || !output_tensor ||
                    !input_tensor->shape() || !filter_tensor->shape() || !output_tensor->shape()) {
                    std::cerr << std::format("Warning: Cannot get tensor shapes for CONV_2D layer {}\n", i);
                    continue;
                }
                
                // Extract dimensions (NHWC format)
                std::size_t batch_size = input_tensor->shape()->Get(0);
                std::size_t input_height = input_tensor->shape()->Get(1);
                std::size_t input_width = input_tensor->shape()->Get(2);
                std::size_t input_channels = input_tensor->shape()->Get(3);
                std::size_t filter_height = filter_tensor->shape()->Get(1);
                std::size_t filter_width = filter_tensor->shape()->Get(2);
                std::size_t output_channels = filter_tensor->shape()->Get(0);
                
                // Get convolution parameters
                TfLitePadding padding = kTfLitePaddingSame;
                int stride_height = 1;
                int stride_width = 1;
                TfLiteFusedActivation fused_activation = kTfLiteActNone;
                int dilation_height = 1;
                int dilation_width = 1;
                
                const tflite::Conv2DOptions* options = op->builtin_options_as_Conv2DOptions();
                if (options) {
                    padding = ConvertPadding(options->padding());
                    stride_height = options->stride_h();
                    stride_width = options->stride_w();
                    fused_activation = ConvertActivation(options->fused_activation_function());
                    dilation_height = options->dilation_h_factor();
                    dilation_width = options->dilation_w_factor();
                }
                
                std::string padding_param = (padding == kTfLitePaddingSame) ? 
                    "PaddingType::SAME" : "PaddingType::VALID";
                std::string activation_param = (fused_activation == kTfLiteActRelu) ? 
                    "ActivationType::RELU" : "ActivationType::NONE";
                
                layer["filter_var"] = filter_var;
                layer["bias_var"] = bias_var;
                layer["batch_size"] = batch_size;
                layer["input_height"] = input_height;
                layer["input_width"] = input_width;
                layer["input_channels"] = input_channels;
                layer["filter_height"] = filter_height;
                layer["filter_width"] = filter_width;
                layer["output_channels"] = output_channels;
                layer["stride_height"] = stride_height;
                layer["stride_width"] = stride_width;
                layer["padding_param"] = padding_param;
                layer["activation_param"] = activation_param;
                layer["dilation_height"] = dilation_height;
                layer["dilation_width"] = dilation_width;
            } else {
                std::cerr << std::format("Warning: CONV_2D layer {} missing filter/bias\n", i);
                continue;
            }
        } else if (op_name == "MAX_POOL_2D") {
            // Get tensor shapes
            const tflite::Tensor* input_tensor = (op_input_tensor_idx >= 0 && op_input_tensor_idx < static_cast<int>(tensors->size()))
                ? tensors->Get(op_input_tensor_idx) : nullptr;
            const tflite::Tensor* output_tensor = (op_output_tensor_idx >= 0 && op_output_tensor_idx < static_cast<int>(tensors->size()))
                ? tensors->Get(op_output_tensor_idx) : nullptr;
            
            if (!input_tensor || !output_tensor ||
                !input_tensor->shape() || !output_tensor->shape()) {
                std::cerr << std::format("Warning: Cannot get tensor shapes for MAX_POOL_2D layer {}\n", i);
                continue;
            }
            
            // Extract dimensions (NHWC format)
            std::size_t batch_size = input_tensor->shape()->Get(0);
            std::size_t input_height = input_tensor->shape()->Get(1);
            std::size_t input_width = input_tensor->shape()->Get(2);
            std::size_t channels = input_tensor->shape()->Get(3);
            
            // Get pooling parameters
            TfLitePadding padding = kTfLitePaddingSame;
            int stride_height = 1;
            int stride_width = 1;
            int filter_height = 2;
            int filter_width = 2;
            TfLiteFusedActivation fused_activation = kTfLiteActNone;
            
            const tflite::Pool2DOptions* options = op->builtin_options_as_Pool2DOptions();
            if (options) {
                padding = ConvertPadding(options->padding());
                stride_height = options->stride_h();
                stride_width = options->stride_w();
                filter_height = options->filter_height();
                filter_width = options->filter_width();
                fused_activation = ConvertActivation(options->fused_activation_function());
            }
            
            std::string padding_param = (padding == kTfLitePaddingSame) ? 
                "PaddingType::SAME" : "PaddingType::VALID";
            std::string activation_param = (fused_activation == kTfLiteActRelu) ? 
                "ActivationType::RELU" : "ActivationType::NONE";
            
            layer["batch_size"] = batch_size;
            layer["input_height"] = input_height;
            layer["input_width"] = input_width;
            layer["channels"] = channels;
            layer["filter_height"] = filter_height;
            layer["filter_width"] = filter_width;
            layer["stride_height"] = stride_height;
            layer["stride_width"] = stride_width;
            layer["padding_param"] = padding_param;
            layer["activation_param"] = activation_param;
        } else if (op_name == "SHAPE") {
            // SHAPE extracts the shape of the input tensor
            const tflite::Tensor* input_tensor = (op_input_tensor_idx >= 0 && op_input_tensor_idx < static_cast<int>(tensors->size()))
                ? tensors->Get(op_input_tensor_idx) : nullptr;
            
            if (!input_tensor || !input_tensor->shape()) {
                std::cerr << std::format("Warning: Cannot get tensor shapes for SHAPE layer {}\n", i);
                continue;
            }
            
            int num_dims = input_tensor->shape()->size();
            std::ostringstream shape_values;
            for (int j = 0; j < num_dims; ++j) {
                shape_values << input_tensor->shape()->Get(j);
                if (j < num_dims - 1) shape_values << ", ";
            }
            
            layer["num_dims"] = num_dims;
            layer["shape_values"] = shape_values.str();
        } else if (op_name == "STRIDED_SLICE" || op_name == "PACK") {
            // Simplified implementations - already have basic fields
            if (op_name == "PACK") {
                int axis = 0;
                const tflite::PackOptions* options = op->builtin_options_as_PackOptions();
                if (options) {
                    axis = options->axis();
                }
                layer["axis"] = axis;
            }
        } else if (op_name == "RESHAPE" || op_name == "FLATTEN") {
            // Already have all needed fields
        } else if (op_name == "ADD") {
            // ADD has two inputs: [input1, input2]
            if (!op_inputs || op_inputs->size() < 2) {
                std::string error_msg = std::format(
                    "ADD layer {} missing inputs: expected 2 inputs, but {}",
                    i, !op_inputs ? "no input list provided" : std::format("only {} provided", op_inputs->size())
                );
                generation_errors.push_back(error_msg);
                continue;
            }
            
            int input1_tensor_idx = op_inputs->Get(0);
            int input2_tensor_idx = op_inputs->Get(1);
            
            if (input1_tensor_idx < 0 || input2_tensor_idx < 0) {
                std::string error_msg = std::format(
                    "ADD layer {} missing inputs: input1={}, input2={} (negative indices indicate optional/unused tensors, which are not supported)",
                    i, input1_tensor_idx, input2_tensor_idx
                );
                generation_errors.push_back(error_msg);
                continue;
            }
            
            // Determine input pointers
            std::string input1_ptr, input2_ptr;
            
            if (input1_tensor_idx == input_tensor_idx) {
                input1_ptr = "input";
            } else if (tensor_to_weight.contains(input1_tensor_idx)) {
                input1_ptr = tensor_to_weight.at(input1_tensor_idx);
            } else {
                input1_ptr = std::format("buffer_{}", input1_tensor_idx);
            }
            
            if (input2_tensor_idx == input_tensor_idx) {
                input2_ptr = "input";
            } else if (tensor_to_weight.contains(input2_tensor_idx)) {
                input2_ptr = tensor_to_weight.at(input2_tensor_idx);
            } else {
                input2_ptr = std::format("buffer_{}", input2_tensor_idx);
            }
            
            std::size_t add_size = output_size_layer;
            
            // Check for fused activation function
            TfLiteFusedActivation fused_activation = kTfLiteActNone;
            const tflite::AddOptions* options = op->builtin_options_as_AddOptions();
            if (options) {
                fused_activation = ConvertActivation(options->fused_activation_function());
            }
            
            std::string activation_param = "ActivationType::NONE";
            if (fused_activation == kTfLiteActRelu) {
                activation_param = "ActivationType::RELU";
            } else if (fused_activation != kTfLiteActNone) {
                std::cerr << std::format("Warning: Unsupported fused activation {} in ADD layer {} (only ReLU supported, using NONE)\n",
                    static_cast<int>(fused_activation), i);
            }
            
            layer["input1_ptr"] = input1_ptr;
            layer["input2_ptr"] = input2_ptr;
            layer["add_size"] = add_size;
            layer["activation_param"] = activation_param;
        } else if (op_name == "DROPOUT") {
            float dropout_rate = 0.0f;
            layer["dropout_rate"] = dropout_rate;
        } else {
            std::cerr << std::format("Warning: Unsupported operation {}\n", op_name);
            continue;
        }
        
        data["layers"].push_back(layer);
    }
    
    // Add intermediate buffers
    for (const auto& [idx, size] : intermediate_buffers) {
        nlohmann::json buffer;
        buffer["index"] = idx;
        buffer["size"] = size;
        data["intermediate_buffers"].push_back(buffer);
    }
    
    // Check for generation errors before rendering
    if (!generation_errors.empty()) {
        std::string error_msg = "\n================================================\n";
        error_msg += "ERROR: Code generation failed due to model structure issues!\n";
        error_msg += "================================================\n\n";
        error_msg += "The following errors occurred during code generation:\n";
        for (const auto& error : generation_errors) {
            error_msg += std::format("  - {}\n", error);
        }
        error_msg += "\nCode generation aborted.\n";
        error_msg += "The generated file may be incomplete or incorrect.\n";
        error_msg += "================================================\n";
        throw CodeGenerationError(error_msg);
    }
    
    // Load and render template
    std::ifstream template_file("templates/model.cpp.inja");
    if (!template_file) {
        throw CodeGenerationError("Cannot open template file templates/model.cpp.inja");
    }
    
    std::string template_content((std::istreambuf_iterator<char>(template_file)),
                           std::istreambuf_iterator<char>());
    
    try {
        inja::Template temp = env.parse(template_content);
        std::string result = env.render(temp, data);
        
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        
        out << result;
        std::cout << std::format("Generated: {}\n", output_path);
    } catch (const std::exception& e) {
        throw CodeGenerationError(
            std::format("Template rendering failed: {}", e.what())
        );
    }
}

// Generate inference.cpp
void GenerateInferenceFile(
    const std::string& output_path,
    const std::string& base_name,
    const tflite::SubGraph* subgraph,
    int32_t input_tensor_idx,
    int32_t output_tensor_idx
) {
    inja::Environment env;
    
    const auto* tensors = subgraph->tensors();
    if (!tensors) {
        throw CodeGenerationError("Cannot access tensors");
    }
    
    const tflite::Tensor* input_tensor = (input_tensor_idx >= 0 && input_tensor_idx < static_cast<int>(tensors->size()))
        ? tensors->Get(input_tensor_idx) : nullptr;
    const tflite::Tensor* output_tensor = (output_tensor_idx >= 0 && output_tensor_idx < static_cast<int>(tensors->size()))
        ? tensors->Get(output_tensor_idx) : nullptr;
    
    if (!input_tensor || !output_tensor) {
        throw CodeGenerationError("Cannot access input/output tensors");
    }
    
    std::size_t input_size = CalculateTensorSize(input_tensor->shape());
    std::size_t output_size = CalculateTensorSize(output_tensor->shape());
    
    nlohmann::json data;
    data["base_name"] = base_name;
    data["input_size"] = input_size;
    data["output_size"] = output_size;
    data["input_shape"] = GetShapeString(input_tensor->shape());
    data["output_shape"] = GetShapeString(output_tensor->shape());
    
    // Create array for iteration with newline info (Inja doesn't support range() function)
    data["input_indices"] = nlohmann::json::array();
    for (std::size_t i = 0; i < input_size; ++i) {
        nlohmann::json idx_obj;
        idx_obj["needs_newline"] = ((i + 1) % 8 == 0);
        idx_obj["is_last"] = (i == input_size - 1);
        data["input_indices"].push_back(idx_obj);
    }
    
    // Load and render template
    std::ifstream template_file("templates/inference.cpp.inja");
    if (!template_file) {
        throw CodeGenerationError("Cannot open template file templates/inference.cpp.inja");
    }
    
    std::string template_content((std::istreambuf_iterator<char>(template_file)),
                           std::istreambuf_iterator<char>());
    
    try {
        inja::Template temp = env.parse(template_content);
        std::string result = env.render(temp, data);
        
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        
        out << result;
        std::cout << std::format("Generated: {}\n", output_path);
    } catch (const std::exception& e) {
        throw CodeGenerationError(
            std::format("Template rendering failed: {}", e.what())
        );
    }
}

// Generate Makefile for the generated code
void GenerateMakefile(
    const std::string& output_path,
    const std::string& base_name
) {
    inja::Environment env;
    
    nlohmann::json data;
    data["base_name"] = base_name;
    
    // Load and render template
    std::ifstream template_file("templates/Makefile.inja");
    if (!template_file) {
        throw CodeGenerationError("Cannot open template file templates/Makefile.inja");
    }
    
    std::string template_content((std::istreambuf_iterator<char>(template_file)),
                           std::istreambuf_iterator<char>());
    
    try {
        inja::Template temp = env.parse(template_content);
        std::string result = env.render(temp, data);
        
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        
        out << result;
        std::cout << std::format("Generated: {}\n", output_path);
    } catch (const std::exception& e) {
        throw CodeGenerationError(
            std::format("Template rendering failed: {}", e.what())
        );
    }
}

// Create directory if it doesn't exist
void CreateDirectory(const std::string& path) {
    namespace fs = std::filesystem;
    try {
        if (fs::exists(path)) {
            if (!fs::is_directory(path)) {
                throw FileSystemError(
                    std::format("Path exists but is not a directory: {}", path)
                );
            }
            // Directory already exists, nothing to do
            return;
        }
        
        // Create directory and all parent directories
        fs::create_directories(path);
    } catch (const fs::filesystem_error& e) {
        throw FileSystemError(
            std::format("Failed to create directory {}: {}", path, e.what())
        );
    }
}

// Recursively delete a directory and all its contents
void DeleteDirectory(const std::string& path) {
    namespace fs = std::filesystem;
    try {
        if (!fs::exists(path)) {
            // Directory doesn't exist, consider it "deleted"
            return;
        }
        
        if (!fs::is_directory(path)) {
            throw FileSystemError(
                std::format("Path is not a directory: {}", path)
            );
        }
        
        // Remove directory and all contents
        fs::remove_all(path);
    } catch (const fs::filesystem_error& e) {
        throw FileSystemError(
            std::format("Failed to delete directory {}: {}", path, e.what())
        );
    }
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.tflite> <base_name> [--output-index=N]" << std::endl;
        std::cerr << "Example: " << argv[0] << " scripts/model.tflite my_model" << std::endl;
        std::cerr << "Example (multi-output): " << argv[0] << " scripts/model.tflite my_model --output-index=1" << std::endl;
        std::cerr << "This will create a directory 'my_model/' containing:" << std::endl;
        std::cerr << "  - my_model_weights.cpp" << std::endl;
        std::cerr << "  - my_model.h" << std::endl;
        std::cerr << "  - my_model.cpp" << std::endl;
        std::cerr << "  - my_model_inference.cpp" << std::endl;
        std::cerr << "  - Makefile" << std::endl;
        std::cerr << "\nTo build: cd my_model && make" << std::endl;
        std::cerr << "\nNote: For models with multiple outputs, use --output-index=N to select" << std::endl;
        std::cerr << "      which output tensor to use (default: 0). Only single input models are supported." << std::endl;
        return 1;
    }
    
    const std::string model_path = argv[1];
    const std::string base_name = argv[2];
    int output_index = 0;  // Default to first output
    bool output_index_specified = false;  // Track if user explicitly provided the flag
    
    // Parse optional output index flag
    if (argc == 4) {
        std::string flag = argv[3];
        if (flag.find("--output-index") == 0) {
            std::size_t eq_pos = flag.find('=');
            if (eq_pos != std::string::npos) {
                // Format: --output-index=N
                std::string index_str = flag.substr(eq_pos + 1);
                try {
                    output_index = std::stoi(index_str);
                    output_index_specified = true;
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid output index: " << index_str << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: --output-index requires a value, e.g., --output-index=1" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown flag: " << flag << std::endl;
            std::cerr << "Use --output-index=N to specify output tensor index" << std::endl;
            return 1;
        }
    }
    
    // Check if file exists using filesystem
    namespace fs = std::filesystem;
    if (!fs::exists(model_path)) {
        std::cerr << std::format("Error: File does not exist: {}\n", model_path);
        return 1;
    }
    
    std::cout << std::format("Loading TFLite model from: {}\n", model_path);
    
    // Read the entire file into memory
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << std::format("Error: Failed to open file: {}\n", model_path);
        return 1;
    }
    
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> buffer(file_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        std::cerr << std::format("Error: Failed to read file: {}\n", model_path);
        return 1;
    }
    
    std::cout << std::format("File read successfully ({} bytes)\n", file_size);
    
    // Verify FlatBuffer
    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!tflite::VerifyModelBuffer(verifier)) {
        std::cerr << "Error: Invalid FlatBuffer format\n";
        return 1;
    }
    
    // Get the model from the FlatBuffer
    const tflite::Model* model = tflite::GetModel(buffer.data());
    if (!model) {
        std::cerr << "Error: Failed to parse model from FlatBuffer\n";
        return 1;
    }
    
    std::cout << "Model loaded successfully!\n";
    
    try {
        // Validate model schema
        std::cout << "Validating model operations..." << std::endl;
        ValidateModelSchema(model);
        std::cout << "Model validation passed - all operations are supported!" << std::endl;
    
        // Get the main subgraph
        const auto* subgraphs = model->subgraphs();
        if (!subgraphs || subgraphs->size() == 0) {
            throw ModelValidationError("No subgraphs found in model");
        }
        
        const tflite::SubGraph* subgraph = subgraphs->Get(0);
        if (!subgraph) {
            throw ModelValidationError("Subgraph is null");
        }
        
        const auto* inputs = subgraph->inputs();
        const auto* outputs = subgraph->outputs();
        const auto* tensors = subgraph->tensors();
        const auto* operators = subgraph->operators();
        
        if (!inputs || !outputs || !tensors || inputs->size() == 0 || outputs->size() == 0) {
            throw ModelValidationError("Invalid model structure - missing inputs/outputs");
        }
        
        // Check input count (only single input supported)
        if (inputs->size() != 1) {
            throw ModelValidationError(
                std::format("Only single input models are supported. This model has {} input(s).", inputs->size())
            );
        }
        
        // Require --output-index flag for multi-output models
        if (outputs->size() > 1 && !output_index_specified) {
            std::string error_msg = std::format(
                "Error: This model has {} output(s).\nYou must specify which output to use with --output-index=N\n\nAvailable outputs:\n",
                outputs->size()
            );
            for (std::size_t i = 0; i < outputs->size(); ++i) {
                int32_t out_idx = outputs->Get(i);
                const tflite::Tensor* out_tensor = (out_idx >= 0 && out_idx < static_cast<int>(tensors->size()))
                    ? tensors->Get(out_idx) : nullptr;
                if (out_tensor) {
                    error_msg += std::format("  [{}] Shape: [{}]", i, GetShapeString(out_tensor->shape()));
                    if (out_tensor->name()) {
                        error_msg += std::format(" Name: {}", out_tensor->name()->c_str());
                    }
                    error_msg += "\n";
                }
            }
            error_msg += std::format("\nRerun with:\n  {} {} {} --output-index=<index>\n", argv[0], model_path, base_name);
            throw ModelValidationError(error_msg);
        }
        
        // Check and validate output index
        if (output_index < 0 || output_index >= static_cast<int>(outputs->size())) {
            std::string error_msg = std::format(
                "Error: Invalid output index {}.\nThis model has {} output(s) (valid indices: 0 to {}).\n",
                output_index, outputs->size(), outputs->size() - 1
            );
            if (outputs->size() > 1) {
                error_msg += std::format("\nTo use a different output, rerun with:\n  {} {} {} --output-index=<index>\n", 
                    argv[0], model_path, base_name);
                error_msg += "Available outputs:\n";
                for (std::size_t i = 0; i < outputs->size(); ++i) {
                    int32_t out_idx = outputs->Get(i);
                    const tflite::Tensor* out_tensor = (out_idx >= 0 && out_idx < static_cast<int>(tensors->size()))
                        ? tensors->Get(out_idx) : nullptr;
                    if (out_tensor) {
                        error_msg += std::format("  [{}] Shape: [{}]", i, GetShapeString(out_tensor->shape()));
                        if (out_tensor->name()) {
                            error_msg += std::format(" Name: {}", out_tensor->name()->c_str());
                        }
                        error_msg += "\n";
                    }
                }
            }
            throw ModelValidationError(error_msg);
        }
        
        int32_t input_tensor_idx = inputs->Get(0);
        int32_t output_tensor_idx = outputs->Get(output_index);
        
        if (outputs->size() > 1) {
            std::cout << std::format("Note: Model has {} outputs. Using output index {} (0-indexed).\n", 
                outputs->size(), output_index);
        }
        
        const tflite::Tensor* input_tensor = (input_tensor_idx >= 0 && input_tensor_idx < static_cast<int>(tensors->size()))
            ? tensors->Get(input_tensor_idx) : nullptr;
        const tflite::Tensor* output_tensor = (output_tensor_idx >= 0 && output_tensor_idx < static_cast<int>(tensors->size()))
            ? tensors->Get(output_tensor_idx) : nullptr;
        
        if (!input_tensor || !output_tensor) {
            throw ModelValidationError("Cannot access input/output tensors");
        }
        
        std::size_t input_size = CalculateTensorSize(input_tensor->shape());
        std::size_t output_size = CalculateTensorSize(output_tensor->shape());
    
        // Create output directory
        std::string output_dir = base_name;
        CreateDirectory(output_dir);
        std::cout << std::format("Created directory: {}\n", output_dir);
    
        // Generate files
        // First create weight mapping
        std::map<int, std::string> tensor_to_weight = CreateWeightMapping(model, subgraph);
        
        // Collect intermediate buffers info
        std::map<int, std::size_t> tensor_sizes;
        tensor_sizes[input_tensor_idx] = input_size;
        tensor_sizes[output_tensor_idx] = output_size;
        
        // Process operators to find all tensor sizes using range-based loops
        if (operators) {
            for (std::size_t i = 0; i < operators->size(); ++i) {
                const tflite::Operator* op = operators->Get(i);
                if (!op) continue;
                
                const auto* op_inputs = op->inputs();
                const auto* op_outputs = op->outputs();
                
                if (op_inputs) {
                    for (std::size_t j = 0; j < op_inputs->size(); ++j) {
                        int tensor_idx = op_inputs->Get(j);
                        if (tensor_idx >= 0 && tensor_idx < static_cast<int>(tensors->size()) &&
                            !tensor_sizes.contains(tensor_idx)) {
                            const tflite::Tensor* tensor = tensors->Get(tensor_idx);
                            if (tensor) {
                                tensor_sizes[tensor_idx] = CalculateTensorSize(tensor->shape());
                            }
                        }
                    }
                }
                
                if (op_outputs) {
                    for (std::size_t j = 0; j < op_outputs->size(); ++j) {
                        int tensor_idx = op_outputs->Get(j);
                        if (tensor_idx >= 0 && tensor_idx < static_cast<int>(tensors->size()) &&
                            !tensor_sizes.contains(tensor_idx)) {
                            const tflite::Tensor* tensor = tensors->Get(tensor_idx);
                            if (tensor) {
                                tensor_sizes[tensor_idx] = CalculateTensorSize(tensor->shape());
                            }
                        }
                    }
                }
            }
        }
        
        std::vector<std::pair<int, std::size_t>> intermediate_buffers;
        // Get set of weight tensor indices to exclude them
        std::set<int> weight_tensor_indices;
        for (const auto& [tensor_idx, weight_name] : tensor_to_weight) {
            weight_tensor_indices.insert(tensor_idx);
        }
        
        // Collect all output tensors from operations
        std::set<int> operation_output_tensors;
        if (operators) {
            for (std::size_t i = 0; i < operators->size(); ++i) {
                const tflite::Operator* op = operators->Get(i);
                if (!op) continue;
                
                const auto* op_outputs = op->outputs();
                if (op_outputs) {
                    for (std::size_t j = 0; j < op_outputs->size(); ++j) {
                        int tensor_idx = op_outputs->Get(j);
                        operation_output_tensors.insert(tensor_idx);
                        // Ensure it's in tensor_sizes
                        if (!tensor_sizes.contains(tensor_idx)) {
                            if (tensor_idx >= 0 && tensor_idx < static_cast<int>(tensors->size())) {
                                const tflite::Tensor* tensor = tensors->Get(tensor_idx);
                                tensor_sizes[tensor_idx] = tensor ? 
                                    (CalculateTensorSize(tensor->shape()) > 0 ? CalculateTensorSize(tensor->shape()) : 1) : 1;
                            } else {
                                tensor_sizes[tensor_idx] = 1;
                            }
                        }
                    }
                }
            }
        }
        
        // Now collect all intermediate buffers
        for (const auto& tensor_idx : operation_output_tensors) {
            if (tensor_idx != input_tensor_idx && tensor_idx != output_tensor_idx) {
                // Skip if it's a weight tensor
                if (!weight_tensor_indices.contains(tensor_idx)) {
                    std::size_t size = tensor_sizes.contains(tensor_idx) ? tensor_sizes[tensor_idx] : 1;
                    intermediate_buffers.push_back({tensor_idx, size});
                }
            }
        }
        
        // Also include any other tensors from tensor_sizes that might have been missed
        for (const auto& [tensor_idx, size] : tensor_sizes) {
            if (tensor_idx != input_tensor_idx && tensor_idx != output_tensor_idx) {
                // Skip if already added or if it's a weight
                bool already_added = std::ranges::any_of(intermediate_buffers,
                    [tensor_idx](const auto& buf) { return buf.first == tensor_idx; });
                
                if (!already_added && !weight_tensor_indices.contains(tensor_idx)) {
                    intermediate_buffers.push_back({tensor_idx, size > 0 ? size : 1});
                }
            }
        }
    
        std::string weights_file = std::format("{}/{}_weights.cpp", output_dir, base_name);
        std::string model_header_file = std::format("{}/{}.h", output_dir, base_name);
        std::string model_file = std::format("{}/{}.cpp", output_dir, base_name);
        std::string inference_file = std::format("{}/{}_inference.cpp", output_dir, base_name);
        std::string makefile = std::format("{}/Makefile", output_dir);
        
        std::cout << "\nGenerating code files...\n";
        
        // Generate weights file
        GenerateWeightsFile(weights_file, base_name, model, subgraph, tensor_to_weight);
        
        // Generate model header
        GenerateModelHeader(model_header_file, base_name, input_size, output_size, intermediate_buffers);
        
        // Generate model file (most critical - can fail)
        GenerateModelFile(model_file, base_name, model, subgraph, tensor_to_weight, intermediate_buffers, 
                           tensor_sizes, input_tensor_idx, output_tensor_idx);
        
        // Generate inference file
        GenerateInferenceFile(inference_file, base_name, subgraph, input_tensor_idx, output_tensor_idx);
        
        // Generate Makefile
        GenerateMakefile(makefile, base_name);
        
        std::cout << "\nCode generation complete!\n";
        std::cout << std::format("Generated files in directory '{}':\n", output_dir);
        std::cout << std::format("  - {}_weights.cpp\n", base_name);
        std::cout << std::format("  - {}.h\n", base_name);
        std::cout << std::format("  - {}.cpp\n", base_name);
        std::cout << std::format("  - {}_inference.cpp\n", base_name);
        std::cout << "  - Makefile\n";
        std::cout << std::format("\nTo build the inference executable:\n  cd {} && make\n", output_dir);
        std::cout << "\nNote: The generated code can be compiled independently without TFLite dependencies.\n";
        std::cout << "This code generator uses direct FlatBuffer inspection, requiring only the schema header.\n";
        
        return 0;
    } catch (const ModelValidationError& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const CodeGenerationError& e) {
        std::cerr << "\n================================================\n";
        std::cerr << "ERROR: Code generation failed!\n";
        std::cerr << "================================================\n";
        std::cerr << e.what() << std::endl;
        std::cerr << "\nCleaning up created artifacts...\n";
        try {
            DeleteDirectory(base_name);
            std::cerr << std::format("Successfully removed directory: {}\n", base_name);
        } catch (const FileSystemError& fs_err) {
            std::cerr << std::format("Warning: Failed to remove directory: {}\n", base_name);
            std::cerr << std::format("Please manually delete: {}\n", base_name);
        }
        std::cerr << "================================================\n";
        return 1;
    } catch (const FileSystemError& e) {
        std::cerr << "\n================================================\n";
        std::cerr << "ERROR: File system operation failed!\n";
        std::cerr << "================================================\n";
        std::cerr << e.what() << std::endl;
        std::cerr << "================================================\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n================================================\n";
        std::cerr << "ERROR: Unexpected error occurred!\n";
        std::cerr << "================================================\n";
        std::cerr << e.what() << std::endl;
        std::cerr << "================================================\n";
        return 1;
    }
}
