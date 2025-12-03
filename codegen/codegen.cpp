// codegen/codegen.cpp
// Code generator that reads TFLite models and generates pure C++ inference code
// Direct FlatBuffer inspection without requiring operator implementations

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <sstream>
#include <map>
#include <set>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

// Template engine
#include <inja/inja.hpp>
#include <nlohmann/json.hpp>

// TensorFlow Lite FlatBuffer schema header
#include "tensorflow/lite/schema/schema_generated.h"
// Only need builtin_op_data for activation/padding enums
#include "tensorflow/lite/c/builtin_op_data.h"

using json = nlohmann::json;

using namespace std;

namespace {

// Helper to get the actual builtin code (handles schema v3 compatibility)
// For v3 models, builtin_code defaults to 0, so we need to check deprecated_builtin_code
// For newer models, builtin_code contains the actual value
tflite::BuiltinOperator GetActualBuiltinCode(const tflite::OperatorCode* op_code) {
    if (!op_code) {
        return tflite::BuiltinOperator_CUSTOM;
    }
    // Use max of both fields to handle schema version compatibility
    // v3 models use deprecated_builtin_code, newer models use builtin_code
    return static_cast<tflite::BuiltinOperator>(
        std::max(static_cast<int>(op_code->builtin_code()),
                 static_cast<int>(op_code->deprecated_builtin_code())));
}

// Helper to get operator name from FlatBuffer
string GetOperatorName(const tflite::OperatorCode* op_code) {
    if (!op_code) {
        return "UNKNOWN";
    }
    
    tflite::BuiltinOperator builtin_code = GetActualBuiltinCode(op_code);
    
    if (builtin_code != tflite::BuiltinOperator_CUSTOM) {
        const char* name = tflite::EnumNameBuiltinOperator(builtin_code);
        return name ? name : "UNKNOWN";
    } else {
        string result = "CUSTOM:";
        if (op_code->custom_code()) {
            result += op_code->custom_code()->c_str();
        }
        return result;
    }
}

// Calculate tensor size from FlatBuffer shape
size_t CalculateTensorSize(const flatbuffers::Vector<int32_t>* shape) {
    if (!shape || shape->size() == 0) {
        return 1;
    }
    
    size_t size = 1;
    for (size_t i = 0; i < shape->size(); ++i) {
        size *= static_cast<size_t>(shape->Get(i));
    }
    return size;
}

// Get shape as string from FlatBuffer
string GetShapeString(const flatbuffers::Vector<int32_t>* shape) {
    if (!shape || shape->size() == 0) {
        return "1";
    }
    
    stringstream ss;
    for (size_t i = 0; i < shape->size(); ++i) {
        ss << shape->Get(i);
        if (i < shape->size() - 1) {
            ss << ", ";
        }
    }
    return ss.str();
}

// Get bytes per element based on tensor type
size_t GetBytesPerElement(tflite::TensorType type) {
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
bool ValidateModelSchema(const tflite::Model* model) {
    if (!model || !model->subgraphs() || model->subgraphs()->size() == 0) {
        cerr << "Error: Invalid model structure - no subgraphs found." << endl;
        return false;
    }

    // Get the main subgraph (usually index 0)
    const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
    if (!subgraph || !subgraph->operators()) {
        cerr << "Error: Invalid model structure - no operators found." << endl;
        return false;
    }

    // Get operator codes
    const flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>* op_codes = 
        model->operator_codes();
    if (!op_codes) {
        cerr << "Error: Invalid model structure - no operator codes found." << endl;
        return false;
    }

    bool ok = true;
    vector<pair<int, string>> unsupported_ops; // (operator_index, operator_name)
    vector<pair<int, int>> unsupported_activations; // (operator_index, activation_code)
    const auto* tensors = subgraph->tensors();

    // Set of supported operators
    set<tflite::BuiltinOperator> supported_ops = {
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
    for (size_t i = 0; i < subgraph->operators()->size(); ++i) {
        const tflite::Operator* op = subgraph->operators()->Get(i);
        if (!op) continue;

        // Get operator code index
        int op_code_index = op->opcode_index();
        if (op_code_index < 0 || op_code_index >= static_cast<int>(op_codes->size())) {
            cerr << "Error: Invalid operator code index " << op_code_index 
                 << " at operator " << i << "." << endl;
            ok = false;
            continue;
        }

        const tflite::OperatorCode* op_code = op_codes->Get(op_code_index);
        if (!op_code) {
            cerr << "Warning: Operator " << i << " has null operator code" << endl;
            ok = false;
            continue;
        }

        // Get actual builtin code (handles schema v3 compatibility)
        // For v3 models, builtin_code defaults to 0, so we need to check deprecated_builtin_code
        // For newer models, builtin_code contains the actual value
        tflite::BuiltinOperator builtin_code = static_cast<tflite::BuiltinOperator>(
            std::max(static_cast<int>(op_code->builtin_code()),
                     static_cast<int>(op_code->deprecated_builtin_code())));

        // Check if operator is supported
        if (supported_ops.find(builtin_code) == supported_ops.end()) {
            string op_name;
            if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
                op_name = string("CUSTOM:") + 
                    (op_code->custom_code() ? op_code->custom_code()->c_str() : "");
                // Check if it's a supported custom operator (DROPOUT or FLATTEN)
                string custom_name = op_code->custom_code() ? op_code->custom_code()->c_str() : "";
                if (custom_name.find("DROPOUT") != string::npos || 
                    custom_name.find("Dropout") != string::npos ||
                    custom_name.find("FLATTEN") != string::npos ||
                    custom_name.find("Flatten") != string::npos) {
                    // Supported custom operator, skip the error
                    continue;
                }
            } else {
                const char* op_name_ptr = tflite::EnumNameBuiltinOperator(builtin_code);
                op_name = op_name_ptr ? op_name_ptr : "UNKNOWN";
            }
            
            // Collect detailed information about this unsupported operator
            string op_details = op_name;
            if (tensors && op->inputs() && op->inputs()->size() > 0) {
                op_details += " (inputs: ";
                for (size_t j = 0; j < op->inputs()->size(); ++j) {
                    int input_idx = op->inputs()->Get(j);
                    if (input_idx >= 0 && input_idx < static_cast<int>(tensors->size())) {
                        const tflite::Tensor* input_tensor = tensors->Get(input_idx);
                        if (input_tensor && input_tensor->shape()) {
                            op_details += "tensor[" + to_string(input_idx) + ":" + 
                                         GetShapeString(input_tensor->shape()) + "]";
                        } else {
                            op_details += "tensor[" + to_string(input_idx) + "]";
                        }
                    } else {
                        op_details += "tensor[" + to_string(input_idx) + "]";
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

    // Print clear error messages
    if (!ok) {
        cerr << "\n================================================" << endl;
        cerr << "ERROR: Model contains unsupported operations!" << endl;
        cerr << "================================================" << endl;
        cerr << "\nThis code generator only supports the following operations:" << endl;
        cerr << "  - FULLY_CONNECTED (with NONE or RELU activation)" << endl;
        cerr << "  - SOFTMAX" << endl;
        cerr << "  - RELU" << endl;
        cerr << "  - CONV_2D (with NONE or RELU activation)" << endl;
        cerr << "  - MAX_POOL_2D" << endl;
        cerr << "  - SHAPE" << endl;
        cerr << "  - STRIDED_SLICE" << endl;
        cerr << "  - PACK" << endl;
        cerr << "  - RESHAPE" << endl;
        cerr << "  - ADD (with NONE or RELU activation)" << endl;
        cerr << "  - DROPOUT (custom operator, no-op during inference)" << endl;
        cerr << "  - FLATTEN (custom operator, converts to reshape)" << endl;
        
        if (!unsupported_ops.empty()) {
            cerr << "\n" << unsupported_ops.size() << " unsupported operator(s) found in model:" << endl;
            cerr << "  (Operator indices are 0-based, in execution order)" << endl;
            for (const auto& [op_idx, op_details] : unsupported_ops) {
                cerr << "  [Operator " << op_idx << "] " << op_details << endl;
            }
            cerr << "\n  Note: These operators appear early in the model graph." << endl;
            cerr << "        The code generator cannot skip them as they may transform" << endl;
            cerr << "        tensor shapes or data that subsequent operations depend on." << endl;
        }
        
        if (!unsupported_activations.empty()) {
            cerr << "\nUnsupported fused activations found:" << endl;
            for (const auto& [op_idx, act_code] : unsupported_activations) {
                cerr << "  - Operator " << op_idx << " has unsupported activation code: " 
                     << act_code << " (only NONE=0 and RELU=1 are supported)" << endl;
            }
        }
        
        cerr << "\nCode generation aborted." << endl;
        cerr << "\nTo fix this issue:" << endl;
        cerr << "  1. Use a model that only contains supported operations, OR" << endl;
        cerr << "  2. Request support for the missing operators to be added to the code generator" << endl;
        cerr << "\nThe model structure cannot be partially generated - all operations" << endl;
        cerr << "must be supported for correct code generation." << endl;
        cerr << "================================================\n" << endl;
    }

    return ok;
}

// Escape identifier for C++
string EscapeIdentifier(const string& name) {
    string result;
    for (char c : name) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || 
            (c >= '0' && c <= '9') || c == '_') {
            result += c;
        } else {
            result += '_';
        }
    }
    return result;
}

// Create mapping from tensor index to weight variable name
map<int, string> CreateWeightMapping(
    const tflite::Model* model,
    const tflite::SubGraph* subgraph
) {
    map<int, string> tensor_to_weight;
    const auto* tensors = subgraph->tensors();
    const auto* buffers = model->buffers();
    
    if (!tensors || !buffers) {
        return tensor_to_weight;
    }
    
    int weight_index = 0;
    
    for (size_t i = 0; i < tensors->size(); ++i) {
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
            string tensor_name = tensor->name() ? tensor->name()->c_str() : "tensor_" + to_string(i);
            tensor_to_weight[static_cast<int>(i)] = "weight_" + to_string(weight_index) + "_" + EscapeIdentifier(tensor_name);
            weight_index++;
        }
    }
    
    return tensor_to_weight;
}

// Generate model_weights.cpp
// Returns true on success, false on failure
bool GenerateWeightsFile(
    const string& output_path,
    const string& base_name,
    const tflite::Model* model,
    const tflite::SubGraph* subgraph,
    const map<int, string>& tensor_to_weight
) {
    inja::Environment env;
    
    const auto* tensors = subgraph->tensors();
    const auto* buffers = model->buffers();
    
    if (!tensors || !buffers) {
        // Create empty file
        ofstream out(output_path);
        if (out) {
            out << "// " << base_name << "_weights.cpp\n";
            out << "// Auto-generated weight data for embedded inference\n\n";
            out << "#include <cstddef>\n\n";
            out << "namespace embedded_ml {\n\n";
            out << "} // namespace embedded_ml\n";
            out.close();
            cout << "Generated: " << output_path << endl;
            return true;
        }
        return false;
    }
    
    json data;
    data["base_name"] = base_name;
    data["weights"] = json::array();
    
    for (size_t i = 0; i < tensors->size(); ++i) {
        if (tensor_to_weight.find(static_cast<int>(i)) == tensor_to_weight.end()) {
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
        
        const size_t num_elements = CalculateTensorSize(tensor->shape());
        string tensor_name = tensor->name() ? tensor->name()->c_str() : "tensor_" + to_string(i);
        string var_name = tensor_to_weight.at(static_cast<int>(i));
        
        // Get buffer data
        const flatbuffers::Vector<uint8_t>* data_vec = buffer->data();
        if (!data_vec || data_vec->size() < num_elements * sizeof(float)) {
            cerr << "Warning: Buffer size mismatch for tensor " << i << endl;
            continue;
        }
        
        const float* float_data = reinterpret_cast<const float*>(data_vec->data());
        
        json weight;
        weight["tensor_index"] = static_cast<int>(i);
        weight["tensor_name"] = tensor_name;
        weight["var_name"] = var_name;
        weight["shape"] = GetShapeString(tensor->shape());
        weight["num_elements"] = num_elements;
        
        // Format float values with precision and track newline positions
        ostringstream value_stream;
        value_stream << fixed << setprecision(9);
        weight["values"] = json::array();
        for (size_t j = 0; j < num_elements; ++j) {
            json value_obj;
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
    ifstream template_file("templates/weights.cpp.inja");
    if (!template_file) {
        cerr << "Error: Cannot open template file templates/weights.cpp.inja" << endl;
        return false;
    }
    
    string template_content((istreambuf_iterator<char>(template_file)),
                           istreambuf_iterator<char>());
    template_file.close();
    
    try {
        inja::Template temp = env.parse(template_content);
        string result = env.render(temp, data);
        
        ofstream out(output_path);
        if (!out) {
            cerr << "Error: Cannot create file " << output_path << endl;
            return false;
        }
        
        out << result;
        out.close();
        cout << "Generated: " << output_path << endl;
        return true;
    } catch (const exception& e) {
        cerr << "Error: Template rendering failed: " << e.what() << endl;
        return false;
    }
}

// Generate model.h (header file)
// Returns true on success, false on failure
bool GenerateModelHeader(
    const string& output_path,
    const string& base_name,
    size_t input_size,
    size_t output_size,
    const vector<pair<int, size_t>>& intermediate_buffers
) {
    inja::Environment env;
    
    // Create include guard name (uppercase, with underscores)
    string guard_name = base_name;
    transform(guard_name.begin(), guard_name.end(), guard_name.begin(), ::toupper);
    for (size_t i = 0; i < guard_name.length(); ++i) {
        if (guard_name[i] == '-' || guard_name[i] == '.') {
            guard_name[i] = '_';
        }
    }
    guard_name += "_MODEL_H";
    
    json data;
    data["base_name"] = base_name;
    data["guard_name"] = guard_name;
    data["input_size"] = input_size;
    data["output_size"] = output_size;
    data["intermediate_buffers"] = json::array();
    
    for (const auto& [idx, size] : intermediate_buffers) {
        json buffer;
        buffer["index"] = idx;
        buffer["size"] = size;
        data["intermediate_buffers"].push_back(buffer);
    }
    
    // Load and render template
    ifstream template_file("templates/model.h.inja");
    if (!template_file) {
        cerr << "Error: Cannot open template file templates/model.h.inja" << endl;
        return false;
    }
    
    string template_content((istreambuf_iterator<char>(template_file)),
                           istreambuf_iterator<char>());
    template_file.close();
    
    try {
        inja::Template temp = env.parse(template_content);
        string result = env.render(temp, data);
        
        ofstream out(output_path);
        if (!out) {
            cerr << "Error: Cannot create file " << output_path << endl;
            return false;
        }
        
        out << result;
        out.close();
        cout << "Generated: " << output_path << endl;
        return true;
    } catch (const exception& e) {
        cerr << "Error: Template rendering failed: " << e.what() << endl;
        return false;
    }
}

// Convert FlatBuffer activation to TfLite activation enum
TfLiteFusedActivation ConvertActivation(tflite::ActivationFunctionType activation) {
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
TfLitePadding ConvertPadding(tflite::Padding padding) {
    switch (padding) {
        case tflite::Padding_SAME: return kTfLitePaddingSame;
        case tflite::Padding_VALID: return kTfLitePaddingValid;
        default: return kTfLitePaddingSame;
    }
}

// Generate model.cpp
// Returns true on success, false on failure
bool GenerateModelFile(
    const string& output_path,
    const string& base_name,
    const tflite::Model* model,
    const tflite::SubGraph* subgraph,
    const map<int, string>& tensor_to_weight,
    const vector<pair<int, size_t>>& intermediate_buffers,
    const map<int, size_t>& tensor_sizes,
    int32_t input_tensor_idx,
    int32_t output_tensor_idx
) {
    inja::Environment env;
    
    bool generation_failed = false;
    vector<string> generation_errors;
    
    const auto* operators = subgraph->operators();
    const auto* operator_codes = model->operator_codes();
    const auto* tensors = subgraph->tensors();
    
    if (!operators || !operator_codes || !tensors) {
        cerr << "Error: Invalid model structure" << endl;
        return false;
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
    json data;
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
    data["layers"] = json::array();
    data["intermediate_buffers"] = json::array();
    
    // Process each operator in order (this is the execution plan)
    for (size_t i = 0; i < operators->size(); ++i) {
        const tflite::Operator* op = operators->Get(i);
        if (!op) continue;
        
        int op_code_index = op->opcode_index();
        if (op_code_index < 0 || op_code_index >= static_cast<int>(operator_codes->size())) {
            continue;
        }
        
        const tflite::OperatorCode* op_code = operator_codes->Get(op_code_index);
        if (!op_code) continue;
        
        string op_name = GetOperatorName(op_code);
        
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
        } else if (op_name.find("DROPOUT") != string::npos || 
                   op_name.find("Dropout") != string::npos) {
            has_dropout = true;
            data["has_dropout"] = true;
            op_name = "DROPOUT";  // Normalize name
        } else if (op_name.find("FLATTEN") != string::npos ||
                   op_name.find("Flatten") != string::npos) {
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
        string input_ptr, output_ptr;
        if (op_input_tensor_idx == input_tensor_idx) {
            input_ptr = "input";
        } else {
            input_ptr = "buffer_" + to_string(op_input_tensor_idx);
        }
        
        if (op_output_tensor_idx == output_tensor_idx) {
            output_ptr = "output";
        } else {
            output_ptr = "buffer_" + to_string(op_output_tensor_idx);
        }
        
        size_t input_size_layer = tensor_sizes.count(op_input_tensor_idx) ? tensor_sizes.at(op_input_tensor_idx) : 0;
        size_t output_size_layer = tensor_sizes.count(op_output_tensor_idx) ? tensor_sizes.at(op_output_tensor_idx) : 0;
        
        // Build layer JSON based on operation type
        json layer;
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
                if (tensor_to_weight.find(weights_tensor_idx) == tensor_to_weight.end() ||
                    tensor_to_weight.find(bias_tensor_idx) == tensor_to_weight.end()) {
                    string error_msg = "FULLY_CONNECTED layer " + to_string(i) + 
                        " missing required weight/bias tensors (weights_idx=" + 
                        to_string(weights_tensor_idx) + ", bias_idx=" + to_string(bias_tensor_idx) + ")";
                    generation_errors.push_back(error_msg);
                    generation_failed = true;
                    continue;
                }
                
                string weights_var = tensor_to_weight.at(weights_tensor_idx);
                string bias_var = tensor_to_weight.at(bias_tensor_idx);
                
                // Determine actual input size from weights shape
                size_t fc_input_size = input_size_layer;
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
                string activation_param = "ActivationType::NONE";
                if (fused_activation == kTfLiteActRelu) {
                    activation_param = "ActivationType::RELU";
                } else if (fused_activation != kTfLiteActNone) {
                    cerr << "Warning: Unsupported fused activation " << static_cast<int>(fused_activation) 
                         << " in FULLY_CONNECTED layer " << i << " (only ReLU supported, using NONE)" << endl;
                }
                
                layer["weights_var"] = weights_var;
                layer["bias_var"] = bias_var;
                layer["fc_input_size"] = fc_input_size;
                layer["activation_param"] = activation_param;
            } else {
                string error_msg = "FULLY_CONNECTED layer " + to_string(i) + 
                    " has insufficient inputs (expected at least 3: input, weights, bias)";
                generation_errors.push_back(error_msg);
                generation_failed = true;
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
                if (tensor_to_weight.find(filter_tensor_idx) == tensor_to_weight.end() ||
                    tensor_to_weight.find(bias_tensor_idx) == tensor_to_weight.end()) {
                    cerr << "Warning: Cannot find filter/bias tensors for CONV_2D layer " << i << endl;
                    continue;
                }
                
                string filter_var = tensor_to_weight.at(filter_tensor_idx);
                string bias_var = tensor_to_weight.at(bias_tensor_idx);
                
                // Get tensor shapes
                const tflite::Tensor* input_tensor = (op_input_tensor_idx >= 0 && op_input_tensor_idx < static_cast<int>(tensors->size())) 
                    ? tensors->Get(op_input_tensor_idx) : nullptr;
                const tflite::Tensor* filter_tensor = (filter_tensor_idx >= 0 && filter_tensor_idx < static_cast<int>(tensors->size()))
                    ? tensors->Get(filter_tensor_idx) : nullptr;
                const tflite::Tensor* output_tensor = (op_output_tensor_idx >= 0 && op_output_tensor_idx < static_cast<int>(tensors->size()))
                    ? tensors->Get(op_output_tensor_idx) : nullptr;
                
                if (!input_tensor || !filter_tensor || !output_tensor ||
                    !input_tensor->shape() || !filter_tensor->shape() || !output_tensor->shape()) {
                    cerr << "Warning: Cannot get tensor shapes for CONV_2D layer " << i << endl;
                    continue;
                }
                
                // Extract dimensions (NHWC format)
                size_t batch_size = input_tensor->shape()->Get(0);
                size_t input_height = input_tensor->shape()->Get(1);
                size_t input_width = input_tensor->shape()->Get(2);
                size_t input_channels = input_tensor->shape()->Get(3);
                size_t filter_height = filter_tensor->shape()->Get(1);
                size_t filter_width = filter_tensor->shape()->Get(2);
                size_t output_channels = filter_tensor->shape()->Get(0);
                
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
                
                string padding_param = (padding == kTfLitePaddingSame) ? 
                    "PaddingType::SAME" : "PaddingType::VALID";
                string activation_param = (fused_activation == kTfLiteActRelu) ? 
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
                cerr << "Warning: CONV_2D layer " << i << " missing filter/bias" << endl;
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
                cerr << "Warning: Cannot get tensor shapes for MAX_POOL_2D layer " << i << endl;
                continue;
            }
            
            // Extract dimensions (NHWC format)
            size_t batch_size = input_tensor->shape()->Get(0);
            size_t input_height = input_tensor->shape()->Get(1);
            size_t input_width = input_tensor->shape()->Get(2);
            size_t channels = input_tensor->shape()->Get(3);
            
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
            
            string padding_param = (padding == kTfLitePaddingSame) ? 
                "PaddingType::SAME" : "PaddingType::VALID";
            string activation_param = (fused_activation == kTfLiteActRelu) ? 
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
                cerr << "Warning: Cannot get tensor shapes for SHAPE layer " << i << endl;
                continue;
            }
            
            int num_dims = input_tensor->shape()->size();
            ostringstream shape_values;
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
                string error_msg = "ADD layer " + to_string(i) + " missing inputs: ";
                error_msg += "expected 2 inputs, but ";
                if (!op_inputs) {
                    error_msg += "no input list provided";
                } else {
                    error_msg += "only " + to_string(op_inputs->size()) + " provided";
                }
                generation_errors.push_back(error_msg);
                generation_failed = true;
                continue;
            }
            
            int input1_tensor_idx = op_inputs->Get(0);
            int input2_tensor_idx = op_inputs->Get(1);
            
            if (input1_tensor_idx < 0 || input2_tensor_idx < 0) {
                string error_msg = "ADD layer " + to_string(i) + " missing inputs: ";
                error_msg += "input1=" + to_string(input1_tensor_idx) + ", input2=" + to_string(input2_tensor_idx);
                error_msg += " (negative indices indicate optional/unused tensors, which are not supported)";
                generation_errors.push_back(error_msg);
                generation_failed = true;
                continue;
            }
            
            // Determine input pointers
            string input1_ptr, input2_ptr;
            
            if (input1_tensor_idx == input_tensor_idx) {
                input1_ptr = "input";
            } else if (tensor_to_weight.find(input1_tensor_idx) != tensor_to_weight.end()) {
                input1_ptr = tensor_to_weight.at(input1_tensor_idx);
            } else {
                input1_ptr = "buffer_" + to_string(input1_tensor_idx);
            }
            
            if (input2_tensor_idx == input_tensor_idx) {
                input2_ptr = "input";
            } else if (tensor_to_weight.find(input2_tensor_idx) != tensor_to_weight.end()) {
                input2_ptr = tensor_to_weight.at(input2_tensor_idx);
            } else {
                input2_ptr = "buffer_" + to_string(input2_tensor_idx);
            }
            
            size_t add_size = output_size_layer;
            
            // Check for fused activation function
            TfLiteFusedActivation fused_activation = kTfLiteActNone;
            const tflite::AddOptions* options = op->builtin_options_as_AddOptions();
            if (options) {
                fused_activation = ConvertActivation(options->fused_activation_function());
            }
            
            string activation_param = "ActivationType::NONE";
            if (fused_activation == kTfLiteActRelu) {
                activation_param = "ActivationType::RELU";
            } else if (fused_activation != kTfLiteActNone) {
                cerr << "Warning: Unsupported fused activation " << static_cast<int>(fused_activation) 
                     << " in ADD layer " << i << " (only ReLU supported, using NONE)" << endl;
            }
            
            layer["input1_ptr"] = input1_ptr;
            layer["input2_ptr"] = input2_ptr;
            layer["add_size"] = add_size;
            layer["activation_param"] = activation_param;
        } else if (op_name == "DROPOUT") {
            float dropout_rate = 0.0f;
            layer["dropout_rate"] = dropout_rate;
        } else {
            cerr << "Warning: Unsupported operation " << op_name << endl;
            continue;
        }
        
        data["layers"].push_back(layer);
    }
    
    // Add intermediate buffers
    for (const auto& [idx, size] : intermediate_buffers) {
        json buffer;
        buffer["index"] = idx;
        buffer["size"] = size;
        data["intermediate_buffers"].push_back(buffer);
    }
    
    // Check for generation errors before rendering
    if (generation_failed) {
        cerr << "\n================================================" << endl;
        cerr << "ERROR: Code generation failed due to model structure issues!" << endl;
        cerr << "================================================" << endl;
        cerr << "\nThe following errors occurred during code generation:" << endl;
        for (const auto& error : generation_errors) {
            cerr << "  - " << error << endl;
        }
        cerr << "\nCode generation aborted." << endl;
        cerr << "The generated file may be incomplete or incorrect." << endl;
        cerr << "================================================\n" << endl;
        return false;
    }
    
    // Load and render template
    ifstream template_file("templates/model.cpp.inja");
    if (!template_file) {
        cerr << "Error: Cannot open template file templates/model.cpp.inja" << endl;
        return false;
    }
    
    string template_content((istreambuf_iterator<char>(template_file)),
                           istreambuf_iterator<char>());
    template_file.close();
    
    inja::Template temp = env.parse(template_content);
    string result = env.render(temp, data);
    
    ofstream out(output_path);
    if (!out) {
        cerr << "Error: Cannot create file " << output_path << endl;
        return false;
    }
    
    out << result;
    out.close();
    
    cout << "Generated: " << output_path << endl;
    return true;
}

// Generate inference.cpp
// Returns true on success, false on failure
bool GenerateInferenceFile(
    const string& output_path,
    const string& base_name,
    const tflite::SubGraph* subgraph,
    int32_t input_tensor_idx,
    int32_t output_tensor_idx
) {
    inja::Environment env;
    
    const auto* tensors = subgraph->tensors();
    if (!tensors) {
        cerr << "Error: Cannot access tensors" << endl;
        return false;
    }
    
    const tflite::Tensor* input_tensor = (input_tensor_idx >= 0 && input_tensor_idx < static_cast<int>(tensors->size()))
        ? tensors->Get(input_tensor_idx) : nullptr;
    const tflite::Tensor* output_tensor = (output_tensor_idx >= 0 && output_tensor_idx < static_cast<int>(tensors->size()))
        ? tensors->Get(output_tensor_idx) : nullptr;
    
    if (!input_tensor || !output_tensor) {
        cerr << "Error: Cannot access input/output tensors" << endl;
        return false;
    }
    
    size_t input_size = CalculateTensorSize(input_tensor->shape());
    size_t output_size = CalculateTensorSize(output_tensor->shape());
    
    json data;
    data["base_name"] = base_name;
    data["input_size"] = input_size;
    data["output_size"] = output_size;
    data["input_shape"] = GetShapeString(input_tensor->shape());
    data["output_shape"] = GetShapeString(output_tensor->shape());
    
    // Create array for iteration with newline info (Inja doesn't support range() function)
    data["input_indices"] = json::array();
    for (size_t i = 0; i < input_size; ++i) {
        json idx_obj;
        idx_obj["needs_newline"] = ((i + 1) % 8 == 0);
        idx_obj["is_last"] = (i == input_size - 1);
        data["input_indices"].push_back(idx_obj);
    }
    
    // Load and render template
    ifstream template_file("templates/inference.cpp.inja");
    if (!template_file) {
        cerr << "Error: Cannot open template file templates/inference.cpp.inja" << endl;
        return false;
    }
    
    string template_content((istreambuf_iterator<char>(template_file)),
                           istreambuf_iterator<char>());
    template_file.close();
    
    try {
        inja::Template temp = env.parse(template_content);
        string result = env.render(temp, data);
        
        ofstream out(output_path);
        if (!out) {
            cerr << "Error: Cannot create file " << output_path << endl;
            return false;
        }
        
        out << result;
        out.close();
        cout << "Generated: " << output_path << endl;
        return true;
    } catch (const exception& e) {
        cerr << "Error: Template rendering failed: " << e.what() << endl;
        return false;
    }
}

// Generate Makefile for the generated code
// Returns true on success, false on failure
bool GenerateMakefile(
    const string& output_path,
    const string& base_name
) {
    inja::Environment env;
    
    json data;
    data["base_name"] = base_name;
    
    // Load and render template
    ifstream template_file("templates/Makefile.inja");
    if (!template_file) {
        cerr << "Error: Cannot open template file templates/Makefile.inja" << endl;
        return false;
    }
    
    string template_content((istreambuf_iterator<char>(template_file)),
                           istreambuf_iterator<char>());
    template_file.close();
    
    try {
        inja::Template temp = env.parse(template_content);
        string result = env.render(temp, data);
        
        ofstream out(output_path);
        if (!out) {
            cerr << "Error: Cannot create file " << output_path << endl;
            return false;
        }
        
        out << result;
        out.close();
        cout << "Generated: " << output_path << endl;
        return true;
    } catch (const exception& e) {
        cerr << "Error: Template rendering failed: " << e.what() << endl;
        return false;
    }
}

// Create directory if it doesn't exist
bool CreateDirectory(const string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        // Directory already exists
        return S_ISDIR(info.st_mode);
    }
    
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0;
#else
    return mkdir(path.c_str(), 0755) == 0;
#endif
}

// Recursively delete a directory and all its contents
bool DeleteDirectory(const string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, consider it "deleted"
        return true;
    }
    
    if (!S_ISDIR(info.st_mode)) {
        // Not a directory
        return false;
    }
    
#ifdef _WIN32
    // Windows: use system command
    // Escape path by wrapping in quotes
    string escaped_path = "\"" + path + "\"";
    string cmd = "rmdir /s /q " + escaped_path;
    int result = system(cmd.c_str());
    return result == 0;
#else
    // Unix/Linux/macOS: use system command
    // Escape path properly for shell (handle spaces and special chars)
    string escaped_path = "\"" + path + "\"";
    string cmd = "rm -rf " + escaped_path;
    int result = system(cmd.c_str());
    return result == 0;
#endif
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        cerr << "Usage: " << argv[0] << " <path_to_model.tflite> <base_name> [--output-index=N]" << endl;
        cerr << "Example: " << argv[0] << " scripts/model.tflite my_model" << endl;
        cerr << "Example (multi-output): " << argv[0] << " scripts/model.tflite my_model --output-index=1" << endl;
        cerr << "This will create a directory 'my_model/' containing:" << endl;
        cerr << "  - my_model_weights.cpp" << endl;
        cerr << "  - my_model.h" << endl;
        cerr << "  - my_model.cpp" << endl;
        cerr << "  - my_model_inference.cpp" << endl;
        cerr << "  - Makefile" << endl;
        cerr << "\nTo build: cd my_model && make" << endl;
        cerr << "\nNote: For models with multiple outputs, use --output-index=N to select" << endl;
        cerr << "      which output tensor to use (default: 0). Only single input models are supported." << endl;
        return 1;
    }
    
    const string model_path = argv[1];
    const string base_name = argv[2];
    int output_index = 0;  // Default to first output
    bool output_index_specified = false;  // Track if user explicitly provided the flag
    
    // Parse optional output index flag
    if (argc == 4) {
        string flag = argv[3];
        if (flag.find("--output-index") == 0) {
            size_t eq_pos = flag.find('=');
            if (eq_pos != string::npos) {
                // Format: --output-index=N
                string index_str = flag.substr(eq_pos + 1);
                try {
                    output_index = stoi(index_str);
                    output_index_specified = true;
                } catch (const exception& e) {
                    cerr << "Error: Invalid output index: " << index_str << endl;
                    return 1;
                }
            } else {
                cerr << "Error: --output-index requires a value, e.g., --output-index=1" << endl;
                return 1;
            }
        } else {
            cerr << "Error: Unknown flag: " << flag << endl;
            cerr << "Use --output-index=N to specify output tensor index" << endl;
            return 1;
        }
    }
    
    // Check if file exists
    ifstream file_check(model_path, ios::binary);
    if (!file_check.good()) {
        cerr << "Error: Cannot open file: " << model_path << endl;
        return 1;
    }
    file_check.close();
    
    cout << "Loading TFLite model from: " << model_path << endl;
    
    // Read the entire file into memory
    ifstream file(model_path, ios::binary | ios::ate);
    if (!file.is_open()) {
        cerr << "Error: Failed to open file: " << model_path << endl;
        return 1;
    }
    
    streamsize file_size = file.tellg();
    file.seekg(0, ios::beg);
    
    vector<uint8_t> buffer(file_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        cerr << "Error: Failed to read file: " << model_path << endl;
        return 1;
    }
    file.close();
    
    cout << "File read successfully (" << file_size << " bytes)" << endl;
    
    // Verify FlatBuffer
    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!tflite::VerifyModelBuffer(verifier)) {
        cerr << "Error: Invalid FlatBuffer format" << endl;
        return 1;
    }
    
    // Get the model from the FlatBuffer
    const tflite::Model* model = tflite::GetModel(buffer.data());
    if (!model) {
        cerr << "Error: Failed to parse model from FlatBuffer" << endl;
        return 1;
    }
    
    cout << "Model loaded successfully!" << endl;
    
    // Validate model schema
    cout << "Validating model operations..." << endl;
    if (!ValidateModelSchema(model)) {
        return 1;
    }
    cout << "Model validation passed - all operations are supported!" << endl;
    
    // Get the main subgraph
    const auto* subgraphs = model->subgraphs();
    if (!subgraphs || subgraphs->size() == 0) {
        cerr << "Error: No subgraphs found in model" << endl;
        return 1;
    }
    
    const tflite::SubGraph* subgraph = subgraphs->Get(0);
    if (!subgraph) {
        cerr << "Error: Subgraph is null" << endl;
        return 1;
    }
    
    const auto* inputs = subgraph->inputs();
    const auto* outputs = subgraph->outputs();
    const auto* tensors = subgraph->tensors();
    const auto* operators = subgraph->operators();
    
    if (!inputs || !outputs || !tensors || inputs->size() == 0 || outputs->size() == 0) {
        cerr << "Error: Invalid model structure - missing inputs/outputs" << endl;
        return 1;
    }
    
    // Check input count (only single input supported)
    if (inputs->size() != 1) {
        cerr << "Error: Only single input models are supported." << endl;
        cerr << "This model has " << inputs->size() << " input(s)." << endl;
        return 1;
    }
    
    // Require --output-index flag for multi-output models
    if (outputs->size() > 1 && !output_index_specified) {
        cerr << "Error: This model has " << outputs->size() << " output(s)." << endl;
        cerr << "You must specify which output to use with --output-index=N" << endl;
        cerr << "\nAvailable outputs:" << endl;
        for (size_t i = 0; i < outputs->size(); ++i) {
            int32_t out_idx = outputs->Get(i);
            const tflite::Tensor* out_tensor = (out_idx >= 0 && out_idx < static_cast<int>(tensors->size()))
                ? tensors->Get(out_idx) : nullptr;
            if (out_tensor) {
                cerr << "  [" << i << "] Shape: [" << GetShapeString(out_tensor->shape()) << "]";
                if (out_tensor->name()) {
                    cerr << " Name: " << out_tensor->name()->c_str();
                }
                cerr << endl;
            }
        }
        cerr << "\nRerun with:" << endl;
        cerr << "  " << argv[0] << " " << model_path << " " << base_name 
             << " --output-index=<index>" << endl;
        return 1;
    }
    
    // Check and validate output index
    if (output_index < 0 || output_index >= static_cast<int>(outputs->size())) {
        cerr << "Error: Invalid output index " << output_index << "." << endl;
        cerr << "This model has " << outputs->size() << " output(s) (valid indices: 0 to " 
             << (outputs->size() - 1) << ")." << endl;
        if (outputs->size() > 1) {
            cerr << "\nTo use a different output, rerun with:" << endl;
            cerr << "  " << argv[0] << " " << model_path << " " << base_name 
                 << " --output-index=<index>" << endl;
            cerr << "Available outputs:" << endl;
            for (size_t i = 0; i < outputs->size(); ++i) {
                int32_t out_idx = outputs->Get(i);
                const tflite::Tensor* out_tensor = (out_idx >= 0 && out_idx < static_cast<int>(tensors->size()))
                    ? tensors->Get(out_idx) : nullptr;
                if (out_tensor) {
                    cerr << "  [" << i << "] Shape: [" << GetShapeString(out_tensor->shape()) << "]";
                    if (out_tensor->name()) {
                        cerr << " Name: " << out_tensor->name()->c_str();
                    }
                    cerr << endl;
                }
            }
        }
        return 1;
    }
    
    int32_t input_tensor_idx = inputs->Get(0);
    int32_t output_tensor_idx = outputs->Get(output_index);
    
    if (outputs->size() > 1) {
        cout << "Note: Model has " << outputs->size() << " outputs. Using output index " 
             << output_index << " (0-indexed)." << endl;
    }
    
    const tflite::Tensor* input_tensor = (input_tensor_idx >= 0 && input_tensor_idx < static_cast<int>(tensors->size()))
        ? tensors->Get(input_tensor_idx) : nullptr;
    const tflite::Tensor* output_tensor = (output_tensor_idx >= 0 && output_tensor_idx < static_cast<int>(tensors->size()))
        ? tensors->Get(output_tensor_idx) : nullptr;
    
    if (!input_tensor || !output_tensor) {
        cerr << "Error: Cannot access input/output tensors" << endl;
        return 1;
    }
    
    size_t input_size = CalculateTensorSize(input_tensor->shape());
    size_t output_size = CalculateTensorSize(output_tensor->shape());
    
    // Create output directory
    string output_dir = base_name;
    if (!CreateDirectory(output_dir)) {
        cerr << "Error: Failed to create directory " << output_dir << endl;
        return 1;
    }
    cout << "Created directory: " << output_dir << endl;
    
    // Generate files
    // First create weight mapping
    map<int, string> tensor_to_weight = CreateWeightMapping(model, subgraph);
    
    // Collect intermediate buffers info
    map<int, size_t> tensor_sizes;
    tensor_sizes[input_tensor_idx] = input_size;
    tensor_sizes[output_tensor_idx] = output_size;
    
    // Process operators to find all tensor sizes
    if (operators) {
        for (size_t i = 0; i < operators->size(); ++i) {
            const tflite::Operator* op = operators->Get(i);
            if (!op) continue;
            
            const auto* op_inputs = op->inputs();
            const auto* op_outputs = op->outputs();
            
            if (op_inputs) {
                for (size_t j = 0; j < op_inputs->size(); ++j) {
                    int tensor_idx = op_inputs->Get(j);
                    if (tensor_idx >= 0 && tensor_idx < static_cast<int>(tensors->size()) &&
                        tensor_sizes.find(tensor_idx) == tensor_sizes.end()) {
                        const tflite::Tensor* tensor = tensors->Get(tensor_idx);
                        if (tensor) {
                            tensor_sizes[tensor_idx] = CalculateTensorSize(tensor->shape());
                        }
                    }
                }
            }
            
            if (op_outputs) {
                for (size_t j = 0; j < op_outputs->size(); ++j) {
                    int tensor_idx = op_outputs->Get(j);
                    if (tensor_idx >= 0 && tensor_idx < static_cast<int>(tensors->size()) &&
                        tensor_sizes.find(tensor_idx) == tensor_sizes.end()) {
                        const tflite::Tensor* tensor = tensors->Get(tensor_idx);
                        if (tensor) {
                            tensor_sizes[tensor_idx] = CalculateTensorSize(tensor->shape());
                        }
                    }
                }
            }
        }
    }
    
    vector<pair<int, size_t>> intermediate_buffers;
    // Get set of weight tensor indices to exclude them
    set<int> weight_tensor_indices;
    for (const auto& [tensor_idx, weight_name] : tensor_to_weight) {
        weight_tensor_indices.insert(tensor_idx);
    }
    
    // Collect all output tensors from operations
    set<int> operation_output_tensors;
    if (operators) {
        for (size_t i = 0; i < operators->size(); ++i) {
            const tflite::Operator* op = operators->Get(i);
            if (!op) continue;
            
            const auto* op_outputs = op->outputs();
            if (op_outputs) {
                for (size_t j = 0; j < op_outputs->size(); ++j) {
                    int tensor_idx = op_outputs->Get(j);
                    operation_output_tensors.insert(tensor_idx);
                    // Ensure it's in tensor_sizes
                    if (tensor_sizes.find(tensor_idx) == tensor_sizes.end()) {
                        if (tensor_idx >= 0 && tensor_idx < static_cast<int>(tensors->size())) {
                            const tflite::Tensor* tensor = tensors->Get(tensor_idx);
                            if (tensor) {
                                size_t size = CalculateTensorSize(tensor->shape());
                                tensor_sizes[tensor_idx] = size > 0 ? size : 1;
                            } else {
                                tensor_sizes[tensor_idx] = 1;
                            }
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
            if (weight_tensor_indices.find(tensor_idx) == weight_tensor_indices.end()) {
                size_t size = tensor_sizes.count(tensor_idx) ? tensor_sizes[tensor_idx] : 1;
                intermediate_buffers.push_back({tensor_idx, size});
            }
        }
    }
    
    // Also include any other tensors from tensor_sizes that might have been missed
    for (const auto& [tensor_idx, size] : tensor_sizes) {
        if (tensor_idx != input_tensor_idx && tensor_idx != output_tensor_idx) {
            // Skip if already added or if it's a weight
            bool already_added = false;
            for (const auto& [buf_idx, buf_size] : intermediate_buffers) {
                if (buf_idx == tensor_idx) {
                    already_added = true;
                    break;
                }
            }
            if (!already_added && weight_tensor_indices.find(tensor_idx) == weight_tensor_indices.end()) {
                intermediate_buffers.push_back({tensor_idx, size > 0 ? size : 1});
            }
        }
    }
    
    string weights_file = output_dir + "/" + base_name + "_weights.cpp";
    string model_header_file = output_dir + "/" + base_name + ".h";
    string model_file = output_dir + "/" + base_name + ".cpp";
    string inference_file = output_dir + "/" + base_name + "_inference.cpp";
    string makefile = output_dir + "/Makefile";
    
    cout << "\nGenerating code files..." << endl;
    
    // Generate weights file
    if (!GenerateWeightsFile(weights_file, base_name, model, subgraph, tensor_to_weight)) {
        cerr << "\n================================================" << endl;
        cerr << "ERROR: Failed to generate weights file!" << endl;
        cerr << "Cleaning up created artifacts..." << endl;
        if (DeleteDirectory(output_dir)) {
            cerr << "Successfully removed directory: " << output_dir << endl;
        } else {
            cerr << "Warning: Failed to remove directory: " << output_dir << endl;
            cerr << "Please manually delete: " << output_dir << endl;
        }
        cerr << "================================================\n" << endl;
        return 1;
    }
    
    // Generate model header
    if (!GenerateModelHeader(model_header_file, base_name, input_size, output_size, intermediate_buffers)) {
        cerr << "\n================================================" << endl;
        cerr << "ERROR: Failed to generate model header file!" << endl;
        cerr << "Cleaning up created artifacts..." << endl;
        if (DeleteDirectory(output_dir)) {
            cerr << "Successfully removed directory: " << output_dir << endl;
        } else {
            cerr << "Warning: Failed to remove directory: " << output_dir << endl;
            cerr << "Please manually delete: " << output_dir << endl;
        }
        cerr << "================================================\n" << endl;
        return 1;
    }
    
    // Generate model file (most critical - can fail)
    if (!GenerateModelFile(model_file, base_name, model, subgraph, tensor_to_weight, intermediate_buffers, 
                           tensor_sizes, input_tensor_idx, output_tensor_idx)) {
        cerr << "\n================================================" << endl;
        cerr << "ERROR: Failed to generate model file!" << endl;
        cerr << "Cleaning up created artifacts..." << endl;
        if (DeleteDirectory(output_dir)) {
            cerr << "Successfully removed directory: " << output_dir << endl;
        } else {
            cerr << "Warning: Failed to remove directory: " << output_dir << endl;
            cerr << "Please manually delete: " << output_dir << endl;
        }
        cerr << "================================================\n" << endl;
        return 1;
    }
    
    // Generate inference file
    if (!GenerateInferenceFile(inference_file, base_name, subgraph, input_tensor_idx, output_tensor_idx)) {
        cerr << "\n================================================" << endl;
        cerr << "ERROR: Failed to generate inference file!" << endl;
        cerr << "Cleaning up created artifacts..." << endl;
        if (DeleteDirectory(output_dir)) {
            cerr << "Successfully removed directory: " << output_dir << endl;
        } else {
            cerr << "Warning: Failed to remove directory: " << output_dir << endl;
            cerr << "Please manually delete: " << output_dir << endl;
        }
        cerr << "================================================\n" << endl;
        return 1;
    }
    
    // Generate Makefile
    if (!GenerateMakefile(makefile, base_name)) {
        cerr << "\n================================================" << endl;
        cerr << "ERROR: Failed to generate Makefile!" << endl;
        cerr << "Cleaning up created artifacts..." << endl;
        if (DeleteDirectory(output_dir)) {
            cerr << "Successfully removed directory: " << output_dir << endl;
        } else {
            cerr << "Warning: Failed to remove directory: " << output_dir << endl;
            cerr << "Please manually delete: " << output_dir << endl;
        }
        cerr << "================================================\n" << endl;
        return 1;
    }
    
    cout << "\nCode generation complete!" << endl;
    cout << "Generated files in directory '" << output_dir << "':" << endl;
    cout << "  - " << base_name << "_weights.cpp" << endl;
    cout << "  - " << base_name << ".h" << endl;
    cout << "  - " << base_name << ".cpp" << endl;
    cout << "  - " << base_name << "_inference.cpp" << endl;
    cout << "  - Makefile" << endl;
    cout << "\nTo build the inference executable:" << endl;
    cout << "  cd " << output_dir << " && make" << endl;
    cout << "\nNote: The generated code can be compiled independently without TFLite dependencies." << endl;
    cout << "This code generator uses direct FlatBuffer inspection, requiring only the schema header." << endl;
    
    return 0;
}
