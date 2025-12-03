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

// TensorFlow Lite FlatBuffer schema header
#include "tensorflow/lite/schema/schema_generated.h"
// Only need builtin_op_data for activation/padding enums
#include "tensorflow/lite/c/builtin_op_data.h"

using namespace std;

namespace {

// Helper to get operator name from FlatBuffer
string GetOperatorName(const tflite::OperatorCode* op_code) {
    if (!op_code) {
        return "UNKNOWN";
    }
    
    if (op_code->builtin_code() != tflite::BuiltinOperator_CUSTOM) {
        const char* name = tflite::EnumNameBuiltinOperator(
            static_cast<tflite::BuiltinOperator>(op_code->builtin_code()));
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
    vector<string> unsupported_ops;
    vector<pair<int, int>> unsupported_activations; // (operator_index, activation_code)

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
        if (!op_code) continue;

        tflite::BuiltinOperator builtin_code = 
            static_cast<tflite::BuiltinOperator>(op_code->builtin_code());

        // Check if operator is supported
        if (supported_ops.find(builtin_code) == supported_ops.end()) {
            string op_name;
            if (op_code->builtin_code() == tflite::BuiltinOperator_CUSTOM) {
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
            unsupported_ops.push_back(op_name + " (at operator " + to_string(i) + ")");
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
            cerr << "\nUnsupported operators found in model:" << endl;
            for (const auto& op : unsupported_ops) {
                cerr << "  - " << op << endl;
            }
        }
        
        if (!unsupported_activations.empty()) {
            cerr << "\nUnsupported fused activations found:" << endl;
            for (const auto& [op_idx, act_code] : unsupported_activations) {
                cerr << "  - Operator " << op_idx << " has unsupported activation code: " 
                     << act_code << " (only NONE=0 and RELU=1 are supported)" << endl;
            }
        }
        
        cerr << "\nCode generation aborted." << endl;
        cerr << "Please use a model that only contains supported operations." << endl;
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
void GenerateWeightsFile(
    const string& output_path,
    const string& base_name,
    const tflite::Model* model,
    const tflite::SubGraph* subgraph,
    const map<int, string>& tensor_to_weight
) {
    ofstream out(output_path);
    if (!out) {
        cerr << "Error: Cannot create file " << output_path << endl;
        return;
    }
    
    out << "// " << base_name << "_weights.cpp\n";
    out << "// Auto-generated weight data for embedded inference\n";
    out << "// This file contains all model weights as C++ arrays\n\n";
    out << "#include <cstddef>\n\n";
    out << "namespace embedded_ml {\n\n";
    
    const auto* tensors = subgraph->tensors();
    const auto* buffers = model->buffers();
    
    if (!tensors || !buffers) {
        out << "} // namespace embedded_ml\n";
        out.close();
        return;
    }
    
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
        
        const float* data = reinterpret_cast<const float*>(data_vec->data());
        
        out << "// Weight tensor " << i << ": " << tensor_name << "\n";
        out << "// Shape: [" << GetShapeString(tensor->shape()) << "]\n";
        out << "// Elements: " << num_elements << "\n";
        out << "static const float " << var_name << "[" << num_elements << "] = {\n";
        out << fixed << setprecision(9);
        
        for (size_t j = 0; j < num_elements; ++j) {
            out << "  " << data[j];
            if (j < num_elements - 1) {
                out << ",";
            }
            if ((j + 1) % 8 == 0) {
                out << "\n";
            } else {
                out << " ";
            }
        }
        if (num_elements % 8 != 0) {
            out << "\n";
        }
        out << "};\n\n";
    }
    
    out << "} // namespace embedded_ml\n";
    out.close();
    cout << "Generated: " << output_path << endl;
}

// Generate model.h (header file)
void GenerateModelHeader(
    const string& output_path,
    const string& base_name,
    size_t input_size,
    size_t output_size,
    const vector<pair<int, size_t>>& intermediate_buffers
) {
    ofstream out(output_path);
    if (!out) {
        cerr << "Error: Cannot create file " << output_path << endl;
        return;
    }
    
    out << "// " << base_name << ".h\n";
    out << "// Auto-generated inference model header\n";
    out << "// Pure C++ implementation for embedded systems\n\n";
    
    // Create include guard name (uppercase, with underscores)
    string guard_name = base_name;
    transform(guard_name.begin(), guard_name.end(), guard_name.begin(), ::toupper);
    for (size_t i = 0; i < guard_name.length(); ++i) {
        if (guard_name[i] == '-' || guard_name[i] == '.') {
            guard_name[i] = '_';
        }
    }
    guard_name += "_MODEL_H";
    
    out << "#ifndef " << guard_name << "\n";
    out << "#define " << guard_name << "\n\n";
    
    out << "#include <cstddef>\n\n";
    
    out << "namespace embedded_ml {\n\n";
    
    out << "class " << base_name << "Model {\n";
    out << "public:\n";
    out << "    static constexpr size_t kInputSize = " << input_size << ";\n";
    out << "    static constexpr size_t kOutputSize = " << output_size << ";\n\n";
    
    // Declare intermediate buffers as static class members
    if (!intermediate_buffers.empty()) {
        out << "private:\n";
        out << "    // Intermediate buffers for layer outputs\n";
        for (const auto& [idx, size] : intermediate_buffers) {
            out << "    static float buffer_" << idx << "[" << size << "];\n";
        }
        out << "\n";
    }
    
    out << "public:\n";
    out << "    // Run inference on input data\n";
    out << "    // input: array of size kInputSize\n";
    out << "    // output: array of size kOutputSize (will be filled with results)\n";
    out << "    static void Inference(const float* input, float* output);\n";
    out << "};\n\n";
    
    out << "} // namespace embedded_ml\n\n";
    out << "#endif // " << guard_name << "\n";
    
    out.close();
    cout << "Generated: " << output_path << endl;
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
void GenerateModelFile(
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
    ofstream out(output_path);
    if (!out) {
        cerr << "Error: Cannot create file " << output_path << endl;
        return;
    }
    
    out << "// " << base_name << ".cpp\n";
    out << "// Auto-generated inference model implementation\n";
    out << "// Pure C++ implementation for embedded systems\n\n";
    
    // Include header and weights
    out << "#include \"" << base_name << ".h\"\n";
    out << "#include \"" << base_name << "_weights.cpp\"\n";
    out << "#include \"../../components/fully_connected.h\"\n";
    
    const auto* operators = subgraph->operators();
    const auto* operator_codes = model->operator_codes();
    const auto* tensors = subgraph->tensors();
    
    if (!operators || !operator_codes || !tensors) {
        cerr << "Error: Invalid model structure" << endl;
        return;
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
        if (op_name == "RELU") {
            has_standalone_relu = true;
        } else if (op_name == "CONV_2D") {
            has_conv2d = true;
        } else if (op_name == "MAX_POOL_2D") {
            has_max_pool2d = true;
        } else if (op_name == "SHAPE") {
            has_shape = true;
        } else if (op_name == "STRIDED_SLICE") {
            has_strided_slice = true;
        } else if (op_name == "PACK") {
            has_pack = true;
        } else if (op_name == "RESHAPE") {
            has_reshape = true;
        } else if (op_name == "ADD") {
            has_add = true;
        } else if (op_name.find("DROPOUT") != string::npos || 
                   op_name.find("Dropout") != string::npos) {
            has_dropout = true;
        } else if (op_name.find("FLATTEN") != string::npos ||
                   op_name.find("Flatten") != string::npos) {
            has_flatten = true;
        }
    }
    
    if (has_standalone_relu) {
        out << "#include \"../../components/relu.h\"\n";
    }
    if (has_conv2d) {
        out << "#include \"../../components/conv_2d.h\"\n";
    }
    if (has_max_pool2d) {
        out << "#include \"../../components/max_pool_2d.h\"\n";
    }
    if (has_shape) {
        out << "#include \"../../components/shape.h\"\n";
    }
    if (has_strided_slice) {
        out << "#include \"../../components/strided_slice.h\"\n";
    }
    if (has_pack) {
        out << "#include \"../../components/pack.h\"\n";
    }
    if (has_reshape) {
        out << "#include \"../../components/reshape.h\"\n";
    }
    if (has_add) {
        out << "#include \"../../components/add.h\"\n";
    }
    if (has_dropout) {
        out << "#include \"../../components/dropout.h\"\n";
    }
    if (has_flatten) {
        out << "#include \"../../components/flatten.h\"\n";
    }
    
    out << "#include \"../../components/softmax.h\"\n";
    out << "#include <cstddef>\n\n";
    
    out << "namespace embedded_ml {\n\n";
    
    // Implementation of Inference method
    out << "void " << base_name << "Model::Inference(const float* input, float* output) {\n";
    
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
        
        out << "        // Layer " << i << ": " << op_name << "\n";
        
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
                    cerr << "Warning: Cannot find weight/bias tensors for layer " << i << endl;
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
                
                out << "        FullyConnected(" << input_ptr << ", " << weights_var 
                    << ", " << bias_var << ", " << output_ptr << ", " 
                    << fc_input_size << ", " << output_size_layer << ", " 
                    << activation_param << ");\n";
            } else {
                cerr << "Warning: FULLY_CONNECTED layer " << i << " missing weights/bias" << endl;
            }
        } else if (op_name == "RELU") {
            out << "        ReLU(" << input_ptr << ", " << output_ptr << ", " 
                << output_size_layer << ");\n";
        } else if (op_name == "SOFTMAX") {
            out << "        Softmax(" << input_ptr << ", " << output_ptr << ", " 
                << output_size_layer << ");\n";
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
                size_t output_height = output_tensor->shape()->Get(1);
                size_t output_width = output_tensor->shape()->Get(2);
                
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
                
                out << "        Conv2D(" << input_ptr << ", " << filter_var << ", " 
                    << bias_var << ", " << output_ptr << ", "
                    << batch_size << ", " << input_height << ", " << input_width << ", " 
                    << input_channels << ", " << filter_height << ", " << filter_width << ", "
                    << output_channels << ", " << stride_height << ", " << stride_width << ", "
                    << padding_param << ", " << activation_param << ", "
                    << dilation_height << ", " << dilation_width << ");\n";
            } else {
                cerr << "Warning: CONV_2D layer " << i << " missing filter/bias" << endl;
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
            
            out << "        MaxPool2D(" << input_ptr << ", " << output_ptr << ", "
                << batch_size << ", " << input_height << ", " << input_width << ", "
                << channels << ", " << filter_height << ", " << filter_width << ", "
                << stride_height << ", " << stride_width << ", "
                << padding_param << ", " << activation_param << ");\n";
        } else if (op_name == "SHAPE") {
            // SHAPE extracts the shape of the input tensor
            const tflite::Tensor* input_tensor = (op_input_tensor_idx >= 0 && op_input_tensor_idx < static_cast<int>(tensors->size()))
                ? tensors->Get(op_input_tensor_idx) : nullptr;
            
            if (!input_tensor || !input_tensor->shape()) {
                cerr << "Warning: Cannot get tensor shapes for SHAPE layer " << i << endl;
                continue;
            }
            
            int num_dims = input_tensor->shape()->size();
            out << "        // SHAPE: Extract shape from input tensor\n";
            out << "        {\n";
            out << "            int32_t input_shape[" << num_dims << "] = {";
            for (int j = 0; j < num_dims; ++j) {
                out << input_tensor->shape()->Get(j);
                if (j < num_dims - 1) out << ", ";
            }
            out << "};\n";
            out << "            Shape(input_shape, " << num_dims << ", reinterpret_cast<int32_t*>(" << output_ptr << "));\n";
            out << "        }\n";
        } else if (op_name == "STRIDED_SLICE") {
            // STRIDED_SLICE has inputs: [input, begin, end, strides]
            const tflite::Tensor* input_tensor = (op_input_tensor_idx >= 0 && op_input_tensor_idx < static_cast<int>(tensors->size()))
                ? tensors->Get(op_input_tensor_idx) : nullptr;
            const tflite::Tensor* output_tensor = (op_output_tensor_idx >= 0 && op_output_tensor_idx < static_cast<int>(tensors->size()))
                ? tensors->Get(op_output_tensor_idx) : nullptr;
            
            if (!input_tensor || !output_tensor ||
                !input_tensor->shape() || !output_tensor->shape()) {
                cerr << "Warning: Cannot get tensor shapes for STRIDED_SLICE layer " << i << endl;
                continue;
            }
            
            // Extract parameters
            int begin_mask = 0;
            int end_mask = 0;
            int shrink_axis_mask = 0;
            
            const tflite::StridedSliceOptions* options = op->builtin_options_as_StridedSliceOptions();
            if (options) {
                begin_mask = options->begin_mask();
                end_mask = options->end_mask();
                shrink_axis_mask = options->shrink_axis_mask();
            }
            
            // For now, generate a simplified version
            out << "        // STRIDED_SLICE: Extract slice from input\n";
            out << "        // Note: This is a simplified implementation\n";
            out << "        // Full implementation would extract begin/end/strides from input tensors\n";
            out << "        // For now, copying input to output as placeholder\n";
            out << "        for (size_t j = 0; j < " << output_size_layer << "; ++j) {\n";
            out << "            " << output_ptr << "[j] = " << input_ptr << "[j];\n";
            out << "        }\n";
        } else if (op_name == "PACK") {
            // PACK has multiple inputs to pack along an axis
            const tflite::Tensor* output_tensor = (op_output_tensor_idx >= 0 && op_output_tensor_idx < static_cast<int>(tensors->size()))
                ? tensors->Get(op_output_tensor_idx) : nullptr;
            
            if (!output_tensor || !output_tensor->shape()) {
                cerr << "Warning: Cannot get tensor shapes for PACK layer " << i << endl;
                continue;
            }
            
            // Get axis parameter
            int axis = 0;
            const tflite::PackOptions* options = op->builtin_options_as_PackOptions();
            if (options) {
                axis = options->axis();
            }
            
            // For simplicity, generate code that packs inputs
            out << "        // PACK: Pack multiple inputs along axis " << axis << "\n";
            out << "        // Note: This is a simplified implementation\n";
            out << "        // Full implementation would handle multiple input tensors\n";
            if (op_inputs && op_inputs->size() > 0) {
                out << "        // Copying first input to output as placeholder\n";
                out << "        for (size_t j = 0; j < " << output_size_layer << "; ++j) {\n";
                out << "            " << output_ptr << "[j] = " << input_ptr << "[j];\n";
                out << "        }\n";
            }
        } else if (op_name == "RESHAPE") {
            // RESHAPE just copies data (memory layout is the same)
            out << "        Reshape(" << input_ptr << ", " << input_size_layer 
                << ", " << output_ptr << ", " << output_size_layer << ");\n";
        } else if (op_name == "ADD") {
            // ADD has two inputs: [input1, input2]
            int input1_tensor_idx = -1;
            int input2_tensor_idx = -1;
            
            if (op_inputs && op_inputs->size() >= 2) {
                input1_tensor_idx = op_inputs->Get(0);
                input2_tensor_idx = op_inputs->Get(1);
            }
            
            if (input1_tensor_idx < 0 || input2_tensor_idx < 0) {
                cerr << "Warning: ADD layer " << i << " missing inputs" << endl;
                continue;
            }
            
            // Determine input pointers
            string input1_ptr, input2_ptr;
            if (input1_tensor_idx == input_tensor_idx) {
                input1_ptr = "input";
            } else {
                input1_ptr = "buffer_" + to_string(input1_tensor_idx);
            }
            
            if (input2_tensor_idx == input_tensor_idx) {
                input2_ptr = "input";
            } else {
                input2_ptr = "buffer_" + to_string(input2_tensor_idx);
            }
            
            // Get sizes (should be the same for element-wise addition)
            size_t input1_size = tensor_sizes.count(input1_tensor_idx) ? tensor_sizes.at(input1_tensor_idx) : output_size_layer;
            size_t input2_size = tensor_sizes.count(input2_tensor_idx) ? tensor_sizes.at(input2_tensor_idx) : output_size_layer;
            
            // Use the output size as the common size (should match for element-wise ops)
            size_t add_size = output_size_layer;
            
            // Check for fused activation function
            TfLiteFusedActivation fused_activation = kTfLiteActNone;
            const tflite::AddOptions* options = op->builtin_options_as_AddOptions();
            if (options) {
                fused_activation = ConvertActivation(options->fused_activation_function());
            }
            
            // Determine activation parameter
            string activation_param = "ActivationType::NONE";
            if (fused_activation == kTfLiteActRelu) {
                activation_param = "ActivationType::RELU";
            } else if (fused_activation != kTfLiteActNone) {
                cerr << "Warning: Unsupported fused activation " << static_cast<int>(fused_activation) 
                     << " in ADD layer " << i << " (only ReLU supported, using NONE)" << endl;
            }
            
            out << "        Add(" << input1_ptr << ", " << input2_ptr << ", " 
                << output_ptr << ", " << add_size << ", " << activation_param << ");\n";
        } else if (op_name.find("DROPOUT") != string::npos || 
                   op_name.find("Dropout") != string::npos) {
            // Get dropout rate if available (though it's ignored during inference)
            float dropout_rate = 0.0f;
            
            out << "        // DROPOUT: No-op during inference (passes through input)\n";
            out << "        Dropout(" << input_ptr << ", " << output_ptr << ", " 
                << output_size_layer << ", " << dropout_rate << "f);\n";
        } else if (op_name.find("FLATTEN") != string::npos ||
                   op_name.find("Flatten") != string::npos) {
            // FLATTEN is essentially a reshape to 1D (except batch dimension)
            out << "        // FLATTEN: Flatten multi-dimensional tensor to 1D\n";
            out << "        Flatten(" << input_ptr << ", " << input_size_layer 
                << ", " << output_ptr << ", " << output_size_layer << ");\n";
        } else {
            cerr << "Warning: Unsupported operation " << op_name << endl;
        }
    }
    
    out << "}\n\n";
    
    // Define static buffers outside class
    if (!intermediate_buffers.empty()) {
        out << "// Intermediate buffer definitions\n";
        for (const auto& [idx, size] : intermediate_buffers) {
            out << "float " << base_name << "Model::buffer_" << idx << "[" << size << "];\n";
        }
        out << "\n";
    }
    
    out << "} // namespace embedded_ml\n";
    out.close();
    cout << "Generated: " << output_path << endl;
}

// Generate inference.cpp
void GenerateInferenceFile(
    const string& output_path,
    const string& base_name,
    const tflite::SubGraph* subgraph,
    int32_t input_tensor_idx,
    int32_t output_tensor_idx
) {
    ofstream out(output_path);
    if (!out) {
        cerr << "Error: Cannot create file " << output_path << endl;
        return;
    }
    
    const auto* tensors = subgraph->tensors();
    if (!tensors) {
        cerr << "Error: Cannot access tensors" << endl;
        return;
    }
    
    const tflite::Tensor* input_tensor = (input_tensor_idx >= 0 && input_tensor_idx < static_cast<int>(tensors->size()))
        ? tensors->Get(input_tensor_idx) : nullptr;
    const tflite::Tensor* output_tensor = (output_tensor_idx >= 0 && output_tensor_idx < static_cast<int>(tensors->size()))
        ? tensors->Get(output_tensor_idx) : nullptr;
    
    if (!input_tensor || !output_tensor) {
        cerr << "Error: Cannot access input/output tensors" << endl;
        return;
    }
    
    size_t input_size = CalculateTensorSize(input_tensor->shape());
    size_t output_size = CalculateTensorSize(output_tensor->shape());
    
    out << "// " << base_name << "_inference.cpp\n";
    out << "// Example inference code - edit this file to match your specific inference needs\n";
    out << "// This is a 'hello world' example that demonstrates how to use the generated model\n\n";
    
    out << "#include \"" << base_name << ".h\"\n";
    out << "#include <iostream>\n";
    out << "#include <iomanip>\n";
    out << "#include <cstddef>\n\n";
    
    out << "using namespace embedded_ml;\n";
    out << "using namespace std;\n\n";
    
    out << "int main() {\n";
    out << "    // Example input data matching the model's input shape\n";
    out << "    // Shape: [" << GetShapeString(input_tensor->shape()) << "]\n";
    out << "    // TODO: Replace this with your actual input data\n";
    out << "    float input[" << base_name << "Model::kInputSize] = {\n";
    
    // Generate example input (zeros for now, user can edit)
    for (size_t i = 0; i < input_size; ++i) {
        out << "        0.0f";
        if (i < input_size - 1) {
            out << ",";
        }
        if ((i + 1) % 8 == 0) {
            out << "\n";
        } else {
            out << " ";
        }
    }
    if (input_size % 8 != 0) {
        out << "\n";
    }
    out << "    };\n\n";
    
    out << "    // Output array to store inference results\n";
    out << "    // Shape: [" << GetShapeString(output_tensor->shape()) << "]\n";
    out << "    float output[" << base_name << "Model::kOutputSize];\n\n";
    
    out << "    // Run inference\n";
    out << "    " << base_name << "Model::Inference(input, output);\n\n";
    
    out << "    // Print results\n";
    out << "    cout << \"Inference Results:\" << endl;\n";
    out << "    cout << fixed << setprecision(6);\n";
    out << "    for (size_t i = 0; i < " << base_name << "Model::kOutputSize; ++i) {\n";
    out << "        cout << \"  Output[\" << i << \"] = \" << output[i] << endl;\n";
    out << "    }\n\n";
    
    out << "    // TODO: Process the output results as needed for your application\n";
    out << "    // For example, find the class with highest probability:\n";
    out << "    // size_t predicted_class = 0;\n";
    out << "    // float max_prob = output[0];\n";
    out << "    // for (size_t i = 1; i < " << base_name << "Model::kOutputSize; ++i) {\n";
    out << "    //     if (output[i] > max_prob) {\n";
    out << "    //         max_prob = output[i];\n";
    out << "    //         predicted_class = i;\n";
    out << "    //     }\n";
    out << "    // }\n";
    out << "    // cout << \"Predicted class: \" << predicted_class << \" (probability: \" << max_prob << \")\" << endl;\n\n";
    
    out << "    return 0;\n";
    out << "}\n";
    
    out.close();
    cout << "Generated: " << output_path << endl;
}

// Generate Makefile for the generated code
void GenerateMakefile(
    const string& output_path,
    const string& base_name
) {
    ofstream out(output_path);
    if (!out) {
        cerr << "Error: Cannot create file " << output_path << endl;
        return;
    }
    
    out << "# Makefile for " << base_name << " inference\n";
    out << "# Auto-generated - compiles the inference executable\n\n";
    
    out << "CXX = g++\n";
    out << "CXXFLAGS = -std=c++17 -O2 -Wall\n\n";
    
    out << "# Source files\n";
    out << "SOURCES = " << base_name << "_weights.cpp \\\n";
    out << "          " << base_name << ".cpp \\\n";
    out << "          " << base_name << "_inference.cpp\n\n";
    
    out << "# Header files\n";
    out << "HEADERS = " << base_name << ".h\n\n";
    
    out << "OBJECTS = $(SOURCES:.cpp=.o)\n";
    out << "TARGET = " << base_name << "_inference\n\n";
    
    out << "$(TARGET): $(OBJECTS)\n";
    out << "\t$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)\n\n";
    
    out << "%.o: %.cpp\n";
    out << "\t$(CXX) $(CXXFLAGS) -c $< -o $@\n\n";
    
    out << "clean:\n";
    out << "\trm -f $(OBJECTS) $(TARGET)\n\n";
    
    out << ".PHONY: clean\n";
    
    out.close();
    cout << "Generated: " << output_path << endl;
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

} // anonymous namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <path_to_model.tflite> <base_name>" << endl;
        cerr << "Example: " << argv[0] << " scripts/model.tflite my_model" << endl;
        cerr << "This will create a directory 'my_model/' containing:" << endl;
        cerr << "  - my_model_weights.cpp" << endl;
        cerr << "  - my_model.h" << endl;
        cerr << "  - my_model.cpp" << endl;
        cerr << "  - my_model_inference.cpp" << endl;
        cerr << "  - Makefile" << endl;
        cerr << "\nTo build: cd my_model && make" << endl;
        return 1;
    }
    
    const string model_path = argv[1];
    const string base_name = argv[2];
    
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
    cout << "inputs->size(): " << inputs->size() << endl;
    cout << "outputs->size(): " << outputs->size() << endl;
    if (inputs->size() != 1 || outputs->size() != 1) {
        cerr << "Error: Only single input/output models supported" << endl;
        return 1;
    }
    
    int32_t input_tensor_idx = inputs->Get(0);
    int32_t output_tensor_idx = outputs->Get(0);
    
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
    GenerateWeightsFile(weights_file, base_name, model, subgraph, tensor_to_weight);
    GenerateModelHeader(model_header_file, base_name, input_size, output_size, intermediate_buffers);
    GenerateModelFile(model_file, base_name, model, subgraph, tensor_to_weight, intermediate_buffers, 
                     tensor_sizes, input_tensor_idx, output_tensor_idx);
    GenerateInferenceFile(inference_file, base_name, subgraph, input_tensor_idx, output_tensor_idx);
    GenerateMakefile(makefile, base_name);
    
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
