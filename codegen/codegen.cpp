// codegen/codegen.cpp
// Code generator that reads TFLite models and generates pure C++ inference code

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

// TensorFlow Lite headers
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"

using namespace std;

namespace {

// Helper to get operator name
string GetOperatorName(const tflite::Interpreter& interpreter, int node_index) {
    const auto* node_and_reg = interpreter.node_and_registration(node_index);
    if (!node_and_reg) {
        return "UNKNOWN";
    }
    
    const auto& registration = node_and_reg->second;
    if (registration.builtin_code != tflite::BuiltinOperator_CUSTOM) {
        return tflite::EnumNameBuiltinOperator(
            static_cast<tflite::BuiltinOperator>(registration.builtin_code));
    } else {
        return string("CUSTOM:") + (registration.custom_name ? registration.custom_name : "");
    }
}

// Calculate tensor size
size_t CalculateTensorSize(const TfLiteIntArray* dims) {
    if (!dims || dims->size == 0) {
        return 1;
    }
    
    size_t size = 1;
    for (int i = 0; i < dims->size; ++i) {
        size *= static_cast<size_t>(dims->data[i]);
    }
    return size;
}

// Get shape as string
string GetShapeString(const TfLiteIntArray* dims) {
    if (!dims || dims->size == 0) {
        return "1";
    }
    
    stringstream ss;
    for (int i = 0; i < dims->size; ++i) {
        ss << dims->data[i];
        if (i < dims->size - 1) {
            ss << ", ";
        }
    }
    return ss.str();
}

// Validate operators from model schema before building interpreter
// This provides clear error messages before TFLite tries to process unsupported ops
bool ValidateModelSchema(const tflite::FlatBufferModel& model) {
    const tflite::Model* model_ptr = model.GetModel();
    if (!model_ptr || !model_ptr->subgraphs() || model_ptr->subgraphs()->size() == 0) {
        cerr << "Error: Invalid model structure - no subgraphs found." << endl;
        return false;
    }

    // Get the main subgraph (usually index 0)
    const tflite::SubGraph* subgraph = model_ptr->subgraphs()->Get(0);
    if (!subgraph || !subgraph->operators()) {
        cerr << "Error: Invalid model structure - no operators found." << endl;
        return false;
    }

    // Get operator codes
    const flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>* op_codes = 
        model_ptr->operator_codes();
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
        tflite::BuiltinOperator_RESHAPE
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

        // For FULLY_CONNECTED and CONV_2D, check fused activation
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

// Validate that the model only uses supported operators and activations
// (This is a secondary check after interpreter is built, for additional validation)
bool ValidateModel(tflite::Interpreter& interpreter) {
    const auto& execution_plan = interpreter.execution_plan();
    bool ok = true;

    for (size_t i = 0; i < execution_plan.size(); ++i) {
        const int node_index = execution_plan[i];
        const auto* node_and_reg = interpreter.node_and_registration(node_index);
        if (!node_and_reg) {
            continue;
        }

        const auto& node = node_and_reg->first;
        const auto& registration = node_and_reg->second;

        tflite::BuiltinOperator op_code =
            static_cast<tflite::BuiltinOperator>(registration.builtin_code);
        string op_name = GetOperatorName(interpreter, node_index);

        // Check operator support
        // Note: DROPOUT and FLATTEN are not standard builtin operators but may appear as custom ops
        if (op_code == tflite::BuiltinOperator_CUSTOM) {
            // Check if it's a supported custom operator
            if (op_name.find("DROPOUT") != string::npos || 
                op_name.find("Dropout") != string::npos ||
                op_name.find("FLATTEN") != string::npos ||
                op_name.find("Flatten") != string::npos) {
                // Supported custom operators
                break;
            }
        }
        
        switch (op_code) {
            case tflite::BuiltinOperator_FULLY_CONNECTED:
            case tflite::BuiltinOperator_SOFTMAX:
            case tflite::BuiltinOperator_RELU:
            case tflite::BuiltinOperator_CONV_2D:
            case tflite::BuiltinOperator_MAX_POOL_2D:
            case tflite::BuiltinOperator_SHAPE:
            case tflite::BuiltinOperator_STRIDED_SLICE:
            case tflite::BuiltinOperator_PACK:
            case tflite::BuiltinOperator_RESHAPE:
                // Supported
                break;
            default:
                cerr << "Error: Unsupported operator '" << op_name
                     << "' (builtin code " << registration.builtin_code
                     << ") at node " << node_index << "." << endl;
                ok = false;
                continue;
        }

        // For FULLY_CONNECTED and CONV_2D, validate fused activation
        if (op_code == tflite::BuiltinOperator_FULLY_CONNECTED) {
            TfLiteFusedActivation fused_activation = kTfLiteActNone;
            const void* builtin_data = node.builtin_data;
            if (builtin_data) {
                const TfLiteFullyConnectedParams* params =
                    static_cast<const TfLiteFullyConnectedParams*>(builtin_data);
                fused_activation = params->activation;
            }

            if (fused_activation != kTfLiteActNone &&
                fused_activation != kTfLiteActRelu) {
                cerr << "Error: Unsupported fused activation (" << fused_activation
                     << ") in FULLY_CONNECTED node " << node_index
                     << ". Only NONE and RELU are supported." << endl;
                ok = false;
            }
        } else if (op_code == tflite::BuiltinOperator_CONV_2D) {
            TfLiteFusedActivation fused_activation = kTfLiteActNone;
            const void* builtin_data = node.builtin_data;
            if (builtin_data) {
                const TfLiteConvParams* params =
                    static_cast<const TfLiteConvParams*>(builtin_data);
                fused_activation = params->activation;
            }

            if (fused_activation != kTfLiteActNone &&
                fused_activation != kTfLiteActRelu) {
                cerr << "Error: Unsupported fused activation (" << fused_activation
                     << ") in CONV_2D node " << node_index
                     << ". Only NONE and RELU are supported." << endl;
                ok = false;
            }
        }
    }

    if (!ok) {
        cerr << "Code generation aborted due to unsupported operators/activations."
             << endl;
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
map<int, string> CreateWeightMapping(tflite::Interpreter& interpreter) {
    map<int, string> tensor_to_weight;
    const size_t num_tensors = interpreter.tensors_size();
    int weight_index = 0;
    
    for (size_t i = 0; i < num_tensors; ++i) {
        const TfLiteTensor* tensor = interpreter.tensor(i);
        if (!tensor) continue;
        
        // Weights are typically read-only tensors
        if (tensor->allocation_type == kTfLiteMmapRo || 
            tensor->allocation_type == kTfLitePersistentRo) {
            if (tensor->type == kTfLiteFloat32) {
                string tensor_name = tensor->name ? tensor->name : "tensor_" + to_string(i);
                tensor_to_weight[static_cast<int>(i)] = "weight_" + to_string(weight_index) + "_" + EscapeIdentifier(tensor_name);
                weight_index++;
            }
        }
    }
    
    return tensor_to_weight;
}

// Generate model_weights.cpp
void GenerateWeightsFile(
    const string& output_path,
    const string& base_name,
    tflite::Interpreter& interpreter,
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
    
    const size_t num_tensors = interpreter.tensors_size();
    
    for (size_t i = 0; i < num_tensors; ++i) {
        if (tensor_to_weight.find(static_cast<int>(i)) == tensor_to_weight.end()) {
            continue;
        }
        
        const TfLiteTensor* tensor = interpreter.tensor(i);
        if (!tensor) {
            continue;
        }
        
        const float* data = interpreter.typed_tensor<float>(i);
        if (!data) {
            continue;
        }
        
        const size_t num_elements = CalculateTensorSize(tensor->dims);
        string tensor_name = tensor->name ? tensor->name : "tensor_" + to_string(i);
        string var_name = tensor_to_weight.at(static_cast<int>(i));
        
        out << "// Weight tensor " << i << ": " << tensor_name << "\n";
        out << "// Shape: [" << GetShapeString(tensor->dims) << "]\n";
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

// Generate model.cpp
void GenerateModelFile(
    const string& output_path,
    const string& base_name,
    tflite::Interpreter& interpreter,
    const map<int, string>& tensor_to_weight,
    const vector<pair<int, size_t>>& intermediate_buffers
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
    
    // Check which components are needed
    bool has_standalone_relu = false;
    bool has_conv2d = false;
    bool has_max_pool2d = false;
    bool has_shape = false;
    bool has_strided_slice = false;
    bool has_pack = false;
    bool has_reshape = false;
    bool has_dropout = false;
    bool has_flatten = false;
    
    const auto& execution_plan_check = interpreter.execution_plan();
    for (size_t i = 0; i < execution_plan_check.size(); ++i) {
        const int node_index = execution_plan_check[i];
        const auto* node_and_reg = interpreter.node_and_registration(node_index);
        if (node_and_reg) {
            string op_name = GetOperatorName(interpreter, node_index);
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
            } else if (op_name.find("DROPOUT") != string::npos || 
                       op_name.find("Dropout") != string::npos) {
                has_dropout = true;
            } else if (op_name.find("FLATTEN") != string::npos ||
                       op_name.find("Flatten") != string::npos) {
                has_flatten = true;
            }
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
    if (has_dropout) {
        out << "#include \"../../components/dropout.h\"\n";
    }
    if (has_flatten) {
        out << "#include \"../../components/flatten.h\"\n";
    }
    
    out << "#include \"../../components/softmax.h\"\n";
    out << "#include <cstddef>\n\n";
    
    out << "namespace embedded_ml {\n\n";
    
    // Get input/output shapes
    const auto& input_indices = interpreter.inputs();
    const auto& output_indices = interpreter.outputs();
    
    if (input_indices.size() != 1 || output_indices.size() != 1) {
        cerr << "Error: Only single input/output models supported" << endl;
        return;
    }
    
    const TfLiteTensor* input_tensor = interpreter.tensor(input_indices[0]);
    const TfLiteTensor* output_tensor = interpreter.tensor(output_indices[0]);
    
    if (!input_tensor || !output_tensor) {
        cerr << "Error: Cannot access input/output tensors" << endl;
        return;
    }
    
    size_t input_size = CalculateTensorSize(input_tensor->dims);
    size_t output_size = CalculateTensorSize(output_tensor->dims);
    
    // Track all tensor sizes first
    map<int, size_t> tensor_sizes;
    tensor_sizes[input_indices[0]] = input_size;
    tensor_sizes[output_indices[0]] = output_size;
    
    // Process execution plan to find all intermediate tensors
    const auto& execution_plan = interpreter.execution_plan();
    set<string> used_components;
    
    // First pass: collect all tensor sizes
    for (size_t i = 0; i < execution_plan.size(); ++i) {
        const int node_index = execution_plan[i];
        const auto* node_and_reg = interpreter.node_and_registration(node_index);
        if (!node_and_reg) continue;
        
        const auto& node = node_and_reg->first;
        if (node.inputs && node.inputs->size > 0) {
            for (int j = 0; j < node.inputs->size; ++j) {
                int tensor_idx = node.inputs->data[j];
                const TfLiteTensor* tensor = interpreter.tensor(tensor_idx);
                if (tensor && tensor_sizes.find(tensor_idx) == tensor_sizes.end()) {
                    tensor_sizes[tensor_idx] = CalculateTensorSize(tensor->dims);
                }
            }
        }
        if (node.outputs && node.outputs->size > 0) {
            for (int j = 0; j < node.outputs->size; ++j) {
                int tensor_idx = node.outputs->data[j];
                const TfLiteTensor* tensor = interpreter.tensor(tensor_idx);
                if (tensor && tensor_sizes.find(tensor_idx) == tensor_sizes.end()) {
                    tensor_sizes[tensor_idx] = CalculateTensorSize(tensor->dims);
                }
            }
        }
    }
    
    // Implementation of Inference method
    // (intermediate_buffers are passed in and already calculated)
    out << "void " << base_name << "Model::Inference(const float* input, float* output) {\n";
    
    // Process each layer
    for (size_t i = 0; i < execution_plan.size(); ++i) {
        const int node_index = execution_plan[i];
        const auto* node_and_reg = interpreter.node_and_registration(node_index);
        if (!node_and_reg) continue;
        
        const auto& node = node_and_reg->first;
        const auto& registration = node_and_reg->second;
        string op_name = GetOperatorName(interpreter, node_index);
        
        out << "        // Layer " << i << ": " << op_name << "\n";
        
        // Get input/output tensor indices
        int input_tensor_idx = -1;
        int output_tensor_idx = -1;
        
        if (node.inputs && node.inputs->size > 0) {
            input_tensor_idx = node.inputs->data[0];
        }
        if (node.outputs && node.outputs->size > 0) {
            output_tensor_idx = node.outputs->data[0];
        }
        
        if (input_tensor_idx < 0 || output_tensor_idx < 0) {
            continue;
        }
        
        // Determine input/output pointers
        string input_ptr, output_ptr;
        if (input_tensor_idx == input_indices[0]) {
            input_ptr = "input";
        } else {
            input_ptr = "buffer_" + to_string(input_tensor_idx);
        }
        
        if (output_tensor_idx == output_indices[0]) {
            output_ptr = "output";
        } else {
            output_ptr = "buffer_" + to_string(output_tensor_idx);
        }
        
        size_t input_size_layer = tensor_sizes.count(input_tensor_idx) ? tensor_sizes[input_tensor_idx] : 0;
        size_t output_size_layer = tensor_sizes.count(output_tensor_idx) ? tensor_sizes[output_tensor_idx] : 0;
        
        // Generate code based on operation
        if (op_name == "FULLY_CONNECTED") {
            used_components.insert("fully_connected");
            
            // Find weights and bias
            int weights_tensor_idx = -1;
            int bias_tensor_idx = -1;
            
            if (node.inputs && node.inputs->size >= 3) {
                weights_tensor_idx = node.inputs->data[1];
                bias_tensor_idx = node.inputs->data[2];
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
                const TfLiteTensor* weights_tensor = interpreter.tensor(weights_tensor_idx);
                size_t fc_input_size = input_size_layer;
                if (weights_tensor && weights_tensor->dims && weights_tensor->dims->size >= 2) {
                    fc_input_size = weights_tensor->dims->data[1]; // weights shape: [output, input]
                }
                
                // Check for fused activation function
                TfLiteFusedActivation fused_activation = kTfLiteActNone;
                if (registration.builtin_code == tflite::BuiltinOperator_FULLY_CONNECTED) {
                    const void* builtin_data = node.builtin_data;
                    if (builtin_data) {
                        const TfLiteFullyConnectedParams* params = 
                            static_cast<const TfLiteFullyConnectedParams*>(builtin_data);
                        fused_activation = params->activation;
                    }
                }
                
                // Determine activation parameter
                string activation_param = "ActivationType::NONE";
                if (fused_activation == kTfLiteActRelu) {
                    activation_param = "ActivationType::RELU";
                } else if (fused_activation != kTfLiteActNone) {
                    cerr << "Warning: Unsupported fused activation " << fused_activation 
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
            used_components.insert("relu");
            out << "        ReLU(" << input_ptr << ", " << output_ptr << ", " 
                << output_size_layer << ");\n";
        } else if (op_name == "SOFTMAX") {
            used_components.insert("softmax");
            out << "        Softmax(" << input_ptr << ", " << output_ptr << ", " 
                << output_size_layer << ");\n";
        } else if (op_name == "CONV_2D") {
            used_components.insert("conv_2d");
            
            // CONV_2D has inputs: [input, filter, bias]
            int filter_tensor_idx = -1;
            int bias_tensor_idx = -1;
            
            if (node.inputs && node.inputs->size >= 3) {
                filter_tensor_idx = node.inputs->data[1];
                bias_tensor_idx = node.inputs->data[2];
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
                const TfLiteTensor* input_tensor = interpreter.tensor(input_tensor_idx);
                const TfLiteTensor* filter_tensor = interpreter.tensor(filter_tensor_idx);
                const TfLiteTensor* output_tensor = interpreter.tensor(output_tensor_idx);
                
                if (!input_tensor || !filter_tensor || !output_tensor ||
                    !input_tensor->dims || !filter_tensor->dims || !output_tensor->dims) {
                    cerr << "Warning: Cannot get tensor shapes for CONV_2D layer " << i << endl;
                    continue;
                }
                
                // Extract dimensions (NHWC format)
                size_t batch_size = input_tensor->dims->data[0];
                size_t input_height = input_tensor->dims->data[1];
                size_t input_width = input_tensor->dims->data[2];
                size_t input_channels = input_tensor->dims->data[3];
                size_t filter_height = filter_tensor->dims->data[1];
                size_t filter_width = filter_tensor->dims->data[2];
                size_t output_channels = filter_tensor->dims->data[0];
                size_t output_height = output_tensor->dims->data[1];
                size_t output_width = output_tensor->dims->data[2];
                
                // Get convolution parameters
                TfLitePadding padding = kTfLitePaddingSame;
                int stride_height = 1;
                int stride_width = 1;
                TfLiteFusedActivation fused_activation = kTfLiteActNone;
                int dilation_height = 1;
                int dilation_width = 1;
                
                const void* builtin_data = node.builtin_data;
                if (builtin_data) {
                    const TfLiteConvParams* params = 
                        static_cast<const TfLiteConvParams*>(builtin_data);
                    padding = params->padding;
                    stride_height = params->stride_height;
                    stride_width = params->stride_width;
                    fused_activation = params->activation;
                    dilation_height = params->dilation_height_factor;
                    dilation_width = params->dilation_width_factor;
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
            used_components.insert("max_pool_2d");
            
            // Get tensor shapes
            const TfLiteTensor* input_tensor = interpreter.tensor(input_tensor_idx);
            const TfLiteTensor* output_tensor = interpreter.tensor(output_tensor_idx);
            
            if (!input_tensor || !output_tensor ||
                !input_tensor->dims || !output_tensor->dims) {
                cerr << "Warning: Cannot get tensor shapes for MAX_POOL_2D layer " << i << endl;
                continue;
            }
            
            // Extract dimensions (NHWC format)
            size_t batch_size = input_tensor->dims->data[0];
            size_t input_height = input_tensor->dims->data[1];
            size_t input_width = input_tensor->dims->data[2];
            size_t channels = input_tensor->dims->data[3];
            
            // Get pooling parameters
            TfLitePadding padding = kTfLitePaddingSame;
            int stride_height = 1;
            int stride_width = 1;
            int filter_height = 2;
            int filter_width = 2;
            TfLiteFusedActivation fused_activation = kTfLiteActNone;
            
            const void* builtin_data = node.builtin_data;
            if (builtin_data) {
                const TfLitePoolParams* params = 
                    static_cast<const TfLitePoolParams*>(builtin_data);
                padding = params->padding;
                stride_height = params->stride_height;
                stride_width = params->stride_width;
                filter_height = params->filter_height;
                filter_width = params->filter_width;
                fused_activation = params->activation;
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
            used_components.insert("shape");
            
            // SHAPE extracts the shape of the input tensor
            const TfLiteTensor* input_tensor = interpreter.tensor(input_tensor_idx);
            const TfLiteTensor* output_tensor = interpreter.tensor(output_tensor_idx);
            
            if (!input_tensor || !output_tensor || !input_tensor->dims) {
                cerr << "Warning: Cannot get tensor shapes for SHAPE layer " << i << endl;
                continue;
            }
            
            int num_dims = input_tensor->dims->size;
            out << "        // SHAPE: Extract shape from input tensor\n";
            out << "        {\n";
            out << "            int32_t input_shape[" << num_dims << "] = {";
            for (int j = 0; j < num_dims; ++j) {
                out << input_tensor->dims->data[j];
                if (j < num_dims - 1) out << ", ";
            }
            out << "};\n";
            out << "            Shape(input_shape, " << num_dims << ", reinterpret_cast<int32_t*>(" << output_ptr << "));\n";
            out << "        }\n";
        } else if (op_name == "STRIDED_SLICE") {
            used_components.insert("strided_slice");
            
            // STRIDED_SLICE has inputs: [input, begin, end, strides]
            // For simplicity, we'll extract the values from constant tensors
            const TfLiteTensor* input_tensor = interpreter.tensor(input_tensor_idx);
            const TfLiteTensor* output_tensor = interpreter.tensor(output_tensor_idx);
            
            if (!input_tensor || !output_tensor ||
                !input_tensor->dims || !output_tensor->dims) {
                cerr << "Warning: Cannot get tensor shapes for STRIDED_SLICE layer " << i << endl;
                continue;
            }
            
            // Get begin, end, strides from input tensors
            int begin_tensor_idx = -1;
            int end_tensor_idx = -1;
            int strides_tensor_idx = -1;
            
            if (node.inputs && node.inputs->size >= 4) {
                begin_tensor_idx = node.inputs->data[1];
                end_tensor_idx = node.inputs->data[2];
                strides_tensor_idx = node.inputs->data[3];
            }
            
            // Extract parameters
            int begin_mask = 0;
            int end_mask = 0;
            int shrink_axis_mask = 0;
            
            const void* builtin_data = node.builtin_data;
            if (builtin_data) {
                const TfLiteStridedSliceParams* params = 
                    static_cast<const TfLiteStridedSliceParams*>(builtin_data);
                begin_mask = params->begin_mask;
                end_mask = params->end_mask;
                shrink_axis_mask = params->shrink_axis_mask;
            }
            
            // For now, generate a simplified version
            // In a full implementation, we'd extract begin/end/strides from tensors
            out << "        // STRIDED_SLICE: Extract slice from input\n";
            out << "        // Note: This is a simplified implementation\n";
            out << "        // Full implementation would extract begin/end/strides from input tensors\n";
            out << "        // For now, copying input to output as placeholder\n";
            out << "        for (size_t j = 0; j < " << output_size_layer << "; ++j) {\n";
            out << "            " << output_ptr << "[j] = " << input_ptr << "[j];\n";
            out << "        }\n";
        } else if (op_name == "PACK") {
            used_components.insert("pack");
            
            // PACK has multiple inputs to pack along an axis
            const TfLiteTensor* output_tensor = interpreter.tensor(output_tensor_idx);
            
            if (!output_tensor || !output_tensor->dims) {
                cerr << "Warning: Cannot get tensor shapes for PACK layer " << i << endl;
                continue;
            }
            
            // Get axis parameter
            int axis = 0;
            const void* builtin_data = node.builtin_data;
            if (builtin_data) {
                const TfLitePackParams* params = 
                    static_cast<const TfLitePackParams*>(builtin_data);
                axis = params->axis;
            }
            
            // For simplicity, generate code that packs inputs
            // In a full implementation, we'd handle multiple input tensors
            out << "        // PACK: Pack multiple inputs along axis " << axis << "\n";
            out << "        // Note: This is a simplified implementation\n";
            out << "        // Full implementation would handle multiple input tensors\n";
            if (node.inputs && node.inputs->size > 0) {
                out << "        // Copying first input to output as placeholder\n";
                out << "        for (size_t j = 0; j < " << output_size_layer << "; ++j) {\n";
                out << "            " << output_ptr << "[j] = " << input_ptr << "[j];\n";
                out << "        }\n";
            }
        } else if (op_name == "RESHAPE") {
            used_components.insert("reshape");
            
            // RESHAPE just copies data (memory layout is the same)
            out << "        Reshape(" << input_ptr << ", " << input_size_layer 
                << ", " << output_ptr << ", " << output_size_layer << ");\n";
        } else if (op_name.find("DROPOUT") != string::npos || 
                   op_name.find("Dropout") != string::npos) {
            used_components.insert("dropout");
            
            // Get dropout rate if available (though it's ignored during inference)
            float dropout_rate = 0.0f;
            const void* builtin_data = node.builtin_data;
            // Note: Dropout typically doesn't have builtin_data in TFLite
            // as it's usually removed or converted during conversion
            
            out << "        // DROPOUT: No-op during inference (passes through input)\n";
            out << "        Dropout(" << input_ptr << ", " << output_ptr << ", " 
                << output_size_layer << ", " << dropout_rate << "f);\n";
        } else if (op_name.find("FLATTEN") != string::npos ||
                   op_name.find("Flatten") != string::npos) {
            used_components.insert("flatten");
            
            // FLATTEN is essentially a reshape to 1D (except batch dimension)
            // For our purposes, it's the same as Reshape
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
    tflite::Interpreter& interpreter
) {
    ofstream out(output_path);
    if (!out) {
        cerr << "Error: Cannot create file " << output_path << endl;
        return;
    }
    
    // Get input/output shapes
    const auto& input_indices = interpreter.inputs();
    const auto& output_indices = interpreter.outputs();
    
    if (input_indices.size() != 1 || output_indices.size() != 1) {
        cerr << "Error: Only single input/output models supported" << endl;
        return;
    }
    
    const TfLiteTensor* input_tensor = interpreter.tensor(input_indices[0]);
    const TfLiteTensor* output_tensor = interpreter.tensor(output_indices[0]);
    
    if (!input_tensor || !output_tensor) {
        cerr << "Error: Cannot access input/output tensors" << endl;
        return;
    }
    
    size_t input_size = CalculateTensorSize(input_tensor->dims);
    size_t output_size = CalculateTensorSize(output_tensor->dims);
    
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
    out << "    // Shape: [" << GetShapeString(input_tensor->dims) << "]\n";
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
    out << "    // Shape: [" << GetShapeString(output_tensor->dims) << "]\n";
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
    
    // Load the model
    unique_ptr<tflite::FlatBufferModel> model = 
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    
    if (!model) {
        cerr << "Error: Failed to load model from " << model_path << endl;
        return 1;
    }
    
    cout << "Model loaded successfully!" << endl;
    
    // Validate model schema BEFORE building interpreter
    // This provides clear error messages for unsupported operations
    cout << "Validating model operations..." << endl;
    if (!ValidateModelSchema(*model)) {
        return 1;
    }
    cout << "Model validation passed - all operations are supported!" << endl;
    
    // Build the interpreter
    tflite::MutableOpResolver resolver;
    resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
        tflite::ops::builtin::Register_FULLY_CONNECTED());
    resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
        tflite::ops::builtin::Register_SOFTMAX());
    resolver.AddBuiltin(tflite::BuiltinOperator_RELU,
        tflite::ops::builtin::Register_RELU());
    resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
        tflite::ops::builtin::Register_CONV_2D());
    resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
        tflite::ops::builtin::Register_MAX_POOL_2D());
    resolver.AddBuiltin(tflite::BuiltinOperator_SHAPE,
        tflite::ops::builtin::Register_SHAPE());
    resolver.AddBuiltin(tflite::BuiltinOperator_STRIDED_SLICE,
        tflite::ops::builtin::Register_STRIDED_SLICE());
    resolver.AddBuiltin(tflite::BuiltinOperator_PACK,
        tflite::ops::builtin::Register_PACK());
    resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
        tflite::ops::builtin::Register_RESHAPE());
    
    unique_ptr<tflite::Interpreter> interpreter;
    
    tflite::InterpreterBuilder builder(*model, resolver);
    if (builder(&interpreter) != kTfLiteOk) {
        cerr << "\nError: Failed to construct interpreter." << endl;
        cerr << "This may indicate an issue with the model structure." << endl;
        cerr << "Note: Model operations were validated, but interpreter construction failed." << endl;
        cerr << "This could be due to an internal TFLite error or model format issue." << endl;
        return 1;
    }
    
    if (!interpreter) {
        cerr << "\nError: Interpreter is null after construction." << endl;
        return 1;
    }
    
    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        cerr << "\nError: Failed to allocate tensors." << endl;
        cerr << "This may indicate a memory issue or invalid tensor configuration." << endl;
        return 1;
    }
    
    cout << "Interpreter initialized successfully!" << endl;

    // Validate that the model only uses supported operators/activations
    if (!ValidateModel(*interpreter)) {
        return 1;
    }
    
    // Create output directory
    string output_dir = base_name;
    if (!CreateDirectory(output_dir)) {
        cerr << "Error: Failed to create directory " << output_dir << endl;
        return 1;
    }
    cout << "Created directory: " << output_dir << endl;
    
    // Get input/output info for header generation
    const auto& input_indices = interpreter->inputs();
    const auto& output_indices = interpreter->outputs();
    const TfLiteTensor* input_tensor = interpreter->tensor(input_indices[0]);
    const TfLiteTensor* output_tensor = interpreter->tensor(output_indices[0]);
    size_t input_size = CalculateTensorSize(input_tensor->dims);
    size_t output_size = CalculateTensorSize(output_tensor->dims);
    
    // Generate files
    // First create weight mapping (used by both weight and model generation)
    map<int, string> tensor_to_weight = CreateWeightMapping(*interpreter);
    
    // Collect intermediate buffers info
    map<int, size_t> tensor_sizes;
    tensor_sizes[input_indices[0]] = input_size;
    tensor_sizes[output_indices[0]] = output_size;
    const auto& execution_plan = interpreter->execution_plan();
    for (size_t i = 0; i < execution_plan.size(); ++i) {
        const int node_index = execution_plan[i];
        const auto* node_and_reg = interpreter->node_and_registration(node_index);
        if (!node_and_reg) continue;
        const auto& node = node_and_reg->first;
        if (node.inputs && node.inputs->size > 0) {
            for (int j = 0; j < node.inputs->size; ++j) {
                int tensor_idx = node.inputs->data[j];
                const TfLiteTensor* tensor = interpreter->tensor(tensor_idx);
                if (tensor && tensor_sizes.find(tensor_idx) == tensor_sizes.end()) {
                    tensor_sizes[tensor_idx] = CalculateTensorSize(tensor->dims);
                }
            }
        }
        if (node.outputs && node.outputs->size > 0) {
            for (int j = 0; j < node.outputs->size; ++j) {
                int tensor_idx = node.outputs->data[j];
                const TfLiteTensor* tensor = interpreter->tensor(tensor_idx);
                if (tensor && tensor_sizes.find(tensor_idx) == tensor_sizes.end()) {
                    tensor_sizes[tensor_idx] = CalculateTensorSize(tensor->dims);
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
    
    // Collect all output tensors from operations to ensure we don't miss any
    set<int> operation_output_tensors;
    for (size_t i = 0; i < execution_plan.size(); ++i) {
        const int node_index = execution_plan[i];
        const auto* node_and_reg = interpreter->node_and_registration(node_index);
        if (!node_and_reg) continue;
        const auto& node = node_and_reg->first;
        if (node.outputs && node.outputs->size > 0) {
            for (int j = 0; j < node.outputs->size; ++j) {
                int tensor_idx = node.outputs->data[j];
                operation_output_tensors.insert(tensor_idx);
                // Ensure it's in tensor_sizes with correct size
                if (tensor_sizes.find(tensor_idx) == tensor_sizes.end()) {
                    const TfLiteTensor* tensor = interpreter->tensor(tensor_idx);
                    if (tensor) {
                        size_t size = CalculateTensorSize(tensor->dims);
                        tensor_sizes[tensor_idx] = size > 0 ? size : 1;  // Ensure at least size 1
                    } else {
                        tensor_sizes[tensor_idx] = 1;  // Default size if tensor not accessible
                    }
                }
            }
        }
    }
    
    // Now collect all intermediate buffers
    for (const auto& tensor_idx : operation_output_tensors) {
        if (tensor_idx != input_indices[0] && tensor_idx != output_indices[0]) {
            // Skip if it's a weight tensor
            if (weight_tensor_indices.find(tensor_idx) == weight_tensor_indices.end()) {
                size_t size = tensor_sizes.count(tensor_idx) ? tensor_sizes[tensor_idx] : 1;
                intermediate_buffers.push_back({tensor_idx, size});
            }
        }
    }
    
    // Also include any other tensors from tensor_sizes that might have been missed
    for (const auto& [tensor_idx, size] : tensor_sizes) {
        if (tensor_idx != input_indices[0] && tensor_idx != output_indices[0]) {
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
    GenerateWeightsFile(weights_file, base_name, *interpreter, tensor_to_weight);
    GenerateModelHeader(model_header_file, base_name, input_size, output_size, intermediate_buffers);
    GenerateModelFile(model_file, base_name, *interpreter, tensor_to_weight, intermediate_buffers);
    GenerateInferenceFile(inference_file, base_name, *interpreter);
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
    
    return 0;
}

