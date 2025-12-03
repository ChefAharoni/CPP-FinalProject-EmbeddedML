// src/main.cpp
// TFLite Model Inspector - Reads and displays structure of a .tflite model
// Direct FlatBuffer inspection without requiring operator implementations

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <cstring>

// TensorFlow Lite FlatBuffer schema header
#include "tensorflow/lite/schema/schema_generated.h"

using namespace std;

// TFLite schema version constant (typically 3)
// This is defined in tensorflow/lite/version.h but we define it here
// to avoid dependency on the full TFLite library
#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION (3)
#endif

namespace {

constexpr int kDisplayWeightsCount = 10;  // Number of weights to show at head/tail

// Helper function to get tensor type name from FlatBuffer enum
string GetTensorTypeName(tflite::TensorType type) {
    switch (type) {
        case tflite::TensorType_FLOAT32: return "FLOAT32";
        case tflite::TensorType_INT32: return "INT32";
        case tflite::TensorType_UINT8: return "UINT8";
        case tflite::TensorType_INT64: return "INT64";
        case tflite::TensorType_STRING: return "STRING";
        case tflite::TensorType_BOOL: return "BOOL";
        case tflite::TensorType_INT16: return "INT16";
        case tflite::TensorType_COMPLEX64: return "COMPLEX64";
        case tflite::TensorType_INT8: return "INT8";
        case tflite::TensorType_FLOAT16: return "FLOAT16";
        case tflite::TensorType_FLOAT64: return "FLOAT64";
        case tflite::TensorType_COMPLEX128: return "COMPLEX128";
        case tflite::TensorType_UINT64: return "UINT64";
        case tflite::TensorType_RESOURCE: return "RESOURCE";
        case tflite::TensorType_VARIANT: return "VARIANT";
        case tflite::TensorType_UINT32: return "UINT32";
        case tflite::TensorType_UINT16: return "UINT16";
        default: return "UNKNOWN";
    }
}

// Helper function to get the actual builtin code (handles schema v3 compatibility)
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

// Helper function to get operator name from FlatBuffer
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

// Print tensor shape from FlatBuffer
void PrintTensorShape(const flatbuffers::Vector<int32_t>* shape) {
    if (!shape || shape->size() == 0) {
        cout << "scalar";
        return;
    }
    
    for (size_t i = 0; i < shape->size(); ++i) {
        cout << shape->Get(i);
        if (i < shape->size() - 1) {
            cout << "x";
        }
    }
}

// Calculate total number of elements in a tensor
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

// Calculate bytes per element based on tensor type
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

// Print weights array (head and tail)
template<typename T>
void PrintWeightsArray(const T* data, size_t num_elements, const string& tensor_name) {
    if (!data || num_elements == 0) {
        cout << "  No data available" << endl;
        return;
    }
    
    const size_t display_count = min(static_cast<size_t>(kDisplayWeightsCount), num_elements);
    
    // Print head
    cout << "  Head (" << display_count << " values): ";
    cout << fixed << setprecision(6);
    for (size_t i = 0; i < display_count; ++i) {
        cout << static_cast<double>(data[i]);
        if (i < display_count - 1) {
            cout << ", ";
        }
    }
    cout << endl;
    
    // Print tail if there are more elements
    if (num_elements > display_count) {
        cout << "  Tail (" << display_count << " values): ";
        for (size_t i = num_elements - display_count; i < num_elements; ++i) {
            cout << static_cast<double>(data[i]);
            if (i < num_elements - 1) {
                cout << ", ";
            }
        }
        cout << endl;
        cout << "  Total elements: " << num_elements << endl;
    } else {
        cout << "  Total elements: " << num_elements << endl;
    }
}

// Print model structure from FlatBuffer
void PrintModelStructure(const tflite::Model* model) {
    if (!model) {
        cerr << "Error: Model is null" << endl;
        return;
    }
    
    cout << "\n=== Model Structure ===" << endl;
    cout << "Model Version: " << model->version() << endl;
    
    const auto* subgraphs = model->subgraphs();
    if (!subgraphs || subgraphs->size() == 0) {
        cerr << "Error: No subgraphs found in model" << endl;
        return;
    }
    
    // Process the main subgraph (usually index 0)
    const tflite::SubGraph* subgraph = subgraphs->Get(0);
    if (!subgraph) {
        cerr << "Error: Subgraph is null" << endl;
        return;
    }
    
    const auto* tensors = subgraph->tensors();
    const auto* inputs = subgraph->inputs();
    const auto* outputs = subgraph->outputs();
    const auto* operators = subgraph->operators();
    const auto* operator_codes = model->operator_codes();
    
    // Print input tensors
    if (inputs && tensors) {
        cout << "\nInput Tensors (" << inputs->size() << " total):" << endl;
        cout << string(80, '-') << endl;
        
        for (size_t i = 0; i < inputs->size(); ++i) {
            int32_t tensor_index = inputs->Get(i);
            if (tensor_index < 0 || tensor_index >= static_cast<int32_t>(tensors->size())) {
                continue;
            }
            
            const tflite::Tensor* tensor = tensors->Get(tensor_index);
            if (!tensor) {
                continue;
            }
            
            cout << "\nInput " << i << " (Tensor " << tensor_index << "):" << endl;
            cout << "  Name: " << (tensor->name() ? tensor->name()->c_str() : "(unnamed)") << endl;
            cout << "  Type: " << GetTensorTypeName(tensor->type()) << endl;
            cout << "  Shape: [";
            PrintTensorShape(tensor->shape());
            cout << "]" << endl;
            
            size_t num_elements = CalculateTensorSize(tensor->shape());
            size_t bytes_per_element = GetBytesPerElement(tensor->type());
            size_t total_bytes = num_elements * bytes_per_element;
            cout << "  Bytes: " << total_bytes << endl;
        }
    }
    
    // Print output tensors
    if (outputs && tensors) {
        cout << "\n\nOutput Tensors (" << outputs->size() << " total):" << endl;
        cout << string(80, '-') << endl;
        
        for (size_t i = 0; i < outputs->size(); ++i) {
            int32_t tensor_index = outputs->Get(i);
            if (tensor_index < 0 || tensor_index >= static_cast<int32_t>(tensors->size())) {
                continue;
            }
            
            const tflite::Tensor* tensor = tensors->Get(tensor_index);
            if (!tensor) {
                continue;
            }
            
            cout << "\nOutput " << i << " (Tensor " << tensor_index << "):" << endl;
            cout << "  Name: " << (tensor->name() ? tensor->name()->c_str() : "(unnamed)") << endl;
            cout << "  Type: " << GetTensorTypeName(tensor->type()) << endl;
            cout << "  Shape: [";
            PrintTensorShape(tensor->shape());
            cout << "]" << endl;
            
            size_t num_elements = CalculateTensorSize(tensor->shape());
            size_t bytes_per_element = GetBytesPerElement(tensor->type());
            size_t total_bytes = num_elements * bytes_per_element;
            cout << "  Bytes: " << total_bytes << endl;
        }
    }
    
    // Print all tensors
    if (tensors) {
        cout << "\n\nAll Tensors (" << tensors->size() << " total):" << endl;
        cout << string(80, '-') << endl;
        
        for (size_t i = 0; i < tensors->size(); ++i) {
            const tflite::Tensor* tensor = tensors->Get(i);
            if (!tensor) {
                continue;
            }
            
            cout << "\nTensor " << i << ":" << endl;
            cout << "  Name: " << (tensor->name() ? tensor->name()->c_str() : "(unnamed)") << endl;
            cout << "  Type: " << GetTensorTypeName(tensor->type()) << endl;
            cout << "  Shape: [";
            PrintTensorShape(tensor->shape());
            cout << "]" << endl;
            
            size_t num_elements = CalculateTensorSize(tensor->shape());
            size_t bytes_per_element = GetBytesPerElement(tensor->type());
            size_t total_bytes = num_elements * bytes_per_element;
            cout << "  Bytes: " << total_bytes << endl;
            cout << "  Buffer Index: " << tensor->buffer() << endl;
        }
    }
    
    // Print operators/layers
    if (operators && operator_codes) {
        cout << "\n\nOperators/Layers (" << operators->size() << " total):" << endl;
        cout << string(80, '-') << endl;
        
        for (size_t i = 0; i < operators->size(); ++i) {
            const tflite::Operator* op = operators->Get(i);
            if (!op) {
                continue;
            }
            
            int32_t op_code_index = op->opcode_index();
            if (op_code_index < 0 || op_code_index >= static_cast<int32_t>(operator_codes->size())) {
                continue;
            }
            
            const tflite::OperatorCode* op_code = operator_codes->Get(op_code_index);
            
            cout << "\nLayer " << i << ":" << endl;
            cout << "  Operator: " << GetOperatorName(op_code) << endl;
            if (op_code) {
                tflite::BuiltinOperator actual_code = GetActualBuiltinCode(op_code);
                cout << "  Builtin Code: " << static_cast<int>(actual_code) 
                     << " (builtin_code=" << op_code->builtin_code() 
                     << ", deprecated_builtin_code=" << op_code->deprecated_builtin_code() << ")" << endl;
                cout << "  Version: " << op_code->version() << endl;
            }
            
            // Input tensors
            const auto* op_inputs = op->inputs();
            if (op_inputs && op_inputs->size() > 0) {
                cout << "  Inputs (" << op_inputs->size() << "): ";
                for (size_t j = 0; j < op_inputs->size(); ++j) {
                    cout << op_inputs->Get(j);
                    if (j < op_inputs->size() - 1) {
                        cout << ", ";
                    }
                }
                cout << endl;
            }
            
            // Output tensors
            const auto* op_outputs = op->outputs();
            if (op_outputs && op_outputs->size() > 0) {
                cout << "  Outputs (" << op_outputs->size() << "): ";
                for (size_t j = 0; j < op_outputs->size(); ++j) {
                    cout << op_outputs->Get(j);
                    if (j < op_outputs->size() - 1) {
                        cout << ", ";
                    }
                }
                cout << endl;
            }
        }
    }
}

// Print weights information from FlatBuffer
void PrintWeights(const tflite::Model* model) {
    if (!model) {
        return;
    }
    
    cout << "\n\n=== Model Weights ===" << endl;
    
    const auto* subgraphs = model->subgraphs();
    const auto* buffers = model->buffers();
    
    if (!subgraphs || subgraphs->size() == 0 || !buffers) {
        cout << "\nNo weight tensors found." << endl;
        return;
    }
    
    const tflite::SubGraph* subgraph = subgraphs->Get(0);
    if (!subgraph) {
        return;
    }
    
    const auto* tensors = subgraph->tensors();
    if (!tensors) {
        return;
    }
    
    bool found_weights = false;
    
    for (size_t i = 0; i < tensors->size(); ++i) {
        const tflite::Tensor* tensor = tensors->Get(i);
        if (!tensor) {
            continue;
        }
        
        // Weight tensors have a buffer index > 0 (0 is typically empty/unused)
        uint32_t buffer_index = tensor->buffer();
        if (buffer_index == 0 || buffer_index >= buffers->size()) {
            continue;
        }
        
        const tflite::Buffer* buffer = buffers->Get(buffer_index);
        if (!buffer || !buffer->data()) {
            continue;
        }
        
        found_weights = true;
        const size_t num_elements = CalculateTensorSize(tensor->shape());
        size_t bytes_per_element = GetBytesPerElement(tensor->type());
        size_t total_bytes = num_elements * bytes_per_element;
        
        cout << "\nTensor " << i << ": " 
                  << (tensor->name() ? tensor->name()->c_str() : "(unnamed)") << endl;
        cout << "  Type: " << GetTensorTypeName(tensor->type()) << endl;
        cout << "  Shape: [";
        PrintTensorShape(tensor->shape());
        cout << "]" << endl;
        cout << "  Elements: " << num_elements << endl;
        cout << "  Size: " << total_bytes << " bytes" << endl;
        cout << "  Buffer Index: " << buffer_index << endl;
        
        // Get buffer data
        const flatbuffers::Vector<uint8_t>* data_vec = buffer->data();
        if (!data_vec || data_vec->size() < total_bytes) {
            cout << "  Warning: Buffer size mismatch" << endl;
            continue;
        }
        
        const uint8_t* raw_data = data_vec->data();
        
        // Print weights based on type
        switch (tensor->type()) {
            case tflite::TensorType_FLOAT32: {
                const float* data = reinterpret_cast<const float*>(raw_data);
                PrintWeightsArray(data, num_elements, 
                    tensor->name() ? tensor->name()->c_str() : "");
                break;
            }
            case tflite::TensorType_INT32: {
                const int32_t* data = reinterpret_cast<const int32_t*>(raw_data);
                PrintWeightsArray(data, num_elements, 
                    tensor->name() ? tensor->name()->c_str() : "");
                break;
            }
            case tflite::TensorType_INT8: {
                const int8_t* data = reinterpret_cast<const int8_t*>(raw_data);
                PrintWeightsArray(data, num_elements, 
                    tensor->name() ? tensor->name()->c_str() : "");
                break;
            }
            case tflite::TensorType_UINT8: {
                const uint8_t* data = raw_data;
                PrintWeightsArray(data, num_elements, 
                    tensor->name() ? tensor->name()->c_str() : "");
                break;
            }
            case tflite::TensorType_FLOAT16: {
                // Note: Float16 may need special handling depending on platform
                cout << "  Float16 weights (raw bytes shown):" << endl;
                const uint16_t* data_u16 = reinterpret_cast<const uint16_t*>(raw_data);
                const size_t display_count = min(
                    static_cast<size_t>(kDisplayWeightsCount), num_elements);
                cout << "  Head: ";
                for (size_t j = 0; j < display_count; ++j) {
                    cout << "0x" << hex << data_u16[j] << dec;
                    if (j < display_count - 1) {
                        cout << ", ";
                    }
                }
                cout << endl;
                if (num_elements > display_count) {
                    cout << "  Total elements: " << num_elements << endl;
                }
                break;
            }
            case tflite::TensorType_INT16: {
                const int16_t* data = reinterpret_cast<const int16_t*>(raw_data);
                PrintWeightsArray(data, num_elements, 
                    tensor->name() ? tensor->name()->c_str() : "");
                break;
            }
            case tflite::TensorType_UINT16: {
                const uint16_t* data = reinterpret_cast<const uint16_t*>(raw_data);
                PrintWeightsArray(data, num_elements, 
                    tensor->name() ? tensor->name()->c_str() : "");
                break;
            }
            default:
                cout << "  Type " << GetTensorTypeName(tensor->type()) 
                          << " not yet supported for weight display" << endl;
                break;
        }
    }
    
    if (!found_weights) {
        cout << "\nNo weight tensors found (tensors with buffer data)." << endl;
    }
}

}  // anonymous namespace

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <path_to_model.tflite>" << endl;
        cerr << "Example: " << argv[0] << " scripts/model.tflite" << endl;
        return 1;
    }
    
    const string model_path = argv[1];
    
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
    
    // Verify model version
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        cerr << "Warning: Model schema version " << model->version() 
             << " doesn't match supported version " << TFLITE_SCHEMA_VERSION << endl;
    }
    
    cout << "Model loaded successfully!" << endl;
    cout << "Model schema version: " << model->version() << endl;
    
    // Print model information
    PrintModelStructure(model);
    PrintWeights(model);
    
    cout << "\n\n=== Summary ===" << endl;
    cout << "Model inspection complete." << endl;
    cout << "This inspection was performed using direct FlatBuffer parsing," << endl;
    cout << "without requiring any operator implementations." << endl;
    
    return 0;
}
