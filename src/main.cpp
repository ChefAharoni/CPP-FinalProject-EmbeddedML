// src/main.cpp
// TFLite Model Inspector - Reads and displays structure of a .tflite model

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <cstring>

// TensorFlow Lite headers
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"

using namespace std;

namespace {

constexpr int kDisplayWeightsCount = 10;  // Number of weights to show at head/tail

// Helper function to get tensor type name
string GetTensorTypeName(TfLiteType type) {
    switch (type) {
        case kTfLiteFloat32: return "FLOAT32";
        case kTfLiteInt32: return "INT32";
        case kTfLiteUInt8: return "UINT8";
        case kTfLiteInt64: return "INT64";
        case kTfLiteString: return "STRING";
        case kTfLiteBool: return "BOOL";
        case kTfLiteInt16: return "INT16";
        case kTfLiteComplex64: return "COMPLEX64";
        case kTfLiteInt8: return "INT8";
        case kTfLiteFloat16: return "FLOAT16";
        case kTfLiteFloat64: return "FLOAT64";
        case kTfLiteComplex128: return "COMPLEX128";
        case kTfLiteUInt64: return "UINT64";
        case kTfLiteResource: return "RESOURCE";
        case kTfLiteVariant: return "VARIANT";
        case kTfLiteUInt32: return "UINT32";
        case kTfLiteUInt16: return "UINT16";
        default: return "UNKNOWN";
    }
}

// Helper function to get allocation type name
string GetAllocationTypeName(TfLiteAllocationType type) {
    switch (type) {
        case kTfLiteMemNone: return "NONE";
        case kTfLiteMmapRo: return "MMAP_RO";
        case kTfLiteDynamic: return "DYNAMIC";
        case kTfLiteArenaRw: return "ARENA_RW";
        case kTfLiteArenaRwPersistent: return "ARENA_RW_PERSISTENT";
        case kTfLitePersistentRo: return "PERSISTENT_RO";
        case kTfLiteCustom: return "CUSTOM";
        default: return "UNKNOWN";
    }
}

// Helper function to get operator name
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

// Print tensor shape
void PrintTensorShape(const TfLiteIntArray* dims) {
    if (!dims || dims->size == 0) {
        cout << "scalar";
        return;
    }
    
    for (int i = 0; i < dims->size; ++i) {
        cout << dims->data[i];
        if (i < dims->size - 1) {
            cout << "x";
        }
    }
}

// Calculate total number of elements in a tensor
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

// Print model structure
void PrintModelStructure(tflite::Interpreter& interpreter) {
    cout << "\n=== Model Structure ===" << endl;
    
    // Print input tensors
    const auto& input_indices = interpreter.inputs();
    cout << "\nInput Tensors (" << input_indices.size() << " total):" << endl;
    cout << string(80, '-') << endl;
    
    for (size_t i = 0; i < input_indices.size(); ++i) {
        const int tensor_index = input_indices[i];
        const TfLiteTensor* tensor = interpreter.tensor(tensor_index);
        if (!tensor) {
            continue;
        }
        
        cout << "\nInput " << i << " (Tensor " << tensor_index << "):" << endl;
        cout << "  Name: " << (tensor->name ? tensor->name : "(unnamed)") << endl;
        cout << "  Type: " << GetTensorTypeName(tensor->type) << endl;
        cout << "  Shape: [";
        PrintTensorShape(tensor->dims);
        cout << "]" << endl;
        cout << "  Allocation: " << GetAllocationTypeName(tensor->allocation_type) << endl;
        cout << "  Bytes: " << tensor->bytes << endl;
    }
    
    // Print output tensors
    const auto& output_indices = interpreter.outputs();
    cout << "\n\nOutput Tensors (" << output_indices.size() << " total):" << endl;
    cout << string(80, '-') << endl;
    
    for (size_t i = 0; i < output_indices.size(); ++i) {
        const int tensor_index = output_indices[i];
        const TfLiteTensor* tensor = interpreter.tensor(tensor_index);
        if (!tensor) {
            continue;
        }
        
        cout << "\nOutput " << i << " (Tensor " << tensor_index << "):" << endl;
        cout << "  Name: " << (tensor->name ? tensor->name : "(unnamed)") << endl;
        cout << "  Type: " << GetTensorTypeName(tensor->type) << endl;
        cout << "  Shape: [";
        PrintTensorShape(tensor->dims);
        cout << "]" << endl;
        cout << "  Allocation: " << GetAllocationTypeName(tensor->allocation_type) << endl;
        cout << "  Bytes: " << tensor->bytes << endl;
    }
    
    // Print all tensors
    const size_t num_tensors = interpreter.tensors_size();
    cout << "\n\nAll Tensors (" << num_tensors << " total):" << endl;
    cout << string(80, '-') << endl;
    
    for (size_t i = 0; i < num_tensors; ++i) {
        const TfLiteTensor* tensor = interpreter.tensor(i);
        if (!tensor) {
            continue;
        }
        
        cout << "\nTensor " << i << ":" << endl;
        cout << "  Name: " << (tensor->name ? tensor->name : "(unnamed)") << endl;
        cout << "  Type: " << GetTensorTypeName(tensor->type) << endl;
        cout << "  Shape: [";
        PrintTensorShape(tensor->dims);
        cout << "]" << endl;
        cout << "  Allocation: " << GetAllocationTypeName(tensor->allocation_type) << endl;
        cout << "  Bytes: " << tensor->bytes << endl;
    }
    
    // Print operators/layers
    const auto& execution_plan = interpreter.execution_plan();
    cout << "\n\nOperators/Layers (" << execution_plan.size() << " total):" << endl;
    cout << string(80, '-') << endl;
    
    for (size_t i = 0; i < execution_plan.size(); ++i) {
        const int node_index = execution_plan[i];
        const auto* node_and_reg = interpreter.node_and_registration(node_index);
        
        if (!node_and_reg) {
            continue;
        }
        
        const auto& node = node_and_reg->first;
        const auto& registration = node_and_reg->second;
        
        cout << "\nLayer " << i << " (Node " << node_index << "):" << endl;
        cout << "  Operator: " << GetOperatorName(interpreter, node_index) << endl;
        cout << "  Builtin Code: " << registration.builtin_code << endl;
        cout << "  Version: " << registration.version << endl;
        
        // Input tensors
        if (node.inputs && node.inputs->size > 0) {
            cout << "  Inputs (" << node.inputs->size << "): ";
            for (int j = 0; j < node.inputs->size; ++j) {
                cout << node.inputs->data[j];
                if (j < node.inputs->size - 1) {
                    cout << ", ";
                }
            }
            cout << endl;
        }
        
        // Output tensors
        if (node.outputs && node.outputs->size > 0) {
            cout << "  Outputs (" << node.outputs->size << "): ";
            for (int j = 0; j < node.outputs->size; ++j) {
                cout << node.outputs->data[j];
                if (j < node.outputs->size - 1) {
                    cout << ", ";
                }
            }
            cout << endl;
        }
    }
}

// Print weights information
void PrintWeights(tflite::Interpreter& interpreter) {
    cout << "\n\n=== Model Weights ===" << endl;
    
    const size_t num_tensors = interpreter.tensors_size();
    bool found_weights = false;
    
    for (size_t i = 0; i < num_tensors; ++i) {
        const TfLiteTensor* tensor = interpreter.tensor(i);
        if (!tensor) {
            continue;
        }
        
        // Weights are typically read-only tensors (MMAP_RO or PERSISTENT_RO)
        if (tensor->allocation_type == kTfLiteMmapRo || 
            tensor->allocation_type == kTfLitePersistentRo) {
            
            found_weights = true;
            const size_t num_elements = CalculateTensorSize(tensor->dims);
            
            cout << "\nTensor " << i << ": " 
                      << (tensor->name ? tensor->name : "(unnamed)") << endl;
            cout << "  Type: " << GetTensorTypeName(tensor->type) << endl;
            cout << "  Shape: [";
            PrintTensorShape(tensor->dims);
            cout << "]" << endl;
            cout << "  Elements: " << num_elements << endl;
            cout << "  Size: " << tensor->bytes << " bytes" << endl;
            
            // Print weights based on type
            switch (tensor->type) {
                case kTfLiteFloat32: {
                    const float* data = interpreter.typed_tensor<float>(i);
                    if (data) {
                        PrintWeightsArray(data, num_elements, tensor->name ? tensor->name : "");
                    }
                    break;
                }
                case kTfLiteInt32: {
                    const int32_t* data = interpreter.typed_tensor<int32_t>(i);
                    if (data) {
                        PrintWeightsArray(data, num_elements, tensor->name ? tensor->name : "");
                    }
                    break;
                }
                case kTfLiteInt8: {
                    const int8_t* data = interpreter.typed_tensor<int8_t>(i);
                    if (data) {
                        PrintWeightsArray(data, num_elements, tensor->name ? tensor->name : "");
                    }
                    break;
                }
                case kTfLiteUInt8: {
                    const uint8_t* data = interpreter.typed_tensor<uint8_t>(i);
                    if (data) {
                        PrintWeightsArray(data, num_elements, tensor->name ? tensor->name : "");
                    }
                    break;
                }
                case kTfLiteFloat16: {
                    // Note: Float16 may need special handling depending on platform
                    cout << "  Float16 weights (raw bytes shown):" << endl;
                    const void* data = tensor->data.data;
                    if (data && num_elements > 0) {
                        const uint16_t* data_u16 = static_cast<const uint16_t*>(data);
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
                    }
                    break;
                }
                default:
                    cout << "  Type " << GetTensorTypeName(tensor->type) 
                              << " not yet supported for weight display" << endl;
                    break;
            }
        }
    }
    
    if (!found_weights) {
        cout << "\nNo weight tensors found (read-only tensors)." << endl;
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
    
    // Load the model
    unique_ptr<tflite::FlatBufferModel> model = 
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    
    if (!model) {
        cerr << "Error: Failed to load model from " << model_path << endl;
        return 1;
    }
    
    cout << "Model loaded successfully!" << endl;
    
    // Build the interpreter
    // tflite::ops::builtin::BuiltinOpResolver resolver;
	tflite::MutableOpResolver resolver;

	resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
		tflite::ops::builtin::Register_FULLY_CONNECTED());
	resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
		tflite::ops::builtin::Register_SOFTMAX());

	// Only add RELU if your model has a separate RELU op.
	// (Many converters fuse relu into FULLY_CONNECTED.)
	resolver.AddBuiltin(tflite::BuiltinOperator_RELU,
		tflite::ops::builtin::Register_RELU());

    unique_ptr<tflite::Interpreter> interpreter;
    
    tflite::InterpreterBuilder builder(*model, resolver);
    if (builder(&interpreter) != kTfLiteOk) {
        cerr << "Error: Failed to construct interpreter." << endl;
        return 1;
    }
    
    if (!interpreter) {
        cerr << "Error: Interpreter is null." << endl;
        return 1;
    }
    
    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        cerr << "Error: Failed to allocate tensors." << endl;
        return 1;
    }
    
    cout << "Interpreter initialized successfully!" << endl;
    
    // Print model information
    PrintModelStructure(*interpreter);
    PrintWeights(*interpreter);
    
    cout << "\n\n=== Summary ===" << endl;
    cout << "Model inspection complete." << endl;
    
    return 0;
}
