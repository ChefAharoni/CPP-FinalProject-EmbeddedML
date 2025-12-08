// codegen/model_utils.cpp
// Model-related utility functions implementation

#include "model_utils.h"
#include <algorithm>
#include <numeric>
#include <ranges>
#include <format>

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

// Escape identifier for C++
std::string EscapeIdentifier(std::string_view name) {
    std::string result;
    result.reserve(name.size());
    
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
        
        uint32_t buffer_index = tensor->buffer();
        if (buffer_index == 0 || buffer_index >= buffers->size()) {
            continue;
        }
        
        const tflite::Buffer* buffer = buffers->Get(buffer_index);
        if (!buffer || !buffer->data()) {
            continue;
        }
        
        if (tensor->type() == tflite::TensorType_FLOAT32) {
            std::string tensor_name = tensor->name() ? tensor->name()->c_str() : std::format("tensor_{}", i);
            tensor_to_weight[static_cast<int>(i)] = std::format("weight_{}_{}", weight_index, EscapeIdentifier(tensor_name));
            weight_index++;
        }
    }
    
    return tensor_to_weight;
}
