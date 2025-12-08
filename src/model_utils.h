// codegen/model_utils.h
// Model-related utility functions for FlatBuffer/TFLite operations

#ifndef CODEGEN_MODEL_UTILS_H
#define CODEGEN_MODEL_UTILS_H

#include <cstddef>
#include <string>
#include <string_view>
#include <map>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <format>
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/builtin_op_data.h"

// Helper to get the actual builtin code (handles schema v3 compatibility)
inline tflite::BuiltinOperator GetActualBuiltinCode(const tflite::OperatorCode* op_code) {
    if (!op_code) {
        return tflite::BuiltinOperator_CUSTOM;
    }
    return static_cast<tflite::BuiltinOperator>(
        std::max(static_cast<int>(op_code->builtin_code()),
                 static_cast<int>(op_code->deprecated_builtin_code())));
}

// Helper to get operator name from FlatBuffer
std::string GetOperatorName(const tflite::OperatorCode* op_code);

// Calculate tensor size from FlatBuffer shape
inline std::size_t CalculateTensorSize(const flatbuffers::Vector<int32_t>* shape) {
    if (!shape || shape->size() == 0) {
        return 1;
    }
    
    auto shape_view = std::ranges::views::iota(0u, shape->size())
        | std::ranges::views::transform([shape](std::size_t i) {
            return static_cast<std::size_t>(shape->Get(i));
        });
    
    return std::accumulate(shape_view.begin(), shape_view.end(), 
                          std::size_t{1}, std::multiplies<>{});
}

// Get shape as string from FlatBuffer
inline std::string GetShapeString(const flatbuffers::Vector<int32_t>* shape) {
    if (!shape || shape->size() == 0) {
        return "1";
    }
    
    auto shape_strs = std::ranges::views::iota(0u, shape->size())
        | std::ranges::views::transform([shape](std::size_t i) {
            return std::to_string(shape->Get(i));
        });
    
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

// Convert FlatBuffer activation to TfLite activation enum
inline constexpr TfLiteFusedActivation ConvertActivation(tflite::ActivationFunctionType activation) {
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
inline constexpr TfLitePadding ConvertPadding(tflite::Padding padding) {
    switch (padding) {
        case tflite::Padding_SAME: return kTfLitePaddingSame;
        case tflite::Padding_VALID: return kTfLitePaddingValid;
        default: return kTfLitePaddingSame;
    }
}

// Escape identifier for C++
std::string EscapeIdentifier(std::string_view name);

// Create mapping from tensor index to weight variable name
std::map<int, std::string> CreateWeightMapping(
    const tflite::Model* model,
    const tflite::SubGraph* subgraph
);

#endif // CODEGEN_MODEL_UTILS_H
