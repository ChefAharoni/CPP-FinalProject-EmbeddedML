// codegen/model_utils.h
// Model-related utility functions for FlatBuffer/TFLite operations

#ifndef CODEGEN_MODEL_UTILS_H
#define CODEGEN_MODEL_UTILS_H

#include <cstddef>
#include <string>
#include <string_view>
#include <map>
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/builtin_op_data.h"

// Helper to get the actual builtin code (handles schema v3 compatibility)
tflite::BuiltinOperator GetActualBuiltinCode(const tflite::OperatorCode* op_code);

// Helper to get operator name from FlatBuffer
std::string GetOperatorName(const tflite::OperatorCode* op_code);

// Calculate tensor size from FlatBuffer shape
std::size_t CalculateTensorSize(const flatbuffers::Vector<int32_t>* shape);

// Get shape as string from FlatBuffer
std::string GetShapeString(const flatbuffers::Vector<int32_t>* shape);

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

// Escape identifier for C++
std::string EscapeIdentifier(std::string_view name);

// Create mapping from tensor index to weight variable name
std::map<int, std::string> CreateWeightMapping(
    const tflite::Model* model,
    const tflite::SubGraph* subgraph
);

#endif // CODEGEN_MODEL_UTILS_H
