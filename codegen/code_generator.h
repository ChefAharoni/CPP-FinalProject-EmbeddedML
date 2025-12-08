// codegen/code_generator.h
// Code generation functions

#ifndef CODEGEN_CODE_GENERATOR_H
#define CODEGEN_CODE_GENERATOR_H

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include "tensorflow/lite/schema/schema_generated.h"

// Generate model_weights.cpp
void GenerateWeightsFile(
    const std::string& output_path,
    const std::string& base_name,
    const tflite::Model* model,
    const tflite::SubGraph* subgraph,
    const std::map<int, std::string>& tensor_to_weight
);

// Generate model.h (header file)
void GenerateModelHeader(
    const std::string& output_path,
    const std::string& base_name,
    std::size_t input_size,
    std::size_t output_size,
    const std::vector<std::pair<int, std::size_t>>& intermediate_buffers
);

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
);

// Generate inference.cpp
void GenerateInferenceFile(
    const std::string& output_path,
    const std::string& base_name,
    const tflite::SubGraph* subgraph,
    int32_t input_tensor_idx,
    int32_t output_tensor_idx
);

// Generate Makefile for the generated code
void GenerateMakefile(
    const std::string& output_path,
    const std::string& base_name
);

#endif // CODEGEN_CODE_GENERATOR_H
