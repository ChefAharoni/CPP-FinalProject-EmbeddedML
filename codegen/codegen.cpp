// codegen/codegen.cpp
// Code generator that reads TFLite models and generates pure C++ inference code
// Direct FlatBuffer inspection without requiring operator implementations

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <format>
#include <filesystem>
#include <ranges>

// TensorFlow includes - we only need these headers to parse the .tflite model
#include "tensorflow/lite/schema/schema_generated.h"

// Module includes
#include "exceptions.h"
#include "model_utils.h"
#include "model_validator.h"
#include "code_generator.h"
#include "filesystem_utils.h"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model.tflite> <base_name> [--output-index=N]" << '\n';
        std::cerr << "Example: " << argv[0] << " scripts/model.tflite my_model" << '\n';
        std::cerr << "Example (multi-output): " << argv[0] << " scripts/model.tflite my_model --output-index=1" << '\n';
        std::cerr << "This will create a directory 'my_model/' containing:" << '\n';
        std::cerr << "  - my_model_weights.cpp" << '\n';
        std::cerr << "  - my_model.h" << '\n';
        std::cerr << "  - my_model.cpp" << '\n';
        std::cerr << "  - my_model_inference.cpp" << '\n';
        std::cerr << "  - Makefile" << '\n';
        std::cerr << "\nTo build: cd my_model && make" << '\n';
        std::cerr << "\nNote: For models with multiple outputs, use --output-index=N to select" << '\n';
        std::cerr << "      which output tensor to use (default: 0). Only single input models are supported." << '\n';
        return 1;
    }
    
    const std::string model_path = argv[1];
    const std::string base_name = argv[2];
    int output_index = 0;
    bool output_index_specified = false;
    
    // Parse optional output index flag
    if (argc == 4) {
        std::string flag = argv[3];
        if (flag.find("--output-index") == 0) {
            std::size_t eq_pos = flag.find('=');
            if (eq_pos != std::string::npos) {
                std::string index_str = flag.substr(eq_pos + 1);
                try {
                    output_index = std::stoi(index_str);
                    output_index_specified = true;
                } catch (const std::exception& e) {
                    std::cerr << "Error: Invalid output index: " << index_str << '\n';
                    return 1;
                }
            } else {
                std::cerr << "Error: --output-index requires a value, e.g., --output-index=1" << '\n';
                return 1;
            }
        } else {
            std::cerr << "Error: Unknown flag: " << flag << '\n';
            std::cerr << "Use --output-index=N to specify output tensor index" << '\n';
            return 1;
        }
    }
    
    // Check if file exists
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
        std::cout << "Validating model operations..." << '\n';
        ValidateModelSchema(model);
        std::cout << "Model validation passed - all operations are supported!" << '\n';
    
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
        std::map<int, std::string> tensor_to_weight = CreateWeightMapping(model, subgraph);
        
        // Collect intermediate buffers info
        std::map<int, std::size_t> tensor_sizes;
        tensor_sizes[input_tensor_idx] = input_size;
        tensor_sizes[output_tensor_idx] = output_size;
        
        // Process operators to find all tensor sizes
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
        std::set<int> weight_tensor_indices;
        for (const auto& [tensor_idx, weight_name] : tensor_to_weight) {
            weight_tensor_indices.insert(tensor_idx);
        }
        
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
        
        // Collect all intermediate buffers
        for (const auto& tensor_idx : operation_output_tensors) {
            if (tensor_idx != input_tensor_idx && tensor_idx != output_tensor_idx) {
                if (!weight_tensor_indices.contains(tensor_idx)) {
                    std::size_t size = tensor_sizes.contains(tensor_idx) ? tensor_sizes[tensor_idx] : 1;
                    intermediate_buffers.push_back({tensor_idx, size});
                }
            }
        }
        
        // Also include any other tensors from tensor_sizes that might have been missed
        for (const auto& [tensor_idx, size] : tensor_sizes) {
            if (tensor_idx != input_tensor_idx && tensor_idx != output_tensor_idx) {
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
        
        // Generate model file
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
        std::cerr << e.what() << '\n';
        return 1;
    } catch (const CodeGenerationError& e) {
        std::cerr << "\n================================================\n";
        std::cerr << "ERROR: Code generation failed!\n";
        std::cerr << "================================================\n";
        std::cerr << e.what() << '\n';
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
        std::cerr << e.what() << '\n';
        std::cerr << "================================================\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n================================================\n";
        std::cerr << "ERROR: Unexpected error occurred!\n";
        std::cerr << "================================================\n";
        std::cerr << e.what() << '\n';
        std::cerr << "================================================\n";
        return 1;
    }
}
