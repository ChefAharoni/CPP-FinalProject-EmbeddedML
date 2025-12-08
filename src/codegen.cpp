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
#include <algorithm>
#include <cctype>

// TensorFlow includes - we only need these headers to parse the .tflite model
#include "tensorflow/lite/schema/schema_generated.h"

// Module includes
#include "exceptions.h"
#include "model_utils.h"
#include "model_validator.h"
#include "code_generator.h"
#include "filesystem_utils.h"

// Helper function to parse inference type from string (case-insensitive)
InferenceType ParseInferenceType(const std::string& str) {
    std::string lower_str = str;
    std::ranges::transform(lower_str, lower_str.begin(),
        [](char c) -> char {
            return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        });
    
    if (lower_str == "none") {
        return InferenceType::None;
    } else if (lower_str == "standard") {
        return InferenceType::Standard;
    } else if (lower_str == "pico-img-bench") {
        return InferenceType::PicoImgBench;
    } else {
        throw std::invalid_argument("Invalid inference type: " + str);
    }
}

// Print short usage message
void PrintShortUsage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <path_to_model.tflite> <base_name> [OPTIONS]" << '\n';
    std::cerr << "Use -h or --help for detailed help" << '\n';
}

// Print detailed help message
void PrintDetailedHelp(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <path_to_model.tflite> <base_name> [OPTIONS]" << '\n';
    std::cerr << "Example: " << program_name << " scripts/model.tflite my_model" << '\n';
    std::cerr << "Example (multi-output): " << program_name << " scripts/model.tflite my_model --output-index=1" << '\n';
    std::cerr << "Example (custom templates): " << program_name << " scripts/model.tflite my_model --template-path=/path/to/templates" << '\n';
    std::cerr << "Example (Pico inference): " << program_name << " scripts/model.tflite my_model --inf=pico-img-bench" << '\n';
    std::cerr << "This will create a directory 'my_model/' containing:" << '\n';
    std::cerr << "  - my_model_weights.cpp" << '\n';
    std::cerr << "  - my_model.h" << '\n';
    std::cerr << "  - my_model.cpp" << '\n';
    std::cerr << "  - my_model_inference.cpp (unless --inf=none)" << '\n';
    std::cerr << "  - Makefile (unless --no-makefile is used)" << '\n';
    std::cerr << "\nTo build: cd my_model && make" << '\n';
    std::cerr << "\nOptions:" << '\n';
    std::cerr << "  --output-index=N    Select output tensor index for multi-output models (default: 0). If you don't know what this is, you probably don't need it." << '\n';
    std::cerr << "  --template-path=PATH Specify custom templates directory (default: ./templates)" << '\n';
    std::cerr << "  --component-path=PATH Specify custom components directory (default: ./components)" << '\n';
    std::cerr << "  --inf=TYPE          Inference script type: none, standard (default), or pico-img-bench" << '\n';
    std::cerr << "  --no-makefile       Skip Makefile generation" << '\n';
    std::cerr << "  --replace           Allow overwriting existing output directory" << '\n';
    std::cerr << "  -h, --help          Show this help message" << '\n';
    std::cerr << "\nNote: Only single input models are supported." << '\n';
    std::cerr << "      If templates are not found in the default location, use --template-path to specify their location." << '\n';
    std::cerr << "      If the output directory already exists, use --replace to overwrite it." << '\n';
}

int main(int argc, char* argv[]) {
    // Check for help flag first
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            PrintDetailedHelp(argv[0]);
            return 0;
        }
    }
    
    if (argc < 3) {
        PrintShortUsage(argv[0]);
        return 1;
    }
    
    const std::string model_path = argv[1];
    const std::string base_name = argv[2];
    int output_index = 0;
    bool output_index_specified = false;
    std::string templates_dir = "templates";
    bool template_path_specified = false;
    std::string components_dir = "components";
    bool component_path_specified = false;
    InferenceType inference_type = InferenceType::Standard;
    bool no_makefile = false;
    bool replace = false;
    
    // Parse optional flags
    for (int i = 3; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag.find("--output-index=") == 0) {
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
            }
        } else if (flag.find("--template-path=") == 0) {
            std::size_t eq_pos = flag.find('=');
            if (eq_pos != std::string::npos) {
                templates_dir = flag.substr(eq_pos + 1);
                template_path_specified = true;
            } else {
                std::cerr << "Error: --template-path requires a value, e.g., --template-path=/path/to/templates" << '\n';
                return 1;
            }
        } else if (flag.find("--component-path=") == 0) {
            std::size_t eq_pos = flag.find('=');
            if (eq_pos != std::string::npos) {
                components_dir = flag.substr(eq_pos + 1);
                component_path_specified = true;
            } else {
                std::cerr << "Error: --component-path requires a value, e.g., --component-path=/path/to/components" << '\n';
                return 1;
            }
        } else if (flag.find("--inf=") == 0) {
            std::size_t eq_pos = flag.find('=');
            if (eq_pos != std::string::npos) {
                std::string inf_type_str = flag.substr(eq_pos + 1);
                try {
                    inference_type = ParseInferenceType(inf_type_str);
                } catch (const std::exception& e) {
                    std::cerr << "Error: " << e.what() << '\n';
                    std::cerr << "Valid inference types are: none, standard, pico-img-bench" << '\n';
                    return 1;
                }
            } else {
                std::cerr << "Error: --inf requires a value, e.g., --inf=standard" << '\n';
                return 1;
            }
        } else if (flag == "--no-makefile") {
            no_makefile = true;
        } else if (flag == "--replace") {
            replace = true;
        } else {
            std::cerr << "Error: Unknown flag: " << flag << '\n';
            std::cerr << "Use -h or --help for help" << '\n';
            return 1;
        }
    }
    
    // Check if file exists
    if (!std::filesystem::exists(model_path)) {
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
    
        // Extract directory and base name from the path
        // If base_name contains a path, use the full path as output_dir and filename as file_base_name
        std::filesystem::path base_path(base_name);
        std::string output_dir = base_name;  // Use full path as output directory
        std::string file_base_name = base_path.filename().empty() ? base_name : base_path.filename().string();
        
        // If base_name has no path component, output_dir and file_base_name are the same
        if (base_path.parent_path().empty()) {
            output_dir = base_name;
            file_base_name = base_name;
        }
        
        // Validate templates directory
        if (!std::filesystem::exists(templates_dir) || !std::filesystem::is_directory(templates_dir)) {
            std::string error_msg = std::format("Templates directory does not exist or is not a directory: {}", templates_dir);
            if (!template_path_specified) {
                error_msg += "\n\nTip: If your templates are in a different location, use --template-path=PATH to specify it.";
                error_msg += std::format("\nExample: {} {} {} --template-path=/path/to/templates", argv[0], model_path, base_name);
            }
            throw CodeGenerationError(error_msg);
        }
        
        // Check for required template files
        std::vector<std::string> required_templates = {
            "weights.cpp.inja",
            "model.h.inja",
            "model.cpp.inja",
            "inference.cpp.inja",
            "Makefile.inja"
        };
        
        for (const auto& template_file : required_templates) {
            std::filesystem::path template_path = std::filesystem::path(templates_dir) / template_file;
            if (!std::filesystem::exists(template_path) || !std::filesystem::is_regular_file(template_path)) {
                std::string error_msg = std::format("Required template file not found: {}", template_path.string());
                if (!template_path_specified) {
                    error_msg += "\n\nTip: If your templates are in a different location, use --template-path=PATH to specify it.";
                    error_msg += std::format("\nExample: {} {} {} --template-path=/path/to/templates", argv[0], model_path, base_name);
                }
                throw CodeGenerationError(error_msg);
            }
        }
        
        std::cout << std::format("Using templates directory: {}\n", templates_dir);
        
        // Validate components directory
        if (!std::filesystem::exists(components_dir) || !std::filesystem::is_directory(components_dir)) {
            std::string error_msg = std::format("Components directory does not exist or is not a directory: {}", components_dir);
            if (!component_path_specified) {
                error_msg += "\n\nTip: If your components are in a different location, use --component-path=PATH to specify it.";
                error_msg += std::format("\nExample: {} {} {} --component-path=/path/to/components", argv[0], model_path, base_name);
            }
            throw CodeGenerationError(error_msg);
        }
        
        std::cout << std::format("Using components directory: {}\n", components_dir);
        
        // Check if output directory exists
        bool output_dir_exists = std::filesystem::exists(output_dir) && std::filesystem::is_directory(output_dir);
        
        if (output_dir_exists && !replace) {
            throw CodeGenerationError(
                std::format("Output directory already exists: {}\nUse --replace to overwrite files in it.", output_dir)
            );
        }
        
        if (!output_dir_exists && replace) {
            throw CodeGenerationError(
                std::format("Output directory does not exist: {}\nCannot use --replace on a non-existent directory.", output_dir)
            );
        }
        
        // Create output directory (will not overwrite if it exists)
        CreateDirectory(output_dir);
        if (output_dir_exists && replace) {
            std::cout << std::format("Using existing directory: {}\n", output_dir);
        } else {
            std::cout << std::format("Created directory: {}\n", output_dir);
        }
    
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
    
        // Determine which components are needed by analyzing the model
        std::set<std::string> needed_components;
        needed_components.insert("fully_connected.h");  // Always needed
        needed_components.insert("softmax.h");  // Always needed
        
        if (operators) {
            const auto* operator_codes = model->operator_codes();
            for (std::size_t i = 0; i < operators->size(); ++i) {
                const tflite::Operator* op = operators->Get(i);
                if (!op) continue;
                
                int op_code_index = op->opcode_index();
                if (op_code_index < 0 || op_code_index >= static_cast<int>(operator_codes->size())) {
                    continue;
                }
                
                const tflite::OperatorCode* op_code = operator_codes->Get(op_code_index);
                if (!op_code) continue;
                
                std::string op_name = GetOperatorName(op_code);
                
                if (op_name == "RELU") {
                    needed_components.insert("relu.h");
                } else if (op_name == "CONV_2D") {
                    needed_components.insert("conv_2d.h");
                } else if (op_name == "MAX_POOL_2D") {
                    needed_components.insert("max_pool_2d.h");
                } else if (op_name == "SHAPE") {
                    needed_components.insert("shape.h");
                } else if (op_name == "STRIDED_SLICE") {
                    needed_components.insert("strided_slice.h");
                } else if (op_name == "PACK") {
                    needed_components.insert("pack.h");
                } else if (op_name == "RESHAPE") {
                    needed_components.insert("reshape.h");
                } else if (op_name == "ADD") {
                    needed_components.insert("add.h");
                } else if (op_name.find("DROPOUT") != std::string::npos || 
                           op_name.find("Dropout") != std::string::npos) {
                    needed_components.insert("dropout.h");
                } else if (op_name.find("FLATTEN") != std::string::npos ||
                           op_name.find("Flatten") != std::string::npos) {
                    needed_components.insert("flatten.h");
                }
            }
        }
        
        // Copy needed components to output directory
        std::string output_components_dir = output_dir + "/components";
        CreateDirectory(output_components_dir);
        std::cout << std::format("Copying components to {}/components/...\n", output_dir);
        
        for (const auto& component_file : needed_components) {
            std::filesystem::path source_path = std::filesystem::path(components_dir) / component_file;
            std::filesystem::path dest_path = std::filesystem::path(output_components_dir) / component_file;
            
            if (!std::filesystem::exists(source_path) || !std::filesystem::is_regular_file(source_path)) {
                throw CodeGenerationError(
                    std::format("Required component file not found: {}", source_path.string())
                );
            }
            
            CopyFile(source_path.string(), dest_path.string());
            std::cout << std::format("  Copied: {}\n", component_file);
        }
        
        std::string weights_file = std::format("{}/{}_weights.cpp", output_dir, file_base_name);
        std::string model_header_file = std::format("{}/{}.h", output_dir, file_base_name);
        std::string model_file = std::format("{}/{}.cpp", output_dir, file_base_name);
        std::string inference_file = std::format("{}/{}_inference.cpp", output_dir, file_base_name);
        std::string makefile = std::format("{}/Makefile", output_dir);
        
        std::cout << "\nGenerating code files...\n";
        
        // Generate weights file
        GenerateWeightsFile(weights_file, file_base_name, model, subgraph, tensor_to_weight, templates_dir);
        
        // Generate model header
        GenerateModelHeader(model_header_file, file_base_name, input_size, output_size, intermediate_buffers, templates_dir);
        
        // Generate model file
        GenerateModelFile(model_file, file_base_name, model, subgraph, tensor_to_weight, intermediate_buffers, 
                           tensor_sizes, input_tensor_idx, output_tensor_idx, templates_dir);
        
        // Generate inference file (if not None)
        GenerateInferenceFile(inference_file, file_base_name, subgraph, input_tensor_idx, output_tensor_idx, inference_type, templates_dir);
        
        // Generate Makefile (unless --no-makefile is set)
        if (!no_makefile) {
            GenerateMakefile(makefile, file_base_name, templates_dir);
        }
        
        std::cout << "\nCode generation complete!\n";
        std::cout << std::format("Generated files in directory '{}':\n", output_dir);
        std::cout << std::format("  - {}_weights.cpp\n", file_base_name);
        std::cout << std::format("  - {}.h\n", file_base_name);
        std::cout << std::format("  - {}.cpp\n", file_base_name);
        if (inference_type != InferenceType::None) {
            std::cout << std::format("  - {}_inference.cpp\n", file_base_name);
        }
        if (!no_makefile) {
            std::cout << "  - Makefile\n";
            if (inference_type != InferenceType::None) {
                std::cout << std::format("\nTo build the inference executable:\n  cd {} && make\n", output_dir);
            }
        } else {
            std::cout << "\nNote: Makefile generation was skipped (--no-makefile flag used).\n";
        }
        if (inference_type == InferenceType::None) {
            std::cout << "\nNote: Inference file generation was skipped (--inf=none flag used).\n";
        }
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
        // Note: We do not delete the directory on error to preserve any existing files
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
