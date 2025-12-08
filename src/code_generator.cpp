// codegen/code_generator.cpp
// Code generation functions implementation

#include "code_generator.h"
#include "exceptions.h"
#include "model_utils.h"
#include <inja/inja.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <format>
#include <stdexcept>
#include <cctype>
#include <ranges>

void GenerateWeightsFile(
    const std::string& output_path,
    const std::string& base_name,
    const tflite::Model* model,
    const tflite::SubGraph* subgraph,
    const std::map<int, std::string>& tensor_to_weight,
    const std::string& templates_dir
) {
    inja::Environment env;
    
    const auto* tensors = subgraph->tensors();
    const auto* buffers = model->buffers();
    
    if (!tensors || !buffers) {
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        out << "// " << base_name << "_weights.cpp\n";
        out << "// Auto-generated weight data for embedded inference\n\n";
        out << "#include <cstddef>\n\n";
        out << "namespace embedded_ml {\n\n";
        out << "} // namespace embedded_ml\n";
        std::cout << std::format("Generated: {}\n", output_path);
        return;
    }
    
    nlohmann::json data;
    data["base_name"] = base_name;
    data["weights"] = nlohmann::json::array();
    
    for (std::size_t i = 0; i < tensors->size(); ++i) {
        if (!tensor_to_weight.contains(static_cast<int>(i))) {
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
        
        const std::size_t num_elements = CalculateTensorSize(tensor->shape());
        std::string tensor_name = tensor->name() ? tensor->name()->c_str() : std::format("tensor_{}", i);
        std::string var_name = tensor_to_weight.at(static_cast<int>(i));
        
        const flatbuffers::Vector<uint8_t>* data_vec = buffer->data();
        if (!data_vec || data_vec->size() < num_elements * sizeof(float)) {
            std::cerr << std::format("Warning: Buffer size mismatch for tensor {}\n", i);
            continue;
        }
        
        const float* float_data = reinterpret_cast<const float*>(data_vec->data());
        
        nlohmann::json weight;
        weight["tensor_index"] = static_cast<int>(i);
        weight["tensor_name"] = tensor_name;
        weight["var_name"] = var_name;
        weight["shape"] = GetShapeString(tensor->shape());
        weight["num_elements"] = num_elements;
        
        std::ostringstream value_stream;
        value_stream << std::fixed << std::setprecision(9);
        weight["values"] = nlohmann::json::array();
        for (std::size_t j = 0; j < num_elements; ++j) {
            nlohmann::json value_obj;
            value_stream.str("");
            value_stream << float_data[j];
            value_obj["value"] = value_stream.str();
            value_obj["needs_newline"] = ((j + 1) % 8 == 0);
            value_obj["is_last"] = (j == num_elements - 1);
            weight["values"].push_back(value_obj);
        }
        
        data["weights"].push_back(weight);
    }
    
    std::string template_path = templates_dir + "/weights.cpp.inja";
    std::ifstream template_file(template_path);
    if (!template_file) {
        throw CodeGenerationError(std::format("Cannot open template file: {}", template_path));
    }
    
    std::string template_content((std::istreambuf_iterator<char>(template_file)),
                           std::istreambuf_iterator<char>());
    
    try {
        inja::Template temp = env.parse(template_content);
        std::string result = env.render(temp, data);
        
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        
        out << result;
        std::cout << std::format("Generated: {}\n", output_path);
    } catch (const std::exception& e) {
        throw CodeGenerationError(
            std::format("Template rendering failed: {}", e.what())
        );
    }
}

void GenerateModelHeader(
    const std::string& output_path,
    const std::string& base_name,
    std::size_t input_size,
    std::size_t output_size,
    const std::vector<std::pair<int, std::size_t>>& intermediate_buffers,
    const std::string& templates_dir
) {
    inja::Environment env;
    
    std::string guard_name = base_name;
    // Transform to uppercase and replace invalid characters in one pass
    std::ranges::transform(guard_name, guard_name.begin(),
        [](char c) -> char {
            if (c == '-' || c == '.') {
                return '_';
            }
            return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        });
    guard_name += "_MODEL_H";
    
    nlohmann::json data;
    data["base_name"] = base_name;
    data["guard_name"] = guard_name;
    data["input_size"] = input_size;
    data["output_size"] = output_size;
    data["intermediate_buffers"] = nlohmann::json::array();
    
    for (const auto& [idx, size] : intermediate_buffers) {
        nlohmann::json buffer;
        buffer["index"] = idx;
        buffer["size"] = size;
        data["intermediate_buffers"].push_back(buffer);
    }
    
    std::string template_path = templates_dir + "/model.h.inja";
    std::ifstream template_file(template_path);
    if (!template_file) {
        throw CodeGenerationError(std::format("Cannot open template file: {}", template_path));
    }
    
    std::string template_content((std::istreambuf_iterator<char>(template_file)),
                           std::istreambuf_iterator<char>());
    
    try {
        inja::Template temp = env.parse(template_content);
        std::string result = env.render(temp, data);
        
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        
        out << result;
        std::cout << std::format("Generated: {}\n", output_path);
    } catch (const std::exception& e) {
        throw CodeGenerationError(
            std::format("Template rendering failed: {}", e.what())
        );
    }
}

void GenerateModelFile(
    const std::string& output_path,
    const std::string& base_name,
    const tflite::Model* model,
    const tflite::SubGraph* subgraph,
    const std::map<int, std::string>& tensor_to_weight,
    const std::vector<std::pair<int, std::size_t>>& intermediate_buffers,
    const std::map<int, std::size_t>& tensor_sizes,
    int32_t input_tensor_idx,
    int32_t output_tensor_idx,
    const std::string& templates_dir
) {
    inja::Environment env;
    
    std::vector<std::string> generation_errors;
    
    const auto* operators = subgraph->operators();
    const auto* operator_codes = model->operator_codes();
    const auto* tensors = subgraph->tensors();
    
    if (!operators || !operator_codes || !tensors) {
        throw CodeGenerationError("Invalid model structure");
    }
    
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
    
    nlohmann::json data;
    data["base_name"] = base_name;
    data["has_standalone_relu"] = false;
    data["has_conv2d"] = false;
    data["has_max_pool2d"] = false;
    data["has_shape"] = false;
    data["has_strided_slice"] = false;
    data["has_pack"] = false;
    data["has_reshape"] = false;
    data["has_add"] = false;
    data["has_dropout"] = false;
    data["has_flatten"] = false;
    data["layers"] = nlohmann::json::array();
    data["intermediate_buffers"] = nlohmann::json::array();
    
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
            has_standalone_relu = true;
            data["has_standalone_relu"] = true;
        } else if (op_name == "CONV_2D") {
            has_conv2d = true;
            data["has_conv2d"] = true;
        } else if (op_name == "MAX_POOL_2D") {
            has_max_pool2d = true;
            data["has_max_pool2d"] = true;
        } else if (op_name == "SHAPE") {
            has_shape = true;
            data["has_shape"] = true;
        } else if (op_name == "STRIDED_SLICE") {
            has_strided_slice = true;
            data["has_strided_slice"] = true;
        } else if (op_name == "PACK") {
            has_pack = true;
            data["has_pack"] = true;
        } else if (op_name == "RESHAPE") {
            has_reshape = true;
            data["has_reshape"] = true;
        } else if (op_name == "ADD") {
            has_add = true;
            data["has_add"] = true;
        } else if (op_name.find("DROPOUT") != std::string::npos || 
                   op_name.find("Dropout") != std::string::npos) {
            has_dropout = true;
            data["has_dropout"] = true;
            op_name = "DROPOUT";
        } else if (op_name.find("FLATTEN") != std::string::npos ||
                   op_name.find("Flatten") != std::string::npos) {
            has_flatten = true;
            data["has_flatten"] = true;
            op_name = "FLATTEN";
        }
        
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
        
        std::string input_ptr, output_ptr;
        if (op_input_tensor_idx == input_tensor_idx) {
            input_ptr = "input";
        } else {
            input_ptr = std::format("buffer_{}", op_input_tensor_idx);
        }
        
        if (op_output_tensor_idx == output_tensor_idx) {
            output_ptr = "output";
        } else {
            output_ptr = std::format("buffer_{}", op_output_tensor_idx);
        }
        
        std::size_t input_size_layer = tensor_sizes.count(op_input_tensor_idx) ? tensor_sizes.at(op_input_tensor_idx) : 0;
        std::size_t output_size_layer = tensor_sizes.count(op_output_tensor_idx) ? tensor_sizes.at(op_output_tensor_idx) : 0;
        
        nlohmann::json layer;
        layer["index"] = static_cast<int>(i);
        layer["op_name"] = op_name;
        layer["input_ptr"] = input_ptr;
        layer["output_ptr"] = output_ptr;
        layer["input_size"] = input_size_layer;
        layer["output_size"] = output_size_layer;
        
        if (op_name == "FULLY_CONNECTED") {
            int weights_tensor_idx = -1;
            int bias_tensor_idx = -1;
            
            if (op_inputs && op_inputs->size() >= 3) {
                weights_tensor_idx = op_inputs->Get(1);
                bias_tensor_idx = op_inputs->Get(2);
            }
            
            if (weights_tensor_idx >= 0 && bias_tensor_idx >= 0) {
                if (!tensor_to_weight.contains(weights_tensor_idx) ||
                    !tensor_to_weight.contains(bias_tensor_idx)) {
                    std::string error_msg = std::format(
                        "FULLY_CONNECTED layer {} missing required weight/bias tensors (weights_idx={}, bias_idx={})",
                        i, weights_tensor_idx, bias_tensor_idx
                    );
                    generation_errors.push_back(error_msg);
                    continue;
                }
                
                std::string weights_var = tensor_to_weight.at(weights_tensor_idx);
                std::string bias_var = tensor_to_weight.at(bias_tensor_idx);
                
                std::size_t fc_input_size = input_size_layer;
                if (weights_tensor_idx >= 0 && weights_tensor_idx < static_cast<int>(tensors->size())) {
                    const tflite::Tensor* weights_tensor = tensors->Get(weights_tensor_idx);
                    if (weights_tensor && weights_tensor->shape() && weights_tensor->shape()->size() >= 2) {
                        fc_input_size = weights_tensor->shape()->Get(1);
                    }
                }
                
                TfLiteFusedActivation fused_activation = kTfLiteActNone;
                const tflite::FullyConnectedOptions* options = 
                    op->builtin_options_as_FullyConnectedOptions();
                if (options) {
                    fused_activation = ConvertActivation(options->fused_activation_function());
                }
                
                std::string activation_param = "ActivationType::NONE";
                if (fused_activation == kTfLiteActRelu) {
                    activation_param = "ActivationType::RELU";
                } else if (fused_activation != kTfLiteActNone) {
                    std::cerr << std::format("Warning: Unsupported fused activation {} in FULLY_CONNECTED layer {} (only ReLU supported, using NONE)\n",
                        static_cast<int>(fused_activation), i);
                }
                
                layer["weights_var"] = weights_var;
                layer["bias_var"] = bias_var;
                layer["fc_input_size"] = fc_input_size;
                layer["activation_param"] = activation_param;
            } else {
                std::string error_msg = std::format(
                    "FULLY_CONNECTED layer {} has insufficient inputs (expected at least 3: input, weights, bias)",
                    i
                );
                generation_errors.push_back(error_msg);
                continue;
            }
        } else if (op_name == "RELU" || op_name == "SOFTMAX") {
            // Simple operations
        } else if (op_name == "CONV_2D") {
            int filter_tensor_idx = -1;
            int bias_tensor_idx = -1;
            
            if (op_inputs && op_inputs->size() >= 3) {
                filter_tensor_idx = op_inputs->Get(1);
                bias_tensor_idx = op_inputs->Get(2);
            }
            
            if (filter_tensor_idx >= 0 && bias_tensor_idx >= 0) {
                if (!tensor_to_weight.contains(filter_tensor_idx) ||
                    !tensor_to_weight.contains(bias_tensor_idx)) {
                    std::cerr << std::format("Warning: Cannot find filter/bias tensors for CONV_2D layer {}\n", i);
                    continue;
                }
                
                std::string filter_var = tensor_to_weight.at(filter_tensor_idx);
                std::string bias_var = tensor_to_weight.at(bias_tensor_idx);
                
                const tflite::Tensor* input_tensor = (op_input_tensor_idx >= 0 && op_input_tensor_idx < static_cast<int>(tensors->size())) 
                    ? tensors->Get(op_input_tensor_idx) : nullptr;
                const tflite::Tensor* filter_tensor = (filter_tensor_idx >= 0 && filter_tensor_idx < static_cast<int>(tensors->size()))
                    ? tensors->Get(filter_tensor_idx) : nullptr;
                const tflite::Tensor* output_tensor = (op_output_tensor_idx >= 0 && op_output_tensor_idx < static_cast<int>(tensors->size()))
                    ? tensors->Get(op_output_tensor_idx) : nullptr;
                
                if (!input_tensor || !filter_tensor || !output_tensor ||
                    !input_tensor->shape() || !filter_tensor->shape() || !output_tensor->shape()) {
                    std::cerr << std::format("Warning: Cannot get tensor shapes for CONV_2D layer {}\n", i);
                    continue;
                }
                
                std::size_t batch_size = input_tensor->shape()->Get(0);
                std::size_t input_height = input_tensor->shape()->Get(1);
                std::size_t input_width = input_tensor->shape()->Get(2);
                std::size_t input_channels = input_tensor->shape()->Get(3);
                std::size_t filter_height = filter_tensor->shape()->Get(1);
                std::size_t filter_width = filter_tensor->shape()->Get(2);
                std::size_t output_channels = filter_tensor->shape()->Get(0);
                
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
                
                std::string padding_param = (padding == kTfLitePaddingSame) ? 
                    "PaddingType::SAME" : "PaddingType::VALID";
                std::string activation_param = (fused_activation == kTfLiteActRelu) ? 
                    "ActivationType::RELU" : "ActivationType::NONE";
                
                layer["filter_var"] = filter_var;
                layer["bias_var"] = bias_var;
                layer["batch_size"] = batch_size;
                layer["input_height"] = input_height;
                layer["input_width"] = input_width;
                layer["input_channels"] = input_channels;
                layer["filter_height"] = filter_height;
                layer["filter_width"] = filter_width;
                layer["output_channels"] = output_channels;
                layer["stride_height"] = stride_height;
                layer["stride_width"] = stride_width;
                layer["padding_param"] = padding_param;
                layer["activation_param"] = activation_param;
                layer["dilation_height"] = dilation_height;
                layer["dilation_width"] = dilation_width;
            } else {
                std::cerr << std::format("Warning: CONV_2D layer {} missing filter/bias\n", i);
                continue;
            }
        } else if (op_name == "MAX_POOL_2D") {
            const tflite::Tensor* input_tensor = (op_input_tensor_idx >= 0 && op_input_tensor_idx < static_cast<int>(tensors->size()))
                ? tensors->Get(op_input_tensor_idx) : nullptr;
            const tflite::Tensor* output_tensor = (op_output_tensor_idx >= 0 && op_output_tensor_idx < static_cast<int>(tensors->size()))
                ? tensors->Get(op_output_tensor_idx) : nullptr;
            
            if (!input_tensor || !output_tensor ||
                !input_tensor->shape() || !output_tensor->shape()) {
                std::cerr << std::format("Warning: Cannot get tensor shapes for MAX_POOL_2D layer {}\n", i);
                continue;
            }
            
            std::size_t batch_size = input_tensor->shape()->Get(0);
            std::size_t input_height = input_tensor->shape()->Get(1);
            std::size_t input_width = input_tensor->shape()->Get(2);
            std::size_t channels = input_tensor->shape()->Get(3);
            
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
            
            std::string padding_param = (padding == kTfLitePaddingSame) ? 
                "PaddingType::SAME" : "PaddingType::VALID";
            std::string activation_param = (fused_activation == kTfLiteActRelu) ? 
                "ActivationType::RELU" : "ActivationType::NONE";
            
            layer["batch_size"] = batch_size;
            layer["input_height"] = input_height;
            layer["input_width"] = input_width;
            layer["channels"] = channels;
            layer["filter_height"] = filter_height;
            layer["filter_width"] = filter_width;
            layer["stride_height"] = stride_height;
            layer["stride_width"] = stride_width;
            layer["padding_param"] = padding_param;
            layer["activation_param"] = activation_param;
        } else if (op_name == "SHAPE") {
            const tflite::Tensor* input_tensor = (op_input_tensor_idx >= 0 && op_input_tensor_idx < static_cast<int>(tensors->size()))
                ? tensors->Get(op_input_tensor_idx) : nullptr;
            
            if (!input_tensor || !input_tensor->shape()) {
                std::cerr << std::format("Warning: Cannot get tensor shapes for SHAPE layer {}\n", i);
                continue;
            }
            
            int num_dims = input_tensor->shape()->size();
            std::ostringstream shape_values;
            for (int j = 0; j < num_dims; ++j) {
                shape_values << input_tensor->shape()->Get(j);
                if (j < num_dims - 1) shape_values << ", ";
            }
            
            layer["num_dims"] = num_dims;
            layer["shape_values"] = shape_values.str();
        } else if (op_name == "STRIDED_SLICE" || op_name == "PACK") {
            if (op_name == "PACK") {
                int axis = 0;
                const tflite::PackOptions* options = op->builtin_options_as_PackOptions();
                if (options) {
                    axis = options->axis();
                }
                layer["axis"] = axis;
            }
        } else if (op_name == "RESHAPE" || op_name == "FLATTEN") {
            // Already have all needed fields
        } else if (op_name == "ADD") {
            if (!op_inputs || op_inputs->size() < 2) {
                std::string error_msg = std::format(
                    "ADD layer {} missing inputs: expected 2 inputs, but {}",
                    i, !op_inputs ? "no input list provided" : std::format("only {} provided", op_inputs->size())
                );
                generation_errors.push_back(error_msg);
                continue;
            }
            
            int input1_tensor_idx = op_inputs->Get(0);
            int input2_tensor_idx = op_inputs->Get(1);
            
            if (input1_tensor_idx < 0 || input2_tensor_idx < 0) {
                std::string error_msg = std::format(
                    "ADD layer {} missing inputs: input1={}, input2={} (negative indices indicate optional/unused tensors, which are not supported)",
                    i, input1_tensor_idx, input2_tensor_idx
                );
                generation_errors.push_back(error_msg);
                continue;
            }
            
            std::string input1_ptr, input2_ptr;
            
            if (input1_tensor_idx == input_tensor_idx) {
                input1_ptr = "input";
            } else if (tensor_to_weight.contains(input1_tensor_idx)) {
                input1_ptr = tensor_to_weight.at(input1_tensor_idx);
            } else {
                input1_ptr = std::format("buffer_{}", input1_tensor_idx);
            }
            
            if (input2_tensor_idx == input_tensor_idx) {
                input2_ptr = "input";
            } else if (tensor_to_weight.contains(input2_tensor_idx)) {
                input2_ptr = tensor_to_weight.at(input2_tensor_idx);
            } else {
                input2_ptr = std::format("buffer_{}", input2_tensor_idx);
            }
            
            std::size_t add_size = output_size_layer;
            
            TfLiteFusedActivation fused_activation = kTfLiteActNone;
            const tflite::AddOptions* options = op->builtin_options_as_AddOptions();
            if (options) {
                fused_activation = ConvertActivation(options->fused_activation_function());
            }
            
            std::string activation_param = "ActivationType::NONE";
            if (fused_activation == kTfLiteActRelu) {
                activation_param = "ActivationType::RELU";
            } else if (fused_activation != kTfLiteActNone) {
                std::cerr << std::format("Warning: Unsupported fused activation {} in ADD layer {} (only ReLU supported, using NONE)\n",
                    static_cast<int>(fused_activation), i);
            }
            
            layer["input1_ptr"] = input1_ptr;
            layer["input2_ptr"] = input2_ptr;
            layer["add_size"] = add_size;
            layer["activation_param"] = activation_param;
        } else if (op_name == "DROPOUT") {
            float dropout_rate = 0.0f;
            layer["dropout_rate"] = dropout_rate;
        } else {
            std::cerr << std::format("Warning: Unsupported operation {}\n", op_name);
            continue;
        }
        
        data["layers"].push_back(layer);
    }
    
    for (const auto& [idx, size] : intermediate_buffers) {
        nlohmann::json buffer;
        buffer["index"] = idx;
        buffer["size"] = size;
        data["intermediate_buffers"].push_back(buffer);
    }
    
    if (!generation_errors.empty()) {
        std::string error_msg = "\n================================================\n";
        error_msg += "ERROR: Code generation failed due to model structure issues!\n";
        error_msg += "================================================\n\n";
        error_msg += "The following errors occurred during code generation:\n";
        for (const auto& error : generation_errors) {
            error_msg += std::format("  - {}\n", error);
        }
        error_msg += "\nCode generation aborted.\n";
        error_msg += "The generated file may be incomplete or incorrect.\n";
        error_msg += "================================================\n";
        throw CodeGenerationError(error_msg);
    }
    
    std::string template_path = templates_dir + "/model.cpp.inja";
    std::ifstream template_file(template_path);
    if (!template_file) {
        throw CodeGenerationError(std::format("Cannot open template file: {}", template_path));
    }
    
    std::string template_content((std::istreambuf_iterator<char>(template_file)),
                           std::istreambuf_iterator<char>());
    
    try {
        inja::Template temp = env.parse(template_content);
        std::string result = env.render(temp, data);
        
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        
        out << result;
        std::cout << std::format("Generated: {}\n", output_path);
    } catch (const std::exception& e) {
        throw CodeGenerationError(
            std::format("Template rendering failed: {}", e.what())
        );
    }
}

void GenerateInferenceFile(
    const std::string& output_path,
    const std::string& base_name,
    const tflite::SubGraph* subgraph,
    int32_t input_tensor_idx,
    int32_t output_tensor_idx,
    InferenceType inference_type,
    const std::string& templates_dir
) {
    // Skip generation if inference type is None
    if (inference_type == InferenceType::None) {
        return;
    }
    
    inja::Environment env;
    
    const auto* tensors = subgraph->tensors();
    if (!tensors) {
        throw CodeGenerationError("Cannot access tensors");
    }
    
    const tflite::Tensor* input_tensor = (input_tensor_idx >= 0 && input_tensor_idx < static_cast<int>(tensors->size()))
        ? tensors->Get(input_tensor_idx) : nullptr;
    const tflite::Tensor* output_tensor = (output_tensor_idx >= 0 && output_tensor_idx < static_cast<int>(tensors->size()))
        ? tensors->Get(output_tensor_idx) : nullptr;
    
    if (!input_tensor || !output_tensor) {
        throw CodeGenerationError("Cannot access input/output tensors");
    }
    
    std::size_t input_size = CalculateTensorSize(input_tensor->shape());
    std::size_t output_size = CalculateTensorSize(output_tensor->shape());
    
    nlohmann::json data;
    data["base_name"] = base_name;
    data["input_size"] = input_size;
    data["output_size"] = output_size;
    data["input_shape"] = GetShapeString(input_tensor->shape());
    data["output_shape"] = GetShapeString(output_tensor->shape());
    
    data["input_indices"] = nlohmann::json::array();
    for (std::size_t i = 0; i < input_size; ++i) {
        nlohmann::json idx_obj;
        idx_obj["needs_newline"] = ((i + 1) % 8 == 0);
        idx_obj["is_last"] = (i == input_size - 1);
        data["input_indices"].push_back(idx_obj);
    }
    
    // Select template based on inference type
    std::string template_filename;
    if (inference_type == InferenceType::Standard) {
        template_filename = "inference.cpp.inja";
    } else if (inference_type == InferenceType::PicoImgBench) {
        template_filename = "inference_pico.cpp.inja";
    } else {
        throw CodeGenerationError("Invalid inference type for generation");
    }
    
    std::string template_path = templates_dir + "/" + template_filename;
    std::ifstream template_file(template_path);
    if (!template_file) {
        throw CodeGenerationError(std::format("Cannot open template file: {}", template_path));
    }
    
    std::string template_content((std::istreambuf_iterator<char>(template_file)),
                           std::istreambuf_iterator<char>());
    
    try {
        inja::Template temp = env.parse(template_content);
        std::string result = env.render(temp, data);
        
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        
        out << result;
        std::cout << std::format("Generated: {}\n", output_path);
    } catch (const std::exception& e) {
        throw CodeGenerationError(
            std::format("Template rendering failed: {}", e.what())
        );
    }
}

void GenerateMakefile(
    const std::string& output_path,
    const std::string& base_name,
    const std::string& templates_dir
) {
    inja::Environment env;
    
    nlohmann::json data;
    data["base_name"] = base_name;
    
    std::string template_path = templates_dir + "/Makefile.inja";
    std::ifstream template_file(template_path);
    if (!template_file) {
        throw CodeGenerationError(std::format("Cannot open template file: {}", template_path));
    }
    
    std::string template_content((std::istreambuf_iterator<char>(template_file)),
                           std::istreambuf_iterator<char>());
    
    try {
        inja::Template temp = env.parse(template_content);
        std::string result = env.render(temp, data);
        
        std::ofstream out(output_path);
        if (!out) {
            throw CodeGenerationError(
                std::format("Cannot create file: {}", output_path)
            );
        }
        
        out << result;
        std::cout << std::format("Generated: {}\n", output_path);
    } catch (const std::exception& e) {
        throw CodeGenerationError(
            std::format("Template rendering failed: {}", e.what())
        );
    }
}
