// codegen/model_validator.cpp
// Model validation logic implementation

#include "model_validator.h"
#include "exceptions.h"
#include "model_utils.h"
#include <vector>
#include <set>
#include <format>

void ValidateModelSchema(const tflite::Model* model) {
    if (!model || !model->subgraphs() || model->subgraphs()->size() == 0) {
        throw ModelValidationError("Invalid model structure - no subgraphs found.");
    }

    const tflite::SubGraph* subgraph = model->subgraphs()->Get(0);
    if (!subgraph || !subgraph->operators()) {
        throw ModelValidationError("Invalid model structure - no operators found.");
    }

    const flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>* op_codes = 
        model->operator_codes();
    if (!op_codes) {
        throw ModelValidationError("Invalid model structure - no operator codes found.");
    }

    bool ok = true;
    std::vector<std::pair<int, std::string>> unsupported_ops;
    std::vector<std::pair<int, int>> unsupported_activations;
    const auto* tensors = subgraph->tensors();

    std::set<tflite::BuiltinOperator> supported_ops = {
        tflite::BuiltinOperator_FULLY_CONNECTED,
        tflite::BuiltinOperator_SOFTMAX,
        tflite::BuiltinOperator_RELU,
        tflite::BuiltinOperator_CONV_2D,
        tflite::BuiltinOperator_MAX_POOL_2D,
        tflite::BuiltinOperator_SHAPE,
        tflite::BuiltinOperator_STRIDED_SLICE,
        tflite::BuiltinOperator_PACK,
        tflite::BuiltinOperator_RESHAPE,
        tflite::BuiltinOperator_ADD
    };

    for (std::size_t i = 0; i < subgraph->operators()->size(); ++i) {
        const tflite::Operator* op = subgraph->operators()->Get(i);
        if (!op) continue;

        int op_code_index = op->opcode_index();
        if (op_code_index < 0 || op_code_index >= static_cast<int>(op_codes->size())) {
            throw ModelValidationError(
                std::format("Invalid operator code index {} at operator {}.", op_code_index, i)
            );
        }

        const tflite::OperatorCode* op_code = op_codes->Get(op_code_index);
        if (!op_code) {
            throw ModelValidationError(
                std::format("Operator {} has null operator code", i)
            );
        }

        tflite::BuiltinOperator builtin_code = static_cast<tflite::BuiltinOperator>(
            std::max(static_cast<int>(op_code->builtin_code()),
                     static_cast<int>(op_code->deprecated_builtin_code())));

        if (!supported_ops.contains(builtin_code)) {
            std::string op_name;
            if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
                op_name = std::string("CUSTOM:") + 
                    (op_code->custom_code() ? op_code->custom_code()->c_str() : "");
                std::string custom_name = op_code->custom_code() ? op_code->custom_code()->c_str() : "";
                if (custom_name.find("DROPOUT") != std::string::npos || 
                    custom_name.find("Dropout") != std::string::npos ||
                    custom_name.find("FLATTEN") != std::string::npos ||
                    custom_name.find("Flatten") != std::string::npos) {
                    continue;
                }
            } else {
                const char* op_name_ptr = tflite::EnumNameBuiltinOperator(builtin_code);
                op_name = op_name_ptr ? op_name_ptr : "UNKNOWN";
            }
            
            std::string op_details = op_name;
            if (tensors && op->inputs() && op->inputs()->size() > 0) {
                op_details += " (inputs: ";
                for (std::size_t j = 0; j < op->inputs()->size(); ++j) {
                    int input_idx = op->inputs()->Get(j);
                    if (input_idx >= 0 && input_idx < static_cast<int>(tensors->size())) {
                        const tflite::Tensor* input_tensor = tensors->Get(input_idx);
                        if (input_tensor && input_tensor->shape()) {
                            op_details += "tensor[" + std::to_string(input_idx) + ":" + 
                                         GetShapeString(input_tensor->shape()) + "]";
                        } else {
                            op_details += "tensor[" + std::to_string(input_idx) + "]";
                        }
                    } else {
                        op_details += "tensor[" + std::to_string(input_idx) + "]";
                    }
                    if (j < op->inputs()->size() - 1) op_details += ", ";
                }
                op_details += ")";
            }
            
            unsupported_ops.push_back({static_cast<int>(i), op_details});
            ok = false;
        }

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
        } else if (builtin_code == tflite::BuiltinOperator_ADD) {
            const tflite::AddOptions* options = 
                op->builtin_options_as_AddOptions();
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

    if (!ok) {
        std::string error_msg = "\n================================================\n";
        error_msg += "ERROR: Model contains unsupported operations!\n";
        error_msg += "================================================\n\n";
        error_msg += "This code generator only supports the following operations:\n";
        error_msg += "  - FULLY_CONNECTED (with NONE or RELU activation)\n";
        error_msg += "  - SOFTMAX\n";
        error_msg += "  - RELU\n";
        error_msg += "  - CONV_2D (with NONE or RELU activation)\n";
        error_msg += "  - MAX_POOL_2D\n";
        error_msg += "  - SHAPE\n";
        error_msg += "  - STRIDED_SLICE\n";
        error_msg += "  - PACK\n";
        error_msg += "  - RESHAPE\n";
        error_msg += "  - ADD (with NONE or RELU activation)\n";
        error_msg += "  - DROPOUT (custom operator, no-op during inference)\n";
        error_msg += "  - FLATTEN (custom operator, converts to reshape)\n";
        
        if (!unsupported_ops.empty()) {
            error_msg += std::format("\n{} unsupported operator(s) found in model:\n", unsupported_ops.size());
            error_msg += "  (Operator indices are 0-based, in execution order)\n";
            for (const auto& [op_idx, op_details] : unsupported_ops) {
                error_msg += std::format("  [Operator {}] {}\n", op_idx, op_details);
            }
            error_msg += "\n  Note: These operators appear early in the model graph.\n";
            error_msg += "        The code generator cannot skip them as they may transform\n";
            error_msg += "        tensor shapes or data that subsequent operations depend on.\n";
        }
        
        if (!unsupported_activations.empty()) {
            error_msg += "\nUnsupported fused activations found:\n";
            for (const auto& [op_idx, act_code] : unsupported_activations) {
                error_msg += std::format("  - Operator {} has unsupported activation code: {} (only NONE=0 and RELU=1 are supported)\n", 
                    op_idx, act_code);
            }
        }
        
        error_msg += "\nCode generation aborted.\n";
        error_msg += "\nTo fix this issue:\n";
        error_msg += "  1. Use a model that only contains supported operations, OR\n";
        error_msg += "  2. Request support for the missing operators to be added to the code generator\n";
        error_msg += "\nThe model structure cannot be partially generated - all operations\n";
        error_msg += "must be supported for correct code generation.\n";
        error_msg += "================================================\n";
        
        throw ModelValidationError(error_msg);
    }
}
