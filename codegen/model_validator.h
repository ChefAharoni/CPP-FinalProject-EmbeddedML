// codegen/model_validator.h
// Model validation logic

#ifndef CODEGEN_MODEL_VALIDATOR_H
#define CODEGEN_MODEL_VALIDATOR_H

#include "tensorflow/lite/schema/schema_generated.h"

// Validate operators from model schema
// This provides clear error messages for unsupported operations
void ValidateModelSchema(const tflite::Model* model);

#endif // CODEGEN_MODEL_VALIDATOR_H
