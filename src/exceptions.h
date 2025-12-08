// codegen/exceptions.h
// Custom exception types for code generation

#ifndef CODEGEN_EXCEPTIONS_H
#define CODEGEN_EXCEPTIONS_H

#include <stdexcept>
#include <string>

// Custom exception types for better error handling
class ModelValidationError : public std::runtime_error {
public:
    explicit ModelValidationError(const std::string& msg) : std::runtime_error(msg) {}
};

class CodeGenerationError : public std::runtime_error {
public:
    explicit CodeGenerationError(const std::string& msg) : std::runtime_error(msg) {}
};

class FileSystemError : public std::runtime_error {
public:
    explicit FileSystemError(const std::string& msg) : std::runtime_error(msg) {}
};

#endif // CODEGEN_EXCEPTIONS_H
