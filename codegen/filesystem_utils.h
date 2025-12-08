// codegen/filesystem_utils.h
// Filesystem operations

#ifndef CODEGEN_FILESYSTEM_UTILS_H
#define CODEGEN_FILESYSTEM_UTILS_H

#include <string>

// Create directory if it doesn't exist
void CreateDirectory(const std::string& path);

// Recursively delete a directory and all its contents
void DeleteDirectory(const std::string& path);

#endif // CODEGEN_FILESYSTEM_UTILS_H
