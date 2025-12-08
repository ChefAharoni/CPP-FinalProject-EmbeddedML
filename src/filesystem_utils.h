// codegen/filesystem_utils.h
// Filesystem operations

#ifndef CODEGEN_FILESYSTEM_UTILS_H
#define CODEGEN_FILESYSTEM_UTILS_H

#include <string>
#include <vector>

// Create directory if it doesn't exist
void CreateDirectory(const std::string& path);

// Recursively delete a directory and all its contents
void DeleteDirectory(const std::string& path);

// Copy a file from source to destination
void CopyFile(const std::string& source, const std::string& destination);

#endif // CODEGEN_FILESYSTEM_UTILS_H
