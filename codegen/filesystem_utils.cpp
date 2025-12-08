// codegen/filesystem_utils.cpp
// Filesystem operations implementation

#include "filesystem_utils.h"
#include "exceptions.h"
#include <filesystem>
#include <format>

void CreateDirectory(const std::string& path) {
    namespace fs = std::filesystem;
    try {
        if (fs::exists(path)) {
            if (!fs::is_directory(path)) {
                throw FileSystemError(
                    std::format("Path exists but is not a directory: {}", path)
                );
            }
            return;
        }
        
        fs::create_directories(path);
    } catch (const fs::filesystem_error& e) {
        throw FileSystemError(
            std::format("Failed to create directory {}: {}", path, e.what())
        );
    }
}

void DeleteDirectory(const std::string& path) {
    namespace fs = std::filesystem;
    try {
        if (!fs::exists(path)) {
            return;
        }
        
        if (!fs::is_directory(path)) {
            throw FileSystemError(
                std::format("Path is not a directory: {}", path)
            );
        }
        
        fs::remove_all(path);
    } catch (const fs::filesystem_error& e) {
        throw FileSystemError(
            std::format("Failed to delete directory {}: {}", path, e.what())
        );
    }
}
