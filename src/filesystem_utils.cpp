// codegen/filesystem_utils.cpp
// Filesystem operations implementation

#include "filesystem_utils.h"
#include "exceptions.h"
#include <filesystem>
#include <format>

void CreateDirectory(const std::string& path) {
    try {
        if (std::filesystem::exists(path)) {
            if (!std::filesystem::is_directory(path)) {
                throw FileSystemError(
                    std::format("Path exists but is not a directory: {}", path)
                );
            }
            return;
        }
        
        std::filesystem::create_directories(path);
    } catch (const std::filesystem::filesystem_error& e) {
        throw FileSystemError(
            std::format("Failed to create directory {}: {}", path, e.what())
        );
    }
}

void DeleteDirectory(const std::string& path) {
    try {
        if (!std::filesystem::exists(path)) {
            return;
        }
        
        if (!std::filesystem::is_directory(path)) {
            throw FileSystemError(
                std::format("Path is not a directory: {}", path)
            );
        }
        
        std::filesystem::remove_all(path);
    } catch (const std::filesystem::filesystem_error& e) {
        throw FileSystemError(
            std::format("Failed to delete directory {}: {}", path, e.what())
        );
    }
}

void CopyFile(const std::string& source, const std::string& destination) {
    try {
        if (!std::filesystem::exists(source)) {
            throw FileSystemError(
                std::format("Source file does not exist: {}", source)
            );
        }
        
        if (!std::filesystem::is_regular_file(source)) {
            throw FileSystemError(
                std::format("Source is not a regular file: {}", source)
            );
        }
        
        // Create destination directory if it doesn't exist
        std::filesystem::path dest_path(destination);
        if (dest_path.has_parent_path()) {
            std::filesystem::create_directories(dest_path.parent_path());
        }
        
        std::filesystem::copy_file(source, destination, std::filesystem::copy_options::overwrite_existing);
    } catch (const std::filesystem::filesystem_error& e) {
        throw FileSystemError(
            std::format("Failed to copy file from {} to {}: {}", source, destination, e.what())
        );
    }
}
