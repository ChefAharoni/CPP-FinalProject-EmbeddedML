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

void CopyFile(const std::string& source, const std::string& destination) {
    namespace fs = std::filesystem;
    try {
        if (!fs::exists(source)) {
            throw FileSystemError(
                std::format("Source file does not exist: {}", source)
            );
        }
        
        if (!fs::is_regular_file(source)) {
            throw FileSystemError(
                std::format("Source is not a regular file: {}", source)
            );
        }
        
        // Create destination directory if it doesn't exist
        fs::path dest_path(destination);
        if (dest_path.has_parent_path()) {
            fs::create_directories(dest_path.parent_path());
        }
        
        fs::copy_file(source, destination, fs::copy_options::overwrite_existing);
    } catch (const fs::filesystem_error& e) {
        throw FileSystemError(
            std::format("Failed to copy file from {} to {}: {}", source, destination, e.what())
        );
    }
}
