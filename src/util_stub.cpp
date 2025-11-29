// util_stub.cpp
// Stub implementation for util::Fingerprint64
// This is needed because TFLite's lsh_projection operator references this function
// which isn't available in the CMake build dependencies

#include <cstdint>
#include <cstring>

namespace util {
uint64_t Fingerprint64(const char* s, size_t len) {
    // Simple hash function - sufficient for model inspection purposes
    // This is a minimal implementation; for actual inference you'd want the full version
    uint64_t hash = 5381;
    for (size_t i = 0; i < len; ++i) {
        hash = ((hash << 5) + hash) + static_cast<unsigned char>(s[i]);
    }
    return hash;
}
}




