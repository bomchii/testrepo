#include "../include/s2_json.h"

#include <cstddef>
#include <cstdint>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    if (size > (1u << 20)) return 0;
    const std::string input(reinterpret_cast<const char *>(data), size);
    std::string normalized;
    std::string error;
    (void)s2::json::normalize_surrogate_pairs(input, normalized, &error);
    return 0;
}
