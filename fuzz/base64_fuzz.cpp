#include "../src/base64.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    if (size > (1u << 20)) return 0;
    const std::string input(reinterpret_cast<const char *>(data), size);
    std::vector<uint8_t> decoded;
    if (!s2::base64_decode(input, decoded)) return 0;

    const std::string encoded = s2::base64_encode(decoded.data(), decoded.size());
    std::vector<uint8_t> roundtrip;
    if (!s2::base64_decode(encoded, roundtrip) || roundtrip != decoded)
        __builtin_trap();
    return 0;
}
