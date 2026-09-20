#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <limits>

namespace s2 {

inline std::string base64_encode(const unsigned char * data, size_t len) {
    static constexpr char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    if (data == nullptr && len != 0) return out;
    const size_t max_input = (std::numeric_limits<size_t>::max() / 4u) * 3u;
    if (len > max_input) return out;
    const size_t groups = len / 3u + (len % 3u != 0u ? 1u : 0u);
    out.reserve(groups * 4u);
    size_t i = 0;
    while (i + 3u <= len) {
        const uint32_t v = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8) |
                           static_cast<uint32_t>(data[i + 2]);
        out.push_back(kTable[(v >> 18) & 63u]);
        out.push_back(kTable[(v >> 12) & 63u]);
        out.push_back(kTable[(v >> 6) & 63u]);
        out.push_back(kTable[v & 63u]);
        i += 3u;
    }
    const size_t rem = len - i;
    if (rem == 1u) {
        const uint32_t v = static_cast<uint32_t>(data[i]) << 16;
        out.push_back(kTable[(v >> 18) & 63u]);
        out.push_back(kTable[(v >> 12) & 63u]);
        out += "==";
    } else if (rem == 2u) {
        const uint32_t v = (static_cast<uint32_t>(data[i]) << 16) |
                           (static_cast<uint32_t>(data[i + 1]) << 8);
        out.push_back(kTable[(v >> 18) & 63u]);
        out.push_back(kTable[(v >> 12) & 63u]);
        out.push_back(kTable[(v >> 6) & 63u]);
        out.push_back('=');
    }
    return out;
}

inline int base64_value(unsigned char c) noexcept {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

// Strict/canonical base64 only: no whitespace, URL alphabet, or misplaced
// padding. This keeps request-size accounting deterministic and rejects input
// that different decoders might interpret differently.
inline bool base64_decode(const std::string & input, std::vector<uint8_t> & out) {
    out.clear();
    if (input.empty()) return true;
    if ((input.size() & 3u) != 0u) return false;

    size_t padding = 0;
    if (!input.empty() && input.back() == '=') ++padding;
    if (input.size() >= 2u && input[input.size() - 2u] == '=') ++padding;
    if (padding > 2u) return false;
    for (size_t i = 0; i + padding < input.size(); ++i) {
        if (input[i] == '=' || base64_value(static_cast<unsigned char>(input[i])) < 0) return false;
    }
    for (size_t i = input.size() - padding; i < input.size(); ++i) {
        if (input[i] != '=') return false;
    }

    const size_t groups = input.size() / 4u;
    if (groups > (static_cast<size_t>(-1) - 2u) / 3u) return false;
    out.reserve(groups * 3u - padding);
    for (size_t g = 0; g < groups; ++g) {
        const size_t i = g * 4u;
        const bool last = g + 1u == groups;
        const int a = base64_value(static_cast<unsigned char>(input[i]));
        const int b = base64_value(static_cast<unsigned char>(input[i + 1u]));
        const int c = input[i + 2u] == '=' ? 0 : base64_value(static_cast<unsigned char>(input[i + 2u]));
        const int d = input[i + 3u] == '=' ? 0 : base64_value(static_cast<unsigned char>(input[i + 3u]));
        if (a < 0 || b < 0 || c < 0 || d < 0) return false;
        if (!last && (input[i + 2u] == '=' || input[i + 3u] == '=')) return false;
        if (input[i + 2u] == '=' && input[i + 3u] != '=') return false;
        if (input[i + 2u] == '=' && (b & 0x0f) != 0) return false; // non-canonical spare bits
        if (input[i + 3u] == '=' && input[i + 2u] != '=' && (c & 0x03) != 0) return false;

        const uint32_t v = (static_cast<uint32_t>(a) << 18) |
                           (static_cast<uint32_t>(b) << 12) |
                           (static_cast<uint32_t>(c) << 6) |
                           static_cast<uint32_t>(d);
        out.push_back(static_cast<uint8_t>((v >> 16) & 0xffu));
        if (input[i + 2u] != '=') out.push_back(static_cast<uint8_t>((v >> 8) & 0xffu));
        if (input[i + 3u] != '=') out.push_back(static_cast<uint8_t>(v & 0xffu));
    }
    return true;
}

} // namespace s2
