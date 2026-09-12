#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace s2 { namespace utf8 {

inline bool decode_one(const std::string & s, size_t pos,
                       uint32_t & cp, size_t & width) noexcept {
    if (pos >= s.size()) return false;
    const unsigned char b0 = static_cast<unsigned char>(s[pos]);
    if (b0 <= 0x7Fu) {
        cp = b0;
        width = 1;
        return true;
    }

    if (b0 >= 0xC2u && b0 <= 0xDFu) {
        if (pos + 1 >= s.size()) return false;
        const unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        if ((b1 & 0xC0u) != 0x80u) return false;
        cp = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
        width = 2;
        return true;
    }

    if (b0 >= 0xE0u && b0 <= 0xEFu) {
        if (pos + 2 >= s.size()) return false;
        const unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        const unsigned char b2 = static_cast<unsigned char>(s[pos + 2]);
        if ((b1 & 0xC0u) != 0x80u || (b2 & 0xC0u) != 0x80u) return false;
        if (b0 == 0xE0u && b1 < 0xA0u) return false;   // overlong
        if (b0 == 0xEDu && b1 >= 0xA0u) return false; // surrogate
        cp = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
        width = 3;
        return true;
    }

    if (b0 >= 0xF0u && b0 <= 0xF4u) {
        if (pos + 3 >= s.size()) return false;
        const unsigned char b1 = static_cast<unsigned char>(s[pos + 1]);
        const unsigned char b2 = static_cast<unsigned char>(s[pos + 2]);
        const unsigned char b3 = static_cast<unsigned char>(s[pos + 3]);
        if ((b1 & 0xC0u) != 0x80u || (b2 & 0xC0u) != 0x80u ||
            (b3 & 0xC0u) != 0x80u) return false;
        if (b0 == 0xF0u && b1 < 0x90u) return false;   // overlong
        if (b0 == 0xF4u && b1 >= 0x90u) return false; // > U+10FFFF
        cp = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) |
             ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
        width = 4;
        return true;
    }

    return false;
}

inline bool is_valid(const std::string & s) noexcept {
    for (size_t pos = 0; pos < s.size();) {
        uint32_t cp = 0;
        size_t width = 0;
        if (!decode_one(s, pos, cp, width)) return false;
        (void)cp;
        pos += width;
    }
    return true;
}

inline bool is_whitespace(uint32_t cp) noexcept {
    return cp == 0x0009u || cp == 0x000Au || cp == 0x000Bu || cp == 0x000Cu ||
           cp == 0x000Du || cp == 0x0020u || cp == 0x0085u || cp == 0x00A0u ||
           cp == 0x1680u || (cp >= 0x2000u && cp <= 0x200Au) || cp == 0x2028u ||
           cp == 0x2029u || cp == 0x202Fu || cp == 0x205Fu || cp == 0x3000u;
}

inline bool has_non_whitespace(const std::string & s) noexcept {
    bool seen_non_whitespace = false;
    for (size_t pos = 0; pos < s.size();) {
        uint32_t cp = 0;
        size_t width = 0;
        if (!decode_one(s, pos, cp, width)) return false;
        if (!is_whitespace(cp)) seen_non_whitespace = true;
        pos += width;
    }
    return seen_non_whitespace;
}

}} // namespace s2::utf8
