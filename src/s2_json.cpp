#include "../include/s2_json.h"
#include "s2_utf8.h"

#include <cstdint>

namespace s2 { namespace json {
namespace {
int hexv(char c) noexcept {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
bool parse_u16(const std::string & s, size_t p, uint16_t & out) noexcept {
    if (p + 6 > s.size() || s[p] != '\\' || s[p + 1] != 'u') return false;
    uint32_t v = 0;
    for (size_t i = 0; i < 4; ++i) {
        const int h = hexv(s[p + 2 + i]);
        if (h < 0) return false;
        v = (v << 4) | static_cast<uint32_t>(h);
    }
    out = static_cast<uint16_t>(v);
    return true;
}
void append_utf8(uint32_t cp, std::string & out) {
    if (cp <= 0x7Fu) out.push_back(static_cast<char>(cp));
    else if (cp <= 0x7FFu) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}
} // namespace

bool normalize_surrogate_pairs(const std::string & input, std::string & output, std::string * error) {
    output.clear();
    auto fail = [&](const char * msg) {
        output.clear();
        if (error) *error = msg;
        return false;
    };
    if (!utf8::is_valid(input)) return fail("JSON contains invalid UTF-8");
    output.reserve(input.size());
    bool in_string = false;
    for (size_t i = 0; i < input.size();) {
        const char c = input[i];
        if (!in_string) {
            output.push_back(c);
            if (c == '"') in_string = true;
            ++i;
            continue;
        }
        if (c == '"') {
            output.push_back(c); in_string = false; ++i; continue;
        }
        if (c != '\\') {
            output.push_back(c); ++i; continue;
        }
        if (i + 1 >= input.size()) return fail("truncated JSON escape");
        const char esc = input[i + 1];
        if (esc != 'u') {
            output.append(input, i, 2);
            i += 2;
            continue;
        }
        uint16_t hi = 0;
        if (!parse_u16(input, i, hi)) return fail("malformed JSON \\u escape");
        if (hi >= 0xD800u && hi <= 0xDBFFu) {
            uint16_t lo = 0;
            if (!parse_u16(input, i + 6, lo) || lo < 0xDC00u || lo > 0xDFFFu)
                return fail("high surrogate is not followed by a low surrogate");
            const uint32_t cp = 0x10000u + ((static_cast<uint32_t>(hi) - 0xD800u) << 10) +
                                (static_cast<uint32_t>(lo) - 0xDC00u);
            append_utf8(cp, output);
            i += 12;
            continue;
        }
        if (hi >= 0xDC00u && hi <= 0xDFFFu)
            return fail("isolated low surrogate in JSON string");
        // Preserve valid BMP escapes verbatim; Crow may decode these normally.
        output.append(input, i, 6);
        i += 6;
    }
    if (in_string) return fail("unterminated JSON string");
    return true;
}

}} // namespace s2::json
