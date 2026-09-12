#include "../include/s2_tokenizer.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static bool decode_hex(const std::string & hex, std::string & out) {
    if ((hex.size() & 1u) != 0) return false;
    out.clear();
    out.reserve(hex.size() / 2);
    for (size_t i = 0; i < hex.size(); i += 2) {
        const int a = hex_value(hex[i]);
        const int b = hex_value(hex[i + 1]);
        if (a < 0 || b < 0) return false;
        out.push_back(static_cast<char>((a << 4) | b));
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_v75_tokenizer <tokenizer.json>\n";
        return 2;
    }
    s2::Tokenizer tokenizer;
    if (!tokenizer.load(argv[1])) return 3;

    std::string line;
    while (std::getline(std::cin, line)) {
        std::string text;
        if (!decode_hex(line, text)) {
            std::cout << "ERR_HEX\n";
            continue;
        }
        const auto ids = tokenizer.encode(text);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << ids[i];
        }
        std::cout << '\n';
    }
    return 0;
}
