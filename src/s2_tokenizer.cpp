// s2_tokenizer.cpp — BPE tokenizer for Qwen3/Fish Speech
#include "../include/s2_tokenizer.h"
#include "../third_party/json.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <sys/types.h>
#endif

using json = nlohmann::json;

namespace s2 {

// ---------------------------------------------------------------------------
// GPT-2 byte-to-unicode table (used by ByteLevel pre-tokenizer in Qwen/tiktoken)
// Maps each byte value (0-255) to a Unicode codepoint.
// Bytes that are printable ASCII or Latin-1 printable map to themselves;
// control bytes map to the range U+0100..U+0143.
// ---------------------------------------------------------------------------
static const uint32_t * byte_to_cp_table() {
    // Function-local static initialization is thread-safe since C++11. Avoid a
    // separate `initialized` flag, which had a data race if Tokenizer::encode()
    // was ever called concurrently outside the server's pipeline mutex.
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        // Bytes that map to themselves: 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF
        for (int b = 0x21; b <= 0x7E; ++b) t[static_cast<size_t>(b)] = static_cast<uint32_t>(b);
        for (int b = 0xA1; b <= 0xAC; ++b) t[static_cast<size_t>(b)] = static_cast<uint32_t>(b);
        for (int b = 0xAE; b <= 0xFF; ++b) t[static_cast<size_t>(b)] = static_cast<uint32_t>(b);
        // Remaining bytes (0x00-0x20, 0x7F-0xA0, 0xAD) map to U+0100..U+0143
        uint32_t extra = 0x0100;
        for (int b = 0x00; b <= 0x20; ++b) t[static_cast<size_t>(b)] = extra++;
        for (int b = 0x7F; b <= 0xA0; ++b) t[static_cast<size_t>(b)] = extra++;
        t[0xAD] = extra;
        return t;
    }();
    return table.data();
}

static std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// Convert a raw UTF-8 string to its byte-level unicode representation
// (each raw byte → one unicode char via the GPT-2 byte-to-unicode table)
static std::string to_byte_level(const std::string & raw) {
    const uint32_t * tbl = byte_to_cp_table();
    std::string result;
    for (unsigned char b : raw) {
        result += cp_to_utf8(tbl[b]);
    }
    return result;
}

// Keep the two BPE symbols unambiguous. Plain concatenation can collide:
// ("ab", "c") and ("a", "bc") would otherwise share the same key.
static std::string merge_key(const std::string & a, const std::string & b) {
    std::string key;
    key.reserve(a.size() + 1 + b.size());
    key.append(a);
    key.push_back('\0');
    key.append(b);
    return key;
}

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------
static std::vector<std::string> utf8_chars(const std::string & text) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = 1;
        unsigned char c = static_cast<unsigned char>(text[i]);
        if      ((c & 0xF8) == 0xF0) len = 4;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xE0) == 0xC0) len = 2;
        if (i + len > text.size()) len = 1;
        chars.push_back(text.substr(i, len));
        i += len;
    }
    return chars;
}

// ---------------------------------------------------------------------------
// Pre-tokenize using the Qwen/Fish tokenizer regex semantics.
//
// tokenizer.json uses Unicode \p{L}/\p{N} categories. Relying on the process
// locale (iswalpha) would make tokenization platform-dependent, so use compact
// generated Unicode category ranges instead.
// ---------------------------------------------------------------------------
#include "s2_unicode_ranges.inc"
#include "s2_unicode_nfc.inc"

// tokenizer.json declares an NFC normalizer.  Special/added tokens in this
// tokenizer are marked normalized=false, so encode() matches them first and
// applies NFC only to the ordinary text spans between them.
static uint8_t nfc_ccc(uint32_t cp) {
    size_t lo = 0, hi = sizeof(kNfcCcc) / sizeof(kNfcCcc[0]);
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (cp < kNfcCcc[mid].cp) hi = mid;
        else if (cp > kNfcCcc[mid].cp) lo = mid + 1;
        else return kNfcCcc[mid].ccc;
    }
    return 0;
}

static const NfcDecompEntry * nfc_decomp(uint32_t cp) {
    size_t lo = 0, hi = sizeof(kNfcDecomp) / sizeof(kNfcDecomp[0]);
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (cp < kNfcDecomp[mid].cp) hi = mid;
        else if (cp > kNfcDecomp[mid].cp) lo = mid + 1;
        else return &kNfcDecomp[mid];
    }
    return nullptr;
}

static uint32_t nfc_compose_pair(uint32_t a, uint32_t b) {
    // Hangul composition is algorithmic (UAX #15).
    constexpr uint32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
    constexpr uint32_t LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount;
    constexpr uint32_t SCount = LCount * NCount;
    if (a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount) {
        return SBase + ((a - LBase) * VCount + (b - VBase)) * TCount;
    }
    if (a >= SBase && a < SBase + SCount && ((a - SBase) % TCount) == 0 &&
        b > TBase && b < TBase + TCount) {
        return a + (b - TBase);
    }

    size_t lo = 0, hi = sizeof(kNfcCompose) / sizeof(kNfcCompose[0]);
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const auto & e = kNfcCompose[mid];
        if (a < e.first || (a == e.first && b < e.second)) hi = mid;
        else if (a > e.first || (a == e.first && b > e.second)) lo = mid + 1;
        else return e.composite;
    }
    return 0;
}

static void nfc_append_utf8(std::string & out, uint32_t cp) {
    // utf8_units() represents an invalid byte as 0x110000 + byte. Preserve it
    // byte-for-byte instead of manufacturing malformed Unicode.
    if (cp > 0x10FFFFu) {
        out.push_back(static_cast<char>(cp - 0x110000u));
    } else if (cp <= 0x7Fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
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


static bool cp_in_ranges(uint32_t cp, const UnicodeRange * ranges, size_t count) {
    size_t lo = 0, hi = count;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (cp < ranges[mid].lo) hi = mid;
        else if (cp > ranges[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

static bool unicode_is_letter(uint32_t cp) {
    return cp_in_ranges(cp, kLetterRanges, sizeof(kLetterRanges) / sizeof(kLetterRanges[0]));
}
static bool unicode_is_number(uint32_t cp) {
    return cp_in_ranges(cp, kNumberRanges, sizeof(kNumberRanges) / sizeof(kNumberRanges[0]));
}
static bool unicode_is_space(uint32_t cp) {
    // Unicode White_Space property used by the regex engine for \s.
    return cp == 0x0009 || cp == 0x000A || cp == 0x000B || cp == 0x000C ||
           cp == 0x000D || cp == 0x0020 || cp == 0x0085 || cp == 0x00A0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

struct Utf8Unit {
    uint32_t cp = 0;
    size_t begin = 0;
    size_t end = 0; // exclusive
};

static std::vector<Utf8Unit> utf8_units(const std::string & text) {
    std::vector<Utf8Unit> out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        const size_t begin = i;
        const unsigned char b0 = static_cast<unsigned char>(text[i]);
        uint32_t cp = b0;
        size_t len = 1;
        if (b0 >= 0xC2 && b0 <= 0xDF && i + 1 < text.size()) {
            const unsigned char b1 = static_cast<unsigned char>(text[i + 1]);
            if ((b1 & 0xC0) == 0x80) {
                cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F); len = 2;
            }
        } else if (b0 >= 0xE0 && b0 <= 0xEF && i + 2 < text.size()) {
            const unsigned char b1 = static_cast<unsigned char>(text[i + 1]);
            const unsigned char b2 = static_cast<unsigned char>(text[i + 2]);
            const bool cont = (b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80;
            const bool scalar = !(b0 == 0xE0 && b1 < 0xA0) && !(b0 == 0xED && b1 >= 0xA0);
            if (cont && scalar) {
                cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F); len = 3;
            }
        } else if (b0 >= 0xF0 && b0 <= 0xF4 && i + 3 < text.size()) {
            const unsigned char b1 = static_cast<unsigned char>(text[i + 1]);
            const unsigned char b2 = static_cast<unsigned char>(text[i + 2]);
            const unsigned char b3 = static_cast<unsigned char>(text[i + 3]);
            const bool cont = (b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80;
            const bool scalar = !(b0 == 0xF0 && b1 < 0x90) && !(b0 == 0xF4 && b1 >= 0x90);
            if (cont && scalar) {
                cp = ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) |
                     ((b2 & 0x3F) << 6) | (b3 & 0x3F); len = 4;
            }
        }
        // Invalid UTF-8 bytes are deliberately treated as non-letter/non-number
        // single units. ByteLevel BPE can still encode their original byte value.
        if (len == 1 && b0 >= 0x80) cp = 0x110000u + b0;
        i += len;
        out.push_back({cp, begin, i});
    }
    return out;
}

static std::string normalize_nfc(const std::string & text) {
    if (text.empty()) return text;

    constexpr uint32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
    constexpr uint32_t VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = 11172;
    std::vector<uint32_t> cps;
    cps.reserve(text.size());

    // Canonical decomposition (the generated table stores full NFD mappings).
    for (const Utf8Unit & u : utf8_units(text)) {
        const uint32_t cp = u.cp;
        if (cp >= SBase && cp < SBase + SCount) {
            const uint32_t sidx = cp - SBase;
            cps.push_back(LBase + sidx / NCount);
            cps.push_back(VBase + (sidx % NCount) / TCount);
            const uint32_t t = sidx % TCount;
            if (t != 0) cps.push_back(TBase + t);
        } else if (const NfcDecompEntry * d = nfc_decomp(cp)) {
            for (uint8_t i = 0; i < d->len; ++i) cps.push_back(kNfcDecompData[d->offset + i]);
        } else {
            cps.push_back(cp);
        }
    }

    // Canonical ordering. Stable insertion is cheap because combining runs are
    // normally tiny, and it correctly handles marks crossing input-codepoint boundaries.
    for (size_t i = 1; i < cps.size(); ++i) {
        const uint8_t cc = nfc_ccc(cps[i]);
        if (cc == 0) continue;
        size_t j = i;
        while (j > 0) {
            const uint8_t prev = nfc_ccc(cps[j - 1]);
            if (prev == 0 || prev <= cc) break;
            std::swap(cps[j], cps[j - 1]);
            --j;
        }
    }

    // Canonical composition.
    if (!cps.empty()) {
        size_t starter_pos = 0;
        uint32_t starter = cps[0];
        uint8_t last_cc = 0;
        size_t i = 1;
        while (i < cps.size()) {
            const uint32_t ch = cps[i];
            const uint8_t cc = nfc_ccc(ch);
            const uint32_t composite = nfc_compose_pair(starter, ch);
            if (composite != 0 && (last_cc == 0 || last_cc < cc)) {
                cps[starter_pos] = composite;
                starter = composite;
                cps.erase(cps.begin() + static_cast<std::ptrdiff_t>(i));
                continue;
            }
            if (cc == 0) {
                starter_pos = i;
                starter = ch;
            }
            last_cc = cc;
            ++i;
        }
    }

    std::string out;
    out.reserve(text.size());
    for (uint32_t cp : cps) nfc_append_utf8(out, cp);
    return out;
}


static bool ascii_ci_equal(uint32_t cp, char c) {
    if (cp >= 'A' && cp <= 'Z') cp += ('a' - 'A');
    return cp == static_cast<uint32_t>(static_cast<unsigned char>(c));
}

static size_t contraction_len(const std::vector<Utf8Unit> & u, size_t i) {
    if (i >= u.size() || u[i].cp != '\'') return 0;
    static const char * suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
    for (const char * suffix : suffixes) {
        size_t j = i + 1;
        const char * p = suffix;
        while (*p && j < u.size() && ascii_ci_equal(u[j].cp, *p)) { ++j; ++p; }
        if (*p == '\0') return j - i;
    }
    return 0;
}

static std::vector<std::string> pre_tokenize(const std::string & text) {
    std::vector<std::string> pieces;
    if (text.empty()) return pieces;
    const auto u = utf8_units(text);

    auto emit = [&](size_t a, size_t b) {
        if (a >= b || b > u.size()) return;
        pieces.push_back(text.substr(u[a].begin, u[b - 1].end - u[a].begin));
    };
    auto is_l = [&](size_t i) { return unicode_is_letter(u[i].cp); };
    auto is_n = [&](size_t i) { return unicode_is_number(u[i].cp); };
    auto is_ws = [&](size_t i) { return unicode_is_space(u[i].cp); };
    auto is_crlf = [&](size_t i) { return u[i].cp == '\r' || u[i].cp == '\n'; };
    auto is_other = [&](size_t i) { return !is_ws(i) && !is_l(i) && !is_n(i); };

    size_t i = 0;
    while (i < u.size()) {
        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (const size_t n = contraction_len(u, i)) {
            emit(i, i + n); i += n; continue;
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if (is_l(i)) {
            size_t j = i + 1; while (j < u.size() && is_l(j)) ++j;
            emit(i, j); i = j; continue;
        }
        if (!is_crlf(i) && !is_l(i) && !is_n(i) && i + 1 < u.size() && is_l(i + 1)) {
            size_t j = i + 2; while (j < u.size() && is_l(j)) ++j;
            emit(i, j); i = j; continue;
        }

        // \p{N} (intentionally one codepoint at a time)
        if (is_n(i)) { emit(i, i + 1); ++i; continue; }

        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        if ((u[i].cp == ' ' && i + 1 < u.size() && is_other(i + 1)) || is_other(i)) {
            size_t j = i;
            if (u[j].cp == ' ') ++j;
            while (j < u.size() && is_other(j)) ++j;
            while (j < u.size() && is_crlf(j)) ++j;
            emit(i, j); i = j; continue;
        }

        if (is_ws(i)) {
            size_t run_end = i;
            size_t last_crlf = std::string::npos;
            while (run_end < u.size() && is_ws(run_end)) {
                if (is_crlf(run_end)) last_crlf = run_end;
                ++run_end;
            }

            // \s*[\r\n]+ -- greedily end at the final newline in this ws run.
            if (last_crlf != std::string::npos) {
                emit(i, last_crlf + 1); i = last_crlf + 1; continue;
            }

            // \s+(?!\S) -- at EOF consume all trailing whitespace; before a
            // non-space consume all but the final ws if there are at least two.
            if (run_end == u.size()) {
                emit(i, run_end); i = run_end; continue;
            }
            if (run_end - i >= 2) {
                emit(i, run_end - 1); i = run_end - 1; continue;
            }

            // \s+
            emit(i, run_end); i = run_end; continue;
        }

        // Should only be reachable for an unusual invalid scalar; preserve it.
        emit(i, i + 1); ++i;
    }
    return pieces;
}

// ---------------------------------------------------------------------------
// BPE merge on a single word
// ---------------------------------------------------------------------------
std::vector<int32_t> Tokenizer::bpe_encode_word(const std::string & word) const {
    if (word.empty()) return {};

    // Apply GPT-2 byte-to-unicode mapping so vocab lookups work correctly.
    // Each raw byte in the word becomes one Unicode char in the byte-level string.
    const std::string bl = to_byte_level(word);

    // Check if the entire byte-level word is a single vocab entry
    auto it = vocab_.find(bl);
    if (it != vocab_.end()) {
        return {it->second};
    }

    // Split byte-level string into individual unicode chars (one per raw byte)
    std::vector<std::string> symbols = utf8_chars(bl);

    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_pos = std::string::npos;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string pair = merge_key(symbols[i], symbols[i + 1]);
            auto it2 = merge_rank_.find(pair);
            if (it2 != merge_rank_.end() && it2->second < best_rank) {
                best_rank = it2->second;
                best_pos = i;
            }
        }

        if (best_pos == std::string::npos) break;

        symbols[best_pos] += symbols[best_pos + 1];
        symbols.erase(symbols.begin() + static_cast<long>(best_pos) + 1);
    }

    std::vector<int32_t> ids;
    for (const auto & sym : symbols) {
        auto it2 = vocab_.find(sym);
        if (it2 != vocab_.end()) {
            ids.push_back(it2->second);
        }
    }
    return ids;
}

// ---------------------------------------------------------------------------
// Implementacion interna compartida por load() y load_from_memory()
// ---------------------------------------------------------------------------
bool Tokenizer::load_from_memory(const char * data, size_t size) {
    constexpr size_t MAX_TOKENIZER_JSON_BYTES = 256u * 1024u * 1024u;
    // A failed/repeated load must never leave a mixture of the previous and new
    // tokenizer state. Build into this clean state and leave loaded_=false on error.
    loaded_ = false;
    config_ = TokenizerConfig{};
    vocab_.clear();
    id_to_token_.clear();
    merges_.clear();
    merge_rank_.clear();
    special_tokens_.clear();
    special_trie_.clear();

    if (data == nullptr || size == 0) {
        std::fprintf(stderr, "[s2_tokenizer] empty tokenizer data\n");
        return false;
    }
    if (size > MAX_TOKENIZER_JSON_BYTES) {
        std::fprintf(stderr, "[s2_tokenizer] tokenizer JSON exceeds 256 MiB safety limit\n");
        return false;
    }

    json j;
    try {
        j = json::parse(data, data + size);

        if (!j.is_object() || !j.contains("model") || !j["model"].is_object() ||
            !j["model"].contains("vocab") || !j["model"]["vocab"].is_object()) {
            throw std::runtime_error("tokenizer.json is missing model.vocab");
        }

    // Load Hugging Face added tokens (special and non-special are atomic)
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto & tok : j["added_tokens"]) {
            std::string content = tok.value("content", "");
            int32_t id = tok.value("id", -1);
            if (!content.empty() && id >= 0) {
                // Hugging Face AddedTokens are atomic even when `special` is
                // false.  This tokenizer has 12 such markers (<think>,
                // <tool_call>, FIM markers, ...), all with normalized=false and
                // no lstrip/rstrip/single_word behavior.  Ignoring them here
                // makes their literal text go through BPE and changes IDs.
                if (tok.value("normalized", false) ||
                    tok.value("single_word", false) ||
                    tok.value("lstrip", false) || tok.value("rstrip", false)) {
                    throw std::runtime_error("unsupported AddedToken normalization/strip semantics");
                }
                vocab_[content] = id;
                id_to_token_[id] = content;
                special_tokens_.push_back({content, id});
            }
        }
    }

    // Load model vocab
    if (j.contains("model") && j["model"].contains("vocab")) {
        for (auto it2 = j["model"]["vocab"].begin(); it2 != j["model"]["vocab"].end(); ++it2) {
            int32_t id = it2.value().get<int32_t>();
            const std::string & key = it2.key();
            if (vocab_.find(key) == vocab_.end()) {
                vocab_[key] = id;
            }
            if (id_to_token_.find(id) == id_to_token_.end()) {
                id_to_token_[id] = key;
            }
        }
    }

    // Load merges
    if (j.contains("model") && j["model"].contains("merges")) {
        int32_t rank = 0;
        for (const auto & merge_item : j["model"]["merges"]) {
            if (merge_item.is_string()) {
                std::string m = merge_item.get<std::string>();
                size_t sp = m.find(' ');
                if (sp != std::string::npos) {
                    std::string a = m.substr(0, sp);
                    std::string b = m.substr(sp + 1);
                    merges_.push_back({a, b});
                    merge_rank_[merge_key(a, b)] = rank++;
                }
            } else if (merge_item.is_array() && merge_item.size() >= 2) {
                std::string a = merge_item[0].get<std::string>();
                std::string b = merge_item[1].get<std::string>();
                merges_.push_back({a, b});
                merge_rank_[merge_key(a, b)] = rank++;
            }
        }
    }

    // Set config from known special token IDs. Some exported tokenizers keep
    // these tokens in model.vocab but forget `special: true` in added_tokens.
    // They still need atomic matching or the literal marker can be BPE-split.
    config_.im_start_id       = token_to_id("<|im_start|>");
    config_.im_end_id         = token_to_id("<|im_end|>");
    config_.voice_id          = token_to_id("<|voice|>");
    config_.pad_id            = token_to_id("<|pad|>");
    config_.eos_id            = token_to_id("<|im_end|>");

        if (vocab_.empty() || config_.im_start_id < 0 || config_.im_end_id < 0 || config_.voice_id < 0) {
            throw std::runtime_error("tokenizer is missing required Fish Speech special tokens");
        }
        // ByteLevel BPE relies on a token for every raw byte. Without this,
        // bpe_encode_word() can silently drop bytes that are absent from a
        // malformed/incompatible vocabulary.
        const uint32_t * byte_map = byte_to_cp_table();
        for (int b = 0; b < 256; ++b) {
            if (vocab_.find(cp_to_utf8(byte_map[b])) == vocab_.end()) {
                throw std::runtime_error("tokenizer ByteLevel vocabulary is incomplete");
            }
        }

        auto ensure_special = [&](const char * token, int32_t id) {
            auto it = std::find_if(special_tokens_.begin(), special_tokens_.end(),
                                   [&](const auto & entry) { return entry.first == token; });
            if (it == special_tokens_.end()) special_tokens_.push_back({token, id});
        };
        ensure_special("<|im_start|>", config_.im_start_id);
        ensure_special("<|im_end|>",   config_.im_end_id);
        ensure_special("<|voice|>",    config_.voice_id);

        // Longer special markers must win when one is a prefix of another.
        std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const std::pair<std::string, int32_t> & a, const std::pair<std::string, int32_t> & b) {
                return a.first.size() > b.first.size();
            });

        special_trie_.clear();
        special_trie_.emplace_back(); // root
        for (const auto & sp : special_tokens_) {
            size_t node = 0;
            for (unsigned char byte : sp.first) {
                auto it = special_trie_[node].next.find(byte);
                if (it == special_trie_[node].next.end()) {
                    const size_t child = special_trie_.size();
                    special_trie_[node].next.emplace(byte, child);
                    special_trie_.emplace_back();
                    node = child;
                } else {
                    node = it->second;
                }
            }
            special_trie_[node].id = sp.second;
        }

        loaded_ = true;
        return true;
    } catch (const std::exception & e) {
        // Clear partial state so callers cannot accidentally use half a tokenizer.
        loaded_ = false;
        vocab_.clear();
        id_to_token_.clear();
        merges_.clear();
        merge_rank_.clear();
        special_tokens_.clear();
        special_trie_.clear();
        std::fprintf(stderr, "[s2_tokenizer] tokenizer load error: %s\n", e.what());
        return false;
    }
}

// ---------------------------------------------------------------------------
// Load tokenizer.json desde disco — wrapper de load_from_memory()
// ---------------------------------------------------------------------------
bool Tokenizer::load(const std::string & path) {
    constexpr uint64_t MAX_TOKENIZER_JSON_BYTES = 256ull * 1024ull * 1024ull;
    if (path.empty()) return false;

    FILE * f = nullptr;
#ifdef _WIN32
    if (path.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
    const int path_len = static_cast<int>(path.size());
    const int wn = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                       path.data(), path_len,
                                       nullptr, 0);
    if (wn > 0) {
        std::wstring wpath(static_cast<size_t>(wn), L'\0');
        if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                path.data(), path_len,
                                wpath.data(), wn) == wn) {
            f = _wfopen(wpath.c_str(), L"rb");
        }
    }
#else
    f = std::fopen(path.c_str(), "rb");
#endif
    if (!f) {
        std::fprintf(stderr, "[s2_tokenizer] failed to open: %s\n", path.c_str());
        return false;
    }
    struct FileGuard { FILE * f; ~FileGuard(){ if (f) std::fclose(f); } } guard{f};

    uint64_t size = 0;
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return false;
    const __int64 end_pos = _ftelli64(f);
    if (end_pos <= 0) return false;
    size = static_cast<uint64_t>(end_pos);
    if (_fseeki64(f, 0, SEEK_SET) != 0) return false;
#else
    if (fseeko(f, 0, SEEK_END) != 0) return false;
    const off_t end_pos = ftello(f);
    if (end_pos <= 0) return false;
    size = static_cast<uint64_t>(end_pos);
    if (fseeko(f, 0, SEEK_SET) != 0) return false;
#endif
    if (size > MAX_TOKENIZER_JSON_BYTES ||
        size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        std::fprintf(stderr, "[s2_tokenizer] tokenizer file is empty/unreadable or exceeds 256 MiB: %s\n", path.c_str());
        return false;
    }

    std::string data;
    try {
        data.resize(static_cast<size_t>(size));
    } catch (const std::bad_alloc &) {
        std::fprintf(stderr, "[s2_tokenizer] insufficient memory for tokenizer: %s\n", path.c_str());
        return false;
    }
    if (!data.empty() && std::fread(data.data(), 1, data.size(), f) != data.size()) {
        std::fprintf(stderr, "[s2_tokenizer] failed to read complete tokenizer: %s\n", path.c_str());
        return false;
    }
    if (std::ferror(f)) return false;
    return load_from_memory(data.data(), data.size());
}

// ---------------------------------------------------------------------------
// Encode text to token IDs
// ---------------------------------------------------------------------------
std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    if (text.empty() || !loaded_) return {};

    std::vector<int32_t> ids;
    auto encode_plain = [&](size_t begin, size_t end) {
        if (end <= begin) return;
        const std::string plain = normalize_nfc(text.substr(begin, end - begin));
        std::vector<std::string> words = pre_tokenize(plain);
        for (const auto & w : words) {
            std::vector<int32_t> w_ids = bpe_encode_word(w);
            ids.insert(ids.end(), w_ids.begin(), w_ids.end());
        }
    };

    // Longest-match traversal through the special-token trie. This is O(text *
    // max_special_length), not O(text * number_of_special_tokens).
    size_t plain_begin = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        int32_t matched_id = -1;
        size_t matched_len = 0;
        if (!special_trie_.empty()) {
            size_t node = 0;
            for (size_t j = pos; j < text.size(); ++j) {
                const unsigned char byte = static_cast<unsigned char>(text[j]);
                auto it = special_trie_[node].next.find(byte);
                if (it == special_trie_[node].next.end()) break;
                node = it->second;
                if (special_trie_[node].id >= 0) {
                    matched_id = special_trie_[node].id;
                    matched_len = j - pos + 1;
                }
            }
        }

        if (matched_id >= 0) {
            encode_plain(plain_begin, pos);
            ids.push_back(matched_id);
            pos += matched_len;
            plain_begin = pos;
        } else {
            ++pos;
        }
    }
    encode_plain(plain_begin, text.size());
    return ids;
}

// ---------------------------------------------------------------------------
// Get Special Token ID
// ---------------------------------------------------------------------------
int32_t Tokenizer::token_to_id(const std::string & token) const {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) {
        return it->second;
    }
    return -1; // Not found
}

} // namespace s2
