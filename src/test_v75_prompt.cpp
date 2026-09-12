#include "../include/s2_prompt.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using s2::PromptHistoryTurn;
using s2::PromptTensor;
using s2::Tokenizer;

static bool equal_tensor(const PromptTensor & a, const PromptTensor & b) {
    return a.rows == b.rows && a.cols == b.cols && a.data == b.data;
}

static int fail(const char * msg) {
    std::cerr << "PROMPT_TEST_FAIL: " << msg << "\n";
    return 1;
}

static bool find_row0_sequence(const PromptTensor & p, const std::vector<int32_t> & seq, size_t * at = nullptr) {
    if (p.rows <= 0 || p.cols <= 0 || seq.empty()) return false;
    const size_t n = static_cast<size_t>(p.cols);
    auto first = p.data.begin();
    auto last = first + static_cast<std::ptrdiff_t>(n);
    auto it = std::search(first, last, seq.begin(), seq.end());
    if (it == last) return false;
    if (at) *at = static_cast<size_t>(it - first);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2) return fail("usage: test_v75_prompt tokenizer.json");
    Tokenizer tok;
    if (!tok.load(argv[1])) return fail("tokenizer load");
    const auto & cfg = tok.config();
    constexpr int32_t C = 10;
    if (cfg.num_codebooks != C) return fail("unexpected codebook count");
    if (cfg.codebook_size < 64 || cfg.semantic_begin_id < 0) return fail("unexpected tokenizer config");

    // Empty history must be exactly V7.4 layout, not merely semantically similar.
    std::vector<int32_t> ref(static_cast<size_t>(C) * 3u);
    for (int32_t cb = 0; cb < C; ++cb)
        for (int32_t t = 0; t < 3; ++t)
            ref[static_cast<size_t>(cb) * 3u + static_cast<size_t>(t)] = 1 + cb * 3 + t;
    const auto old_no_ref = s2::build_prompt(tok, "مرحبا؟你好。नमस्ते।", "", nullptr, C, 0);
    const auto new_no_ref = s2::build_prompt_with_history(tok, "مرحبا؟你好。नमस्ते।", "", nullptr, C, 0, {});
    if (!equal_tensor(old_no_ref, new_no_ref)) return fail("empty-history no-ref changed V7.4 layout");
    const auto old_ref = s2::build_prompt(tok, "тест", "<|speaker:0|>hola", ref.data(), C, 3);
    const auto new_ref = s2::build_prompt_with_history(tok, "тест", "<|speaker:0|>hola", ref.data(), C, 3, {});
    if (!equal_tensor(old_ref, new_ref)) return fail("empty-history reference changed V7.4 layout");

    PromptHistoryTurn h1;
    h1.text = "<|speaker:0|>مرحبا؟";
    h1.n_frames = 4;
    h1.codes.resize(static_cast<size_t>(C) * 4u);
    for (int32_t cb = 0; cb < C; ++cb)
        for (int32_t t = 0; t < 4; ++t)
            h1.codes[static_cast<size_t>(cb) * 4u + static_cast<size_t>(t)] = 10 + cb * 7 + t;

    PromptHistoryTurn h2;
    h2.text = "<|speaker:1|>你好。[whisper in small voice]";
    h2.n_frames = 2;
    h2.codes.resize(static_cast<size_t>(C) * 2u);
    for (int32_t cb = 0; cb < C; ++cb)
        for (int32_t t = 0; t < 2; ++t)
            h2.codes[static_cast<size_t>(cb) * 2u + static_cast<size_t>(t)] = 100 + cb * 5 + t;

    const std::vector<PromptHistoryTurn> hist{h1, h2};
    const auto p = s2::build_prompt_with_history(tok, "<|speaker:0|>नमस्ते।", "", nullptr, C, 0, hist);
    if (p.rows != C + 1 || p.cols <= 0 || p.data.size() != static_cast<size_t>(p.rows) * static_cast<size_t>(p.cols))
        return fail("multi-turn dimensions");

    // All user text remains byte-transparent to the tokenizer path and appears in row 0.
    size_t h1_text_at = 0, h2_text_at = 0, current_text_at = 0;
    if (!find_row0_sequence(p, tok.encode(h1.text), &h1_text_at)) return fail("history text 1 missing");
    if (!find_row0_sequence(p, tok.encode(h2.text), &h2_text_at)) return fail("history text 2 missing");
    if (!find_row0_sequence(p, tok.encode("<|speaker:0|>नमस्ते।"), &current_text_at)) return fail("current text missing");
    if (!(h1_text_at < h2_text_at && h2_text_at < current_text_at)) return fail("history ordering");

    // Locate each VQ block through semantic row 0 and verify every codebook row.
    auto verify_vq = [&](const PromptHistoryTurn & h) -> bool {
        std::vector<int32_t> sem(static_cast<size_t>(h.n_frames));
        for (int32_t t = 0; t < h.n_frames; ++t)
            sem[static_cast<size_t>(t)] = h.codes[static_cast<size_t>(t)] + cfg.semantic_begin_id;
        size_t at = 0;
        if (!find_row0_sequence(p, sem, &at)) return false;
        for (int32_t cb = 0; cb < C; ++cb) {
            const size_t row_base = static_cast<size_t>(cb + 1) * static_cast<size_t>(p.cols);
            for (int32_t t = 0; t < h.n_frames; ++t) {
                const int32_t got = p.data[row_base + at + static_cast<size_t>(t)];
                const int32_t want = h.codes[static_cast<size_t>(cb) * static_cast<size_t>(h.n_frames) + static_cast<size_t>(t)];
                if (got != want) return false;
            }
        }
        return true;
    };
    if (!verify_vq(h1) || !verify_vq(h2)) return fail("history VQ layout mismatch");

    // Only a syntactically valid speaker tag suppresses the default speaker:0 prefix.
    // Fish Speech upstream uses <|speaker:\d+|>, so malformed lookalikes must stay literal.
    const auto speaker0 = tok.encode("<|speaker:0|>");
    const auto malformed_ref = s2::build_prompt(tok, "next", "<|speaker:x|>literal", ref.data(), C, 3);
    if (!find_row0_sequence(malformed_ref, speaker0)) return fail("malformed speaker tag suppressed default reference speaker");
    const auto valid_ref = s2::build_prompt(tok, "next", "<|speaker:7|>literal", ref.data(), C, 3);
    const auto speaker7 = tok.encode("<|speaker:7|>");
    if (!find_row0_sequence(valid_ref, speaker7)) return fail("valid reference speaker tag missing");

    // A reference and history can coexist; the reference VQ must still be accepted.
    const auto both = s2::build_prompt_with_history(tok, "next", "reference transcript", ref.data(), C, 3, {h1});
    if (both.cols <= 0 || both.rows != C + 1) return fail("reference + history rejected");

    // Hostile/malformed history must fail closed, never silently truncate/reshape.
    PromptHistoryTurn bad = h1;
    bad.codes.pop_back();
    if (s2::build_prompt_with_history(tok, "x", "", nullptr, C, 0, {bad}).cols != 0) return fail("bad code shape accepted");
    bad = h1;
    bad.codes[0] = cfg.codebook_size;
    if (s2::build_prompt_with_history(tok, "x", "", nullptr, C, 0, {bad}).cols != 0) return fail("out-of-range code accepted");
    bad = h1;
    bad.n_frames = 0;
    if (s2::build_prompt_with_history(tok, "x", "", nullptr, C, 0, {bad}).cols != 0) return fail("zero-frame history accepted");
    bad = h1;
    bad.text.clear();
    if (s2::build_prompt_with_history(tok, "x", "", nullptr, C, 0, {bad}).cols != 0) return fail("empty history text accepted");

    // Partial external reference remains an error even with history.
    if (s2::build_prompt_with_history(tok, "x", "transcript", nullptr, C, 0, {h1}).cols != 0)
        return fail("partial reference accepted");

    std::cout << "PROMPT_TEST_PASS history=2 multilingual=1\n";
    return 0;
}
