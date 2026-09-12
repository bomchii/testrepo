#include "../include/s2_prompt.h"
#include "../include/s2_text.h"
#include <iostream>
#include <cstring>
#include <limits>

namespace s2 {

// ---------------------------------------------------------------------------
// build_prompt() — matches Python build_prompt_tensor() in ggml_pure.py
// ---------------------------------------------------------------------------
//
// Layout of the returned PromptTensor (rows x cols):
//   Row 0:        token IDs in vocabulary space.
//                 - Text positions: raw BPE token IDs.
//                 - VQ positions:   prompt_codes[0][t] + semantic_begin_id
//   Rows 1..num_cb: codebook values (0 for text positions).
//                 - VQ positions:   prompt_codes[cb][t]  (0-indexed, cb = 0..num_cb-1)
//
// When prompt_codes / prompt_text are absent (no reference audio), a simplified
// prompt is built without the VQ section.
// ---------------------------------------------------------------------------

PromptTensor build_prompt(
    const Tokenizer & tokenizer,
    const std::string & text,
    const std::string & prompt_text,
    const int32_t * prompt_codes,  // (num_codebooks, T_prompt) row-major, or nullptr
    int32_t num_codebooks,
    int32_t T_prompt
) {
    PromptTensor result;
    if (!tokenizer.is_loaded() || num_codebooks <= 0 || num_codebooks > 1024 || T_prompt < 0) {
        return result;
    }
    // A reference is all-or-nothing. Silently ignoring codes because their
    // transcript is empty produces the wrong voice while reporting success.
    const bool any_reference_arg = (prompt_codes != nullptr || T_prompt > 0 || !prompt_text.empty());
    const bool complete_reference = (prompt_codes != nullptr && T_prompt > 0 && !prompt_text.empty());
    if (any_reference_arg && !complete_reference) return result;

    result.rows = num_codebooks + 1;

    const TokenizerConfig & cfg = tokenizer.config();
    const int32_t im_end_id     = cfg.im_end_id;
    const int32_t voice_id      = cfg.voice_id;
    const int32_t sem_begin     = cfg.semantic_begin_id;
    const int32_t codebook_size = cfg.codebook_size;
    if (im_end_id < 0 || voice_id < 0 || sem_begin < 0 || codebook_size <= 0 ||
        cfg.vocab_size <= 0 ||
        static_cast<int64_t>(sem_begin) + codebook_size > cfg.vocab_size) {
        return result;
    }

    // Derive newline from the loaded tokenizer instead of hard-coding Qwen's
    // current ID (198). This keeps prompt construction correct for compatible
    // tokenizer variants and also detects an incomplete tokenizer cleanly.
    const std::vector<int32_t> NEWLINE = tokenizer.encode("\n");
    if (NEWLINE.empty()) return result;

    bool has_reference = (prompt_codes != nullptr && T_prompt > 0 && !prompt_text.empty());
    const bool prompt_has_speaker_tag = contains_valid_speaker_tag(prompt_text);

    std::vector<int32_t> sys_pre;
    std::vector<int32_t> sys_post;

    if (has_reference) {
        // System section: introduce voice reference
        auto app = [&](std::vector<int32_t> & dst, const std::vector<int32_t> & src) {
            dst.insert(dst.end(), src.begin(), src.end());
        };

        app(sys_pre, tokenizer.encode("<|im_start|>system"));
        app(sys_pre, NEWLINE);
        app(sys_pre, tokenizer.encode("convert the provided text to speech reference to the following:\n\nText:\n"));
        if (!prompt_has_speaker_tag) {
            app(sys_pre, tokenizer.encode("<|speaker:0|>"));
        }
        app(sys_pre, tokenizer.encode(prompt_text));
        app(sys_pre, tokenizer.encode("\n\nSpeech:\n"));
        // VQ section goes here (T_prompt frames)

        app(sys_post, { im_end_id });
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode("<|im_start|>user"));
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode(text));
        app(sys_post, { im_end_id });
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode("<|im_start|>assistant"));
        app(sys_post, NEWLINE);
        app(sys_post, { voice_id });
    } else {
        // No reference audio: simpler prompt
        auto app = [&](std::vector<int32_t> & dst, const std::vector<int32_t> & src) {
            dst.insert(dst.end(), src.begin(), src.end());
        };

        // sys_pre is empty, everything goes into sys_post
        app(sys_post, tokenizer.encode("<|im_start|>system"));
        app(sys_post, NEWLINE);
        // Match Fish Speech S2's no-reference system prompt.  Using a generic
        // assistant prompt changes the conditioning distribution and makes
        // random-timbre synthesis less stable than the reference implementation.
        app(sys_post, tokenizer.encode("convert the provided text to speech"));
        app(sys_post, { im_end_id });
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode("<|im_start|>user"));
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode(text));
        app(sys_post, { im_end_id });
        app(sys_post, NEWLINE);
        app(sys_post, tokenizer.encode("<|im_start|>assistant"));
        app(sys_post, NEWLINE);
        app(sys_post, { voice_id });
    }

    // total columns -- calculate in size_t first so malformed/huge inputs cannot
    // overflow int32_t and become a massive allocation.
    const size_t total_size = sys_pre.size() + (has_reference ? static_cast<size_t>(T_prompt) : 0u) + sys_post.size();
    if (total_size == 0 || total_size > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        total_size > std::numeric_limits<size_t>::max() / static_cast<size_t>(result.rows)) {
        return {};
    }
    const int32_t total_len = static_cast<int32_t>(total_size);
    result.cols = total_len;

    // Allocate and zero-fill
    result.data.assign(static_cast<size_t>(result.rows) * total_size, 0);

    int32_t pos = 0;

    // Write sys_pre into row 0 (rows 1..num_cb remain 0)
    for (int32_t i = 0; i < (int32_t)sys_pre.size(); ++i) {
        result.data[0 * total_len + pos + i] = sys_pre[i];
    }
    pos += (int32_t)sys_pre.size();

    // Write VQ section
    if (has_reference && T_prompt > 0) {
        // Validate before touching the output.  This function is public and
        // should not rely solely on VoiceProfile/codec callers to provide IDs
        // suitable for ggml_get_rows().  Use size_t indexing so cb*T_prompt
        // cannot overflow signed int32 arithmetic.
        const size_t prompt_stride = static_cast<size_t>(T_prompt);
        for (int32_t cb = 0; cb < num_codebooks; ++cb) {
            const size_t base = static_cast<size_t>(cb) * prompt_stride;
            for (int32_t t = 0; t < T_prompt; ++t) {
                const int32_t code = prompt_codes[base + static_cast<size_t>(t)];
                if (code < 0 || code >= codebook_size) return {};
            }
        }

        // Row 0: semantic token IDs = prompt_codes[0][t] + semantic_begin_id
        for (int32_t t = 0; t < T_prompt; ++t) {
            result.data[static_cast<size_t>(pos + t)] =
                prompt_codes[static_cast<size_t>(t)] + sem_begin;
        }
        // Rows 1..num_cb: prompt_codes[cb][t] (codebook-space 0-indexed)
        for (int32_t cb = 0; cb < num_codebooks; ++cb) {
            const size_t src_base = static_cast<size_t>(cb) * prompt_stride;
            const size_t dst_base = static_cast<size_t>(cb + 1) * total_size + static_cast<size_t>(pos);
            for (int32_t t = 0; t < T_prompt; ++t) {
                result.data[dst_base + static_cast<size_t>(t)] =
                    prompt_codes[src_base + static_cast<size_t>(t)];
            }
        }
        pos += T_prompt;
    }

    // Write sys_post into row 0
    for (int32_t i = 0; i < (int32_t)sys_post.size(); ++i) {
        result.data[0 * total_len + pos + i] = sys_post[i];
    }
    // rows 1..num_cb at sys_post positions remain 0

    return result;
}


PromptTensor build_prompt_with_history(
    const Tokenizer & tokenizer,
    const std::string & text,
    const std::string & prompt_text,
    const int32_t * prompt_codes,
    int32_t num_codebooks,
    int32_t T_prompt,
    const std::vector<PromptHistoryTurn> & history
) {
    if (history.empty()) {
        return build_prompt(tokenizer, text, prompt_text, prompt_codes,
                            num_codebooks, T_prompt);
    }

    PromptTensor out;
    if (!tokenizer.is_loaded() || num_codebooks <= 0 || num_codebooks > 1024 || T_prompt < 0)
        return out;
    const bool any_ref = prompt_codes != nullptr || T_prompt > 0 || !prompt_text.empty();
    const bool have_ref = prompt_codes != nullptr && T_prompt > 0 && !prompt_text.empty();
    if (any_ref && !have_ref) return out;

    const auto & cfg = tokenizer.config();
    if (cfg.im_end_id < 0 || cfg.voice_id < 0 || cfg.semantic_begin_id < 0 ||
        cfg.codebook_size <= 0 || cfg.vocab_size <= 0 ||
        static_cast<int64_t>(cfg.semantic_begin_id) + cfg.codebook_size > cfg.vocab_size)
        return out;
    const auto newline = tokenizer.encode("\n");
    if (newline.empty()) return out;

    const int32_t rows = num_codebooks + 1;
    std::vector<std::vector<int32_t>> r(static_cast<size_t>(rows));
    auto append_tokens = [&](const std::vector<int32_t> & ids) -> bool {
        if (ids.empty()) return true;
        if (r[0].size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - ids.size()) return false;
        r[0].insert(r[0].end(), ids.begin(), ids.end());
        for (int32_t row = 1; row < rows; ++row) r[static_cast<size_t>(row)].resize(r[0].size(), 0);
        return true;
    };
    auto append_text = [&](const std::string & value) -> bool {
        const auto ids = tokenizer.encode(value);
        if (!value.empty() && ids.empty()) return false;
        return append_tokens(ids);
    };
    auto append_id = [&](int32_t id) -> bool { return append_tokens(std::vector<int32_t>{id}); };
    auto append_vq = [&](const int32_t * codes, int32_t frames) -> bool {
        if (!codes || frames <= 0) return false;
        const size_t old = r[0].size();
        const size_t f = static_cast<size_t>(frames);
        if (old > static_cast<size_t>(std::numeric_limits<int32_t>::max()) - f) return false;
        for (int32_t cb = 0; cb < num_codebooks; ++cb) {
            const size_t base = static_cast<size_t>(cb) * f;
            for (int32_t t = 0; t < frames; ++t) {
                const int32_t code = codes[base + static_cast<size_t>(t)];
                if (code < 0 || code >= cfg.codebook_size) return false;
            }
        }
        r[0].resize(old + f, 0);
        for (int32_t row = 1; row < rows; ++row) r[static_cast<size_t>(row)].resize(old + f, 0);
        for (int32_t t = 0; t < frames; ++t) {
            r[0][old + static_cast<size_t>(t)] = codes[static_cast<size_t>(t)] + cfg.semantic_begin_id;
            for (int32_t cb = 0; cb < num_codebooks; ++cb) {
                r[static_cast<size_t>(cb + 1)][old + static_cast<size_t>(t)] =
                    codes[static_cast<size_t>(cb) * f + static_cast<size_t>(t)];
            }
        }
        return true;
    };
    auto user_turn = [&](const std::string & value) -> bool {
        return append_text("<|im_start|>user") && append_tokens(newline) &&
               append_text(value) && append_id(cfg.im_end_id) && append_tokens(newline);
    };
    auto assistant_prefix = [&]() -> bool {
        return append_text("<|im_start|>assistant") && append_tokens(newline) && append_id(cfg.voice_id);
    };

    if (have_ref) {
        if (!append_text("<|im_start|>system") || !append_tokens(newline) ||
            !append_text("convert the provided text to speech reference to the following:\n\nText:\n")) return {};
        if (!contains_valid_speaker_tag(prompt_text) && !append_text("<|speaker:0|>")) return {};
        if (!append_text(prompt_text) || !append_text("\n\nSpeech:\n") ||
            !append_vq(prompt_codes, T_prompt) || !append_id(cfg.im_end_id) || !append_tokens(newline)) return {};
    } else {
        if (!append_text("<|im_start|>system") || !append_tokens(newline) ||
            !append_text("convert the provided text to speech") || !append_id(cfg.im_end_id) ||
            !append_tokens(newline)) return {};
    }

    for (const auto & turn : history) {
        if (turn.n_frames <= 0 || turn.text.empty()) return {};
        const uint64_t expected = static_cast<uint64_t>(num_codebooks) * static_cast<uint64_t>(turn.n_frames);
        if (expected > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            turn.codes.size() != static_cast<size_t>(expected)) return {};
        if (!user_turn(turn.text) || !assistant_prefix() ||
            !append_vq(turn.codes.data(), turn.n_frames) || !append_id(cfg.im_end_id) || !append_tokens(newline)) return {};
    }
    if (!user_turn(text) || !assistant_prefix()) return {};

    if (r[0].empty() || r[0].size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) return {};
    const size_t cols = r[0].size();
    if (cols > std::numeric_limits<size_t>::max() / static_cast<size_t>(rows)) return {};
    out.rows = rows;
    out.cols = static_cast<int32_t>(cols);
    out.data.resize(static_cast<size_t>(rows) * cols);
    for (int32_t row = 0; row < rows; ++row) {
        if (r[static_cast<size_t>(row)].size() != cols) return {};
        std::copy(r[static_cast<size_t>(row)].begin(), r[static_cast<size_t>(row)].end(),
                  out.data.begin() + static_cast<size_t>(row) * cols);
    }
    return out;
}

} // namespace s2
