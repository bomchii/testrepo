#pragma once
// s2_prompt.h — Prompt tensor builder for Fish Speech S2

#include "s2_tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace s2 {

struct PromptTensor {
    std::vector<int32_t> data;
    int32_t rows = 0;
    int32_t cols = 0;
};

struct PromptHistoryTurn {
    std::string text;
    std::vector<int32_t> codes; // (num_codebooks, n_frames), row-major
    int32_t n_frames = 0;
};

PromptTensor build_prompt(
    const Tokenizer & tokenizer,
    const std::string & text,
    const std::string & prompt_text,
    const int32_t * prompt_codes,
    int32_t num_codebooks,
    int32_t T_prompt
);

// Multi-turn extension. Empty history is guaranteed to use build_prompt()
// exactly, preserving V7.4 token layout/behavior.
PromptTensor build_prompt_with_history(
    const Tokenizer & tokenizer,
    const std::string & text,
    const std::string & prompt_text,
    const int32_t * prompt_codes,
    int32_t num_codebooks,
    int32_t T_prompt,
    const std::vector<PromptHistoryTurn> & history
);

} // namespace s2
