#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace s2 {
// Conservative request text normalization for TTS. Preserves UTF-8 content,
// collapses whitespace, and trims leading/trailing whitespace.
std::string normalize_tts_text(const std::string & text);

struct TextSegment {
    std::string text;
    int32_t speaker_id = -1;
    bool has_speaker = false;
};

// Unicode-transparent sentence segmentation. Invalid UTF-8 returns {}.
// Valid <|speaker:N|> tags are repeated on every segment belonging to that
// speaker; malformed speaker tags remain literal text.
std::vector<TextSegment> split_text_segments(const std::string & text,
                                             int32_t min_chars = 0);
// Long-form chunking layered on top of sentence segmentation. target_chars is
// measured in visible Unicode code points (Extend/Format do not inflate it).
// Chunks preserve speaker tags and never split inside balanced [ ... ] blocks,
// UTF-8 code points, combining sequences, or ZWJ/ZWNJ formatting runs.
std::vector<TextSegment> split_text_chunks(const std::string & text,
                                           int32_t target_chars,
                                           int32_t min_chunk_chars = 0);
// True only for a syntactically valid <|speaker:N|> tag with N in int32 range.
// Malformed lookalikes must not change prompt-conditioning semantics.
bool contains_valid_speaker_tag(const std::string & text) noexcept;
size_t distinct_speaker_count(const std::vector<TextSegment> & segments);
} // namespace s2
