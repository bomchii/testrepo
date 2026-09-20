#include "../include/s2_audio.h"

#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    // Keep the target focused on parser/decoder robustness rather than unbounded
    // corpus expansion. Runtime reference paths enforce 30 seconds too.
    if (size > (2u << 20)) return 0;
    s2::AudioData audio;
    (void)s2::load_audio_from_memory_limited(data, size, audio, 44100, 30);
    return 0;
}
