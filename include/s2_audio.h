#pragma once
// s2_audio.h — WAV/MP3 audio I/O for s2.cpp

#include <cstdint>
#include <string>
#include <vector>

namespace s2 {

struct AudioData {
    std::vector<float> samples;   // mono interleaved float32
    int32_t            sample_rate = 0;
};

// Read an audio file (WAV or MP3). Returns mono float32.
bool audio_read(const std::string & path, AudioData & out);

// Write mono float32 audio to WAV file.
bool audio_write_wav(const std::string & path, const float * data, size_t n_samples, int32_t sample_rate);

// Resample mono float32 audio from src_rate to dst_rate (linear interpolation).
std::vector<float> audio_resample(const float * data, size_t n_samples, int32_t src_rate, int32_t dst_rate);

// Trim trailing silence from audio.
std::vector<float> audio_trim_trailing_silence(const float * data, size_t n_samples,
                                               int32_t sample_rate,
                                               float threshold = 0.01f,
                                               float min_silence_duration = 0.08f);

// Helper wrappers used by the pipeline
bool load_audio(const std::string & path, AudioData & out, int32_t target_sample_rate = 0);
bool load_audio_from_memory(const void * data, size_t bytes, AudioData & out, int32_t target_sample_rate = 0);
bool save_audio(const std::string & path, const std::vector<float> & data, int32_t sample_rate,
                bool trim_silence = false, bool normalize_peak = false);

} // namespace s2
