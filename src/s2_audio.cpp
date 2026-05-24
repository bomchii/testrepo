// s2_audio.cpp — WAV/MP3 audio I/O
#include "../include/s2_audio.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

// dr_libs implementations (header-only, define once)
#define DR_WAV_IMPLEMENTATION
#include "../third_party/dr_wav.h"

#define DR_MP3_IMPLEMENTATION
#include "../third_party/dr_mp3.h"

namespace s2 {

bool audio_read(const std::string & path, AudioData & out) {
    out.samples.clear();
    out.sample_rate = 0;

    // Try WAV first
    {
        unsigned int channels = 0;
        unsigned int sample_rate = 0;
        drwav_uint64 total_frames = 0;
        float * data = drwav_open_file_and_read_pcm_frames_f32(
            path.c_str(), &channels, &sample_rate, &total_frames, nullptr);
        if (data != nullptr) {
            out.sample_rate = static_cast<int32_t>(sample_rate);
            out.samples.resize(static_cast<size_t>(total_frames));
            if (channels == 1) {
                std::memcpy(out.samples.data(), data, total_frames * sizeof(float));
            } else {
                for (drwav_uint64 i = 0; i < total_frames; ++i) {
                    float sum = 0.0f;
                    for (unsigned int ch = 0; ch < channels; ++ch) {
                        sum += data[i * channels + ch];
                    }
                    out.samples[i] = sum / static_cast<float>(channels);
                }
            }
            drwav_free(data, nullptr);
            return true;
        }
    }

    // Try MP3
    {
        drmp3_config config = {};
        drmp3_uint64 total_frames = 0;
        float * data = drmp3_open_file_and_read_pcm_frames_f32(
            path.c_str(), &config, &total_frames, nullptr);
        if (data != nullptr) {
            out.sample_rate = static_cast<int32_t>(config.sampleRate);
            out.samples.resize(static_cast<size_t>(total_frames));
            if (config.channels == 1) {
                std::memcpy(out.samples.data(), data, total_frames * sizeof(float));
            } else {
                for (drmp3_uint64 i = 0; i < total_frames; ++i) {
                    float sum = 0.0f;
                    for (unsigned int ch = 0; ch < config.channels; ++ch) {
                        sum += data[i * config.channels + ch];
                    }
                    out.samples[i] = sum / static_cast<float>(config.channels);
                }
            }
            drmp3_free(data, nullptr);
            return true;
        }
    }

    std::fprintf(stderr, "[s2_audio] failed to read audio file: %s\n", path.c_str());
    return false;
}

bool audio_read_from_memory(const void * in_data, size_t in_data_size, AudioData & out) {
    out.samples.clear();
    out.sample_rate = 0;

    // Try WAV
    {
        unsigned int channels = 0;
        unsigned int sample_rate = 0;
        drwav_uint64 total_frames = 0;
        float * data = drwav_open_memory_and_read_pcm_frames_f32(
            in_data, in_data_size, &channels, &sample_rate, &total_frames, nullptr);
        if (data != nullptr) {
            out.sample_rate = static_cast<int32_t>(sample_rate);
            out.samples.resize(static_cast<size_t>(total_frames));
            if (channels == 1) {
                std::memcpy(out.samples.data(), data, total_frames * sizeof(float));
            } else {
                for (drwav_uint64 i = 0; i < total_frames; ++i) {
                    float sum = 0.0f;
                    for (unsigned int ch = 0; ch < channels; ++ch) {
                        sum += data[i * channels + ch];
                    }
                    out.samples[i] = sum / static_cast<float>(channels);
                }
            }
            drwav_free(data, nullptr);
            return true;
        }
    }

    // Try MP3
    {
        drmp3_config config = {};
        drmp3_uint64 total_frames = 0;
        float * data = drmp3_open_memory_and_read_pcm_frames_f32(
            in_data, in_data_size, &config, &total_frames, nullptr);
        if (data != nullptr) {
            out.sample_rate = static_cast<int32_t>(config.sampleRate);
            out.samples.resize(static_cast<size_t>(total_frames));
            if (config.channels == 1) {
                std::memcpy(out.samples.data(), data, total_frames * sizeof(float));
            } else {
                for (drmp3_uint64 i = 0; i < total_frames; ++i) {
                    float sum = 0.0f;
                    for (unsigned int ch = 0; ch < config.channels; ++ch) {
                        sum += data[i * config.channels + ch];
                    }
                    out.samples[i] = sum / static_cast<float>(config.channels);
                }
            }
            drmp3_free(data, nullptr);
            return true;
        }
    }

    std::fprintf(stderr, "[s2_audio] failed to decode audio from memory\n");
    return false;
}

bool audio_write_wav(const std::string & path, const float * data, size_t n_samples, int32_t sample_rate) {
    drwav wav;
    drwav_data_format format = {};
    format.container     = drwav_container_riff;
    format.format        = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels      = 1;
    format.sampleRate    = static_cast<drwav_uint32>(sample_rate);
    format.bitsPerSample = 32;

    if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) {
        std::fprintf(stderr, "[s2_audio] failed to open WAV for writing: %s\n", path.c_str());
        return false;
    }

    drwav_uint64 written = drwav_write_pcm_frames(&wav, n_samples, data);
    drwav_uninit(&wav);

    if (written != static_cast<drwav_uint64>(n_samples)) {
        std::fprintf(stderr, "[s2_audio] WAV write incomplete: %llu / %zu frames\n",
                     (unsigned long long)written, n_samples);
        return false;
    }
    return true;
}

std::vector<float> audio_resample(const float * data, size_t n_samples, int32_t src_rate, int32_t dst_rate) {
    if (src_rate == dst_rate || n_samples == 0) {
        return std::vector<float>(data, data + n_samples);
    }

    const double ratio = static_cast<double>(dst_rate) / static_cast<double>(src_rate);
    const size_t out_len = static_cast<size_t>(std::ceil(n_samples * ratio));

    std::vector<float> out(out_len);
    for (size_t i = 0; i < out_len; ++i) {
        const double src_pos = i / ratio;
        const size_t idx = static_cast<size_t>(src_pos);
        const double frac = src_pos - idx;

        if (idx + 1 < n_samples) {
            out[i] = static_cast<float>(data[idx] * (1.0 - frac) + data[idx + 1] * frac);
        } else if (idx < n_samples) {
            out[i] = data[idx];
        } else {
            out[i] = 0.0f;
        }
    }
    return out;
}

std::vector<float> audio_trim_trailing_silence(const float * data, size_t n_samples,
                                               int32_t sample_rate,
                                               float threshold,
                                               float min_silence_duration) {
    if (n_samples == 0) return std::vector<float>();
    if (sample_rate <= 0) sample_rate = 44100;

    const size_t min_silence_samples = static_cast<size_t>(min_silence_duration * sample_rate);
    const size_t keep_tail_samples   = static_cast<size_t>(0.01f * sample_rate);
    size_t last_audio_idx = n_samples;

    for (size_t i = n_samples - 1; i > 0; --i) {
        if (std::fabs(data[i]) > threshold) {
            size_t silence_after = n_samples - 1 - i;
            if (silence_after >= min_silence_samples) {
                last_audio_idx = std::min(n_samples, i + 1 + keep_tail_samples);
            }
            break;
        }
    }

    if (last_audio_idx == n_samples) {
        bool has_audio = false;
        for (size_t i = 0; i < n_samples; ++i) {
            if (std::fabs(data[i]) > threshold) { has_audio = true; break; }
        }
        if (!has_audio) return std::vector<float>();
        return std::vector<float>(data, data + n_samples);
    }

    const size_t min_audio_samples = static_cast<size_t>(0.1f * sample_rate);
    if (last_audio_idx < min_audio_samples) {
        last_audio_idx = std::min(min_audio_samples, n_samples);
    }

    return std::vector<float>(data, data + last_audio_idx);
}

bool load_audio(const std::string & path, AudioData & out, int32_t target_sample_rate) {
    if (!audio_read(path, out)) {
        return false;
    }
    if (target_sample_rate > 0 && out.sample_rate != target_sample_rate) {
        out.samples = audio_resample(out.samples.data(), out.samples.size(), out.sample_rate, target_sample_rate);
        out.sample_rate = target_sample_rate;
    }
    return true;
}

bool load_audio_from_memory(const void * data, size_t bytes, AudioData & out, int32_t target_sample_rate) {
    if (!audio_read_from_memory(data, bytes, out)) {
        return false;
    }
    if (target_sample_rate > 0 && out.sample_rate != target_sample_rate) {
        out.samples = audio_resample(out.samples.data(), out.samples.size(), out.sample_rate, target_sample_rate);
        out.sample_rate = target_sample_rate;
    }
    return true;
}

bool save_audio(const std::string & path, const std::vector<float> & data, int32_t sample_rate,
                bool trim_silence, bool normalize_peak) {
    std::vector<float> output_data = data;

    if (trim_silence && !output_data.empty()) {
        auto trimmed = audio_trim_trailing_silence(output_data.data(), output_data.size(), sample_rate);
        if (!trimmed.empty()) output_data = std::move(trimmed);
    }

    if (normalize_peak && !output_data.empty()) {
        float peak = 0.0f;
        for (float s : output_data) {
            float a = std::fabs(s);
            if (a > peak) peak = a;
        }
        if (peak > 1e-6f) {
            float scale = 0.95f / peak;
            for (float & s : output_data) s *= scale;
        }
    }

    return audio_write_wav(path, output_data.data(), output_data.size(), sample_rate);
}

} // namespace s2
