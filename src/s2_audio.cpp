// s2_audio.cpp — WAV/MP3 audio I/O
#include "../include/s2_audio.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

// dr_libs implementations (header-only, define once)
#define DR_WAV_IMPLEMENTATION
#include "../third_party/dr_wav.h"

#define DR_MP3_IMPLEMENTATION
#include "../third_party/dr_mp3.h"

namespace s2 {

namespace {
constexpr size_t AUDIO_DECODE_CHUNK_FRAMES = 4096;
constexpr size_t MAX_DECODED_MONO_FRAMES = 64u * 1024u * 1024u; // 256 MiB float32 mono
constexpr unsigned int MAX_AUDIO_CHANNELS = 32;
constexpr size_t MAX_AUDIO_MEMORY_INPUT = 512u * 1024u * 1024u;

#ifdef _WIN32
static bool utf8_to_wide(const std::string & s, std::wstring & out) {
    out.clear();
    if (s.empty() || s.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
    const int s_len = static_cast<int>(s.size());
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(),
                                      s_len, nullptr, 0);
    if (n <= 0) return false;
    out.resize(static_cast<size_t>(n));
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(),
                               s_len, out.data(), n) == n;
}
#endif

static uint32_t read_le32_audio(const unsigned char * p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static bool validate_riff_layout_memory(const unsigned char * data, size_t size) {
    if (!data) return true;
    if (size >= 4 && std::memcmp(data, "RIFF", 4) == 0 && size < 12) return false;
    if (size < 12) return true; // not enough to classify as another supported format
    if (std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0)
        return true; // MP3/RF64/W64/etc. are handled by their decoders

    const uint64_t declared_total = 8ull + read_le32_audio(data + 4);
    if (declared_total < 12ull || declared_total > static_cast<uint64_t>(size)) return false;

    uint64_t pos = 12;
    while (pos < declared_total) {
        if (declared_total - pos < 8ull) return false;
        const auto * h = data + static_cast<size_t>(pos);
        const uint64_t chunk_size = read_le32_audio(h + 4);
        uint64_t next = pos + 8ull + chunk_size;
        if (next < pos || next > declared_total) return false;
        if (chunk_size & 1u) {
            if (next == declared_total) return false; // missing RIFF pad byte
            ++next;
        }
        if (next > declared_total) return false;
        pos = next;
    }
    return pos == declared_total;
}

static FILE * open_audio_file_utf8(const std::string & path, const char * mode) {
#ifdef _WIN32
    std::wstring wpath;
    if (!utf8_to_wide(path, wpath)) return nullptr;
    std::wstring wmode;
    while (*mode) wmode.push_back(static_cast<wchar_t>(*mode++));
    return _wfopen(wpath.c_str(), wmode.c_str());
#else
    return std::fopen(path.c_str(), mode);
#endif
}

static bool seek_audio_file(FILE * f, uint64_t offset) {
#ifdef _WIN32
    return offset <= static_cast<uint64_t>(std::numeric_limits<__int64>::max()) &&
           _fseeki64(f, static_cast<__int64>(offset), SEEK_SET) == 0;
#else
    return offset <= static_cast<uint64_t>(std::numeric_limits<off_t>::max()) &&
           fseeko(f, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

static bool get_audio_file_size(FILE * f, uint64_t & size) {
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return false;
    const __int64 n = _ftelli64(f);
    if (n < 0) return false;
    size = static_cast<uint64_t>(n);
#else
    if (fseeko(f, 0, SEEK_END) != 0) return false;
    const off_t n = ftello(f);
    if (n < 0) return false;
    size = static_cast<uint64_t>(n);
#endif
    return true;
}

static bool audio_file_size_within_limit(const std::string & path, uint64_t max_bytes) {
    FILE * f = open_audio_file_utf8(path, "rb");
    if (!f) return true; // let the decoder produce the normal open failure
    struct Guard { FILE * f; ~Guard(){ if (f) std::fclose(f); } } guard{f};
    uint64_t physical = 0;
    return get_audio_file_size(f, physical) && physical <= max_bytes;
}

// dr_wav intentionally tolerates some truncated RIFF files and can reduce its
// visible frame count to the bytes that remain. For reference audio we want a
// damaged file to fail, not silently clone from a partial recording. Validate
// declared RIFF/chunk bounds against the physical file before handing it off.
static bool validate_riff_layout_file(const std::string & path) {
    FILE * f = open_audio_file_utf8(path, "rb");
    if (!f) return true; // decoder will report the actual open error
    struct Guard { FILE * f; ~Guard(){ if (f) std::fclose(f); } } guard{f};

    uint64_t physical = 0;
    if (!get_audio_file_size(f, physical)) return false;
    if (physical < 12) {
        if (physical >= 4) {
            unsigned char magic[4] = {};
            if (!seek_audio_file(f, 0) || std::fread(magic, 1, sizeof(magic), f) != sizeof(magic)) return false;
            if (std::memcmp(magic, "RIFF", 4) == 0) return false;
        }
        return true;
    }
    unsigned char head[12] = {};
    if (!seek_audio_file(f, 0) || std::fread(head, 1, sizeof(head), f) != sizeof(head)) return false;
    if (std::memcmp(head, "RIFF", 4) != 0 || std::memcmp(head + 8, "WAVE", 4) != 0) return true;

    const uint64_t declared_total = 8ull + read_le32_audio(head + 4);
    if (declared_total < 12ull || declared_total > physical) return false;
    uint64_t pos = 12;
    unsigned char chunk[8] = {};
    while (pos < declared_total) {
        if (declared_total - pos < 8ull || !seek_audio_file(f, pos) ||
            std::fread(chunk, 1, sizeof(chunk), f) != sizeof(chunk)) return false;
        const uint64_t chunk_size = read_le32_audio(chunk + 4);
        uint64_t next = pos + 8ull + chunk_size;
        if (next < pos || next > declared_total) return false;
        if (chunk_size & 1u) {
            if (next == declared_total) return false;
            ++next;
        }
        if (next > declared_total) return false;
        pos = next;
    }
    return pos == declared_total;
}

static bool append_downmixed(const float * interleaved, size_t frames, unsigned int channels,
                             AudioData & out) {
    if (!interleaved || channels == 0 || channels > MAX_AUDIO_CHANNELS ||
        frames > MAX_DECODED_MONO_FRAMES - out.samples.size()) return false;
    const size_t old = out.samples.size();
    out.samples.resize(old + frames);
    if (channels == 1) {
        for (size_t i = 0; i < frames; ++i) {
            const float v = interleaved[i];
            if (!std::isfinite(v)) return false;
            out.samples[old + i] = v;
        }
    } else {
        for (size_t i = 0; i < frames; ++i) {
            double sum = 0.0;
            for (unsigned int ch = 0; ch < channels; ++ch) {
                const float v = interleaved[i * channels + ch];
                if (!std::isfinite(v)) return false;
                sum += v;
            }
            const float mono = static_cast<float>(sum / static_cast<double>(channels));
            if (!std::isfinite(mono)) return false;
            out.samples[old + i] = mono;
        }
    }
    return true;
}

static bool decode_wav_stream(drwav & wav, AudioData & out) {
    if (wav.channels == 0 || wav.channels > MAX_AUDIO_CHANNELS || wav.sampleRate < 1000u ||
        wav.sampleRate > 768000u ||
        wav.totalPCMFrameCount > MAX_DECODED_MONO_FRAMES) return false;
    out.sample_rate = static_cast<int32_t>(wav.sampleRate);
    out.samples.clear();
    out.samples.reserve(static_cast<size_t>(wav.totalPCMFrameCount));
    std::vector<float> chunk(AUDIO_DECODE_CHUNK_FRAMES * static_cast<size_t>(wav.channels));
    drwav_uint64 total = 0;
    while (total < wav.totalPCMFrameCount) {
        const drwav_uint64 want = std::min<drwav_uint64>(
            AUDIO_DECODE_CHUNK_FRAMES, wav.totalPCMFrameCount - total);
        const drwav_uint64 got = drwav_read_pcm_frames_f32(&wav, want, chunk.data());
        if (got == 0) break;
        if (!append_downmixed(chunk.data(), static_cast<size_t>(got), wav.channels, out)) return false;
        total += got;
    }
    return total == wav.totalPCMFrameCount;
}

static bool decode_mp3_stream(drmp3 & mp3, AudioData & out) {
    if (mp3.channels == 0 || mp3.channels > MAX_AUDIO_CHANNELS || mp3.sampleRate < 1000u ||
        mp3.sampleRate > 768000u) return false;
    if (mp3.totalPCMFrameCount != DRMP3_UINT64_MAX && mp3.totalPCMFrameCount > MAX_DECODED_MONO_FRAMES)
        return false;
    out.sample_rate = static_cast<int32_t>(mp3.sampleRate);
    out.samples.clear();
    if (mp3.totalPCMFrameCount != DRMP3_UINT64_MAX)
        out.samples.reserve(static_cast<size_t>(mp3.totalPCMFrameCount));
    std::vector<float> chunk(AUDIO_DECODE_CHUNK_FRAMES * static_cast<size_t>(mp3.channels));
    for (;;) {
        const drmp3_uint64 got = drmp3_read_pcm_frames_f32(&mp3, AUDIO_DECODE_CHUNK_FRAMES, chunk.data());
        if (got == 0) break;
        if (!append_downmixed(chunk.data(), static_cast<size_t>(got), mp3.channels, out)) return false;
    }
    return !out.samples.empty();
}
} // namespace

bool audio_read(const std::string & path, AudioData & out) {
    out.samples.clear();
    out.sample_rate = 0;
    if (path.empty()) return false;
    try {
        if (!audio_file_size_within_limit(path, MAX_AUDIO_MEMORY_INPUT)) {
            std::fprintf(stderr, "[s2_audio] audio input exceeds 512 MiB limit: %s\n", path.c_str());
            return false;
        }
        if (!validate_riff_layout_file(path)) {
            std::fprintf(stderr, "[s2_audio] invalid/truncated RIFF/WAV container: %s\n", path.c_str());
            return false;
        }
        // Try WAV first. Decode incrementally so compressed/hostile inputs cannot
        // force a full-file PCM allocation inside dr_wav before our limits run.
        drwav wav{};
        bool wav_open = false;
#ifdef _WIN32
        std::wstring wpath;
        if (!utf8_to_wide(path, wpath)) return false;
        wav_open = drwav_init_file_w(&wav, wpath.c_str(), nullptr) != 0;
#else
        wav_open = drwav_init_file(&wav, path.c_str(), nullptr) != 0;
#endif
        if (wav_open) {
            const bool ok = decode_wav_stream(wav, out);
            drwav_uninit(&wav);
            if (!ok) { out = {}; return false; }
            return true;
        }

        drmp3 mp3{};
        bool mp3_open = false;
#ifdef _WIN32
        mp3_open = drmp3_init_file_w(&mp3, wpath.c_str(), nullptr) != 0;
#else
        mp3_open = drmp3_init_file(&mp3, path.c_str(), nullptr) != 0;
#endif
        if (mp3_open) {
            const bool ok = decode_mp3_stream(mp3, out);
            drmp3_uninit(&mp3);
            if (!ok) { out = {}; return false; }
            return true;
        }
    } catch (const std::bad_alloc &) {
        std::fprintf(stderr, "[s2_audio] audio decode exceeded available memory: %s\n", path.c_str());
        out = {};
        return false;
    }

    std::fprintf(stderr, "[s2_audio] failed to read audio file: %s\n", path.c_str());
    return false;
}

bool audio_read_from_memory(const void * in_data, size_t in_data_size, AudioData & out) {
    out.samples.clear();
    out.sample_rate = 0;
    if (!in_data || in_data_size == 0 || in_data_size > MAX_AUDIO_MEMORY_INPUT) return false;
    if (!validate_riff_layout_memory(static_cast<const unsigned char *>(in_data), in_data_size)) return false;
    try {
        drwav wav{};
        if (drwav_init_memory(&wav, in_data, in_data_size, nullptr)) {
            const bool ok = decode_wav_stream(wav, out);
            drwav_uninit(&wav);
            if (!ok) { out = {}; return false; }
            return true;
        }

        drmp3 mp3{};
        if (drmp3_init_memory(&mp3, in_data, in_data_size, nullptr)) {
            const bool ok = decode_mp3_stream(mp3, out);
            drmp3_uninit(&mp3);
            if (!ok) { out = {}; return false; }
            return true;
        }
    } catch (const std::bad_alloc &) {
        out = {};
        return false;
    }

    std::fprintf(stderr, "[s2_audio] failed to decode audio from memory\n");
    return false;
}

bool audio_write_wav(const std::string & path, const float * data, size_t n_samples, int32_t sample_rate) {
    if (path.empty() || sample_rate <= 0 || (n_samples > 0 && data == nullptr)) return false;
    // Classic RIFF stores chunk sizes in uint32. Refuse an oversized file
    // rather than letting the header wrap. Also do not persist NaN/Inf PCM.
    constexpr uint64_t MAX_RIFF_DATA = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - 36u;
    if (n_samples > MAX_RIFF_DATA / sizeof(float)) {
        std::fprintf(stderr, "[s2_audio] WAV exceeds classic RIFF 4 GiB limit\n");
        return false;
    }
    for (size_t i = 0; i < n_samples; ++i) {
        if (!std::isfinite(data[i])) {
            std::fprintf(stderr, "[s2_audio] refusing to write NaN/Inf sample at %zu\n", i);
            return false;
        }
    }
    drwav wav;
    drwav_data_format format = {};
    format.container     = drwav_container_riff;
    format.format        = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels      = 1;
    format.sampleRate    = static_cast<drwav_uint32>(sample_rate);
    format.bitsPerSample = 32;

    bool write_open = false;
#ifdef _WIN32
    std::wstring wpath;
    if (!utf8_to_wide(path, wpath)) return false;
    write_open = drwav_init_file_write_w(&wav, wpath.c_str(), &format, nullptr) != 0;
#else
    write_open = drwav_init_file_write(&wav, path.c_str(), &format, nullptr) != 0;
#endif
    if (!write_open) {
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
    if (n_samples == 0) return {};
    if (!data || src_rate <= 0 || dst_rate <= 0) return {};
    if (src_rate == dst_rate) return std::vector<float>(data, data + n_samples);

    const double ratio = static_cast<double>(dst_rate) / static_cast<double>(src_rate);
    const long double requested = std::ceil(static_cast<long double>(n_samples) * ratio);
    if (!(ratio > 0.0) || !std::isfinite(ratio) || requested <= 0.0L ||
        requested > static_cast<long double>(std::numeric_limits<size_t>::max())) return {};
    if (requested > static_cast<long double>(MAX_DECODED_MONO_FRAMES)) return {};
    const size_t out_len = static_cast<size_t>(requested);

    std::vector<float> out;
    try { out.resize(out_len); } catch (const std::bad_alloc &) { return {}; }
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
    if (n_samples == 0) return {};
    if (!data || sample_rate <= 0 || !std::isfinite(threshold) || threshold < 0.0f ||
        !std::isfinite(min_silence_duration) || min_silence_duration < 0.0f) return {};

    const long double silence_req = static_cast<long double>(min_silence_duration) * sample_rate;
    if (silence_req > static_cast<long double>(std::numeric_limits<size_t>::max())) return {};
    const size_t min_silence_samples = static_cast<size_t>(silence_req);
    const size_t keep_tail_samples   = static_cast<size_t>(0.01L * sample_rate);
    size_t last_audio_idx = n_samples;

    // Include sample 0 in the backwards scan.  The old `i > 0` loop failed to
    // trim a clip whose only non-silent sample happened to be the first one.
    for (size_t i = n_samples; i-- > 0;) {
        if (std::isfinite(data[i]) && std::fabs(data[i]) > threshold) {
            const size_t silence_after = n_samples - 1 - i;
            if (silence_after >= min_silence_samples) {
                const size_t after_audio = i + 1;
                last_audio_idx = keep_tail_samples > n_samples - after_audio
                    ? n_samples
                    : after_audio + keep_tail_samples;
            }
            break;
        }
    }

    if (last_audio_idx == n_samples) {
        bool has_audio = false;
        for (size_t i = 0; i < n_samples; ++i) {
            if (std::isfinite(data[i]) && std::fabs(data[i]) > threshold) { has_audio = true; break; }
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
    if (target_sample_rate < 0 || target_sample_rate > 768000) return false;
    if (!audio_read(path, out) || out.sample_rate <= 0 || out.samples.empty()) return false;
    if (target_sample_rate > 0 && out.sample_rate != target_sample_rate) {
        auto resampled = audio_resample(out.samples.data(), out.samples.size(), out.sample_rate, target_sample_rate);
        if (resampled.empty()) return false;
        out.samples = std::move(resampled);
        out.sample_rate = target_sample_rate;
    }
    return true;
}

bool load_audio_from_memory(const void * data, size_t bytes, AudioData & out, int32_t target_sample_rate) {
    if (target_sample_rate < 0 || target_sample_rate > 768000) return false;
    if (!audio_read_from_memory(data, bytes, out) || out.sample_rate <= 0 || out.samples.empty()) return false;
    if (target_sample_rate > 0 && out.sample_rate != target_sample_rate) {
        auto resampled = audio_resample(out.samples.data(), out.samples.size(), out.sample_rate, target_sample_rate);
        if (resampled.empty()) return false;
        out.samples = std::move(resampled);
        out.sample_rate = target_sample_rate;
    }
    return true;
}

bool save_audio(const std::string & path, const std::vector<float> & data, int32_t sample_rate,
                bool trim_silence, bool normalize_peak) {
    if (path.empty() || sample_rate <= 0) return false;
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
