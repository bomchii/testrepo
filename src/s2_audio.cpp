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
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <unistd.h>
#  include <fcntl.h>
#  include <sys/stat.h>
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

struct WavFileWriter {
    FILE * fp = nullptr;
    bool io_error = false;
};

static size_t wav_file_write_cb(void * user, const void * data, size_t bytes) {
    auto * w = static_cast<WavFileWriter *>(user);
    if (!w || !w->fp || (!data && bytes != 0)) return 0;
    const size_t n = std::fwrite(data, 1, bytes, w->fp);
    if (n != bytes || std::ferror(w->fp)) w->io_error = true;
    return n;
}

static drwav_bool32 wav_file_seek_cb(void * user, int offset, drwav_seek_origin origin) {
    auto * w = static_cast<WavFileWriter *>(user);
    if (!w || !w->fp) return DRWAV_FALSE;
    const int whence = origin == DRWAV_SEEK_SET ? SEEK_SET : SEEK_CUR;
    if (std::fseek(w->fp, offset, whence) != 0) {
        w->io_error = true;
        return DRWAV_FALSE;
    }
    return DRWAV_TRUE;
}

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
        // PCM/ADPCM WAV format chunks have a mandatory 16-byte base header.
        // Reject undersized fmt chunks before dr_wav sees them. Besides being
        // malformed, these have triggered upstream parser/fuzzer findings.
        if (std::memcmp(h, "fmt ", 4) == 0 && chunk_size < 16ull) return false;
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

static FILE * open_audio_output_utf8(const std::string & path) {
#ifdef _WIN32
    std::wstring wpath;
    if (!utf8_to_wide(path, wpath)) return nullptr;
    return _wfopen(wpath.c_str(), L"wb");
#else
    if (path.empty()) return nullptr;
    const int flags = O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC;
    const int fd = ::open(path.c_str(), flags, S_IRUSR | S_IWUSR);
    if (fd < 0) return nullptr;
    FILE * f = ::fdopen(fd, "wb");
    if (!f) { ::close(fd); return nullptr; }
    return f;
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
        if (std::memcmp(chunk, "fmt ", 4) == 0 && chunk_size < 16ull) return false;
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
                             AudioData & out, size_t max_frames = MAX_DECODED_MONO_FRAMES) {
    max_frames = std::min(max_frames, MAX_DECODED_MONO_FRAMES);
    if (!interleaved || channels == 0 || channels > MAX_AUDIO_CHANNELS ||
        out.samples.size() > max_frames || frames > max_frames - out.samples.size()) return false;
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

static size_t decode_frame_limit(unsigned int sample_rate, int32_t max_seconds) noexcept {
    if (max_seconds <= 0) return MAX_DECODED_MONO_FRAMES;
    const uint64_t requested = static_cast<uint64_t>(sample_rate) * static_cast<uint64_t>(max_seconds);
    return static_cast<size_t>(std::min<uint64_t>(requested, MAX_DECODED_MONO_FRAMES));
}

static bool decode_wav_stream(drwav & wav, AudioData & out, int32_t max_seconds = 0) {
    const size_t max_frames = decode_frame_limit(wav.sampleRate, max_seconds);
    if (wav.channels == 0 || wav.channels > MAX_AUDIO_CHANNELS || wav.sampleRate < 1000u ||
        wav.sampleRate > 768000u || wav.totalPCMFrameCount > max_frames) return false;
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
        if (!append_downmixed(chunk.data(), static_cast<size_t>(got), wav.channels, out, max_frames)) return false;
        total += got;
    }
    return total == wav.totalPCMFrameCount;
}

static bool decode_mp3_stream(drmp3 & mp3, AudioData & out, int32_t max_seconds = 0) {
    const size_t max_frames = decode_frame_limit(mp3.sampleRate, max_seconds);
    if (mp3.channels == 0 || mp3.channels > MAX_AUDIO_CHANNELS || mp3.sampleRate < 1000u ||
        mp3.sampleRate > 768000u) return false;
    if (mp3.totalPCMFrameCount != DRMP3_UINT64_MAX && mp3.totalPCMFrameCount > max_frames)
        return false;
    out.sample_rate = static_cast<int32_t>(mp3.sampleRate);
    out.samples.clear();
    if (mp3.totalPCMFrameCount != DRMP3_UINT64_MAX)
        out.samples.reserve(static_cast<size_t>(mp3.totalPCMFrameCount));
    std::vector<float> chunk(AUDIO_DECODE_CHUNK_FRAMES * static_cast<size_t>(mp3.channels));
    for (;;) {
        const drmp3_uint64 got = drmp3_read_pcm_frames_f32(&mp3, AUDIO_DECODE_CHUNK_FRAMES, chunk.data());
        if (got == 0) break;
        if (!append_downmixed(chunk.data(), static_cast<size_t>(got), mp3.channels, out, max_frames)) return false;
    }
    return !out.samples.empty();
}
} // namespace

static bool audio_read_impl(const std::string & path, AudioData & out, int32_t max_seconds) {
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
            const bool ok = decode_wav_stream(wav, out, max_seconds);
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
            const bool ok = decode_mp3_stream(mp3, out, max_seconds);
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

static bool audio_read_from_memory_impl(const void * in_data, size_t in_data_size, AudioData & out, int32_t max_seconds) {
    out.samples.clear();
    out.sample_rate = 0;
    if (!in_data || in_data_size == 0 || in_data_size > MAX_AUDIO_MEMORY_INPUT) return false;
    if (!validate_riff_layout_memory(static_cast<const unsigned char *>(in_data), in_data_size)) return false;
    try {
        drwav wav{};
        if (drwav_init_memory(&wav, in_data, in_data_size, nullptr)) {
            const bool ok = decode_wav_stream(wav, out, max_seconds);
            drwav_uninit(&wav);
            if (!ok) { out = {}; return false; }
            return true;
        }

        drmp3 mp3{};
        if (drmp3_init_memory(&mp3, in_data, in_data_size, nullptr)) {
            const bool ok = decode_mp3_stream(mp3, out, max_seconds);
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

bool audio_read(const std::string & path, AudioData & out) {
    return audio_read_impl(path, out, 0);
}

static bool audio_read_from_memory(const void * data, size_t bytes, AudioData & out) {
    return audio_read_from_memory_impl(data, bytes, out, 0);
}

bool audio_write_wav(const std::string & path, const float * data, size_t n_samples, int32_t sample_rate) {
    if (path.empty() || sample_rate < 1000 || sample_rate > 768000 ||
        (n_samples > 0 && data == nullptr)) return false;
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

    FILE * fp = open_audio_output_utf8(path);
    if (!fp) {
        std::fprintf(stderr, "[s2_audio] failed to open WAV for writing: %s\n", path.c_str());
        return false;
    }

    WavFileWriter writer{fp, false};
    drwav wav{};
    drwav_data_format format{};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 1;
    format.sampleRate = static_cast<drwav_uint32>(sample_rate);
    format.bitsPerSample = 32;

    bool ok = drwav_init_write(&wav, &format, wav_file_write_cb, wav_file_seek_cb,
                               &writer, nullptr) != 0;
    drwav_uint64 written = 0;
    drwav_result finalize_result = DRWAV_SUCCESS;
    if (ok) {
        written = drwav_write_pcm_frames(&wav, static_cast<drwav_uint64>(n_samples), data);
        finalize_result = drwav_uninit(&wav);
        if (written != static_cast<drwav_uint64>(n_samples) || finalize_result != DRWAV_SUCCESS)
            ok = false;
    }
    if (writer.io_error || std::ferror(fp)) ok = false;
    // stdio may buffer a successful fwrite even when the device is full. Both
    // flush and close are part of the write transaction and must be observed.
    if (std::fflush(fp) != 0 || std::ferror(fp)) ok = false;
    if (std::fclose(fp) != 0) ok = false;

    if (!ok) {
        std::fprintf(stderr, "[s2_audio] WAV write/finalize/flush/close failed: %s\n", path.c_str());
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

std::vector<float> audio_time_stretch(const float * data, size_t n_samples, int32_t sample_rate, float speed) {
    if (!data || n_samples == 0 || sample_rate <= 0 || !std::isfinite(speed) ||
        speed < 0.5f || speed > 2.0f) return {};
    if (std::fabs(speed - 1.0f) < 1e-5f) return std::vector<float>(data, data + n_samples);

    // Small WSOLA/SOLA implementation for mono speech.  We choose overlapping
    // analysis windows by normalized correlation, then crossfade them at a
    // fixed synthesis hop.  This changes duration without changing sample rate.
    const size_t win = std::max<size_t>(64, static_cast<size_t>(sample_rate * 0.030));
    const size_t overlap = win / 2;
    const size_t synth_hop = win - overlap;
    const double analysis_hop_f = static_cast<double>(synth_hop) * speed;
    const size_t search = std::max<size_t>(8, static_cast<size_t>(sample_rate * 0.010));
    const size_t expected_out = static_cast<size_t>(std::ceil(static_cast<double>(n_samples) / speed));
    if (expected_out == 0 || expected_out > MAX_DECODED_MONO_FRAMES) return {};
    if (n_samples <= win + search) {
        // Very short clips do not contain enough context for WSOLA matching.
        // Linear duration interpolation is safer than returning nothing.
        std::vector<float> out(expected_out);
        for (size_t i = 0; i < expected_out; ++i) {
            const double pos = std::min<double>(static_cast<double>(n_samples - 1),
                                                static_cast<double>(i) * static_cast<double>(speed));
            const size_t a = static_cast<size_t>(pos);
            const size_t b = std::min(a + 1, n_samples - 1);
            const float f = static_cast<float>(pos - a);
            out[i] = data[a] * (1.0f - f) + data[b] * f;
        }
        return out;
    }

    std::vector<float> out;
    try { out.assign(expected_out + win + search, 0.0f); }
    catch (const std::bad_alloc &) { return {}; }
    std::vector<float> weight(out.size(), 0.0f);

    auto add_window = [&](size_t in_pos, size_t out_pos) {
        const size_t count = std::min(win, n_samples - in_pos);
        for (size_t j = 0; j < count && out_pos + j < out.size(); ++j) {
            // Hann window avoids discontinuities at both edges.
            const float w = 0.5f - 0.5f * std::cos(static_cast<float>(2.0 * 3.14159265358979323846 * j / std::max<size_t>(1, win - 1)));
            out[out_pos + j] += data[in_pos + j] * w;
            weight[out_pos + j] += w;
        }
    };

    size_t in_pos = 0;
    size_t out_pos = 0;
    add_window(0, 0);
    double desired_in = analysis_hop_f;
    out_pos += synth_hop;

    while (out_pos < expected_out && desired_in + win < static_cast<double>(n_samples + search)) {
        const size_t center = static_cast<size_t>(std::llround(desired_in));
        const size_t lo = center > search ? center - search : 0;
        const size_t hi = std::min(n_samples > win ? n_samples - win : 0, center + search);
        size_t best = std::min(center, hi);
        long double best_corr = -2.0L;

        // Compare candidate input overlap against the already synthesized tail.
        for (size_t cand = lo; cand <= hi; ++cand) {
            long double dot = 0.0L, aa = 0.0L, bb = 0.0L;
            for (size_t j = 0; j < overlap && out_pos + j < out.size() && cand + j < n_samples; ++j) {
                const float existing = weight[out_pos + j] > 1e-9f ? out[out_pos + j] / weight[out_pos + j] : 0.0f;
                const float incoming = data[cand + j];
                dot += static_cast<long double>(existing) * incoming;
                aa += static_cast<long double>(existing) * existing;
                bb += static_cast<long double>(incoming) * incoming;
            }
            const long double denom = std::sqrt(std::max<long double>(aa * bb, 1e-24L));
            const long double corr = denom > 0.0L ? dot / denom : 0.0L;
            if (corr > best_corr) { best_corr = corr; best = cand; }
            if (cand == hi) break; // avoid size_t wrap
        }
        in_pos = best;
        add_window(in_pos, out_pos);
        out_pos += synth_hop;
        desired_in += analysis_hop_f;
        if (in_pos + 1 >= n_samples) break;
    }

    out.resize(expected_out);
    weight.resize(expected_out);
    for (size_t i = 0; i < out.size(); ++i) {
        if (weight[i] > 1e-9f) out[i] /= weight[i];
        else {
            const size_t src = std::min(n_samples - 1, static_cast<size_t>(std::min<double>(
                static_cast<double>(n_samples - 1),
                static_cast<double>(i) * static_cast<double>(speed))));
            out[i] = data[src];
        }
        if (!std::isfinite(out[i])) out[i] = 0.0f;
    }
    return out;
}

void audio_normalize_loudness(std::vector<float> & audio, float target_dbfs) {
    if (audio.empty() || !std::isfinite(target_dbfs) || target_dbfs > 0.0f || target_dbfs < -80.0f) return;
    long double sum_sq = 0.0L;
    size_t n = 0;
    for (float x : audio) {
        if (!std::isfinite(x)) continue;
        sum_sq += static_cast<long double>(x) * x;
        ++n;
    }
    if (n == 0) return;
    const long double rms = std::sqrt(sum_sq / static_cast<long double>(n));
    if (!(rms > 1e-9L)) return;
    const long double target = std::pow(10.0L, static_cast<long double>(target_dbfs) / 20.0L);
    long double gain = target / rms;
    long double peak = 0.0L;
    for (float x : audio) if (std::isfinite(x)) peak = std::max(peak, std::fabs(static_cast<long double>(x)));
    if (peak > 0.0L && peak * gain > 0.999L) gain = 0.999L / peak;
    for (float & x : audio) {
        if (!std::isfinite(x)) x = 0.0f;
        else x = static_cast<float>(std::clamp(static_cast<long double>(x) * gain, -0.999L, 0.999L));
    }
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

    const size_t min_audio_samples = static_cast<size_t>(0.1L * sample_rate);
    if (last_audio_idx == n_samples) {
        bool has_audio = false;
        for (size_t i = 0; i < n_samples; ++i) {
            if (std::isfinite(data[i]) && std::fabs(data[i]) > threshold) { has_audio = true; break; }
        }
        if (!has_audio) {
            // An all-silent clip is still a valid clip. Returning {} used to be
            // ambiguous with invalid input, and both callers consequently kept
            // the *entire* silent input when trimming was requested. Keep a
            // short non-empty tail instead so WAV output remains valid while
            // --trim-silence actually does what it says.
            const size_t keep = std::min(min_audio_samples, n_samples);
            return std::vector<float>(data, data + keep);
        }
        return std::vector<float>(data, data + n_samples);
    }

    if (last_audio_idx < min_audio_samples) {
        last_audio_idx = std::min(min_audio_samples, n_samples);
    }

    return std::vector<float>(data, data + last_audio_idx);
}

bool load_audio(const std::string & path, AudioData & out, int32_t target_sample_rate) {
    if (target_sample_rate != 0 && (target_sample_rate < 1000 || target_sample_rate > 768000)) return false;
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
    if (target_sample_rate != 0 && (target_sample_rate < 1000 || target_sample_rate > 768000)) return false;
    if (!audio_read_from_memory(data, bytes, out) || out.sample_rate <= 0 || out.samples.empty()) return false;
    if (target_sample_rate > 0 && out.sample_rate != target_sample_rate) {
        auto resampled = audio_resample(out.samples.data(), out.samples.size(), out.sample_rate, target_sample_rate);
        if (resampled.empty()) return false;
        out.samples = std::move(resampled);
        out.sample_rate = target_sample_rate;
    }
    return true;
}

bool load_audio_limited(const std::string & path, AudioData & out, int32_t target_sample_rate, int32_t max_seconds) {
    if (max_seconds <= 0 || target_sample_rate < 0 || target_sample_rate > 768000 ||
        (target_sample_rate != 0 && target_sample_rate < 1000)) return false;
    if (!audio_read_impl(path, out, max_seconds) || out.sample_rate <= 0 || out.samples.empty()) return false;
    if (target_sample_rate > 0 && out.sample_rate != target_sample_rate) {
        auto resampled = audio_resample(out.samples.data(), out.samples.size(), out.sample_rate, target_sample_rate);
        const uint64_t max_out = static_cast<uint64_t>(target_sample_rate) * static_cast<uint64_t>(max_seconds);
        if (resampled.empty() || static_cast<uint64_t>(resampled.size()) > max_out) return false;
        out.samples = std::move(resampled);
        out.sample_rate = target_sample_rate;
    }
    return true;
}

bool load_audio_from_memory_limited(const void * data, size_t bytes, AudioData & out,
                                    int32_t target_sample_rate, int32_t max_seconds) {
    if (max_seconds <= 0 || target_sample_rate < 0 || target_sample_rate > 768000 ||
        (target_sample_rate != 0 && target_sample_rate < 1000)) return false;
    if (!audio_read_from_memory_impl(data, bytes, out, max_seconds) || out.sample_rate <= 0 || out.samples.empty()) return false;
    if (target_sample_rate > 0 && out.sample_rate != target_sample_rate) {
        auto resampled = audio_resample(out.samples.data(), out.samples.size(), out.sample_rate, target_sample_rate);
        const uint64_t max_out = static_cast<uint64_t>(target_sample_rate) * static_cast<uint64_t>(max_seconds);
        if (resampled.empty() || static_cast<uint64_t>(resampled.size()) > max_out) return false;
        out.samples = std::move(resampled);
        out.sample_rate = target_sample_rate;
    }
    return true;
}

bool save_audio(const std::string & path, const std::vector<float> & data, int32_t sample_rate,
                bool trim_silence, bool normalize_peak) {
    if (path.empty() || sample_rate < 1000 || sample_rate > 768000) return false;
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
