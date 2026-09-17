#include "s2_pipeline.h"
#include "s2_json.h"
#include "s2_utf8.h"
#if defined(GGML_USE_HIP)
#  include <hip/hip_runtime.h>
#elif defined(GGML_USE_CUDA)
#  include <cuda_runtime.h>
#endif
#ifndef CROW_ENFORCE_WS_SPEC
#define CROW_ENFORCE_WS_SPEC
#endif
#include <crow.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <mutex>
#include <limits>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <cctype>
#include <atomic>
#include <memory>
#include <unordered_map>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <ws2tcpip.h>
#elif defined(__APPLE__)
#  include <mach-o/dyld.h>
#  include <limits.h>
#  include <arpa/inet.h>
#else
#  include <unistd.h>
#  include <limits.h>
#  include <arpa/inet.h>
#endif

// tokenizer_data.h es generado por el workflow antes de compilar.
// Contiene: extern const char   tokenizer_json_data[];
//           extern const size_t tokenizer_json_size;
// Si no existe (build local sin el workflow), se usa el archivo en disco.
#if __has_include("tokenizer_data.h")
#  include "tokenizer_data.h"
#  define S2_TOKENIZER_EMBEDDED 1
#endif

// Devuelve el directorio donde vive el ejecutable, con separador final.
// Ej: "C:\\Users\\you\\s2\\"  o  "/home/you/s2/"
static std::string get_exe_dir() {
#ifdef _WIN32
    wchar_t buf[32768] = {};
    const DWORD cap = static_cast<DWORD>(std::size(buf));
    DWORD len = GetModuleFileNameW(nullptr, buf, cap);
    if (len == 0 || len >= cap) return "";
    // Convertir UTF-16 -> UTF-8 sin asumir que la conversion siempre cabe.
    const int utf8_len = WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(len), nullptr, 0, nullptr, nullptr);
    if (utf8_len <= 0) return "";
    std::string path(static_cast<size_t>(utf8_len), '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(len), path.data(), utf8_len, nullptr, nullptr) != utf8_len)
        return "";
    // Recortar hasta el ultimo separador
    auto sep = path.find_last_of("/\\");
    return (sep == std::string::npos) ? "" : path.substr(0, sep + 1);
#elif defined(__APPLE__)
    char stack_buf[PATH_MAX] = {};
    uint32_t size = sizeof(stack_buf);
    std::string path;
    if (_NSGetExecutablePath(stack_buf, &size) == 0) {
        path.assign(stack_buf);
    } else {
        std::vector<char> dyn(static_cast<size_t>(size) + 1u, '\0');
        uint32_t dyn_size = static_cast<uint32_t>(dyn.size());
        if (_NSGetExecutablePath(dyn.data(), &dyn_size) != 0) return "";
        path.assign(dyn.data());
    }
    auto sep = path.rfind('/');
    return (sep == std::string::npos) ? "" : path.substr(0, sep + 1);
#else
    char buf[PATH_MAX] = {};
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0 || static_cast<size_t>(len) >= sizeof(buf) - 1) return "";
    std::string path(buf, static_cast<size_t>(len));
    auto sep = path.rfind('/');
    return (sep == std::string::npos) ? "" : path.substr(0, sep + 1);
#endif
}

static int parse_int_arg(const char * raw) {
    if (!raw) throw std::invalid_argument("missing numeric value");
    const std::string value(raw);
    size_t pos = 0;
    const int parsed = std::stoi(value, &pos, 10);
    if (pos != value.size()) throw std::invalid_argument("trailing characters in integer: " + value);
    return parsed;
}

static uint64_t parse_u64_arg(const char * raw) {
    if (!raw) throw std::invalid_argument("missing unsigned integer value");
    const std::string value(raw);
    if (value.empty() || value[0] == '-') throw std::invalid_argument("expected uint64: " + value);
    size_t pos = 0;
    const unsigned long long parsed = std::stoull(value, &pos, 10);
    if (pos != value.size()) throw std::invalid_argument("trailing characters in uint64: " + value);
    return static_cast<uint64_t>(parsed);
}

#ifdef _WIN32
static bool utf8_to_wide_path(const std::string & value, std::wstring & out) {
    out.clear();
    if (value.empty() || value.size() > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
    const int value_len = static_cast<int>(value.size());
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                      value.data(), value_len,
                                      nullptr, 0);
    if (n <= 0) return false;
    out.resize(static_cast<size_t>(n));
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                               value.data(), value_len,
                               out.data(), n) == n;
}
#endif

static bool is_ip_address_literal(const std::string & value) {
    if (value.empty()) return false;
    unsigned char storage[16] = {};
#ifdef _WIN32
    return InetPtonA(AF_INET, value.c_str(), storage) == 1 ||
           InetPtonA(AF_INET6, value.c_str(), storage) == 1;
#else
    return inet_pton(AF_INET, value.c_str(), storage) == 1 ||
           inet_pton(AF_INET6, value.c_str(), storage) == 1;
#endif
}

static float parse_float_arg(const char * raw) {
    if (!raw) throw std::invalid_argument("missing numeric value");
    const std::string value(raw);
    size_t pos = 0;
    const float parsed = std::stof(value, &pos);
    if (pos != value.size()) throw std::invalid_argument("trailing characters in number: " + value);
    return parsed;
}

static bool json_is_integer(const crow::json::rvalue & value) {
    if (value.t() != crow::json::type::Number) return false;
    const auto nt = value.nt();
    return nt == crow::json::num_type::Signed_integer ||
           nt == crow::json::num_type::Unsigned_integer;
}

static int32_t checked_json_i32(const crow::json::rvalue & value) {
    if (!json_is_integer(value)) throw std::invalid_argument("JSON field must be an integer number");
    if (value.nt() == crow::json::num_type::Unsigned_integer) {
        const uint64_t v = value.u();
        if (v > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
            throw std::out_of_range("JSON integer is outside int32 range");
        return static_cast<int32_t>(v);
    }
    const int64_t v = value.i();
    if (v < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
        v > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
        throw std::out_of_range("JSON integer is outside int32 range");
    return static_cast<int32_t>(v);
}

static uint64_t checked_json_u64(const crow::json::rvalue & value) {
    if (!json_is_integer(value)) throw std::invalid_argument("JSON field must be an integer number");
    if (value.nt() == crow::json::num_type::Unsigned_integer) return value.u();
    const int64_t v = value.i();
    if (v < 0) throw std::out_of_range("JSON uint64 must be non-negative");
    return static_cast<uint64_t>(v);
}

static uint64_t checked_json_fish_seed(const crow::json::rvalue & value) {
    // Fish Speech serializes an unspecified seed as null; s2.cpp uses 0 for random.
    if (value.t() == crow::json::type::Null) return 0;
    return checked_json_u64(value);
}

static int32_t checked_json_fish_max_new_tokens(const crow::json::rvalue & value) {
    const int32_t v = checked_json_i32(value);
    // Fish Speech/WebUI defines 0 as no explicit limit. Keep a finite defensive
    // ceiling here; Pipeline clamps this again to the model context before decode.
    return v == 0 ? 32768 : v;
}

static void validate_fish_json_subset(const crow::json::rvalue & json, bool http_buffered) {
    // Known Fish fields that materially change generation/output but do not yet
    // have an equivalent implementation here must fail explicitly. Silently
    // accepting them is worse than a clear subset error because the client may
    // believe it requested different prosody/chunk continuity/normalization.
    static constexpr const char * unsupported_semantic_fields[] = {
        "early_stop_threshold",
        "normalize",
        "sample_rate",
        "mp3_bitrate",
        "opus_bitrate",
        "use_memory_cache",
    };
    for (const char * field : unsupported_semantic_fields) {
        if (json.has(field)) {
            throw std::invalid_argument(
                std::string("Fish field '") + field +
                "' is not supported by this API subset and must not be ignored");
        }
    }

    if (json.has("references")) {
        const auto & refs = json["references"];
        if (refs.t() != crow::json::type::List)
            throw std::invalid_argument("Fish field 'references' must be an array");
        if (refs.size() != 0)
            throw std::invalid_argument(
                "inline Fish 'references' are not supported by this JSON API; "
                "use reference_audio+prompt_text or reference_id/voice");
    }
    if (json.has("streaming")) {
        const auto & streaming = json["streaming"];
        if (streaming.t() != crow::json::type::True &&
            streaming.t() != crow::json::type::False)
            throw std::invalid_argument("Fish field 'streaming' must be a boolean");
        if (http_buffered && streaming.b())
            throw std::invalid_argument(
                "HTTP streaming=true is not supported; use /ws/tts for streaming audio");
        if (!http_buffered && !streaming.b())
            throw std::invalid_argument(
                "streaming=false is incompatible with /ws/tts; use an HTTP synthesis endpoint");
    }

    // Route-specific fields must not be accepted as silent no-ops. HTTP returns
    // one buffered WAV/PCM response; WebSocket always streams framed PCM.
    if (http_buffered && json.has("stream_stride"))
        throw std::invalid_argument(
            "stream_stride is WebSocket-only; use /ws/tts or --stream-decode-stride as its server default");
    if (!http_buffered && (json.has("format") || json.has("response_format")))
        throw std::invalid_argument(
            "format/response_format are HTTP-only; /ws/tts always streams PCM");
}

static crow::json::rvalue load_json_strict(const std::string & raw) {
    std::string normalized, error;
    if (!s2::json::normalize_surrogate_pairs(raw, normalized, &error))
        throw std::invalid_argument(error);
    return crow::json::load(normalized);
}

constexpr size_t MAX_JSON_REQUEST_BYTES = 8u * 1024u * 1024u;

struct ScopedTempPath {
    std::string path;
    ~ScopedTempPath() {
        if (path.empty()) return;
#ifdef _WIN32
        // Destructors are implicitly noexcept. UTF-8 -> UTF-16 conversion may
        // allocate, so cleanup must absorb allocation/conversion failures
        // instead of terminating the server while unwinding another error.
        try {
            std::wstring wpath;
            if (utf8_to_wide_path(path, wpath)) DeleteFileW(wpath.c_str());
        } catch (...) {
        }
#else
        std::remove(path.c_str());
#endif
    }
};

static uint16_t read_le_u16(const unsigned char * p) {
    return static_cast<uint16_t>(p[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8);
}

static uint32_t read_le_u32(const unsigned char * p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

// synthesize_to_file() creates one exact 44-byte PCM WAV layout. Validate it
// before returning bytes to a client so a truncated temp file cannot look like
// a successful empty/partial response.
static bool read_generated_wav(const std::string & path, bool pcm_only, std::string & out) {
    out.clear();
    std::ifstream f;
#ifdef _WIN32
    std::wstring wpath;
    if (!utf8_to_wide_path(path, wpath)) return false;
    f.open(wpath.c_str(), std::ios::binary);
#else
    f.open(path, std::ios::binary);
#endif
    if (!f) return false;

    unsigned char header[44] = {};
    if (!f.read(reinterpret_cast<char *>(header), sizeof(header))) return false;
    if (std::memcmp(header + 0, "RIFF", 4) != 0 ||
        std::memcmp(header + 8, "WAVE", 4) != 0 ||
        std::memcmp(header + 12, "fmt ", 4) != 0 ||
        std::memcmp(header + 36, "data", 4) != 0) return false;

    const uint32_t fmt_size     = read_le_u32(header + 16);
    const uint16_t audio_format = read_le_u16(header + 20);
    const uint16_t channels     = read_le_u16(header + 22);
    const uint32_t sample_rate  = read_le_u32(header + 24);
    const uint32_t byte_rate    = read_le_u32(header + 28);
    const uint16_t block_align  = read_le_u16(header + 32);
    const uint16_t bits         = read_le_u16(header + 34);
    const uint32_t riff_size    = read_le_u32(header + 4);
    const uint32_t data_size    = read_le_u32(header + 40);

    // synthesize_to_file() always emits canonical mono signed-16 PCM.  Validate
    // the complete format contract before stripping the 44-byte header for the
    // raw-PCM API. This also catches a corrupted header that merely has the
    // expected RIFF/data magic bytes.
    if (fmt_size != 16u || audio_format != 1u || channels != 1u ||
        sample_rate == 0u || sample_rate > std::numeric_limits<uint32_t>::max() / 2u ||
        bits != 16u || block_align != 2u ||
        byte_rate != sample_rate * 2u || (data_size % block_align) != 0u) return false;
    if (data_size > std::numeric_limits<uint32_t>::max() - 36u ||
        riff_size != 36u + data_size) return false;
    const uint64_t expected64 = 44ull + static_cast<uint64_t>(data_size);
    if (expected64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        expected64 > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) return false;

    f.seekg(0, std::ios::end);
    const std::streamoff actual = f.tellg();
    if (actual < 0 || static_cast<uint64_t>(actual) != expected64) return false;

    if (pcm_only) {
        out.resize(static_cast<size_t>(data_size));
        f.seekg(44, std::ios::beg);
        if (data_size > 0 && !f.read(out.data(), static_cast<std::streamsize>(data_size))) return false;
    } else {
        out.resize(static_cast<size_t>(expected64));
        f.seekg(0, std::ios::beg);
        if (!out.empty() && !f.read(out.data(), static_cast<std::streamsize>(out.size()))) return false;
    }
    return true;
}

static bool validate_pipeline_params(const s2::PipelineParams & p,
                                     std::string & error,
                                     bool require_text = false) {
    auto fail = [&](const std::string & msg) { error = msg; return false; };
    if (!s2::utf8::is_valid(p.text)) return fail("text must be valid UTF-8");
    if (!s2::utf8::is_valid(p.prompt_text)) return fail("prompt_text must be valid UTF-8");
    if (require_text && !s2::utf8::has_non_whitespace(p.text)) return fail("text must not be empty or whitespace-only");
    if (!p.prompt_audio_path.empty() && !s2::utf8::has_non_whitespace(p.prompt_text))
        return fail("reference_audio/--prompt-audio requires a non-empty prompt_text/--prompt-text");
    if (p.text.size() > 1024u * 1024u) return fail("text exceeds 1 MiB request limit");
    if (p.prompt_text.size() > 1024u * 1024u) return fail("prompt_text exceeds 1 MiB request limit");
    if (p.prompt_audio_path.size() > 32768u) return fail("reference audio path is too long");
    if (p.voice_id.size() > 128u) return fail("voice id is too long");
    if (!p.voice_id.empty()) {
        for (unsigned char c : p.voice_id) {
            const bool ascii_alnum = (c >= 'a' && c <= 'z') ||
                                     (c >= 'A' && c <= 'Z') ||
                                     (c >= '0' && c <= '9');
            if (!(ascii_alnum || c == '_' || c == '-'))
                return fail("voice id may contain only ASCII letters, digits, '_' and '-'");
        }
    }
    if (p.gen.n_threads < 1 || p.gen.n_threads > 256) return fail("threads must be between 1 and 256");
    if (p.gen.max_new_tokens < 1 || p.gen.max_new_tokens > 32768) return fail("max_tokens must be between 1 and 32768");
    if (p.max_tokens_per_segment < 1 || p.max_tokens_per_segment > 32768) return fail("max_seg_tokens must be between 1 and 32768");
    if (!std::isfinite(p.gen.temperature) || p.gen.temperature < 0.0f || p.gen.temperature > 10.0f)
        return fail("temperature must be finite and between 0 and 10");
    if (!std::isfinite(p.gen.top_p) || p.gen.top_p <= 0.0f || p.gen.top_p > 1.0f)
        return fail("top_p must be in (0, 1]");
    if (p.gen.top_k < 0 || p.gen.top_k > 1000000) return fail("top_k must be between 0 and 1000000");
    if (p.gen.min_tokens_before_end < 0 || p.gen.min_tokens_before_end > 32768)
        return fail("min_end_tokens must be between 0 and 32768");
    if (p.gen.min_tokens_before_end >= p.gen.max_new_tokens)
        return fail("min_end_tokens must be smaller than max_tokens");
    if (p.gen.repetition_penalty < 1.0f || !std::isfinite(p.gen.repetition_penalty) || p.gen.repetition_penalty > 10.0f)
        return fail("repetition_penalty must be finite and between 1.0 and 10.0");
    if (p.gen.repetition_window < 0 || p.gen.repetition_window > 32768)
        return fail("repetition_window must be between 0 and 32768");
    if (p.multi_turn_history < 0 || p.multi_turn_history > 1024)
        return fail("multi_turn_history must be between 0 and 1024");
    if (p.gen.ras_window_size < 0 || p.gen.ras_window_size > 32768)
        return fail("ras_window must be between 0 and 32768");
    if (!std::isfinite(p.gen.ras_high_temp) || p.gen.ras_high_temp < 0.0f || p.gen.ras_high_temp > 10.0f)
        return fail("ras_temp must be finite and between 0 and 10");
    if (!std::isfinite(p.gen.ras_high_top_p) || p.gen.ras_high_top_p <= 0.0f || p.gen.ras_high_top_p > 1.0f)
        return fail("ras_top_p must be in (0, 1]");
    if (p.codec_chunk_frames < 0) return fail("codec_chunk must be >= 0");
    if (p.codec_overlap_frames < 0) return fail("codec_overlap must be >= 0");
    if (p.min_seg_chars < 0 || p.min_seg_chars > 1000000) return fail("min_seg_chars must be between 0 and 1000000");
    if (p.chunk_length != 0 && (p.chunk_length < 100 || p.chunk_length > 300))
        return fail("chunk_length must be 0 (off) or between 100 and 300");
    if (p.min_chunk_length < 0 || p.min_chunk_length > 100)
        return fail("min_chunk_length must be between 0 and 100");
    if (p.chunk_length == 0 && p.min_chunk_length != 0)
        return fail("min_chunk_length requires chunk_length");
    if (!std::isfinite(p.prosody_volume_db) || p.prosody_volume_db < -20.0f || p.prosody_volume_db > 20.0f)
        return fail("prosody volume must be finite and between -20 and 20 dB");
    if (p.stream_decode_stride_frames < -1 || p.stream_decode_stride_frames > 32768)
        return fail("stream_stride must be -1 (disabled), 0 (auto), or 1..32768");
    if (p.vulkan_device < -1) return fail("model device must be -1 (CPU) or >= 0");
    if (p.codec_vulkan_device < -2) return fail("codec device must be -2 (inherit), -1 (CPU), or >= 0");
    return true;
}

int main(int argc, char** argv) {
    const std::string exe_dir = get_exe_dir();

    s2::PipelineParams params;
    // -----------------------------------------------------------------------
    // Defaults conservadores -- funcionan en cualquier PC sin GPU Vulkan.
    // El usuario activa GPU explicitamente con -v / --codec-vulkan.
    // -----------------------------------------------------------------------
    params.model_path           = exe_dir + "model.gguf";
    params.codec_model_path     = exe_dir + "codec.gguf";
    params.tokenizer_path       = exe_dir + "tokenizer.json";
    params.vulkan_device        = -1;   // CPU por defecto (seguro en cualquier PC)
    params.codec_vulkan_device  = -2;   // heredar device del modelo; -1 fuerza CPU
    params.gen.n_threads        = 4;
    params.gen.max_new_tokens   = 1024;
    params.gen.temperature      = 0.7f;
    params.gen.top_p            = 0.7f;
    params.gen.top_k            = 30;
    params.segment_sentences    = false; // OFF por defecto -- el usuario activa con --segment
    params.codec_chunk_frames   = 0;     // 0 = automatico (se calcula en runtime)
    params.codec_overlap_frames  = 0;     // 0 = historial automatico del codec
    params.min_seg_chars         = 0;     // 0 = sin filtro de longitud minima
    params.chunk_length          = 0;     // 0 = long-form Fish chunking OFF
    params.min_chunk_length      = 0;
    params.condition_on_previous_chunks = true;
    params.prosody_volume_db     = 0.0f;
    // GenerateParams defaults (tambien en s2_generate.h)
    params.gen.temperature       = 0.7f;
    params.gen.top_p             = 0.7f;
    params.gen.top_k             = 30;
    params.gen.min_tokens_before_end = 64;
    params.gen.ras_window_size   = 10;
    params.gen.ras_high_temp     = 1.0f;
    params.gen.ras_high_top_p    = 0.9f;
    params.gen.seed              = 0;
    params.gen.repetition_penalty = 1.0f;
    params.gen.repetition_window = 64;
    params.multi_turn_history     = 0;
    params.warmup                 = false;
    params.base_dir             = exe_dir;
    params.output_path          = "";     // vacio = usar archivo temporal Crow
    params.trim_silence         = false;
    params.voice_storage_dir    = exe_dir + "voices";
    params.stream_decode_stride_frames = 0;

    int port = 8080;
    // Crow defaults to 0.0.0.0. This API accepts local filesystem paths for
    // reference audio, so bind to loopback unless the user explicitly opts in.
    std::string bind_host = "127.0.0.1";
    bool list_voices = false;

    // --- Parse des arguments ---
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        // Strict numeric helpers reject non-numeric input and trailing garbage; std::stoi/std::stof
        // still provide range checking for oversized values. Guard every iteration so a
        // bad value (e.g. "-v abc", "--port 9e99", or "--port --segment") yields a
        // clean error + exit code 1 instead of an uncaught exception -> std::terminate()
        // -> process abort. (Critical Pattern #2 — was missing from the parse loop.)
        try {
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (arg == "--model-codec" && i + 1 < argc) {
            params.codec_model_path = argv[++i];
        } else if ((arg == "-t" || arg == "--tokenizer") && i + 1 < argc) {
            params.tokenizer_path = argv[++i];
        } else if ((arg == "-v" || arg == "--vulkan") && i + 1 < argc) {
            params.vulkan_device = parse_int_arg(argv[++i]);
        } else if (arg == "--codec-vulkan" && i + 1 < argc) {
            params.codec_vulkan_device = parse_int_arg(argv[++i]);
        } else if (arg == "--segment") {
            params.segment_sentences = true;
        } else if (arg == "--codec-chunk" && i + 1 < argc) {
            params.codec_chunk_frames = parse_int_arg(argv[++i]);
        } else if (arg == "--codec-overlap" && i + 1 < argc) {
            params.codec_overlap_frames = parse_int_arg(argv[++i]);
        } else if (arg == "--min-seg-chars" && i + 1 < argc) {
            params.min_seg_chars = parse_int_arg(argv[++i]);
        } else if (arg == "--chunk-length" && i + 1 < argc) {
            params.chunk_length = parse_int_arg(argv[++i]);
        } else if (arg == "--min-chunk-length" && i + 1 < argc) {
            params.min_chunk_length = parse_int_arg(argv[++i]);
        } else if (arg == "--condition-on-previous-chunks") {
            params.condition_on_previous_chunks = true;
        } else if (arg == "--no-condition-on-previous-chunks") {
            params.condition_on_previous_chunks = false;
        } else if (arg == "--prosody-volume" && i + 1 < argc) {
            params.prosody_volume_db = parse_float_arg(argv[++i]);
        } else if ((arg == "--temperature" || arg == "--temp") && i + 1 < argc) {
            params.gen.temperature = parse_float_arg(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            params.gen.top_p = parse_float_arg(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            params.gen.top_k = parse_int_arg(argv[++i]);
        } else if (arg == "--min-end-tokens" && i + 1 < argc) {
            params.gen.min_tokens_before_end = parse_int_arg(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            params.gen.seed = parse_u64_arg(argv[++i]);
        } else if (arg == "--repetition-penalty" && i + 1 < argc) {
            params.gen.repetition_penalty = parse_float_arg(argv[++i]);
        } else if (arg == "--repetition-window" && i + 1 < argc) {
            params.gen.repetition_window = parse_int_arg(argv[++i]);
        } else if (arg == "--multi-turn-history" && i + 1 < argc) {
            params.multi_turn_history = parse_int_arg(argv[++i]);
        } else if (arg == "--warmup") {
            params.warmup = true;
        } else if (arg == "--ras-window" && i + 1 < argc) {
            params.gen.ras_window_size = parse_int_arg(argv[++i]);
        } else if (arg == "--ras-temp" && i + 1 < argc) {
            params.gen.ras_high_temp = parse_float_arg(argv[++i]);
        } else if (arg == "--ras-top-p" && i + 1 < argc) {
            params.gen.ras_high_top_p = parse_float_arg(argv[++i]);
        } else if (arg == "--max-seg-tokens" && i + 1 < argc) {
            params.max_tokens_per_segment = parse_int_arg(argv[++i]);
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = parse_int_arg(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            bind_host = argv[++i];
        } else if ((arg == "-threads" || arg == "--threads") && i + 1 < argc) {
            params.gen.n_threads = parse_int_arg(argv[++i]);
        } else if ((arg == "--max-tokens") && i + 1 < argc) {
            params.gen.max_new_tokens = parse_int_arg(argv[++i]);
        } else if ((arg == "--text") && i + 1 < argc) {
            params.text = argv[++i];
        } else if ((arg == "-pa" || arg == "--prompt-audio") && i + 1 < argc) {
            params.prompt_audio_path = argv[++i];
        } else if ((arg == "-pt" || arg == "--prompt-text") && i + 1 < argc) {
            params.prompt_text = argv[++i];
        } else if (arg == "--voice" && i + 1 < argc) {
            params.voice_id = argv[++i];
        } else if (arg == "--save-voice") {
            params.save_voice = true;
        } else if (arg == "--voice-dir" && i + 1 < argc) {
            params.voice_storage_dir = argv[++i];
        } else if (arg == "--list-voices") {
            list_voices = true;
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            params.output_path = argv[++i];
        } else if (arg == "--trim-silence") {
            params.trim_silence = true;
        } else if (arg == "--no-trim-silence") {
            params.trim_silence = false;
        } else if (arg == "--stream-decode-stride" && i + 1 < argc) {
            params.stream_decode_stride_frames = parse_int_arg(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout <<
R"S2HELP(s2 -- Fish Speech TTS server + CLI (CPU / Vulkan / CUDA / AMD / Metal)
Local Fish Speech synthesis, voice cloning, saved .s2voice profiles, HTTP API,
and WebSocket PCM streaming. The same CLI is used by every backend-specific build.

USAGE:
  s2 [options]                         Start the HTTP/WebSocket server.
  s2 [options] --output out.wav        Synthesize once to WAV and exit.
  s2 [options] --save-voice ...        Encode/save a .s2voice profile.
  s2 [options] --list-voices           List saved profiles and exit.
  s2 --help                            Show this help and exit.

  Release filenames:
    Windows: s2-windows-cpu-x86-64.exe, s2-windows-vulkan-x86-64.exe,
             s2-windows-cuda-x86-64.exe, s2-windows-amd-x86-64.exe
    Linux:   s2-linux-cpu-x86-64, s2-linux-vulkan-x86-64,
             s2-linux-cuda-x86-64, s2-linux-amd-x86-64
    macOS:   s2-macos-metal-arm64, s2-macos-cpu-x86-64

QUICK START - DOWNLOADED RELEASE:
  There is no --server flag. Server mode is the default when --output is absent.
  Replace the model filenames below with the files you downloaded.

  Windows CPU:
    s2-windows-cpu-x86-64.exe --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v -1 --port 8080
  Windows Vulkan, GPU 0:
    s2-windows-vulkan-x86-64.exe --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --codec-vulkan 0 --port 8080
  Windows CUDA, GPU 0:
    s2-windows-cuda-x86-64.exe --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --port 8080
  Windows AMD, GPU 0:
    s2-windows-amd-x86-64.exe --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --port 8080
  Linux CPU:
    ./s2-linux-cpu-x86-64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v -1 --port 8080
  Linux Vulkan, GPU 0:
    ./s2-linux-vulkan-x86-64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --codec-vulkan 0 --port 8080
  Linux CUDA, GPU 0:
    ./s2-linux-cuda-x86-64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --port 8080
  Linux AMD, GPU 0:
    ./s2-linux-amd-x86-64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --port 8080
  macOS Intel CPU:
    ./s2-macos-cpu-x86-64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v -1 --port 8080
  macOS Metal (Apple Silicon/arm64):
    ./s2-macos-metal-arm64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --port 8080

  All ten release targets listen on http://127.0.0.1:8080 by default. Check the server with:
    curl http://127.0.0.1:8080/v1/health
  On Windows PowerShell, use curl.exe instead of curl if curl is an alias.
  If one combined GGUF contains both parts, pass that same file to --model and
  --model-codec. The codec follows the transformer device by default.

  Packaged runtime launchers:
    s2-windows-cuda-x86-64.exe --runtime-info   Show embedded CUDA runtime/cache details.
    s2-windows-cuda-x86-64.exe --clean-runtime  Remove inactive extracted runtime caches.
    s2-windows-amd-x86-64.exe --runtime-info    Show embedded AMD runtime/cache details.
    s2-linux-amd-x86-64 --runtime-info           Show embedded AMD runtime/cache details.

COMMAND MODES / FUNCTIONS:
  Server mode (default)
    Mode precedence is: --help exits immediately; --list-voices exits before
    model loading; --save-voice saves and exits unless --output is also present;
    --output synthesizes once and exits. Otherwise s2 starts Crow HTTP/WebSocket.
    Generation, voice, segmentation, sampling and codec options become server
    defaults and supported request JSON fields can override them per request.

  One-shot synthesis
    --output <path> writes one WAV file and exits without starting the server.
    Input comes from --text; if --text is omitted, UTF-8 text is read from stdin.

  Save a voice profile
    --save-voice requires --voice <id>, --prompt-audio <path> and
    --prompt-text <text>. It encodes the reference and saves <id>.s2voice.
    With no --output it exits after saving. With --output it saves first, then
    performs the requested one-shot synthesis.

  List voice profiles
    --list-voices lists IDs under --voice-dir and exits without loading models.

MORE EXAMPLES:
  Server on a different port (still localhost-only):
    s2 --model model.gguf --model-codec codec.gguf --port 8081

  One-shot synthesis:
    s2 --model model.gguf --model-codec codec.gguf \
       --text "Hello world." --output hello.wav

  One-shot synthesis from stdin:
    echo "Hello from stdin." | s2 --model model.gguf --model-codec codec.gguf \
       --output hello.wav

  Clone directly from reference audio:
    s2 --model model.gguf --model-codec codec.gguf \
       --prompt-audio ref.wav --prompt-text "Exact reference transcript." \
       --text "Hello in the reference voice." --output cloned.wav

  Save and reuse a voice profile:
    s2 --model model.gguf --model-codec codec.gguf \
       --prompt-audio ref.wav --prompt-text "Exact reference transcript." \
       --voice narrator --save-voice
    s2 --model model.gguf --model-codec codec.gguf \
       --voice narrator --text "Hello again." --output reused.wav

ALL COMMAND-LINE OPTIONS:

  Models / tokenizer:
    -m, --model <path>            Transformer/full GGUF path.
                                  Default: model.gguf next to executable.
        --model-codec <path>      Codec/full GGUF path.
                                  Default: codec.gguf next to executable.
    -t, --tokenizer <path>        tokenizer.json path for non-embedded builds.
                                  Release builds normally embed the tokenizer;
                                  otherwise default is tokenizer.json next to exe.

  Backend / device selection:
    -v, --vulkan <N>              Transformer device selector (legacy name).
                                  -1 = CPU (default), 0+ = compiled GPU device.
                                  Vulkan: Vulkan index; CUDA: CUDA index;
                                  Metal: any 0+ request is normalized to device 0.
                                  CPU-only builds ignore GPU requests with warning.
        --codec-vulkan <N>        Codec device selector (legacy name).
                                  -2 = inherit transformer device (default)
                                  -1 = force CPU
                                   0+ = compiled GPU device (Metal -> device 0)

  Server:
    -p, --port <N>                Listen port. Default: 8080. Range: 1..65535.
        --host <IP>               Bind IPv4/IPv6 address literal only.
                                  Default: 127.0.0.1. Max text length: 64 chars.
                                  Use 0.0.0.0 only for intentional LAN exposure.

  One-shot input/output:
        --text <text>             Input for --output mode. Max: 1 MiB.
                                  If omitted with --output, text is read from stdin.
    -o, --output <path>           Write one WAV file and exit; no server is started.
        --trim-silence            Trim only final trailing silence.
        --no-trim-silence         Disable trailing-silence trim (default; also
                                  overrides an earlier --trim-silence).

  Reference audio / voice cloning:
    -pa, --prompt-audio <path>    Reference WAV or MP3 path. Max path: 32768 bytes.
                                  Requires a non-empty --prompt-text. Decoded
                                  reference duration is limited to 30 seconds.
    -pt, --prompt-text <text>     Exact transcript of reference audio. Max: 1 MiB.
        --voice <id>              Load saved .s2voice ID. Max: 128 chars; only
                                  ASCII letters, digits, '_' and '-' are allowed.
                                  Explicit --prompt-audio takes precedence.
        --save-voice              Save encoded reference as --voice <id>.
                                  Requires --voice, --prompt-audio, --prompt-text.
        --voice-dir <path>        .s2voice storage directory.
                                  Default: voices/ next to executable.
        --list-voices             List saved profile IDs and exit before model load.

  Generation / segmentation:
    -threads, --threads <N>       CPU threads. Default: 4. Range: 1..256.
        --max-tokens <N>          Generation budget. Default: 1024.
                                  Range: 1..32768. In segmented mode it also caps
                                  each segment's effective budget.
        --segment                 Enable Unicode-aware sentence segmentation.
                                  Without a reference/saved voice, the first generated
                                  segment becomes a request-local voice anchor so later
                                  segments keep the same random timbre.
        --max-seg-tokens <N>      Per-segment budget. Default: 300.
                                  Range: 1..32768; effective segment budget is
                                  min(--max-tokens, --max-seg-tokens, context).
        --min-seg-chars <N>       Merge very short sentence pieces. Default: 0.
                                  Range: 0..1000000; 0 disables minimum length.
        --chunk-length <N>        Fish-style long-form chunk target in visible
                                  Unicode characters. Default: 0 = disabled.
                                  Range when enabled: 100..300. Peak PCM RAM stays
                                  bounded; the staged WAV grows with final audio but
                                  no second raw-PCM copy is created.
        --min-chunk-length <N>    Minimum visible chars for long-form chunks.
                                  Default: 0. Range: 0..100; requires chunking.
        --condition-on-previous-chunks
                                  Keep bounded VQ/acoustic context between chunks
                                  (default).
        --no-condition-on-previous-chunks
                                  Disable automatic inter-chunk VQ conditioning.
        --prosody-volume <dB>     Output gain after final trim. Default: 0 dB.
                                  Range: -20..20 dB.

  Sampling:
        --temperature <F>         Sampling temperature. Alias: --temp.
        --temp <F>                Default: 0.7. Range: finite 0..10; 0 = greedy.
        --top-p <F>               Nucleus threshold. Default: 0.7. Range: (0, 1].
        --top-k <N>               Top-k cutoff. Default: 30. Range: 0..1000000.
        --min-end-tokens <N>      Minimum generated tokens before EOS is allowed.
                                  Default: 64. Range: 0..32768 and must be
                                  strictly smaller than --max-tokens. It is also
                                  clamped to the effective generation budget.
        --seed <uint64>           Request seed. Default: 0 = random source.
                                  1..UINT64_MAX are deterministic; segmented
                                  requests derive a stable sub-seed per segment.
        --repetition-penalty <F>  Explicit recent-token penalty. Default: 1.0
                                  (disabled). Range: finite 1.0..10.0.
        --repetition-window <N>   Tokens considered by repetition penalty.
                                  Default: 64. Range: 0..32768; 0 disables it.

  Conversation / startup:
        --multi-turn-history <N>  Maximum prior text->VQ turns to retain.
                                  Default: 0 = V7.4 behavior for normal text;
                                  multi-speaker input still keeps conversational
                                  VQ context while it fits the model context.
                                  Explicit range: 0..1024 turns.
        --warmup                  Run one isolated deterministic model+codec
                                  warmup before serving/synthesis. Default: OFF.
                                  It never reuses the real voice/reference state.

  RAS (Repetition Aware Sampling):
        --ras-window <N>          Recent-token repetition window. Default: 10.
                                  Range: 0..32768; 0 disables the window.
        --ras-temp <F>            Resample temperature. Default: 1.0.
                                  Range: finite 0..10.
        --ras-top-p <F>           Resample nucleus threshold. Default: 0.9.
                                  Range: (0, 1].

  Codec memory / streaming:
        --codec-chunk <N>         Maximum codec frames per decode window.
                                  Default: 0 = automatic. Range: >= 0.
                                  Positive values can reduce peak memory/VRAM.
        --codec-overlap <N>       Decoder boundary history/holdback frames.
                                  Default: 0 = codec-derived automatic history.
                                  Range: >= 0. Positive values explicitly override
                                  history and can trade continuity for lower memory.
        --stream-decode-stride <N>
                                  WebSocket codec decode cadence in frames.
                                  -1 = disable stride streaming (emit per text segment)
                                   0 = automatic 4-frame cadence (default)
                                  1..32768 = explicit cadence.

  Other:
    -h, --help                    Show this help and exit successfully.

OPTION INTERACTIONS:
  * --prompt-audio requires --prompt-text and takes precedence over --voice.
  * --codec-vulkan -2 inherits the transformer device after parsing.
  * --min-end-tokens must be lower than --max-tokens; runtime additionally
    clamps it below a smaller effective context/segment budget when necessary.
  * text/prompt_text must be valid UTF-8. --segment handles no-space CJK text,
    repeated CJK punctuation and Unicode sentence boundaries without inserting
    artificial ASCII spaces. Keep --min-seg-chars=0 when tiny CJK utterances
    should remain independent. Without a reference voice, the first generated
    segment anchors the random timbre for the rest of that request. Internal
    stride/segment boundaries are not trimmed.
  * Numeric parsing is strict: trailing junk, NaN/Inf where not allowed, and
    out-of-range values fail with exit code 1 instead of being silently accepted.

SERVER ENDPOINTS (default mode):
  GET  /                         Service info.
  GET  /health                   Historical plain-text health check.
  GET  /v1/health                Fish Speech-compatible JSON health check.
  GET  /v1/models                Local model service entry.
  POST /v1/tts                   Fish-Audio-style synthesis.
  POST /v1/audio/speech          OpenAI-style synthesis subset.
  POST /synthesize               Legacy synthesis alias.
  GET  /v1/voices                List saved voices.
  POST /v1/voices/<id>           Save voice from server-local audio_path+transcript.
  GET  /v1/voices/<id>           Read saved voice metadata.
  DELETE /v1/voices/<id>         Delete a saved voice.
  WS   /ws/tts                   Incremental mono PCM int16 streaming.

SERVER REQUEST EXAMPLES (default http://127.0.0.1:8080):
  Status/model info:
    curl http://127.0.0.1:8080/
    curl http://127.0.0.1:8080/health
    curl http://127.0.0.1:8080/v1/health
    curl http://127.0.0.1:8080/v1/models

  Native WAV TTS:
    curl -X POST http://127.0.0.1:8080/v1/tts -H "Content-Type: application/json" -d '{"text":"Hello from s2.","format":"wav"}' -o output.wav

  Long-form WAV with bounded inter-chunk voice context:
    curl -X POST http://127.0.0.1:8080/v1/tts -H "Content-Type: application/json" -d '{"text":"A long passage...","format":"wav","chunk_length":300,"min_chunk_length":50,"condition_on_previous_chunks":true}' -o long.wav

  Raw PCM (sample rate is returned in X-Sample-Rate):
    curl -D pcm-headers.txt -X POST http://127.0.0.1:8080/v1/tts -H "Content-Type: application/json" -d '{"text":"PCM request.","format":"pcm"}' -o output.pcm

  OpenAI-style WAV:
    curl -X POST http://127.0.0.1:8080/v1/audio/speech -H "Content-Type: application/json" -d '{"input":"Hello.","response_format":"wav"}' -o output.wav

  Legacy alias:
    curl -X POST http://127.0.0.1:8080/synthesize -H "Content-Type: application/json" -d '{"text":"Hello.","format":"wav"}' -o output.wav

  Direct reference clone (reference_audio is a server-local path):
    curl -X POST http://127.0.0.1:8080/v1/tts -H "Content-Type: application/json" -d '{"text":"Clone me.","reference_audio":"/path/reference.wav","prompt_text":"Exact transcript."}' -o cloned.wav

  Saved voices:
    curl -X POST http://127.0.0.1:8080/v1/voices/narrator -H "Content-Type: application/json" -d '{"audio_path":"/path/reference.wav","transcript":"Reference transcript."}'
    curl http://127.0.0.1:8080/v1/voices
    curl http://127.0.0.1:8080/v1/voices/narrator
    curl -X POST http://127.0.0.1:8080/v1/tts -H "Content-Type: application/json" -d '{"text":"Use narrator.","voice":"narrator","format":"wav"}' -o narrator.wav
    curl -X DELETE http://127.0.0.1:8080/v1/voices/narrator

  WebSocket streaming (websocat/wscat are separate tools):
    websocat ws://127.0.0.1:8080/ws/tts
    # send: {"text":"Streaming request.","segment":true,"stream_stride":0}
    wscat -c ws://127.0.0.1:8080/ws/tts
    # then enter the same JSON text message.
    Binary replies are [2-byte flags][PCM int16 LE]; strip the flags per frame.

  Windows PowerShell: use curl.exe if curl is mapped to a PowerShell alias.

HTTP/WS SYNTHESIS FIELDS:
  Common: text/input, segment, reference_audio, prompt_text, voice/reference_id,
  temperature, top_p, top_k, seed, repetition_penalty, repetition_window,
  multi_turn_history, threads, max_tokens/max_new_tokens, max_seg_tokens,
  min_end_tokens, ras_window, ras_temp, ras_top_p, codec_chunk, codec_overlap,
  min_seg_chars, chunk_length, min_chunk_length, condition_on_previous_chunks,
  prosody (volume; speed currently only 1.0), latency (normal only),
  trim_silence, and Fish streaming route-consistency flag.
  HTTP only: format/response_format = wav|pcm. stream_stride is rejected.
  WebSocket only: stream_stride. format/response_format are rejected because
  /ws/tts always emits framed mono PCM int16.
  Rejected Fish fields/modes: non-empty references[], early_stop_threshold,
  normalize, sample_rate, mp3_bitrate, opus_bitrate, use_memory_cache,
  prosody.normalize_loudness, latency balanced/low, and prosody.speed != 1.0.
  These fail explicitly instead of being silently ignored.
  Application request/message limit: 8 MiB; text and prompt_text: 1 MiB each.
  Crow 1.3.4 buffers HTTP bodies before route handlers; for a pre-buffer HTTP
  limit on non-loopback deployments, enforce a body limit in the reverse proxy.
  Crow 1.3.4 applies the WebSocket payload limit to the complete reassembled
  message, including fragmented messages; the app checks JSON size again.
  There is no built-in authentication. Put authentication/access control in a
  reverse proxy before exposing a non-loopback bind to untrusted clients.
  WebSocket clients must follow RFC 6455 masking; non-conforming clients close.

WEBSOCKET /ws/tts OUTPUT:
  Binary: [2-byte little-endian flags][mono PCM int16 little-endian samples]
          flags bit 0 = final output boundary.
  Final text JSON: {"done":true,"segments":N,"sample_rate":RATE}
  With stride decoding, multiple PCM chunks may be emitted inside one text segment.

Run the README's complete CLI/API reference for examples and detailed backend notes.
)S2HELP";
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            std::cerr << "Use --help for usage information." << std::endl;
            return 1;
        }
        } catch (const std::exception & e) {
            std::cerr << "Invalid value for option '" << arg << "': " << e.what() << std::endl;
            std::cerr << "Use --help for usage information." << std::endl;
            return 1;
        }
    }

    // --list-voices: listar perfiles guardados y salir
    if (list_voices) {
        try {
            s2::VoiceProfileManager voice_mgr_tmp;
            voice_mgr_tmp.set_storage_dir(params.voice_storage_dir);
            std::vector<std::string> ids = voice_mgr_tmp.list();
            std::sort(ids.begin(), ids.end());
            if (ids.empty()) {
                std::cout << "No saved voice profiles in: " << params.voice_storage_dir << "\n";
            } else {
                std::cout << "Saved voice profiles (" << params.voice_storage_dir << "):\n";
                for (const std::string & id : ids) {
                    std::cout << "  " << id << "\n";
                }
            }
            return 0;
        } catch (const std::exception & e) {
            std::cerr << "Failed to list voice profiles: " << e.what() << "\n";
            return 1;
        }
    }

    // --save-voice validations
    if (params.save_voice) {
        if (params.voice_id.empty()) {
            std::cerr << "Error: --save-voice requires --voice <id>.\n";
            return 1;
        }
        if (params.prompt_audio_path.empty() || params.prompt_text.empty()) {
            std::cerr << "Error: --save-voice requires both --prompt-audio and --prompt-text.\n";
            return 1;
        }
    }

    {
        std::string validation_error;
        if (port < 1 || port > 65535) {
            std::cerr << "Error: port must be between 1 and 65535.\n";
            return 1;
        }
        if (bind_host.size() > 64 || !is_ip_address_literal(bind_host)) {
            std::cerr << "Error: --host must be a valid IPv4 or IPv6 address literal (for example 127.0.0.1 or ::1).\n";
            return 1;
        }
        if (!validate_pipeline_params(params, validation_error, false)) {
            std::cerr << "Error: " << validation_error << "\n";
            return 1;
        }
    }

    // -2 is an internal/default sentinel meaning "use the same device as the
    // transformer". Resolve it before backend-specific validation. Explicit -1
    // always remains CPU, so `-v 0 --codec-vulkan -1` now works as documented.
    if (params.codec_vulkan_device == -2) {
        params.codec_vulkan_device = params.vulkan_device;
    }

    auto gpu_str = [](int d) -> std::string {
        return d < 0 ? "CPU" : ("GPU " + std::to_string(d));
    };

    // Detectar flags de backend incorrectos y advertir al usuario
#if defined(GGML_USE_HIP)
    // ROCm/HIP uses GGML's CUDA-compatible host API internally, but device
    // discovery must use HIP -- never NVIDIA's CUDA runtime.
    {
        int hip_dev_count = 0;
        hipError_t hip_err = hipGetDeviceCount(&hip_dev_count);
        if (hip_err != hipSuccess || hip_dev_count == 0) {
            if (params.vulkan_device >= 0 || params.codec_vulkan_device >= 0) {
                std::cerr << "[Warning] No ROCm/HIP devices found. Requested GPU components will run on CPU.\n";
            }
            params.vulkan_device = -1;
            params.codec_vulkan_device = -1;
        } else {
            auto clamp_hip_device = [&](int32_t & device, const char * component) {
                if (device >= hip_dev_count) {
                    std::cerr << "[Warning] ROCm/HIP device " << device << " requested for " << component
                              << " is unavailable (" << hip_dev_count << " device(s)); using device 0.\n";
                    device = 0;
                }
            };
            clamp_hip_device(params.vulkan_device, "model");
            clamp_hip_device(params.codec_vulkan_device, "codec");
        }
    }
#elif defined(GGML_USE_CUDA) && !defined(GGML_USE_VULKAN)
    // CUDA and Vulkan builds use the same generic device fields internally.
    // Keep transformer and codec selection independent: model CPU + codec GPU
    // and model GPU + codec CPU are both valid configurations.
    {
        int cuda_dev_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&cuda_dev_count);
        if (cuda_err != cudaSuccess || cuda_dev_count == 0) {
            if (params.vulkan_device >= 0 || params.codec_vulkan_device >= 0) {
                std::cerr << "[Warning] No CUDA devices found. Requested GPU components will run on CPU.\n";
            }
            params.vulkan_device = -1;
            params.codec_vulkan_device = -1;
        } else {
            auto clamp_cuda_device = [&](int32_t & device, const char * component) {
                if (device >= cuda_dev_count) {
                    std::cerr << "[Warning] CUDA device " << device << " requested for " << component
                              << " is unavailable (" << cuda_dev_count << " device(s)); using device 0.\n";
                    device = 0;
                }
            };
            clamp_cuda_device(params.vulkan_device, "model");
            clamp_cuda_device(params.codec_vulkan_device, "codec");
        }
    }
#elif defined(GGML_USE_VULKAN) && !defined(GGML_USE_CUDA)
    // Build Vulkan: no hay nada especial que advertir
    (void)0;
#elif defined(GGML_USE_METAL) && !defined(GGML_USE_CUDA) && !defined(GGML_USE_VULKAN)
    // Metal exposes one logical GPU. Preserve independent CPU/GPU choices for
    // transformer and codec, but normalize any positive GPU index to device 0.
    if (params.vulkan_device > 0) {
        std::cerr << "[Warning] Metal has a single device; using device 0 for the model.\n";
        params.vulkan_device = 0;
    }
    if (params.codec_vulkan_device > 0) {
        std::cerr << "[Warning] Metal has a single device; using device 0 for the codec.\n";
        params.codec_vulkan_device = 0;
    }
#else
    // Build CPU puro: advertir si el usuario pide GPU
    if (params.vulkan_device >= 0 || params.codec_vulkan_device >= 0) {
        std::cerr << "[Warning] This build has no GPU backend compiled.\n"
                  << "          -v and --codec-vulkan flags are ignored. Running on CPU.\n";
        params.vulkan_device       = -1;
        params.codec_vulkan_device = -1;
    }
#endif

    std::cout << "\nConfiguration:\n"
              << "  Model:         " << params.model_path << "\n"
              << "  Codec:         " << (params.codec_model_path.empty() ? params.model_path : params.codec_model_path) << "\n"
              << "  Model GPU:     " << gpu_str(params.vulkan_device) << "\n"
              << "  Codec GPU:     " << gpu_str(params.codec_vulkan_device) << "\n"
              << "  Bind address:  " << bind_host << "\n"
              << "  Port:          " << port << "\n"
              << "  CPU threads:   " << params.gen.n_threads << "\n"
              << "  Max tokens:    " << params.gen.max_new_tokens << "\n"
              << "  Seg tokens:    " << params.max_tokens_per_segment << " (per segment)\n"
              << "  Segmentation:  " << (params.segment_sentences ? "ON" : "OFF (use --segment to enable)") << "\n"
              << "  Codec chunk:   " << (params.codec_chunk_frames == 0 ? "auto" : std::to_string(params.codec_chunk_frames) + " frames") << "\n"
              << "  Codec overlap: " << params.codec_overlap_frames << " frames\n"
              << "  Min seg chars: " << (params.min_seg_chars == 0 ? "off" : std::to_string(params.min_seg_chars) + " chars") << "\n"
              << "  Temperature:   " << params.gen.temperature << "\n"
              << "  Top-p:        " << params.gen.top_p << "\n"
              << "  Top-k:         " << params.gen.top_k << "\n"
              << "  Seed:          " << params.gen.seed << (params.gen.seed == 0 ? " (random)" : " (deterministic)") << "\n"
              << "  Repetition:    " << params.gen.repetition_penalty << " / window " << params.gen.repetition_window << "\n"
              << "  Multi-turn:    " << params.multi_turn_history << " turns\n"
              << "  Warmup:        " << (params.warmup ? "ON" : "OFF") << "\n"
              << "  RAS window:    " << params.gen.ras_window_size << " tokens\n"
              << "  RAS temp:      " << params.gen.ras_high_temp << "\n";

    // --- Charger le modele ---
    s2::Pipeline pipeline;
    // Pipeline/model/codec/KV state is mutable and not re-entrant. Crow is multithreaded,
    // so serialize all operations that touch the shared pipeline.
    std::mutex pipeline_mutex;

#ifdef S2_TOKENIZER_EMBEDDED
    // Tokenizer embebido: usar los bytes del array generado por el workflow.
    // No se necesita tokenizer.json en disco.
    params.tokenizer_data      = reinterpret_cast<const char*>(tokenizer_json_data);
    params.tokenizer_data_size = static_cast<size_t>(tokenizer_json_size);
    std::cout << "  Tokenizer:     [embedded in exe, "
              << tokenizer_json_size << " bytes]" << std::endl;
#else
    std::cout << "  Tokenizer:    " << params.tokenizer_path << std::endl;
#endif

    if (!pipeline.init(params)) {
        std::cerr << "Pipeline initialization failed." << std::endl;
        return 1;
    }
    if (params.warmup) {
        std::cout << "[Warmup] Running isolated deterministic model+codec warmup...\n";
        if (!pipeline.warmup(params)) {
            std::cerr << "Warmup failed.\n";
            return 1;
        }
        std::cout << "[Warmup] OK\n";
    }

    // --save-voice is a one-shot CLI operation, not persistent server state.
    // Encode/save immediately so it works without dummy synthesis text/output.
    // If --output is also present, continue below and synthesize once afterwards.
    if (params.save_voice) {
        s2::PipelineParams voice_params = params;
        voice_params.save_voice = false;
        std::vector<int32_t> codes;
        int32_t T_prompt = 0;
        bool encoded_ok = false;
        try {
            encoded_ok = pipeline.encode_reference(voice_params, codes, T_prompt);
        } catch (const std::bad_alloc &) {
            std::cerr << "Failed to encode reference audio for --save-voice: out of memory.\n";
            return 1;
        } catch (const std::exception & e) {
            std::cerr << "Failed to encode reference audio for --save-voice: " << e.what() << "\n";
            return 1;
        } catch (...) {
            std::cerr << "Failed to encode reference audio for --save-voice: unexpected error.\n";
            return 1;
        }
        if (!encoded_ok || codes.empty() || T_prompt <= 0 ||
            codes.size() != static_cast<size_t>(pipeline.num_codebooks()) * static_cast<size_t>(T_prompt)) {
            std::cerr << "Failed to encode reference audio for --save-voice.\n";
            return 1;
        }

        s2::VoiceProfile profile;
        profile.transcript = params.prompt_text;
        profile.codes = std::move(codes);
        profile.T_prompt = T_prompt;
        profile.num_codebooks = pipeline.num_codebooks();
        profile.codebook_size = pipeline.codebook_size();
        profile.sample_rate = pipeline.sample_rate();

        try {
            s2::VoiceProfileManager mgr;
            mgr.set_storage_dir(params.voice_storage_dir);
            if (!mgr.save(params.voice_id, profile)) {
                std::cerr << "Failed to save voice profile '" << params.voice_id << "'.\n";
                return 1;
            }
        } catch (const std::exception & e) {
            std::cerr << "Failed to save voice profile '" << params.voice_id << "': " << e.what() << "\n";
            return 1;
        }
        std::cout << "Saved voice profile '" << params.voice_id << "' (" << T_prompt << " frames).\n";
        params.save_voice = false;
        if (params.output_path.empty()) return 0;
    }

    // --- Modo CLI: --output path ---
    // Si se especifico --output, sintetizar una vez, guardar y salir (sin servidor HTTP).
    if (!params.output_path.empty()) {
        if (params.text.empty()) {
            // Leer stdin si no se paso texto
            std::cout << "Reading text from stdin (Ctrl+D to finish)...\n";
            std::string line;
            while (std::getline(std::cin, line)) {
                if (!params.text.empty()) params.text += " ";
                params.text += line;
            }
        }
        if (params.text.empty()) {
            std::cerr << "Error: --output requires text. Pipe it or set --text.\n";
            return 1;
        }
        {
            std::string validation_error;
            if (!validate_pipeline_params(params, validation_error, true)) {
                std::cerr << "Error: " << validation_error << "\n";
                return 1;
            }
        }
        // segment_sentences ya configurado por --segment si el usuario lo paso
        try {
            if (!pipeline.synthesize(params)) {
                std::cerr << "Synthesis failed.\n";
                return 1;
            }
        } catch (const std::bad_alloc &) {
            std::cerr << "Synthesis failed: out of memory.\n";
            return 1;
        } catch (const std::exception & e) {
            std::cerr << "Synthesis failed: " << e.what() << "\n";
            return 1;
        } catch (...) {
            std::cerr << "Synthesis failed: unexpected error.\n";
            return 1;
        }
        std::cout << "Done: " << params.output_path << "\n";
        return 0;
    }

    // --- Serveur HTTP ---
    crow::SimpleApp app;

    // Respuestas > 1MB se streamean automaticamente (sin timeout).
    // Los WAV de audio suelen ser varios MB -- sin esto pueden cortar.
    app.stream_threshold(1024 * 1024); // 1 MB
    // Crow 1.3.4 enforces this limit against the complete reassembled WebSocket
    // message, including fragmented messages. onmessage keeps the same complete
    // JSON-size check as an application-level defense in depth.
    app.websocket_max_payload(MAX_JSON_REQUEST_BYTES);

    // Per-connection cancellation state lets a disconnect stop an expensive
    // synthesis even if Crow's asynchronous send itself does not throw.
    std::mutex ws_state_mutex;
    std::unordered_map<crow::websocket::connection *, std::shared_ptr<std::atomic_bool>> ws_alive;
    auto mark_ws_closed = [&](crow::websocket::connection & conn) {
        std::lock_guard<std::mutex> lock(ws_state_mutex);
        auto it = ws_alive.find(&conn);
        if (it != ws_alive.end()) {
            it->second->store(false, std::memory_order_relaxed);
            ws_alive.erase(it);
        }
    };

    // ================================================================
    // Helper : traitement commun de synthese
    // ================================================================
    auto do_synthesize = [&](const crow::json::rvalue& json) -> crow::response {
        bool request_validated = false;
        try {
            s2::PipelineParams synth_params = params;
            validate_fish_json_subset(json, true);

            // Fish Audio uses "text"; OpenAI-compatible clients usually use "input".
            // Treat aliases as one logical field and reject contradictory requests.
            if (json.has("text") && json.has("input")) {
                const std::string a = json["text"].s();
                const std::string b = json["input"].s();
                if (a != b) throw std::invalid_argument("conflicting text and input");
                synth_params.text = a;
            } else if (json.has("text")) {
                synth_params.text = json["text"].s();
            } else if (json.has("input")) {
                synth_params.text = json["input"].s();
            } else {
                return crow::response(400, "Missing 'text' or 'input' field");
            }

            if (json.has("temperature"))   synth_params.gen.temperature = static_cast<float>(json["temperature"].d());
            if (json.has("top_p"))         synth_params.gen.top_p = static_cast<float>(json["top_p"].d());
            if (json.has("top_k"))         synth_params.gen.top_k = checked_json_i32(json["top_k"]);
            if (json.has("seed"))          synth_params.gen.seed = checked_json_fish_seed(json["seed"]);
            if (json.has("repetition_penalty")) synth_params.gen.repetition_penalty = static_cast<float>(json["repetition_penalty"].d());
            if (json.has("repetition_window")) synth_params.gen.repetition_window = checked_json_i32(json["repetition_window"]);
            if (json.has("multi_turn_history")) synth_params.multi_turn_history = checked_json_i32(json["multi_turn_history"]);
            if (json.has("threads"))       synth_params.gen.n_threads = checked_json_i32(json["threads"]);
            // Native s2.cpp uses max_tokens; Fish Speech/Fish Audio uses max_new_tokens.
            // Accept both, but never silently choose one when a client sends conflicting values.
            if (json.has("max_tokens") && json.has("max_new_tokens")) {
                const int32_t a = checked_json_i32(json["max_tokens"]);
                const int32_t b = checked_json_fish_max_new_tokens(json["max_new_tokens"]);
                if (a != b) throw std::invalid_argument("conflicting max_tokens and max_new_tokens");
                synth_params.gen.max_new_tokens = a;
            } else if (json.has("max_tokens")) {
                synth_params.gen.max_new_tokens = checked_json_i32(json["max_tokens"]);
            } else if (json.has("max_new_tokens")) {
                synth_params.gen.max_new_tokens = checked_json_fish_max_new_tokens(json["max_new_tokens"]);
            }
            if (json.has("max_seg_tokens")) synth_params.max_tokens_per_segment = checked_json_i32(json["max_seg_tokens"]);
            if (json.has("segment"))       synth_params.segment_sentences = json["segment"].b();
            if (json.has("codec_chunk"))   synth_params.codec_chunk_frames = checked_json_i32(json["codec_chunk"]);
            if (json.has("codec_overlap")) synth_params.codec_overlap_frames = checked_json_i32(json["codec_overlap"]);
            if (json.has("min_seg_chars")) synth_params.min_seg_chars = checked_json_i32(json["min_seg_chars"]);
            if (json.has("chunk_length")) synth_params.chunk_length = checked_json_i32(json["chunk_length"]);
            if (json.has("min_chunk_length")) synth_params.min_chunk_length = checked_json_i32(json["min_chunk_length"]);
            if (json.has("condition_on_previous_chunks")) {
                const auto & v = json["condition_on_previous_chunks"];
                if (v.t() != crow::json::type::True && v.t() != crow::json::type::False)
                    throw std::invalid_argument("condition_on_previous_chunks must be a boolean");
                synth_params.condition_on_previous_chunks = v.b();
            }
            if (json.has("latency")) {
                const std::string latency = json["latency"].s();
                if (latency != "normal")
                    throw std::invalid_argument("latency modes balanced/low are not implemented; use normal");
            }
            if (json.has("prosody")) {
                const auto & pr = json["prosody"];
                if (pr.t() != crow::json::type::Object)
                    throw std::invalid_argument("prosody must be an object");
                if (pr.has("speed")) {
                    if (pr["speed"].t() != crow::json::type::Number)
                        throw std::invalid_argument("prosody.speed must be a number");
                    const double speed = pr["speed"].d();
                    if (!std::isfinite(speed) || speed < 0.5 || speed > 2.0)
                        throw std::invalid_argument("prosody.speed must be between 0.5 and 2.0");
                    if (std::abs(speed - 1.0) > 1e-12)
                        throw std::invalid_argument("prosody.speed other than 1.0 is not implemented without pitch-preserving time stretch");
                }
                if (pr.has("volume")) {
                    if (pr["volume"].t() != crow::json::type::Number)
                        throw std::invalid_argument("prosody.volume must be a number");
                    const double volume = pr["volume"].d();
                    if (!std::isfinite(volume) || volume < -20.0 || volume > 20.0)
                        throw std::invalid_argument("prosody.volume must be between -20 and 20 dB");
                    synth_params.prosody_volume_db = static_cast<float>(volume);
                }
                if (pr.has("normalize_loudness"))
                    throw std::invalid_argument("prosody.normalize_loudness is not implemented");
            }
            if (json.has("min_end_tokens")) synth_params.gen.min_tokens_before_end = checked_json_i32(json["min_end_tokens"]);
            if (json.has("ras_window"))    synth_params.gen.ras_window_size = checked_json_i32(json["ras_window"]);
            if (json.has("ras_temp"))      synth_params.gen.ras_high_temp = static_cast<float>(json["ras_temp"].d());
            if (json.has("ras_top_p"))     synth_params.gen.ras_high_top_p = static_cast<float>(json["ras_top_p"].d());
            if (json.has("prompt_text"))   synth_params.prompt_text = json["prompt_text"].s();
            if (json.has("reference_audio")) synth_params.prompt_audio_path = json["reference_audio"].s();
            // Fish Audio names a saved voice reference_id; s2.cpp historically uses voice.
            const bool has_reference_id =
                json.has("reference_id") && json["reference_id"].t() != crow::json::type::Null;
            if (json.has("voice") && has_reference_id) {
                const std::string a = json["voice"].s();
                const std::string b = json["reference_id"].s();
                if (a != b) throw std::invalid_argument("conflicting voice and reference_id");
                synth_params.voice_id = a;
            } else if (json.has("voice")) {
                synth_params.voice_id = json["voice"].s();
            } else if (has_reference_id) {
                synth_params.voice_id = json["reference_id"].s();
            }
            if (json.has("trim_silence"))  synth_params.trim_silence = json["trim_silence"].b();
            if (json.has("stream_stride")) synth_params.stream_decode_stride_frames = checked_json_i32(json["stream_stride"]);

            std::string validation_error;
            if (!validate_pipeline_params(synth_params, validation_error, true)) {
                return crow::response(400, validation_error);
            }

            // Native clients use "format"; OpenAI-compatible clients use
            // "response_format". This implementation intentionally supports
            // only WAV and raw PCM rather than silently returning WAV as MP3/Opus.
            std::string format = "wav";
            auto normalized_format = [](std::string value) {
                std::transform(value.begin(), value.end(), value.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                return value;
            };
            if (json.has("format") && json.has("response_format")) {
                const std::string a = normalized_format(json["format"].s());
                const std::string b = normalized_format(json["response_format"].s());
                if (a != b) throw std::invalid_argument("conflicting format and response_format");
                format = a;
            } else if (json.has("format")) {
                format = normalized_format(json["format"].s());
            } else if (json.has("response_format")) {
                format = normalized_format(json["response_format"].s());
            }
            if (format != "wav" && format != "pcm") {
                return crow::response(400, "Unsupported format. Supported formats: wav, pcm");
            }

            // From this point onward, exceptions are execution/storage failures,
            // not malformed client input. Keep that distinction for HTTP status.
            request_validated = true;

            // Preserve the HTTP contract: a syntactically valid saved voice id
            // that does not exist is a client-visible 404. If the file exists but
            // is corrupt/unreadable, Pipeline::get_ref_codes() fails later and the
            // request remains a 500. An explicit reference_audio has priority over
            // voice/reference_id and therefore does not require a saved profile.
            if (!synth_params.voice_id.empty() && synth_params.prompt_audio_path.empty()) {
                s2::VoiceProfileManager mgr;
                mgr.set_storage_dir(synth_params.voice_storage_dir);
                if (!mgr.exists(synth_params.voice_id)) {
                    return crow::response(404, "Voice not found: " + synth_params.voice_id);
                }
            }

            // The Pipeline/model/codec/KV cache are shared and mutable. Parse and
            // validate before taking the lock so malformed requests do not block TTS.
            std::string wav_path;
            {
                std::lock_guard<std::mutex> pipeline_lock(pipeline_mutex);
                if (!pipeline.synthesize_to_file(synth_params, wav_path)) {
                    return crow::response(500, "Synthesis failed (check requested voice/reference and server log)");
                }
            }

            ScopedTempPath temp_wav{wav_path};
            crow::response res;
            std::string payload;
            if (!read_generated_wav(wav_path, format == "pcm", payload)) {
                return crow::response(500, "Generated WAV file is truncated or invalid");
            }
            if (format == "pcm") {
                res.set_header("Content-Type", "audio/pcm");
                res.set_header("X-Sample-Rate", std::to_string(pipeline.sample_rate()));
            } else {
                res.set_header("Content-Type", "audio/wav");
                res.set_header("Content-Disposition", "attachment; filename=\"audio.wav\"");
                res.set_header("Content-Length", std::to_string(payload.size()));
            }
            res.body = std::move(payload);
            return res;
        } catch (const std::bad_alloc &) {
            return crow::response(500, "Server ran out of memory while preparing the response");
        } catch (const std::exception & e) {
            const int status = request_validated ? 500 : 400;
            const char * prefix = request_validated ? "Synthesis failed: " : "Invalid request: ";
            return crow::response(status, std::string(prefix) + e.what());
        } catch (...) {
            return crow::response(request_validated ? 500 : 400,
                                  request_validated ? "Synthesis failed" : "Invalid request");
        }
    };

    // Strict JSON normalization may throw before Crow sees the body (for example,
    // malformed UTF-16 surrogate escapes). Convert all client parse/type failures
    // into 4xx here instead of letting them escape the route callback.
    auto handle_synthesis_request = [&](const crow::request & req) -> crow::response {
        if (req.body.size() > MAX_JSON_REQUEST_BYTES) {
            return crow::response(413, "JSON request body too large");
        }
        try {
            auto json = load_json_strict(req.body);
            if (!json) {
                return crow::response(400, "Invalid JSON");
            }
            return do_synthesize(json);
        } catch (const std::bad_alloc &) {
            return crow::response(500, "Server ran out of memory while parsing the request");
        } catch (const std::exception & e) {
            return crow::response(400, std::string("Invalid JSON request: ") + e.what());
        } catch (...) {
            return crow::response(400, "Invalid JSON request");
        }
    };

    // ================================================================
    // Route Fish Audio compatible : POST /v1/tts
    // ================================================================
    CROW_ROUTE(app, "/v1/tts")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        return handle_synthesis_request(req);
    });

    // ================================================================
    // Route legacy : POST /synthesize (compatibilite avec vos tests)
    // ================================================================
    CROW_ROUTE(app, "/synthesize")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        return handle_synthesis_request(req);
    });

    // ================================================================
    // Route OpenAI compatible : POST /v1/audio/speech
    // ================================================================
    CROW_ROUTE(app, "/v1/audio/speech")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        return handle_synthesis_request(req);
    });

    // ================================================================
    // Health check & info
    // ================================================================
    CROW_ROUTE(app, "/v1/models")
    .methods("GET"_method)
    ([&]() {
        crow::json::wvalue resp;
        resp["object"] = "list";
        crow::json::wvalue model;
        model["id"] = "s2-pro-local";
        model["object"] = "model";
        model["owned_by"] = "local";
        resp["data"][0] = std::move(model);
        return crow::response(200, resp);
    });

    // ================================================================
    // Voice profile management -- GET/POST/DELETE /v1/voices
    // ================================================================

    // GET /v1/voices -- list all saved voice profiles
    CROW_ROUTE(app, "/v1/voices")
    .methods("GET"_method)
    ([&]() {
        try {
            std::lock_guard<std::mutex> pipeline_lock(pipeline_mutex);
            s2::VoiceProfileManager mgr;
            mgr.set_storage_dir(params.voice_storage_dir);
            std::vector<std::string> ids = mgr.list();
            std::sort(ids.begin(), ids.end());

            crow::json::wvalue resp;
            resp["object"] = "list";
            int idx = 0;
            for (const std::string & id : ids) {
                crow::json::wvalue item;
                item["id"] = id;
                item["object"] = "voice";
                resp["data"][idx++] = std::move(item);
            }
            resp["count"] = static_cast<int64_t>(ids.size());
            return crow::response(200, resp);
        } catch (const std::exception & e) {
            return crow::response(500, std::string("Failed to list voices: ") + e.what());
        }
    });

    // POST /v1/voices/<id> -- save a voice profile from a local reference audio.
    // Body JSON: { "transcript": "...", "audio_path": "/abs/path/to/ref.wav" }
    CROW_ROUTE(app, "/v1/voices/<string>")
    .methods("POST"_method)
    ([&](const crow::request& req, const std::string & voice_id) {
        bool request_validated = false;
        try {
            if (req.body.size() > MAX_JSON_REQUEST_BYTES) return crow::response(413, "JSON request body too large");
            auto json = load_json_strict(req.body);
            if (!json) return crow::response(400, "Invalid JSON");
            if (!json.has("audio_path") || !json.has("transcript")) {
                return crow::response(400, "Required fields: audio_path, transcript");
            }

            s2::PipelineParams vp = params;
            vp.prompt_audio_path = json["audio_path"].s();
            vp.prompt_text = json["transcript"].s();
            vp.voice_id = voice_id;
            vp.save_voice = false; // this route saves exactly once below
            vp.voice_storage_dir = params.voice_storage_dir;
            if (vp.prompt_audio_path.empty() || vp.prompt_text.empty()) {
                return crow::response(400, "audio_path and transcript must not be empty");
            }

            std::string validation_error;
            if (!validate_pipeline_params(vp, validation_error, false)) {
                return crow::response(400, validation_error);
            }
            request_validated = true;

            std::lock_guard<std::mutex> pipeline_lock(pipeline_mutex);
            std::vector<int32_t> codes;
            int32_t T_prompt = 0;
            if (!pipeline.encode_reference(vp, codes, T_prompt) || codes.empty() || T_prompt <= 0) {
                return crow::response(400, "Failed to load/encode reference audio");
            }
            if (codes.size() != static_cast<size_t>(pipeline.num_codebooks()) * static_cast<size_t>(T_prompt)) {
                return crow::response(500, "Encoded voice has inconsistent dimensions");
            }

            s2::VoiceProfileManager mgr;
            mgr.set_storage_dir(params.voice_storage_dir);
            s2::VoiceProfile profile;
            profile.transcript = vp.prompt_text;
            profile.codes = std::move(codes);
            profile.T_prompt = T_prompt;
            profile.num_codebooks = pipeline.num_codebooks();
            profile.codebook_size = pipeline.codebook_size();
            profile.sample_rate = pipeline.sample_rate();
            if (!mgr.save(voice_id, profile)) {
                return crow::response(500, "Failed to save voice profile");
            }

            crow::json::wvalue resp;
            resp["id"] = voice_id;
            resp["object"] = "voice";
            resp["T_prompt"] = T_prompt;
            resp["saved"] = true;
            return crow::response(201, resp);
        } catch (const std::bad_alloc &) {
            return crow::response(500, "Server ran out of memory while creating the voice profile");
        } catch (const std::invalid_argument & e) {
            const int status = request_validated ? 500 : 400;
            const char * prefix = request_validated
                ? "Failed to create voice profile: "
                : "Invalid voice request: ";
            return crow::response(status, std::string(prefix) + e.what());
        } catch (const std::exception & e) {
            const int status = request_validated ? 500 : 400;
            const char * prefix = request_validated
                ? "Failed to create voice profile: "
                : "Invalid voice request: ";
            return crow::response(status, std::string(prefix) + e.what());
        } catch (...) {
            return crow::response(request_validated ? 500 : 400,
                                  request_validated ? "Failed to create voice profile"
                                                    : "Invalid voice request");
        }
    });

    // GET /v1/voices/<id> -- get metadata for a single voice profile
    CROW_ROUTE(app, "/v1/voices/<string>")
    .methods("GET"_method)
    ([&](const std::string & voice_id) {
        try {
            std::lock_guard<std::mutex> pipeline_lock(pipeline_mutex);
            s2::VoiceProfileManager mgr;
            mgr.set_storage_dir(params.voice_storage_dir);
            if (!mgr.exists(voice_id)) {
                return crow::response(404, "Voice not found: " + voice_id);
            }
            s2::VoiceProfile profile = mgr.load(voice_id);
            crow::json::wvalue resp;
            resp["id"] = voice_id;
            resp["object"] = "voice";
            resp["transcript"] = profile.transcript;
            resp["T_prompt"] = profile.T_prompt;
            resp["num_codebooks"] = profile.num_codebooks;
            resp["codebook_size"] = profile.codebook_size;
            resp["sample_rate"] = profile.sample_rate;
            return crow::response(200, resp);
        } catch (const std::bad_alloc &) {
            return crow::response(500, "Server ran out of memory while loading the voice profile");
        } catch (const std::invalid_argument & e) {
            return crow::response(400, std::string("Invalid voice id: ") + e.what());
        } catch (const std::exception & e) {
            // A profile that exists but cannot be parsed/read is a server-side
            // storage error, not a missing resource.
            return crow::response(500, std::string("Failed to load voice profile: ") + e.what());
        }
    });

    // DELETE /v1/voices/<id> -- delete a saved voice profile
    CROW_ROUTE(app, "/v1/voices/<string>")
    .methods("DELETE"_method)
    ([&](const std::string & voice_id) {
        try {
            std::lock_guard<std::mutex> pipeline_lock(pipeline_mutex);
            s2::VoiceProfileManager mgr;
            mgr.set_storage_dir(params.voice_storage_dir);
            if (!mgr.remove(voice_id)) {
                crow::json::wvalue err;
                err["error"] = "Voice not found: " + voice_id;
                return crow::response(404, err);
            }
            crow::json::wvalue resp;
            resp["id"] = voice_id;
            resp["deleted"] = true;
            return crow::response(200, resp);
        } catch (const std::invalid_argument & e) {
            return crow::response(400, std::string("Invalid voice id: ") + e.what());
        } catch (const std::exception & e) {
            return crow::response(500, std::string("Failed to delete voice: ") + e.what());
        }
    });

    // ================================================================
    CROW_ROUTE(app, "/health")
    .methods("GET"_method)
    ([]() {
        return crow::response(200, "OK");
    });

    // Fish Speech's documented local-server health endpoint. Keep /health as
    // the historical alias so existing s2.cpp deployments do not break.
    CROW_ROUTE(app, "/v1/health")
    .methods("GET"_method)
    ([]() {
        crow::json::wvalue status;
        status["status"] = "ok";
        return crow::response(200, status);
    });

    CROW_ROUTE(app, "/")
    ([&port, &bind_host]() {
        crow::json::wvalue info;
        info["status"] = "running";
        info["host"] = bind_host;
        info["port"] = port;
        info["endpoints"][0] = "/v1/tts";
        info["endpoints"][1] = "/synthesize";
        info["endpoints"][2] = "/v1/audio/speech";
        info["endpoints"][3] = "/v1/models";
        info["endpoints"][4] = "/v1/voices";
        info["endpoints"][5] = "/health";
        info["endpoints"][6] = "/v1/health";
        info["endpoints"][7] = "/v1/voices/<id>";
        info["endpoints"][8] = "/ws/tts";
        return crow::response(200, info);
    });

    // ================================================================
    // WebSocket /ws/tts -- streaming PCM durante la generacion.
    //
    // Protocolo (JSON sobre WebSocket):
    //
    //   Cliente -> Servidor:
    //     { "text": "...", "segment": true, "reference_audio": "path" }
    //
    //   Servidor -> Cliente (mensajes binarios):
    //     [2 bytes little-endian: flags] [PCM int16 LE, mono, codec sample rate]
    //     flags bit0 = is_last (1 si es el ultimo segmento)
    //
    //   Servidor -> Cliente (mensaje de texto al finalizar):
    //     { "done": true, "segments": N, "sample_rate": RATE }
    //
    //   Servidor -> Cliente (en caso de error):
    //     { "error": "descripcion" }
    //
    // Con stride activo, el cliente puede empezar a reproducir PCM antes de
    // que termine la oracion actual; con stride desactivado se envia por segmento.
    // ================================================================
    CROW_WEBSOCKET_ROUTE(app, "/ws/tts")
    .onopen([&](crow::websocket::connection& conn) {
        auto alive = std::make_shared<std::atomic_bool>(true);
        {
            std::lock_guard<std::mutex> lock(ws_state_mutex);
            ws_alive[&conn] = std::move(alive);
        }
        std::cout << "[WS] Client connected: " << conn.get_remote_ip() << "\n";
    })
    .onclose([&](crow::websocket::connection& conn, const std::string& reason, uint16_t) {
        mark_ws_closed(conn);
        std::cout << "[WS] Client disconnected: " << reason << "\n";
    })
    .onmessage([&](crow::websocket::connection& conn,
                   const std::string& data,
                   bool is_binary) {
        std::shared_ptr<std::atomic_bool> alive;
        {
            std::lock_guard<std::mutex> lock(ws_state_mutex);
            auto it = ws_alive.find(&conn);
            if (it != ws_alive.end()) alive = it->second;
        }
        if (!alive) return;

        auto safe_send_text = [&](const std::string & msg) -> bool {
            if (!alive->load(std::memory_order_relaxed)) return false;
            try { conn.send_text(msg); return true; } catch (...) { return false; }
        };
        if (is_binary) {
            safe_send_text("{\"error\": \"expected JSON text message\"}");
            return;
        }
        if (data.size() > MAX_JSON_REQUEST_BYTES) {
            safe_send_text("{\"error\": \"JSON request body too large\"}");
            return;
        }

        try {
            auto json = load_json_strict(data);
            if (!json || (!json.has("text") && !json.has("input"))) {
                safe_send_text("{\"error\": \"missing 'text' or 'input' field\"}");
                return;
            }

            s2::PipelineParams ws_params = params;
            validate_fish_json_subset(json, false);
            if (json.has("text") && json.has("input")) {
                const std::string a = json["text"].s();
                const std::string b = json["input"].s();
                if (a != b) throw std::invalid_argument("conflicting text and input");
                ws_params.text = a;
            } else if (json.has("text")) {
                ws_params.text = json["text"].s();
            } else {
                ws_params.text = json["input"].s();
            }
            if (json.has("segment"))          ws_params.segment_sentences = json["segment"].b();
            if (json.has("temperature"))      ws_params.gen.temperature = static_cast<float>(json["temperature"].d());
            if (json.has("top_p"))            ws_params.gen.top_p = static_cast<float>(json["top_p"].d());
            if (json.has("top_k"))            ws_params.gen.top_k = checked_json_i32(json["top_k"]);
            if (json.has("seed"))             ws_params.gen.seed = checked_json_fish_seed(json["seed"]);
            if (json.has("repetition_penalty")) ws_params.gen.repetition_penalty = static_cast<float>(json["repetition_penalty"].d());
            if (json.has("repetition_window")) ws_params.gen.repetition_window = checked_json_i32(json["repetition_window"]);
            if (json.has("multi_turn_history")) ws_params.multi_turn_history = checked_json_i32(json["multi_turn_history"]);
            if (json.has("threads"))          ws_params.gen.n_threads = checked_json_i32(json["threads"]);
            if (json.has("max_tokens") && json.has("max_new_tokens")) {
                const int32_t a = checked_json_i32(json["max_tokens"]);
                const int32_t b = checked_json_fish_max_new_tokens(json["max_new_tokens"]);
                if (a != b) throw std::invalid_argument("conflicting max_tokens and max_new_tokens");
                ws_params.gen.max_new_tokens = a;
            } else if (json.has("max_tokens")) {
                ws_params.gen.max_new_tokens = checked_json_i32(json["max_tokens"]);
            } else if (json.has("max_new_tokens")) {
                ws_params.gen.max_new_tokens = checked_json_fish_max_new_tokens(json["max_new_tokens"]);
            }
            if (json.has("max_seg_tokens"))   ws_params.max_tokens_per_segment = checked_json_i32(json["max_seg_tokens"]);
            if (json.has("reference_audio"))  ws_params.prompt_audio_path = json["reference_audio"].s();
            if (json.has("codec_chunk"))      ws_params.codec_chunk_frames = checked_json_i32(json["codec_chunk"]);
            if (json.has("codec_overlap"))    ws_params.codec_overlap_frames = checked_json_i32(json["codec_overlap"]);
            if (json.has("min_seg_chars"))    ws_params.min_seg_chars = checked_json_i32(json["min_seg_chars"]);
            if (json.has("chunk_length")) ws_params.chunk_length = checked_json_i32(json["chunk_length"]);
            if (json.has("min_chunk_length")) ws_params.min_chunk_length = checked_json_i32(json["min_chunk_length"]);
            if (json.has("condition_on_previous_chunks")) {
                const auto & v = json["condition_on_previous_chunks"];
                if (v.t() != crow::json::type::True && v.t() != crow::json::type::False)
                    throw std::invalid_argument("condition_on_previous_chunks must be a boolean");
                ws_params.condition_on_previous_chunks = v.b();
            }
            if (json.has("latency")) {
                const std::string latency = json["latency"].s();
                if (latency != "normal")
                    throw std::invalid_argument("latency modes balanced/low are not implemented; use normal");
            }
            if (json.has("prosody")) {
                const auto & pr = json["prosody"];
                if (pr.t() != crow::json::type::Object) throw std::invalid_argument("prosody must be an object");
                if (pr.has("speed")) {
                    if (pr["speed"].t() != crow::json::type::Number) throw std::invalid_argument("prosody.speed must be a number");
                    const double speed = pr["speed"].d();
                    if (!std::isfinite(speed) || speed < 0.5 || speed > 2.0) throw std::invalid_argument("prosody.speed must be between 0.5 and 2.0");
                    if (std::abs(speed - 1.0) > 1e-12) throw std::invalid_argument("prosody.speed other than 1.0 is not implemented without pitch-preserving time stretch");
                }
                if (pr.has("volume")) {
                    if (pr["volume"].t() != crow::json::type::Number) throw std::invalid_argument("prosody.volume must be a number");
                    const double volume = pr["volume"].d();
                    if (!std::isfinite(volume) || volume < -20.0 || volume > 20.0) throw std::invalid_argument("prosody.volume must be between -20 and 20 dB");
                    ws_params.prosody_volume_db = static_cast<float>(volume);
                }
                if (pr.has("normalize_loudness")) throw std::invalid_argument("prosody.normalize_loudness is not implemented");
            }
            if (json.has("min_end_tokens"))   ws_params.gen.min_tokens_before_end = checked_json_i32(json["min_end_tokens"]);
            if (json.has("ras_window"))       ws_params.gen.ras_window_size = checked_json_i32(json["ras_window"]);
            if (json.has("ras_temp"))         ws_params.gen.ras_high_temp = static_cast<float>(json["ras_temp"].d());
            if (json.has("ras_top_p"))        ws_params.gen.ras_high_top_p = static_cast<float>(json["ras_top_p"].d());
            if (json.has("prompt_text"))      ws_params.prompt_text = json["prompt_text"].s();
            const bool has_reference_id =
                json.has("reference_id") && json["reference_id"].t() != crow::json::type::Null;
            if (json.has("voice") && has_reference_id) {
                const std::string a = json["voice"].s();
                const std::string b = json["reference_id"].s();
                if (a != b) throw std::invalid_argument("conflicting voice and reference_id");
                ws_params.voice_id = a;
            } else if (json.has("voice")) {
                ws_params.voice_id = json["voice"].s();
            } else if (has_reference_id) {
                ws_params.voice_id = json["reference_id"].s();
            }
            if (json.has("trim_silence"))     ws_params.trim_silence = json["trim_silence"].b();
            if (json.has("stream_stride"))    ws_params.stream_decode_stride_frames = checked_json_i32(json["stream_stride"]);

            std::string validation_error;
            if (!validate_pipeline_params(ws_params, validation_error, true)) {
                crow::json::wvalue err;
                err["error"] = validation_error;
                safe_send_text(err.dump());
                return;
            }

            int32_t segment_count = 0;
            s2::StreamCallback cb = [&](const int16_t* pcm, size_t n_samples, bool is_last) -> bool {
                try {
                    if (!alive->load(std::memory_order_relaxed)) return false;
                    if ((n_samples > 0 && pcm == nullptr) ||
                        n_samples > (std::numeric_limits<size_t>::max() - 2u) / 2u) return false;
                    uint16_t flags = is_last ? 1u : 0u;
                    std::string msg(2 + n_samples * 2, '\0');
                    msg[0] = static_cast<char>(flags & 0xFF);
                    msg[1] = static_cast<char>((flags >> 8) & 0xFF);
                    if (n_samples > 0) std::memcpy(msg.data() + 2, pcm, n_samples * 2);
                    conn.send_binary(msg);
                    return true;
                } catch (...) {
                    return false;
                }
            };

            bool ok = false;
            {
                std::lock_guard<std::mutex> pipeline_lock(pipeline_mutex);
                s2::CancelCallback should_continue = [alive]() -> bool {
                    return alive->load(std::memory_order_relaxed);
                };
                ok = pipeline.synthesize_streaming(ws_params, cb, &segment_count, should_continue);
            }

            crow::json::wvalue done_msg;
            if (ok) {
                done_msg["done"] = true;
                done_msg["segments"] = segment_count;
                done_msg["sample_rate"] = pipeline.sample_rate();
            } else {
                done_msg["error"] = "synthesis failed (check requested voice/reference and server log)";
                done_msg["segments"] = segment_count;
            }
            safe_send_text(done_msg.dump());
        } catch (const std::bad_alloc &) {
            safe_send_text("{\"error\": \"server ran out of memory\"}");
        } catch (const std::exception & e) {
            crow::json::wvalue err;
            err["error"] = std::string("invalid request: ") + e.what();
            safe_send_text(err.dump());
        } catch (...) {
            safe_send_text("{\"error\": \"invalid request\"}");
        }
    })
    .onerror([&](crow::websocket::connection& conn, const std::string& error_message) {
        mark_ws_closed(conn);
        std::cerr << "[WS] Connection error: " << error_message << "\n";
    });

    std::cout << "\nEndpoints:\n"
              << "  POST /v1/tts           (Fish Audio compatible)\n"
              << "  POST /synthesize       (legacy)\n"
              << "  POST /v1/audio/speech  (OpenAI compatible)\n"
              << "  GET  /v1/models\n"
              << "  GET  /v1/voices        (list saved voices)\n"
              << "  POST /v1/voices/<id>   (save voice)\n"
              << "  GET  /v1/voices/<id>   (voice metadata)\n"
              << "  DELETE /v1/voices/<id> (delete voice)\n"
              << "  GET  /health\n"
              << "  GET  /v1/health\n"
              << "  WS   /ws/tts           (streaming -- minimum latency)\n\n";

    if (bind_host != "127.0.0.1" && bind_host != "::1") {
        std::cerr << "[Security warning] Server is binding to " << bind_host
                  << ". API requests can reference local audio paths; expose this only to trusted clients.\n";
    }
    std::cout << "Server listening on " << bind_host << ":" << port << "...\n";
    try {
        app.bindaddr(bind_host).port(static_cast<uint16_t>(port)).multithreaded().run();
    } catch (const std::exception & e) {
        std::cerr << "Server startup/runtime error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Server startup/runtime error: unknown exception\n";
        return 1;
    }
    return 0;
}
