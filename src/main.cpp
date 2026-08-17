#include "s2_pipeline.h"
#if defined(GGML_USE_CUDA)
#  include <cuda_runtime.h>
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

static int32_t checked_json_i32(const crow::json::rvalue & value) {
    // Crow exposes JSON integers as int64_t. Narrowing first and validating the
    // int32_t afterwards lets values such as 4294967297 wrap to 1 and bypass
    // request limits. Range-check before the cast instead.
    const int64_t v = value.i();
    if (v < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
        v > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::out_of_range("JSON integer is outside int32 range");
    }
    return static_cast<int32_t>(v);
}

constexpr size_t MAX_JSON_REQUEST_BYTES = 8u * 1024u * 1024u;

struct ScopedTempPath {
    std::string path;
    ~ScopedTempPath() {
        if (path.empty()) return;
#ifdef _WIN32
        std::wstring wpath;
        if (utf8_to_wide_path(path, wpath)) DeleteFileW(wpath.c_str());
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
    auto has_non_ws = [](const std::string & value) {
        return std::any_of(value.begin(), value.end(), [](unsigned char c) { return !std::isspace(c); });
    };
    if (require_text && !has_non_ws(p.text)) return fail("text must not be empty or whitespace-only");
    if (!p.prompt_audio_path.empty() && !has_non_ws(p.prompt_text))
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
    if (p.gen.ras_window_size < 0 || p.gen.ras_window_size > 32768)
        return fail("ras_window must be between 0 and 32768");
    if (!std::isfinite(p.gen.ras_high_temp) || p.gen.ras_high_temp < 0.0f || p.gen.ras_high_temp > 10.0f)
        return fail("ras_temp must be finite and between 0 and 10");
    if (!std::isfinite(p.gen.ras_high_top_p) || p.gen.ras_high_top_p <= 0.0f || p.gen.ras_high_top_p > 1.0f)
        return fail("ras_top_p must be in (0, 1]");
    if (p.codec_chunk_frames < 0) return fail("codec_chunk must be >= 0");
    if (p.codec_overlap_frames < 0) return fail("codec_overlap must be >= 0");
    if (p.min_seg_chars < 0 || p.min_seg_chars > 1000000) return fail("min_seg_chars must be between 0 and 1000000");
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
    params.codec_overlap_frames  = 0;     // 0 = sin overlap (recomendado con VRAM ajustada)
    params.min_seg_chars         = 0;     // 0 = sin filtro de longitud minima
    // GenerateParams defaults (tambien en s2_generate.h)
    params.gen.temperature       = 0.7f;
    params.gen.top_p             = 0.7f;
    params.gen.top_k             = 30;
    params.gen.min_tokens_before_end = 64;
    params.gen.ras_window_size   = 10;
    params.gen.ras_high_temp     = 1.0f;
    params.gen.ras_high_top_p    = 0.9f;
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
        } else if ((arg == "--temperature" || arg == "--temp") && i + 1 < argc) {
            params.gen.temperature = parse_float_arg(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            params.gen.top_p = parse_float_arg(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            params.gen.top_k = parse_int_arg(argv[++i]);
        } else if (arg == "--min-end-tokens" && i + 1 < argc) {
            params.gen.min_tokens_before_end = parse_int_arg(argv[++i]);
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
R"(s2 -- Fish Speech TTS server + CLI  (CPU / Vulkan / CUDA / Metal)
HTTP+WebSocket server and CLI tool for local voice cloning with Fish Speech models.

QUICK START:
  Server -- CPU (works everywhere):
    s2.exe --model s2-pro-q4_k_m-transformer-only.gguf \
           --model-codec s2-pro-q4_k_m-codec-only.gguf

  Server -- RTX 3050 laptop (4 GB VRAM, iGPU on index 0):
    s2.exe --model s2-pro-q4_k_m-transformer-only.gguf \
           --model-codec s2-pro-q4_k_m-codec-only.gguf \
           -v 1 --codec-vulkan 1 --segment --codec-chunk 32 \
           --max-seg-tokens 300 --min-seg-chars 60 \
           --temperature 0.8 --top-p 0.8 --top-k 40

  CLI -- synthesize once to a file (no server):
    s2.exe --model ... --model-codec ... -v 1 --codec-vulkan 1 \
           --prompt-audio ref.wav --prompt-text "Reference transcript." \
           --text "Hello, this is a cloned voice." \
           --trim-silence --output hello.wav

  Save a voice profile, then reuse it by name:
    s2.exe --model ... --model-codec ... -v 1 --codec-vulkan 1 \
           --prompt-audio ref.wav --prompt-text "Reference transcript." \
           --voice my_voice --save-voice
    s2.exe --model ... --model-codec ... -v 1 --codec-vulkan 1 \
           --voice my_voice --text "Hello from saved voice." --output out.wav

  List saved voice profiles:
    s2.exe --list-voices

OPTIONS:

  Models:
    -m,  --model <path>          Path to the transformer GGUF model file.
                                 Default: model.gguf next to the executable.
         --model-codec <path>    Path to the codec GGUF model file.
                                 Default: codec.gguf next to the executable.
    -t,  --tokenizer <path>      Path to tokenizer.json.
                                 Default: tokenizer embedded inside the exe.

  GPU / device selection:
    -v,  --vulkan <N>            GPU device index for the transformer model.
                                 In Vulkan builds this selects the Vulkan device.
                                 In CUDA builds this selects the CUDA device.
                                 In Metal builds, 0 enables the Metal device.
                                   -1  CPU (default -- works on any machine)
                                    0  first GPU
                                    1  second GPU (use on laptops where
                                       index 0 is the Intel/AMD iGPU)
         --codec-vulkan <N>      Codec device index (legacy option name).
                                  -2  inherit transformer device (default)
                                  -1  force codec to CPU
                                  >=0 select that Vulkan/CUDA GPU; Metal uses device 0
                                 Note: q4_k_m codec only works on GPU; CPU
                                 fallback requires f16 or f32 codec weights.

  NOTE FOR CUDA BUILDS (s2-cuda.exe):
    -v and --codec-vulkan select transformer and codec devices independently.
    Example: model+codec on the first NVIDIA GPU:
      s2-cuda.exe --model ... --model-codec ... -v 0 --segment --codec-chunk 32
    Example: transformer on CUDA, codec forced to CPU:
      s2-cuda.exe --model ... --model-codec ... -v 0 --codec-vulkan -1

  Server:
    -p,  --port <N>              HTTP port to listen on. Default: 8080.
         --host <IP>             Bind IP address. Default: 127.0.0.1 (local only).
                                 Use 0.0.0.0 only if you intentionally want LAN access.

  Reference audio (voice cloning):
    -pa, --prompt-audio <path>   Path to reference WAV/MP3 for voice cloning.
    -pt, --prompt-text <text>    Transcript of the reference audio.
                                 Required when using --save-voice.
                                 Helps the model align prosody to the reference.

  Voice profiles (save and reuse encoded reference voices):
         --voice <id>            Load a saved voice profile by name.
                                 Skips re-encoding the reference audio.
                                 Ignored if --prompt-audio is also given.
         --save-voice            Encode --prompt-audio and save it as a profile.
                                 Requires --voice <id>, --prompt-audio,
                                 and --prompt-text.
         --voice-dir <path>      Directory for .s2voice profile files.
                                 Default: voices/ next to the executable.
         --list-voices           List saved voice profiles and exit.

  CLI output (no HTTP server):
    -o,  --output <path>         Synthesize once, write WAV to <path>, exit.
                                 Combine with --text for the input text, or
                                 pipe text to stdin if --text is omitted.
         --text <text>           Input text for CLI mode (--output).
         --trim-silence          Trim trailing silence from the output WAV.
         --no-trim-silence       Keep trailing silence in the output (default).

  Generation limits:
         --threads <N>           CPU threads for CPU-bound ops. Default: 4.
         --max-tokens <N>        Max tokens per request (no --segment).
                                 Default: 1024. One token ~= ~11 ms of audio.
         --max-seg-tokens <N>    Max tokens per sentence with --segment.
                                 Controls KV-cache size; lower = less VRAM.
                                 Default: 300 (~3.3 s per sentence).

  Segmentation (recommended for long texts or limited VRAM):
         --segment               Split text into sentences before generating.
                                 Each sentence uses its own KV cache, capping
                                 VRAM to the longest sentence, not the full text.
                                 Enable per-request: { "segment": true }
         --min-seg-chars <N>     Merge segments shorter than N characters with
                                 the next one. Prevents unnatural short clips.
                                 Default: 0 (no minimum). Recommended: 60-90.

  Codec chunking (advanced -- tune for your VRAM budget):
         --codec-chunk <N>       Codec frames decoded per GPU call. Smaller =
                                 less peak VRAM, slightly more overhead.
                                 Default: 0 (auto, ~120 frames).
                                 RTX 3050 4 GB with transformer loaded: use 32.
         --codec-overlap <N>     Overlap frames between codec chunks.
                                 Smooths chunk boundaries but costs more VRAM
                                 (graph scales with chunk+overlap). Keep at 0
                                 on 4 GB GPUs with transformer in VRAM.
                                 Default: 0.

  Sampling:
         --temperature <F>       Sampling temperature. Higher = more expressive,
                                 less stable. 0 = greedy. Default: 0.7. Range: 0-10.
         --top-p <F>             Nucleus sampling threshold.
                                 Default: 0.7. Range: 0.1-1.0.
         --top-k <N>             Top-k candidates before applying top-p.
                                 Default: 30.
         --min-end-tokens <N>    Min tokens before EOS is allowed. Prevents
                                 empty output on short texts. Default: 64.

  RAS -- Repetition Aware Sampling (anti-repetition):
    Detects when the model loops on a token and resamples it at higher
    temperature. Use if the output stutters or repeats syllables.
         --ras-window <N>        Recent-token window to watch. Default: 10.
         --ras-temp <F>          Temperature for the resample. Default: 1.0.
         --ras-top-p <F>         Top-p for the resample. Default: 0.9.

  Streaming (WebSocket /ws/tts):
         --stream-decode-stride <N>
                                 Codec decode cadence in frames. Lower values
                                 reduce first-chunk latency at the cost of more
                                 decode calls. 0 = auto (4 frames), -1 disables
                                 stride decoding. Default: 0.

  Other:
    -h,  --help                  Show this help and exit.

HTTP ENDPOINTS:
  POST /v1/tts                   Fish Audio-compatible TTS endpoint.
  POST /v1/audio/speech          OpenAI-compatible TTS endpoint.
  POST /synthesize               Legacy endpoint.
  GET  /v1/models                List available models.
  GET  /health                   Health check.

  Request body (JSON) -- all fields optional except "text":
    {
      "text":           "Text to synthesize.",
      "format":         "wav",
      "segment":        true,
      "prompt_text":    "Transcript of the reference audio.",
      "voice":          "my_voice",
      "temperature":    0.7,
      "top_p":          0.7,
      "top_k":          30,
      "min_end_tokens": 64,
      "ras_window":     10,
      "ras_temp":       1.0,
      "ras_top_p":      0.9,
      "codec_chunk":    32,
      "codec_overlap":  0,
      "min_seg_chars":  60,
      "trim_silence":   false,
      "stream_stride":  0
    }

WEBSOCKET ENDPOINT -- /ws/tts  (streaming, minimum latency):
  Send JSON, receive binary PCM frames as each sentence finishes.
  Binary message format: [2-byte flags LE][PCM int16 LE, mono, 44100 Hz]
    flags bit 0 = is_last (1 on the final segment)
  Final message (text): {"done": true, "segments": N, "sample_rate": 44100}

  Python example:
    import asyncio, json, websockets

    async def tts():
        async with websockets.connect("ws://localhost:8080/ws/tts") as ws:
            await ws.send(json.dumps({
                "text": "Hello world. How are you today?",
                "segment": True,
                "voice": "my_voice"
            }))
            while True:
                msg = await ws.recv()
                if isinstance(msg, str):
                    print(json.loads(msg))   # {"done": true, "segments": 2, ...}
                    break
                pcm = msg[2:]               # strip 2-byte flags header
                # feed pcm (int16, mono, 44100 Hz) to your audio output

HTTP EXAMPLES:
  Basic synthesis:
    curl -X POST http://localhost:8080/v1/tts \
         -H "Content-Type: application/json" \
         -d '{"text":"Hello world.","segment":true}' \
         --output audio.wav

  With a saved voice:
    curl -X POST http://localhost:8080/v1/tts \
         -H "Content-Type: application/json" \
         -d '{"text":"Hello.","voice":"my_voice","trim_silence":true}' \
         --output audio.wav

  Save a voice via HTTP:
    curl -X POST http://localhost:8080/v1/voices/my_voice \
         -H "Content-Type: application/json" \
         -d '{"audio_path":"C:/refs/speaker.wav","transcript":"Reference text."}'

  List saved voices:
    curl http://localhost:8080/v1/voices

  Delete a voice:
    curl -X DELETE http://localhost:8080/v1/voices/my_voice
)";
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
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_VULKAN)
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
              << "  Top-k:        " << params.gen.top_k << "\n"
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
    // Enforce the WebSocket JSON cap before Crow buffers an oversized message.
    // The per-message size check in onmessage remains as defense in depth.
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
        try {
            s2::PipelineParams synth_params = params;

            // Fish Audio uses "text"; OpenAI-compatible clients usually use "input".
            if (json.has("text")) {
                synth_params.text = json["text"].s();
            } else if (json.has("input")) {
                synth_params.text = json["input"].s();
            } else {
                return crow::response(400, "Missing 'text' or 'input' field");
            }

            if (json.has("temperature"))   synth_params.gen.temperature = static_cast<float>(json["temperature"].d());
            if (json.has("top_p"))         synth_params.gen.top_p = static_cast<float>(json["top_p"].d());
            if (json.has("top_k"))         synth_params.gen.top_k = checked_json_i32(json["top_k"]);
            if (json.has("threads"))       synth_params.gen.n_threads = checked_json_i32(json["threads"]);
            if (json.has("max_tokens"))    synth_params.gen.max_new_tokens = checked_json_i32(json["max_tokens"]);
            if (json.has("max_seg_tokens")) synth_params.max_tokens_per_segment = checked_json_i32(json["max_seg_tokens"]);
            if (json.has("segment"))       synth_params.segment_sentences = json["segment"].b();
            if (json.has("codec_chunk"))   synth_params.codec_chunk_frames = checked_json_i32(json["codec_chunk"]);
            if (json.has("codec_overlap")) synth_params.codec_overlap_frames = checked_json_i32(json["codec_overlap"]);
            if (json.has("min_seg_chars")) synth_params.min_seg_chars = checked_json_i32(json["min_seg_chars"]);
            if (json.has("min_end_tokens")) synth_params.gen.min_tokens_before_end = checked_json_i32(json["min_end_tokens"]);
            if (json.has("ras_window"))    synth_params.gen.ras_window_size = checked_json_i32(json["ras_window"]);
            if (json.has("ras_temp"))      synth_params.gen.ras_high_temp = static_cast<float>(json["ras_temp"].d());
            if (json.has("ras_top_p"))     synth_params.gen.ras_high_top_p = static_cast<float>(json["ras_top_p"].d());
            if (json.has("prompt_text"))   synth_params.prompt_text = json["prompt_text"].s();
            if (json.has("reference_audio")) synth_params.prompt_audio_path = json["reference_audio"].s();
            if (json.has("voice"))         synth_params.voice_id = json["voice"].s();
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
            if (json.has("format")) format = json["format"].s();
            else if (json.has("response_format")) format = json["response_format"].s();
            std::transform(format.begin(), format.end(), format.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (format != "wav" && format != "pcm") {
                return crow::response(400, "Unsupported format. Supported formats: wav, pcm");
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
            return crow::response(400, std::string("Invalid request: ") + e.what());
        } catch (...) {
            return crow::response(400, "Invalid request");
        }
    };

    // ================================================================
    // Route Fish Audio compatible : POST /v1/tts
    // ================================================================
    CROW_ROUTE(app, "/v1/tts")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        if (req.body.size() > MAX_JSON_REQUEST_BYTES) return crow::response(413, "JSON request body too large");
        auto json = crow::json::load(req.body);
        if (!json) {
            return crow::response(400, "Invalid JSON");
        }
        return do_synthesize(json);
    });

    // ================================================================
    // Route legacy : POST /synthesize (compatibilite avec vos tests)
    // ================================================================
    CROW_ROUTE(app, "/synthesize")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        if (req.body.size() > MAX_JSON_REQUEST_BYTES) return crow::response(413, "JSON request body too large");
        auto json = crow::json::load(req.body);
        if (!json) {
            return crow::response(400, "Invalid JSON");
        }
        return do_synthesize(json);
    });

    // ================================================================
    // Route OpenAI compatible : POST /v1/audio/speech
    // ================================================================
    CROW_ROUTE(app, "/v1/audio/speech")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        if (req.body.size() > MAX_JSON_REQUEST_BYTES) return crow::response(413, "JSON request body too large");
        auto json = crow::json::load(req.body);
        if (!json) {
            return crow::response(400, "Invalid JSON");
        }
        return do_synthesize(json);
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
            resp["count"] = static_cast<int>(ids.size());
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
        try {
            if (req.body.size() > MAX_JSON_REQUEST_BYTES) return crow::response(413, "JSON request body too large");
            auto json = crow::json::load(req.body);
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
            return crow::response(400, std::string("Invalid voice request: ") + e.what());
        } catch (const std::exception & e) {
            return crow::response(400, std::string("Invalid voice request: ") + e.what());
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
            return crow::response(404, std::string("Voice not found: ") + e.what());
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
        return crow::response(200, info);
    });

    // ================================================================
    // WebSocket /ws/tts -- streaming real por segmento de oracion.
    //
    // Protocolo (JSON sobre WebSocket):
    //
    //   Cliente -> Servidor:
    //     { "text": "...", "segment": true, "reference_audio": "path" }
    //
    //   Servidor -> Cliente (mensajes binarios):
    //     [2 bytes little-endian: flags] [PCM int16 LE, mono, 44100Hz]
    //     flags bit0 = is_last (1 si es el ultimo segmento)
    //
    //   Servidor -> Cliente (mensaje de texto al finalizar):
    //     { "done": true, "segments": N, "sample_rate": 44100 }
    //
    //   Servidor -> Cliente (en caso de error):
    //     { "error": "descripcion" }
    //
    // El cliente puede empezar a reproducir el primer mensaje binario
    // antes de que lleguen los siguientes -- latencia = primera oracion.
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
            auto json = crow::json::load(data);
            if (!json || !json.has("text")) {
                safe_send_text("{\"error\": \"missing 'text' field\"}");
                return;
            }

            s2::PipelineParams ws_params = params;
            ws_params.text = json["text"].s();
            if (json.has("segment"))          ws_params.segment_sentences = json["segment"].b();
            if (json.has("temperature"))      ws_params.gen.temperature = static_cast<float>(json["temperature"].d());
            if (json.has("top_p"))            ws_params.gen.top_p = static_cast<float>(json["top_p"].d());
            if (json.has("top_k"))            ws_params.gen.top_k = checked_json_i32(json["top_k"]);
            if (json.has("threads"))          ws_params.gen.n_threads = checked_json_i32(json["threads"]);
            if (json.has("max_tokens"))       ws_params.gen.max_new_tokens = checked_json_i32(json["max_tokens"]);
            if (json.has("max_seg_tokens"))   ws_params.max_tokens_per_segment = checked_json_i32(json["max_seg_tokens"]);
            if (json.has("reference_audio"))  ws_params.prompt_audio_path = json["reference_audio"].s();
            if (json.has("codec_chunk"))      ws_params.codec_chunk_frames = checked_json_i32(json["codec_chunk"]);
            if (json.has("codec_overlap"))    ws_params.codec_overlap_frames = checked_json_i32(json["codec_overlap"]);
            if (json.has("min_seg_chars"))    ws_params.min_seg_chars = checked_json_i32(json["min_seg_chars"]);
            if (json.has("min_end_tokens"))   ws_params.gen.min_tokens_before_end = checked_json_i32(json["min_end_tokens"]);
            if (json.has("ras_window"))       ws_params.gen.ras_window_size = checked_json_i32(json["ras_window"]);
            if (json.has("ras_temp"))         ws_params.gen.ras_high_temp = static_cast<float>(json["ras_temp"].d());
            if (json.has("ras_top_p"))        ws_params.gen.ras_high_top_p = static_cast<float>(json["ras_top_p"].d());
            if (json.has("prompt_text"))      ws_params.prompt_text = json["prompt_text"].s();
            if (json.has("voice"))            ws_params.voice_id = json["voice"].s();
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
                ok = pipeline.synthesize_streaming(ws_params, cb, &segment_count);
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
              << "  GET  /health\n"
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
