#include "s2_pipeline.h"
#include "s2_json.h"
#include "s2_utf8.h"
#include "base64.h"
#include "server_limits.h"
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
#include <string_view>
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
#include <condition_variable>
#include <future>
#include <functional>
#include <utility>
#include <deque>
#include <optional>
#include <cstdlib>
#include <cerrno>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <ws2tcpip.h>
#  include <process.h>
#elif defined(__APPLE__)
#  include <mach-o/dyld.h>
#  include <crt_externs.h>
#  include <limits.h>
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <spawn.h>
#  include <unistd.h>
#else
#  include <unistd.h>
#  include <limits.h>
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <spawn.h>
#endif

#if !defined(_WIN32) && !defined(__APPLE__)
extern char **environ;
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

static bool resolve_bind_host(const std::string & value, std::string & resolved) {
    resolved.clear();
    if (value.empty() || value.size() > 254) return false;
#ifdef _WIN32
    static std::once_flag winsock_once;
    static int winsock_rc = 0;
    std::call_once(winsock_once, [] {
        WSADATA wsa{};
        winsock_rc = WSAStartup(MAKEWORD(2, 2), &wsa);
    });
    if (winsock_rc != 0) return false;
#endif
    // Preserve literal-address behavior independently of AI_ADDRCONFIG. In
    // particular, ::1 must remain bindable on hosts without a global IPv6
    // address. DNS resolution is needed only for actual hostnames.
    in_addr ipv4{};
    if (inet_pton(AF_INET, value.c_str(), &ipv4) == 1) { resolved = value; return true; }
    in6_addr ipv6{};
    if (inet_pton(AF_INET6, value.c_str(), &ipv6) == 1) { resolved = value; return true; }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = 0;
    addrinfo * result = nullptr;
    if (getaddrinfo(value.c_str(), nullptr, &hints, &result) != 0 || !result) return false;

    char host[NI_MAXHOST] = {};
    bool ok = false;
    for (int pass = 0; pass < 2 && !ok; ++pass) {
        const int wanted = pass == 0 ? AF_INET : AF_INET6;
        for (addrinfo * ai = result; ai; ai = ai->ai_next) {
            if (ai->ai_family != wanted) continue;
            if (getnameinfo(ai->ai_addr, static_cast<int>(ai->ai_addrlen),
                            host, sizeof(host), nullptr, 0, NI_NUMERICHOST) == 0) {
                resolved = host;
                ok = true;
                break;
            }
        }
    }
    freeaddrinfo(result);
    return ok;
}

static bool is_loopback_address_literal(const std::string & value) noexcept {
    in_addr ipv4{};
    if (inet_pton(AF_INET, value.c_str(), &ipv4) == 1) {
        return reinterpret_cast<const unsigned char *>(&ipv4)[0] == 127u;
    }
    in6_addr ipv6{};
    if (inet_pton(AF_INET6, value.c_str(), &ipv6) == 1) {
        static const unsigned char loopback[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};
        return std::memcmp(&ipv6, loopback, sizeof(loopback)) == 0;
    }
    return false;
}

static bool api_token_is_safe(std::string_view value) noexcept {
    if (value.size() < 16u || value.size() > 4096u) return false;
    for (unsigned char c : value) if (c <= 0x20u || c == 0x7fu) return false;
    return true;
}

static bool constant_time_equal(std::string_view a, std::string_view b) noexcept {
    const size_t n = std::max(a.size(), b.size());
    unsigned int diff = static_cast<unsigned int>(a.size() ^ b.size());
    for (size_t i = 0; i < n; ++i) {
        const unsigned char av = i < a.size() ? static_cast<unsigned char>(a[i]) : 0u;
        const unsigned char bv = i < b.size() ? static_cast<unsigned char>(b[i]) : 0u;
        diff |= static_cast<unsigned int>(av ^ bv);
    }
    return diff == 0u;
}

static std::string ascii_lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

static bool request_has_json_content_type(const crow::request & req) {
    std::string value = req.get_header_value("Content-Type");
    if (value.empty()) return false;
    const size_t semi = value.find(';');
    if (semi != std::string::npos) value.resize(semi);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) ++first;
    value.erase(0, first);
    return ascii_lower_copy(value) == "application/json";
}

static bool request_has_bearer_token(const crow::request & req, std::string_view expected) noexcept {
    if (expected.empty()) return true;
    try {
        const std::string auth = req.get_header_value("Authorization");
        constexpr std::string_view prefix = "Bearer ";
        if (auth.size() <= prefix.size() || auth.compare(0, prefix.size(), prefix.data(), prefix.size()) != 0) return false;
        return constant_time_equal(std::string_view(auth).substr(prefix.size()), expected);
    } catch (...) { return false; }
}

static bool split_authority_host(const std::string & authority, std::string & host) {
    host.clear();
    if (authority.empty() || authority.find('@') != std::string::npos) return false;
    if (authority.front() == '[') {
        const size_t close = authority.find(']');
        if (close == std::string::npos) return false;
        host = authority.substr(1, close - 1);
        if (close + 1 < authority.size() && authority[close + 1] != ':') return false;
    } else {
        const size_t first_colon = authority.find(':');
        const size_t last_colon = authority.rfind(':');
        if (first_colon != std::string::npos && first_colon != last_colon) return false;
        host = authority.substr(0, first_colon);
    }
    host = ascii_lower_copy(host);
    while (!host.empty() && host.back() == '.') host.pop_back();
    return !host.empty();
}

static bool request_origin_allowed(const crow::request & req, bool loopback_bind) {
    const std::string origin = req.get_header_value("Origin");
    if (origin.empty()) return true;
    const std::string lower = ascii_lower_copy(origin);
    size_t start = std::string::npos;
    if (lower.rfind("http://", 0) == 0) start = 7;
    else if (lower.rfind("https://", 0) == 0) start = 8;
    else return false;
    const size_t end = origin.find('/', start);
    const std::string authority = origin.substr(start, end == std::string::npos ? std::string::npos : end - start);
    const std::string request_host = req.get_header_value("Host");
    if (authority.empty() || request_host.empty() || ascii_lower_copy(authority) != ascii_lower_copy(request_host)) return false;
    if (!loopback_bind) return true;
    std::string host;
    if (!split_authority_host(authority, host)) return false;
    if (host == "localhost" || host == "::1") return true;
    in_addr ipv4{};
    return inet_pton(AF_INET, host.c_str(), &ipv4) == 1 && reinterpret_cast<const unsigned char *>(&ipv4)[0] == 127u;
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

static bool checked_json_bool(const crow::json::rvalue & value, const char * field) {
    if (value.t() != crow::json::type::True && value.t() != crow::json::type::False)
        throw std::invalid_argument(std::string(field) + " must be a boolean");
    return value.b();
}

static bool checked_json_memory_cache(const crow::json::rvalue & value) {
    if (value.t() == crow::json::type::True || value.t() == crow::json::type::False)
        return value.b();
    if (value.t() != crow::json::type::String)
        throw std::invalid_argument("use_memory_cache must be 'on'/'off' or boolean");
    std::string mode = value.s();
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode == "on" || mode == "true" || mode == "1") return true;
    if (mode == "off" || mode == "false" || mode == "0") return false;
    throw std::invalid_argument("use_memory_cache must be 'on' or 'off'");
}

static void parse_inline_references(const crow::json::rvalue & refs, s2::PipelineParams & params) {
    if (refs.t() != crow::json::type::List)
        throw std::invalid_argument("Fish field 'references' must be an array");
    if (refs.size() > 8)
        throw std::invalid_argument("Fish field 'references' supports at most 8 entries");
    params.inline_references.clear();
    size_t total_audio = 0, total_text = 0;
    for (size_t i = 0; i < refs.size(); ++i) {
        const auto & ref = refs[i];
        if (ref.t() != crow::json::type::Object)
            throw std::invalid_argument("each references[] entry must be an object");
        if (!ref.has("audio") || !ref.has("text"))
            throw std::invalid_argument("each references[] entry requires audio and text");
        if (ref["audio"].t() != crow::json::type::String || ref["text"].t() != crow::json::type::String)
            throw std::invalid_argument("references[].audio/text must be strings in JSON");
        s2::InlineReference item;
        const std::string encoded = ref["audio"].s();
        if (!s2::base64_decode(encoded, item.audio) || item.audio.empty())
            throw std::invalid_argument("references[].audio must be non-empty canonical base64 audio bytes");
        item.text = ref["text"].s();
        if (!s2::utf8::is_valid(item.text) || !s2::utf8::has_non_whitespace(item.text))
            throw std::invalid_argument("references[].text must be non-empty valid UTF-8");
        if (item.audio.size() > 4u * 1024u * 1024u)
            throw std::invalid_argument("one inline reference exceeds the 4 MiB decoded-audio limit");
        if (item.text.size() > 1024u * 1024u)
            throw std::invalid_argument("one inline reference transcript exceeds 1 MiB");
        if (total_audio > 6u * 1024u * 1024u - item.audio.size())
            throw std::invalid_argument("combined inline reference audio exceeds 6 MiB");
        if (total_text > 1024u * 1024u - item.text.size())
            throw std::invalid_argument("combined inline reference transcripts exceed 1 MiB");
        total_audio += item.audio.size();
        total_text += item.text.size();
        params.inline_references.push_back(std::move(item));
    }
}

static void apply_fish_fields(const crow::json::rvalue & json, s2::PipelineParams & p, bool websocket) {
    if (json.has("early_stop_threshold")) {
        if (json["early_stop_threshold"].t() != crow::json::type::Number)
            throw std::invalid_argument("early_stop_threshold must be a number");
        p.gen.early_stop_threshold = static_cast<float>(json["early_stop_threshold"].d());
    }
    if (json.has("normalize")) p.normalize_text = checked_json_bool(json["normalize"], "normalize");
    if (json.has("sample_rate")) p.output_sample_rate = checked_json_i32(json["sample_rate"]);
    if (json.has("use_memory_cache")) p.reference_memory_cache = checked_json_memory_cache(json["use_memory_cache"]);
    if (json.has("references")) parse_inline_references(json["references"], p);

    if (json.has("prosody")) {
        const auto & pr = json["prosody"];
        if (pr.t() != crow::json::type::Object) throw std::invalid_argument("prosody must be an object");
        if (pr.has("speed")) {
            if (pr["speed"].t() != crow::json::type::Number) throw std::invalid_argument("prosody.speed must be a number");
            p.prosody_speed = static_cast<float>(pr["speed"].d());
        }
        if (pr.has("volume")) {
            if (pr["volume"].t() != crow::json::type::Number) throw std::invalid_argument("prosody.volume must be a number");
            p.prosody_volume_db = static_cast<float>(pr["volume"].d());
        }
        if (pr.has("normalize_loudness"))
            p.normalize_loudness = checked_json_bool(pr["normalize_loudness"], "prosody.normalize_loudness");
    }

    if (json.has("latency")) {
        if (json["latency"].t() != crow::json::type::String)
            throw std::invalid_argument("latency must be normal, balanced, or low");
        std::string latency = json["latency"].s();
        std::transform(latency.begin(), latency.end(), latency.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (latency != "normal" && latency != "balanced" && latency != "low")
            throw std::invalid_argument("latency must be normal, balanced, or low");
        if (!websocket && latency != "normal")
            throw std::invalid_argument("latency=balanced/low is supported by the local low-latency WebSocket path; buffered HTTP supports latency=normal");
        if (websocket && !json.has("stream_stride"))
            p.stream_decode_stride_frames = latency == "low" ? 2 : (latency == "balanced" ? 4 : 8);
    }
}

static void validate_fish_json_subset(const crow::json::rvalue & json, bool http_buffered) {
    if (json.has("streaming")) {
        const auto & streaming = json["streaming"];
        if (streaming.t() != crow::json::type::True && streaming.t() != crow::json::type::False)
            throw std::invalid_argument("Fish field 'streaming' must be a boolean");
        if (http_buffered && streaming.b())
            throw std::invalid_argument("HTTP streaming=true is not supported; use /ws/tts for streaming audio");
        if (!http_buffered && !streaming.b())
            throw std::invalid_argument("streaming=false is incompatible with /ws/tts; use an HTTP synthesis endpoint");
    }
    if (http_buffered && json.has("stream_stride"))
        throw std::invalid_argument("stream_stride is WebSocket-only; use /ws/tts or --stream-decode-stride as its server default");
    if (!http_buffered && (json.has("format") || json.has("response_format") ||
                           json.has("mp3_bitrate") || json.has("opus_bitrate")))
        throw std::invalid_argument("format/response_format and encoded-audio bitrate fields are HTTP-only; /ws/tts streams PCM");
}

static crow::json::rvalue load_json_strict(const std::string & raw) {
    std::string normalized, error;
    if (!s2::json::normalize_surrogate_pairs(raw, normalized, &error))
        throw std::invalid_argument(error);
    return crow::json::load(normalized);
}

constexpr size_t MAX_JSON_REQUEST_BYTES = 8u * 1024u * 1024u;
constexpr size_t MAX_TEXT_REQUEST_BYTES = 1024u * 1024u;
constexpr uint64_t MAX_BATCH_REQUESTED_TOKENS = 32768u;
constexpr size_t MAX_BATCH_BASE64_BYTES = 128u * 1024u * 1024u;

static bool read_stdin_text_bounded(std::istream & input, std::string & out, std::string & error) {
    out.clear();
    error.clear();
    char buffer[8192];
    while (input) {
        input.read(buffer, static_cast<std::streamsize>(sizeof(buffer)));
        const std::streamsize got = input.gcount();
        if (got <= 0) break;
        const size_t n = static_cast<size_t>(got);
        if (n > MAX_TEXT_REQUEST_BYTES - out.size()) {
            error = "stdin text exceeds 1 MiB request limit";
            out.clear();
            return false;
        }
        out.append(buffer, n);
    }
    if (input.bad()) {
        error = "failed to read text from stdin";
        out.clear();
        return false;
    }
    return true;
}

static bool base64_encoded_size(size_t input_bytes, size_t & encoded_bytes) noexcept {
    if (input_bytes > std::numeric_limits<size_t>::max() - 2u) return false;
    const size_t groups = (input_bytes + 2u) / 3u;
    if (groups > std::numeric_limits<size_t>::max() / 4u) return false;
    encoded_bytes = groups * 4u;
    return true;
}

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
static uint64_t read_le_u64(const unsigned char * p) {
    return static_cast<uint64_t>(read_le_u32(p)) |
           (static_cast<uint64_t>(read_le_u32(p + 4)) << 32);
}

static bool read_binary_file(const std::string & path, std::string & out) {
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
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    if (n < 0 || static_cast<uint64_t>(n) > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        static_cast<uint64_t>(n) > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) return false;
    out.resize(static_cast<size_t>(n));
    f.seekg(0, std::ios::beg);
    return out.empty() || static_cast<bool>(f.read(out.data(), static_cast<std::streamsize>(out.size())));
}

static bool read_generated_wave(const std::string & path, bool pcm_only, std::string & out,
                                uint32_t & sample_rate_out, bool & rf64_out) {
    out.clear(); sample_rate_out = 0; rf64_out = false;
    std::string bytes;
    if (!read_binary_file(path, bytes) || bytes.size() < 44u) return false;
    const auto * b = reinterpret_cast<const unsigned char *>(bytes.data());
    const bool riff = std::memcmp(b, "RIFF", 4) == 0;
    const bool rf64 = std::memcmp(b, "RF64", 4) == 0;
    if ((!riff && !rf64) || std::memcmp(b + 8, "WAVE", 4) != 0) return false;

    size_t fmt_off = 0, data_off = 0;
    uint64_t data_size64 = 0, riff_size64 = 0;
    bool have_ds64 = false;
    size_t pos = 12;
    while (pos + 8u <= bytes.size()) {
        const uint32_t chunk_size = read_le_u32(b + pos + 4u);
        const size_t payload = pos + 8u;
        if (payload > bytes.size()) return false;
        const bool rf64_data_sentinel = rf64 && std::memcmp(b + pos, "data", 4) == 0 && chunk_size == 0xffffffffu;
        if (!rf64_data_sentinel && static_cast<uint64_t>(chunk_size) > bytes.size() - payload) return false;
        if (std::memcmp(b + pos, "ds64", 4) == 0) {
            if (chunk_size < 28u) return false;
            riff_size64 = read_le_u64(b + payload);
            data_size64 = read_le_u64(b + payload + 8u);
            have_ds64 = true;
        } else if (std::memcmp(b + pos, "fmt ", 4) == 0) {
            if (chunk_size < 16u) return false;
            fmt_off = payload;
        } else if (std::memcmp(b + pos, "data", 4) == 0) {
            data_off = payload;
            if (!rf64) data_size64 = chunk_size;
            break;
        }
        const uint64_t next = static_cast<uint64_t>(payload) + chunk_size + (chunk_size & 1u);
        if (next > bytes.size()) return false;
        pos = static_cast<size_t>(next);
    }
    if (!fmt_off || !data_off || (rf64 && !have_ds64)) return false;
    if (fmt_off + 16u > bytes.size()) return false;
    const uint16_t audio_format = read_le_u16(b + fmt_off);
    const uint16_t channels = read_le_u16(b + fmt_off + 2u);
    const uint32_t sample_rate = read_le_u32(b + fmt_off + 4u);
    const uint32_t byte_rate = read_le_u32(b + fmt_off + 8u);
    const uint16_t block_align = read_le_u16(b + fmt_off + 12u);
    const uint16_t bits = read_le_u16(b + fmt_off + 14u);
    if (audio_format != 1u || channels != 1u || sample_rate == 0u || bits != 16u ||
        block_align != 2u || sample_rate > std::numeric_limits<uint32_t>::max() / 2u ||
        byte_rate != sample_rate * 2u || (data_size64 % 2u) != 0u) return false;
    if (data_size64 > bytes.size() - data_off) return false;
    if (data_off + data_size64 != bytes.size()) return false;
    if (riff) {
        if (read_le_u32(b + 4u) != bytes.size() - 8u) return false;
    } else {
        if (read_le_u32(b + 4u) != 0xffffffffu || riff_size64 != bytes.size() - 8u) return false;
    }
    sample_rate_out = sample_rate;
    rf64_out = rf64;
    if (pcm_only) out.assign(bytes.data() + data_off, static_cast<size_t>(data_size64));
    else out = std::move(bytes);
    return true;
}

static int run_process(const std::vector<std::string> & args) {
    if (args.empty()) return -1;
#ifdef _WIN32
    // Use the wide CRT spawn API so --ffmpeg, %TEMP% and generated paths do
    // not depend on the active ANSI code page.
    std::vector<std::wstring> wide_args;
    wide_args.reserve(args.size());
    for (const auto & a : args) {
        std::wstring w;
        if (!utf8_to_wide_path(a, w)) return -1;
        wide_args.push_back(std::move(w));
    }
    std::vector<const wchar_t *> argv;
    argv.reserve(wide_args.size() + 1u);
    for (const auto & a : wide_args) argv.push_back(a.c_str());
    argv.push_back(nullptr);
    return static_cast<int>(_wspawnvp(_P_WAIT, argv[0], argv.data()));
#else
    // fork()+non-async-signal-safe C++ work in the child can deadlock when the
    // HTTP server is multithreaded. posix_spawnp performs process creation
    // without running our allocator/iostream state in a post-fork child.
    std::vector<char *> argv;
    argv.reserve(args.size() + 1u);
    for (const auto & a : args) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    pid_t pid = -1;
    #  if defined(__APPLE__)
    char ** envp = *_NSGetEnviron();
#  else
    char ** envp = environ;
#  endif
    const int spawn_rc = posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), envp);
    if (spawn_rc != 0) { errno = spawn_rc; return -1; }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

static bool is_mono_ogg_opus(const std::string & payload) noexcept {
    if (payload.size() < 27u || payload.compare(0, 4, "OggS") != 0) return false;
    const size_t head = payload.find("OpusHead");
    if (head == std::string::npos || payload.size() - head < 19u) return false;
    const unsigned char version = static_cast<unsigned char>(payload[head + 8u]);
    const unsigned char channels = static_cast<unsigned char>(payload[head + 9u]);
    const unsigned char mapping_family = static_cast<unsigned char>(payload[head + 18u]);
    // RFC 7845 output from our -ac 1 encoder must be a mono Opus identification
    // header. Reject a generic Ogg stream or unexpected multichannel output.
    return version != 0u && version <= 15u && channels == 1u && mapping_family == 0u;
}

static bool encode_with_ffmpeg(const std::string & ffmpeg_bin, const std::string & wav_path,
                               const std::string & format, int bitrate_kbps, int output_rate,
                               std::string & payload, std::string & error) {
    payload.clear(); error.clear();
    const std::string ext = format == "mp3" ? ".mp3" : ".opus";
    ScopedTempPath encoded{wav_path + ext};
    std::vector<std::string> args = {
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", wav_path, "-map_metadata", "-1", "-vn", "-ac", "1",
        "-ar", std::to_string(output_rate)
    };
    if (bitrate_kbps != -1000) {
        args.push_back("-b:a");
        args.push_back(std::to_string(bitrate_kbps) + "k");
    }
    args.push_back("-f");
    args.push_back(format == "mp3" ? "mp3" : "opus");
    args.push_back(encoded.path);
    const int rc = run_process(args);
    if (rc != 0) {
        error = "ffmpeg failed for " + format + " (exit " + std::to_string(rc) + ")";
        return false;
    }
    if (!read_binary_file(encoded.path, payload) || payload.empty()) {
        error = "ffmpeg produced no readable " + format + " output";
        return false;
    }
    if (format == "mp3") {
        const bool id3 = payload.size() >= 3u && payload.compare(0, 3, "ID3") == 0;
        const bool frame = payload.size() >= 2u && static_cast<unsigned char>(payload[0]) == 0xffu &&
                           (static_cast<unsigned char>(payload[1]) & 0xe0u) == 0xe0u;
        if (!id3 && !frame) { error = "ffmpeg output is not recognizable MP3"; return false; }
    } else if (!is_mono_ogg_opus(payload)) {
        error = "ffmpeg output is not a mono Ogg/Opus stream with a valid OpusHead"; return false;
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
    if (p.text.size() > MAX_TEXT_REQUEST_BYTES) return fail("text exceeds 1 MiB request limit");
    if (p.prompt_text.size() > MAX_TEXT_REQUEST_BYTES) return fail("prompt_text exceeds 1 MiB request limit");
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
    if (!std::isfinite(p.gen.early_stop_threshold) ||
        (p.gen.early_stop_threshold != -1.0f && p.gen.early_stop_threshold != 1.0f))
        return fail("early_stop_threshold values that change legacy Fish multi-sample early-stop semantics require tensor-batched generation; this single-sample engine accepts only -1 (disabled) or 1.0 (neutral/all-finished)");
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
    if (p.chunk_length != 0 && (p.chunk_length < 100 || p.chunk_length > 1000))
        return fail("chunk_length must be 0 (off) or between 100 and 1000");
    if (p.min_chunk_length < 0 || p.min_chunk_length > 100)
        return fail("min_chunk_length must be between 0 and 100");
    if (p.chunk_length == 0 && p.min_chunk_length != 0)
        return fail("min_chunk_length requires chunk_length");
    if (!std::isfinite(p.prosody_volume_db) || p.prosody_volume_db < -20.0f || p.prosody_volume_db > 20.0f)
        return fail("prosody volume must be finite and between -20 and 20 dB");
    if (!std::isfinite(p.prosody_speed) || p.prosody_speed < 0.5f || p.prosody_speed > 2.0f)
        return fail("prosody speed must be finite and between 0.5 and 2.0");
    if (p.output_sample_rate != 0 && (p.output_sample_rate < 8000 || p.output_sample_rate > 192000))
        return fail("sample_rate must be 0 (native) or between 8000 and 192000");
    if (p.inline_references.size() > 8) return fail("at most 8 inline references are supported");
    if (p.stream_decode_stride_frames < -1 || p.stream_decode_stride_frames > 32768)
        return fail("stream_stride must be -1 (disabled), 0 (auto), or 1..32768");
    if (p.vulkan_device < -1) return fail("model device must be -1 (CPU) or >= 0");
    if (p.codec_vulkan_device < -2) return fail("codec device must be -2 (inherit), -1 (CPU), or >= 0");
    return true;
}

class BoundedTaskPool {
public:
    BoundedTaskPool(size_t thread_count, size_t max_queue)
        : max_queue_(std::max<size_t>(1u, max_queue)) {
        thread_count = std::max<size_t>(1u, thread_count);
        try {
            workers_.reserve(thread_count);
            for (size_t i = 0; i < thread_count; ++i)
                workers_.emplace_back([this] { worker_loop(); });
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            cv_.notify_all();
            for (auto & worker : workers_) if (worker.joinable()) worker.join();
            throw;
        }
    }

    BoundedTaskPool(const BoundedTaskPool &) = delete;
    BoundedTaskPool & operator=(const BoundedTaskPool &) = delete;
    ~BoundedTaskPool() { shutdown(); }

    template <class F>
    bool submit(F && fn) noexcept {
        try {
            std::function<void()> job(std::forward<F>(fn));
            if (!job) return false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stopping_ || queue_.size() >= max_queue_) return false;
                queue_.push_back(std::move(job));
            }
            cv_.notify_one();
            return true;
        } catch (...) {
            return false;
        }
    }

    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                // A constructor-failure cleanup may have set the flag already;
                // still join any threads owned by a fully-constructed object.
            }
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto & worker : workers_) if (worker.joinable()) worker.join();
        workers_.clear();
    }

private:
    void worker_loop() noexcept {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            try { job(); } catch (...) { /* task owns user-visible error handling */ }
        }
    }

    const size_t max_queue_;
    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
};

class PipelinePool {
public:
    class Lease {
    public:
        Lease() = default;
        Lease(PipelinePool * owner, size_t index) : owner_(owner), index_(index) {}
        Lease(const Lease &) = delete;
        Lease & operator=(const Lease &) = delete;
        Lease(Lease && other) noexcept : owner_(other.owner_), index_(other.index_) { other.owner_ = nullptr; }
        Lease & operator=(Lease && other) noexcept {
            if (this != &other) { release(); owner_ = other.owner_; index_ = other.index_; other.owner_ = nullptr; }
            return *this;
        }
        ~Lease() { release(); }
        explicit operator bool() const noexcept { return owner_ != nullptr; }
        s2::Pipeline * operator->() const { return owner_->pipelines_[index_]; }
        s2::Pipeline & operator*() const { return *owner_->pipelines_[index_]; }
    private:
        void release() {
            if (!owner_) return;
            owner_->release(index_);
            owner_ = nullptr;
        }
        PipelinePool * owner_ = nullptr;
        size_t index_ = 0;
    };

    explicit PipelinePool(std::vector<s2::Pipeline *> pipelines) : pipelines_(std::move(pipelines)) {
        for (size_t i = 0; i < pipelines_.size(); ++i) available_.push_back(i);
    }

    Lease acquire(const std::atomic_bool * keep_waiting = nullptr) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (keep_waiting) {
            while (available_.empty() && keep_waiting->load(std::memory_order_relaxed))
                cv_.wait_for(lock, std::chrono::milliseconds(50));
            if (!keep_waiting->load(std::memory_order_relaxed)) return Lease();
        } else {
            cv_.wait(lock, [&] { return !available_.empty(); });
        }
        const size_t index = available_.back();
        available_.pop_back();
        return Lease(this, index);
    }

    size_t capacity() const noexcept { return pipelines_.size(); }

private:
    friend class Lease;
    void release(size_t index) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            available_.push_back(index);
        }
        cv_.notify_one();
    }
    std::vector<s2::Pipeline *> pipelines_;
    std::vector<size_t> available_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

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
    params.prosody_speed         = 1.0f;
    params.normalize_loudness    = false;
    params.normalize_text        = false;
    params.output_sample_rate    = 0;
    params.output_rf64           = false;
    params.reference_memory_cache = true;
    // GenerateParams defaults (tambien en s2_generate.h)
    params.gen.temperature       = 0.7f;
    params.gen.top_p             = 0.7f;
    params.gen.top_k             = 30;
    params.gen.min_tokens_before_end = 64;
    params.gen.early_stop_threshold = -1.0f;
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
    int server_workers = 1;
    int request_rate_per_minute = 240;
    int request_burst = 60;
    int max_http_inflight = 16;
    int max_ws_connections = 64;
    std::string ffmpeg_bin = "ffmpeg";
    if (const char * env_ffmpeg = std::getenv("S2_FFMPEG"); env_ffmpeg && *env_ffmpeg) ffmpeg_bin = env_ffmpeg;
    // Crow defaults to 0.0.0.0. This API accepts local filesystem paths for
    // reference audio, so bind to loopback unless the user explicitly opts in.
    std::string bind_host = "127.0.0.1";
    bool allow_remote = false;
    bool list_voices = false;
    bool normalize_cli_explicit = false;

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
        } else if (arg == "--prosody-speed" && i + 1 < argc) {
            params.prosody_speed = parse_float_arg(argv[++i]);
        } else if (arg == "--normalize") {
            params.normalize_text = true;
            normalize_cli_explicit = true;
        } else if (arg == "--no-normalize") {
            params.normalize_text = false;
            normalize_cli_explicit = true;
        } else if (arg == "--normalize-loudness") {
            params.normalize_loudness = true;
        } else if (arg == "--no-normalize-loudness") {
            params.normalize_loudness = false;
        } else if (arg == "--sample-rate" && i + 1 < argc) {
            params.output_sample_rate = parse_int_arg(argv[++i]);
        } else if (arg == "--rf64") {
            params.output_rf64 = true;
        } else if (arg == "--no-rf64") {
            params.output_rf64 = false;
        } else if ((arg == "--temperature" || arg == "--temp") && i + 1 < argc) {
            params.gen.temperature = parse_float_arg(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            params.gen.top_p = parse_float_arg(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            params.gen.top_k = parse_int_arg(argv[++i]);
        } else if (arg == "--min-end-tokens" && i + 1 < argc) {
            params.gen.min_tokens_before_end = parse_int_arg(argv[++i]);
        } else if (arg == "--early-stop-threshold" && i + 1 < argc) {
            params.gen.early_stop_threshold = parse_float_arg(argv[++i]);
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
        } else if (arg == "--allow-remote") {
            allow_remote = true;
        } else if (arg == "--workers" && i + 1 < argc) {
            server_workers = parse_int_arg(argv[++i]);
        } else if (arg == "--request-rate" && i + 1 < argc) {
            request_rate_per_minute = parse_int_arg(argv[++i]);
        } else if (arg == "--request-burst" && i + 1 < argc) {
            request_burst = parse_int_arg(argv[++i]);
        } else if (arg == "--max-http-inflight" && i + 1 < argc) {
            max_http_inflight = parse_int_arg(argv[++i]);
        } else if (arg == "--max-ws-connections" && i + 1 < argc) {
            max_ws_connections = parse_int_arg(argv[++i]);
        } else if (arg == "--ffmpeg" && i + 1 < argc) {
            ffmpeg_bin = argv[++i];
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
  s2 [options] --output out.wav        Synthesize once to RIFF/WAV (or RF64) and exit.
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
    --output <path> writes one PCM16 RIFF/WAV file (RF64 with --rf64) and exits
    without starting the server.
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
        --host <name-or-IP>       Server-only bind IPv4/IPv6 literal or DNS hostname.
                                  Default: 127.0.0.1. Hostnames resolve at server startup.
        --allow-remote            Explicitly permit a non-loopback --host. Remote binds
                                  also require S2_API_TOKEN + Bearer authentication.
                                  Localhost does not require this flag or a token.
        --workers <N>             Server-only independent pipeline replicas for
                                  concurrent inference. Default: 1. Range: 1..16. Model/codec
                                  memory usage scales roughly with worker count.
        --request-rate <N>        Process-wide accepted-request budget per minute.
                                  Default: 240. Range: 1..100000. Applies to HTTP and WS messages.
        --request-burst <N>       Token-bucket burst capacity. Default: 60. Range: 1..10000.
        --max-http-inflight <N>   Maximum simultaneous synthesis/voice-save HTTP requests.
                                  Default: 16. Range: 1..4096. Excess requests get HTTP 503.
        --max-ws-connections <N>  Maximum accepted WebSocket connections. Default: 64.
                                  Range: 1..4096. Excess handshakes get HTTP 503.
        --ffmpeg <path>           Server-only ffmpeg executable used for HTTP MP3/Opus output.
                                  Default: S2_FFMPEG or ffmpeg from PATH.

  One-shot input/output:
        --text <text>             Input for --output mode. Max: 1 MiB.
                                  If omitted with --output, text is read from stdin.
    -o, --output <path>           Write PCM16 RIFF/WAV (RF64 with --rf64) and exit;
                                  no server is started.
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
                                  Range when enabled: 100..1000. Peak PCM RAM stays
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
        --prosody-speed <F>       WSOLA-style tempo change. Default: 1.0.
                                  Range: 0.5..2.0; tiny clips use interpolation.
        --normalize               Normalize Unicode whitespace in request text.
        --no-normalize            Disable request-text normalization (default).
        --normalize-loudness      Normalize final RMS loudness before gain.
        --no-normalize-loudness   Disable loudness normalization (default).
        --sample-rate <N>         Resample output. 0 = codec-native (default),
                                  otherwise 8000..192000 Hz.
        --rf64                    Write RF64 rather than RIFF/WAV in one-shot/file paths.
        --no-rf64                 Use classic RIFF/WAV output (default).

  Sampling:
        --temperature <F>         Sampling temperature. Alias: --temp.
        --temp <F>                Default: 0.7. Range: finite 0..10; 0 = greedy.
        --top-p <F>               Nucleus threshold. Default: 0.7. Range: (0, 1].
        --top-k <N>               Top-k cutoff. Default: 30. Range: 0..1000000.
        --min-end-tokens <N>      Minimum generated tokens before EOS is allowed.
                                  Default: 64. Range: 0..32768 and must be
                                  strictly smaller than --max-tokens. It is also
                                  clamped to the effective generation budget.
        --early-stop-threshold <F>
                                  Legacy Fish compatibility. -1 disables (default);
                                  1.0 is neutral/all-finished. Fractional values
                                  require true multi-sample tensor batching and are
                                  rejected rather than reinterpreted.
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
        --codec-overlap <N>       Decoder left-history frames for streaming boundaries.
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
  POST /v1/tts/batch             Batch synthesis (1..32 items).
                                  JSON returns one base64 audio output per item;
                                  decode to a complete WAV/RF64/MP3/Ogg-Opus file
                                  (PCM format returns raw mono PCM16 bytes).
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
    curl -X POST http://127.0.0.1:8080/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"s2-pro-local","input":"Hello.","response_format":"wav"}' -o output.wav
    model is optional, but if present must be s2-pro-local. voice selects a
    locally saved voice. instructions and stream_format are rejected because
    this local OpenAI-style subset does not implement them.

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
  early_stop_threshold, normalize, sample_rate, use_memory_cache, references[],
  prosody (volume, speed, normalize_loudness), latency, trim_silence, and the
  Fish streaming route-consistency flag.
  HTTP only: format/response_format = wav|pcm|rf64|mp3|opus plus mp3_bitrate
  and opus_bitrate. Buffered HTTP defaults to WAV even when CLI --rf64 is set;
  request format=rf64 explicitly when needed. stream_stride is rejected. Buffered HTTP accepts only
  latency=normal. MP3 supports 64/128/192 kbps and rates 8/11.025/12/16/
  22.05/24/32/44.1/48 kHz. Opus supports -1000(auto)/24/32/48/64 kbps and
  48 kHz output. MP3/Opus use optional ffmpeg.
  WebSocket only: stream_stride and latency=balanced/low. Explicit balanced/low
  or stream_stride>=0 cannot be combined with speed!=1, loudness normalization,
  or non-native sample_rate because those need whole-segment postprocessing.
  format/response_format and encoded-audio bitrates are rejected because
  /ws/tts always emits framed mono PCM int16. Inline references[] accept base64
  audio plus transcript text; use_memory_cache controls reference-code caching.
  Explicit WS latency maps normal/balanced/low to 8/4/2 codec frames unless
  stream_stride is supplied; generic stream_stride=0 remains auto 4 frames.
  Fish requests default normalize=true unless CLI normalization was explicitly
  selected; use_memory_cache defaults off per Fish request.
  Application request/message limit: 8 MiB; text and prompt_text: 1 MiB each.
  JSON POST routes require Content-Type: application/json. Browser requests with
  Origin must be same-origin with Host. Localhost needs no token by default. A
  non-loopback bind requires BOTH --allow-remote and S2_API_TOKEN (16..4096
  visible non-space bytes); clients must send Authorization: Bearer <token>.
  Setting S2_API_TOKEN on loopback enables optional Bearer auth there too. Built-in
  request-rate/in-flight/WebSocket limits are process-wide safety rails, not per-IP
  fairness controls. Crow buffers HTTP bodies before route handlers, so use a reverse
  proxy for a true pre-buffer body limit, TLS, and per-client/IP rate/connection
  limits on LAN/public deployments. Crow 1.3.4 applies the WebSocket payload limit to the complete
  reassembled message, including fragmented messages; the WebSocket upgrade uses
  the same auth/Origin policy.
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

    // Network/server-only options must never block one-shot synthesis or the
    // save-only voice command. Validate and resolve them only when Crow will
    // actually be started. Pipeline/generation options remain validated for all
    // modes below.
    const bool will_start_server = params.output_path.empty() && !params.save_voice;
    std::string resolved_bind_host = bind_host;
    bool server_loopback_bind = true;
    std::string server_api_token;
    if (will_start_server) {
        if (port < 1 || port > 65535) {
            std::cerr << "Error: port must be between 1 and 65535.\n";
            return 1;
        }
        if (server_workers < 1 || server_workers > 16) {
            std::cerr << "Error: --workers must be between 1 and 16.\n";
            return 1;
        }
        if (request_rate_per_minute < 1 || request_rate_per_minute > 100000) {
            std::cerr << "Error: --request-rate must be between 1 and 100000 per minute.\n";
            return 1;
        }
        if (request_burst < 1 || request_burst > 10000) {
            std::cerr << "Error: --request-burst must be between 1 and 10000.\n";
            return 1;
        }
        if (max_http_inflight < 1 || max_http_inflight > 4096) {
            std::cerr << "Error: --max-http-inflight must be between 1 and 4096.\n";
            return 1;
        }
        if (max_ws_connections < 1 || max_ws_connections > 4096) {
            std::cerr << "Error: --max-ws-connections must be between 1 and 4096.\n";
            return 1;
        }
        if (ffmpeg_bin.empty() || ffmpeg_bin.size() > 32768u) {
            std::cerr << "Error: --ffmpeg path/name must not be empty or excessively long.\n";
            return 1;
        }
        if (!resolve_bind_host(bind_host, resolved_bind_host)) {
            std::cerr << "Error: --host could not be resolved to a bindable IPv4/IPv6 address: " << bind_host << "\n";
            return 1;
        }
        server_loopback_bind = is_loopback_address_literal(resolved_bind_host);
        const char * token_env = std::getenv("S2_API_TOKEN");
        if (token_env && *token_env) server_api_token = token_env;
        if (!server_api_token.empty() && !api_token_is_safe(server_api_token)) {
            std::cerr << "Error: S2_API_TOKEN must be 16..4096 visible non-space bytes.\n";
            return 1;
        }
        if (!server_loopback_bind && !allow_remote) {
            std::cerr << "Error: non-loopback --host requires explicit --allow-remote.\n";
            return 1;
        }
        if (!server_loopback_bind && server_api_token.empty()) {
            std::cerr << "Error: --allow-remote with a non-loopback --host requires S2_API_TOKEN (16+ non-space bytes).\n";
            return 1;
        }
    }
    {
        std::string validation_error;
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
              << "  Codec GPU:     " << gpu_str(params.codec_vulkan_device) << "\n";
    if (will_start_server) {
        std::cout << "  Bind address:  " << bind_host << (resolved_bind_host != bind_host ? " -> " + resolved_bind_host : "") << "\n"
                  << "  Port:          " << port << "\n"
                  << "  Workers:       " << server_workers << " (independent pipeline replicas)\n";
    }
    std::cout << "  CPU threads:   " << params.gen.n_threads << "\n"
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
            // Read raw stdin incrementally so multiline input keeps its exact
            // line boundaries and cannot grow beyond the same 1 MiB text limit
            // enforced for HTTP/WS requests.
            std::cout << "Reading text from stdin (Ctrl+D to finish)...\n";
            std::string stdin_error;
            if (!read_stdin_text_bounded(std::cin, params.text, stdin_error)) {
                std::cerr << "Error: " << stdin_error << ".\n";
                return 1;
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

    // Server workers each own a complete mutable model/codec/KV state. This avoids
    // concurrent entry into one Pipeline while allowing true request-level parallelism.
    std::vector<std::unique_ptr<s2::Pipeline>> extra_pipelines;
    std::vector<s2::Pipeline *> server_pipelines;
    server_pipelines.push_back(&pipeline);
    for (int worker = 1; worker < server_workers; ++worker) {
        auto replica = std::make_unique<s2::Pipeline>();
        std::cout << "[Workers] Initializing pipeline " << (worker + 1) << "/" << server_workers << "...\n";
        if (!replica->init(params)) {
            std::cerr << "Worker pipeline initialization failed.\n";
            return 1;
        }
        if (params.warmup && !replica->warmup(params)) {
            std::cerr << "Worker pipeline warmup failed.\n";
            return 1;
        }
        server_pipelines.push_back(replica.get());
        extra_pipelines.push_back(std::move(replica));
    }
    PipelinePool pipeline_pool(std::move(server_pipelines));
    std::mutex voice_storage_mutex;

    // --- Serveur HTTP ---
    crow::SimpleApp app;

    // Respuestas > 1MB se streamean automaticamente (sin timeout).
    // Los WAV de audio suelen ser varios MB -- sin esto pueden cortar.
    app.stream_threshold(1024 * 1024); // 1 MB
    // Crow 1.3.4 enforces this limit against the complete reassembled WebSocket
    // message, including fragmented messages. onmessage keeps the same complete
    // JSON-size check as an application-level defense in depth.
    app.websocket_max_payload(MAX_JSON_REQUEST_BYTES);

    const bool server_auth_required = !server_api_token.empty();
    s2::server::TokenBucketRateLimiter request_rate_limiter(static_cast<size_t>(request_rate_per_minute),
                                                static_cast<size_t>(request_burst));
    std::atomic<size_t> active_http_expensive{0};
    std::atomic<size_t> accepted_ws_connections{0};
    auto enforce_http_security = [&](const crow::request & req, bool require_json) -> std::optional<crow::response> {
        if (server_auth_required && !request_has_bearer_token(req, server_api_token)) {
            crow::response res(401, "Unauthorized");
            res.set_header("WWW-Authenticate", "Bearer realm=\"s2.cpp\"");
            return res;
        }
        if (!request_origin_allowed(req, server_loopback_bind))
            return crow::response(403, "Cross-origin browser request rejected");
        if (require_json && !request_has_json_content_type(req))
            return crow::response(415, "Content-Type must be application/json");
        if (!request_rate_limiter.allow()) {
            crow::response res(429, "Request rate limit exceeded; retry later");
            res.set_header("Retry-After", "1");
            return res;
        }
        return std::nullopt;
    };

    // Crow invokes onmessage from its I/O context. Never run model inference in
    // that callback: send_binary/send_text post work back to the same context, so
    // blocking it would defeat incremental WebSocket delivery. Each connection
    // gets cancellation + busy state, while a bounded pool runs heavy synthesis.
    struct WsConnectionState {
        std::atomic_bool alive{true};
        std::atomic_bool busy{false};
        // onaccept reserves one slot before Crow constructs the connection.
        // This flag makes close/error/shutdown release that reservation exactly once.
        std::atomic_bool counted_connection{false};
        std::mutex send_mutex;
    };
    // Crow preserves onaccept userdata on the websocket connection. A sentinel
    // lets error/close release a connection slot even if the handshake fails
    // before onopen can register WsConnectionState.
    static int ws_reserved_slot_sentinel = 0;
    std::mutex ws_state_mutex;
    std::unordered_map<crow::websocket::connection *, std::shared_ptr<WsConnectionState>> ws_states;
    std::unique_ptr<BoundedTaskPool> ws_tasks;
    std::unique_ptr<BoundedTaskPool> batch_tasks;
    try {
        ws_tasks = std::make_unique<BoundedTaskPool>(pipeline_pool.capacity(), 32u);
        const size_t batch_queue = std::max<size_t>(32u, pipeline_pool.capacity() * 8u);
        batch_tasks = std::make_unique<BoundedTaskPool>(pipeline_pool.capacity(), batch_queue);
    } catch (const std::exception & e) {
        std::cerr << "Failed to initialize bounded server worker pools: " << e.what() << "\n";
        return 1;
    }
    auto mark_ws_closed = [&](crow::websocket::connection & conn) {
        std::shared_ptr<WsConnectionState> state;
        bool reserved_without_state = false;
        {
            // Serialize registry changes with onopen so a close/error cannot land
            // between consuming the userdata sentinel and publishing the state.
            std::lock_guard<std::mutex> lock(ws_state_mutex);
            auto it = ws_states.find(&conn);
            if (it != ws_states.end()) {
                state = it->second;
                ws_states.erase(it);
                conn.userdata(nullptr);
            } else if (conn.userdata() == &ws_reserved_slot_sentinel) {
                conn.userdata(nullptr);
                reserved_without_state = true;
            }
        }
        if (state) {
            {
                std::lock_guard<std::mutex> send_lock(state->send_mutex);
                state->alive.store(false, std::memory_order_relaxed);
            }
            if (state->counted_connection.exchange(false, std::memory_order_acq_rel))
                accepted_ws_connections.fetch_sub(1u, std::memory_order_acq_rel);
            return;
        }
        // onaccept may have reserved a slot immediately before a handshake error.
        // Crow stores this pointer on the connection before start(), so release it
        // exactly once even when onopen never registered state.
        if (reserved_without_state)
            accepted_ws_connections.fetch_sub(1u, std::memory_order_acq_rel);
    };
    auto mark_all_ws_closed = [&]() {
        std::vector<std::shared_ptr<WsConnectionState>> states;
        {
            std::lock_guard<std::mutex> lock(ws_state_mutex);
            states.reserve(ws_states.size());
            for (auto & entry : ws_states) states.push_back(entry.second);
            ws_states.clear();
        }
        // Never hold the registry mutex while waiting for an in-flight send.
        // Once alive=false, queued/running synthesis cancels without touching
        // Crow connection pointers after the server has stopped.
        for (auto & state : states) {
            if (!state) continue;
            std::lock_guard<std::mutex> send_lock(state->send_mutex);
            state->alive.store(false, std::memory_order_relaxed);
            if (state->counted_connection.exchange(false, std::memory_order_acq_rel))
                accepted_ws_connections.fetch_sub(1u, std::memory_order_acq_rel);
        }
    };

    enum class ApiFlavor { Legacy, Fish, OpenAI };

    // ================================================================
    // Helper : traitement commun de synthese
    // ================================================================
    auto do_synthesize = [&](const crow::json::rvalue& json, ApiFlavor flavor) -> crow::response {
        bool request_validated = false;
        try {
            if (json.t() != crow::json::type::Object) return crow::response(400, "JSON request must be an object");
            s2::PipelineParams synth_params = params;
            if (flavor == ApiFlavor::Fish) {
                // Match Fish request defaults without changing historical CLI,
                // legacy HTTP, or OpenAI-compatible defaults.
                if (!normalize_cli_explicit) synth_params.normalize_text = true;
                synth_params.reference_memory_cache = false;
            }
            validate_fish_json_subset(json, true);
            apply_fish_fields(json, synth_params, false);

            if (flavor == ApiFlavor::OpenAI) {
                if (json.has("model")) {
                    if (json["model"].t() != crow::json::type::String)
                        throw std::invalid_argument("OpenAI model must be a string");
                    if (json["model"].s() != "s2-pro-local")
                        throw std::invalid_argument("Unsupported OpenAI model; this local server exposes only s2-pro-local");
                }
                if (json.has("instructions"))
                    throw std::invalid_argument("OpenAI instructions is not implemented by this local synthesis subset");
                if (json.has("stream_format"))
                    throw std::invalid_argument("OpenAI stream_format is not implemented by this local synthesis subset");
                if (json.has("stream")) {
                    const bool stream = checked_json_bool(json["stream"], "stream");
                    if (stream) throw std::invalid_argument("OpenAI stream=true is not supported on buffered HTTP; use /ws/tts");
                }
            }
            if (flavor == ApiFlavor::OpenAI && json.has("speed")) {
                if (json["speed"].t() != crow::json::type::Number)
                    throw std::invalid_argument("speed must be a number");
                const float speed = static_cast<float>(json["speed"].d());
                if (json.has("prosody") && json["prosody"].t() == crow::json::type::Object && json["prosody"].has("speed")) {
                    const float nested = static_cast<float>(json["prosody"]["speed"].d());
                    if (std::fabs(speed - nested) > 1e-6f)
                        throw std::invalid_argument("conflicting speed and prosody.speed");
                }
                synth_params.prosody_speed = speed;
            }

            if (json.has("text") && json.has("input")) {
                const std::string a = json["text"].s();
                const std::string b = json["input"].s();
                if (a != b) throw std::invalid_argument("conflicting text and input");
                synth_params.text = a;
            } else if (json.has("text")) synth_params.text = json["text"].s();
            else if (json.has("input")) synth_params.text = json["input"].s();
            else return crow::response(400, "Missing 'text' or 'input' field");

            if (json.has("segment")) synth_params.segment_sentences = checked_json_bool(json["segment"], "segment");
            if (json.has("temperature")) synth_params.gen.temperature = static_cast<float>(json["temperature"].d());
            if (json.has("top_p")) synth_params.gen.top_p = static_cast<float>(json["top_p"].d());
            if (json.has("top_k")) synth_params.gen.top_k = checked_json_i32(json["top_k"]);
            if (json.has("seed")) synth_params.gen.seed = checked_json_fish_seed(json["seed"]);
            if (json.has("repetition_penalty")) synth_params.gen.repetition_penalty = static_cast<float>(json["repetition_penalty"].d());
            if (json.has("repetition_window")) synth_params.gen.repetition_window = checked_json_i32(json["repetition_window"]);
            if (json.has("multi_turn_history")) synth_params.multi_turn_history = checked_json_i32(json["multi_turn_history"]);
            if (json.has("threads")) {
                const int32_t requested_threads = checked_json_i32(json["threads"]);
                if (requested_threads > params.gen.n_threads)
                    throw std::invalid_argument("threads cannot exceed the server --threads limit");
                synth_params.gen.n_threads = requested_threads;
            }
            if (json.has("max_tokens") && json.has("max_new_tokens")) {
                const int32_t a = checked_json_i32(json["max_tokens"]);
                const int32_t b = checked_json_fish_max_new_tokens(json["max_new_tokens"]);
                if (a != b) throw std::invalid_argument("conflicting max_tokens and max_new_tokens");
                synth_params.gen.max_new_tokens = a;
            } else if (json.has("max_tokens")) synth_params.gen.max_new_tokens = checked_json_i32(json["max_tokens"]);
            else if (json.has("max_new_tokens")) synth_params.gen.max_new_tokens = checked_json_fish_max_new_tokens(json["max_new_tokens"]);
            if (json.has("max_seg_tokens")) synth_params.max_tokens_per_segment = checked_json_i32(json["max_seg_tokens"]);
            if (json.has("codec_chunk")) synth_params.codec_chunk_frames = checked_json_i32(json["codec_chunk"]);
            if (json.has("codec_overlap")) synth_params.codec_overlap_frames = checked_json_i32(json["codec_overlap"]);
            if (json.has("min_seg_chars")) synth_params.min_seg_chars = checked_json_i32(json["min_seg_chars"]);
            if (json.has("chunk_length")) synth_params.chunk_length = checked_json_i32(json["chunk_length"]);
            if (json.has("min_chunk_length")) synth_params.min_chunk_length = checked_json_i32(json["min_chunk_length"]);
            if (json.has("condition_on_previous_chunks"))
                synth_params.condition_on_previous_chunks = checked_json_bool(json["condition_on_previous_chunks"], "condition_on_previous_chunks");
            if (json.has("min_end_tokens")) synth_params.gen.min_tokens_before_end = checked_json_i32(json["min_end_tokens"]);
            if (json.has("ras_window")) synth_params.gen.ras_window_size = checked_json_i32(json["ras_window"]);
            if (json.has("ras_temp")) synth_params.gen.ras_high_temp = static_cast<float>(json["ras_temp"].d());
            if (json.has("ras_top_p")) synth_params.gen.ras_high_top_p = static_cast<float>(json["ras_top_p"].d());
            if (json.has("prompt_text")) synth_params.prompt_text = json["prompt_text"].s();
            if (json.has("reference_audio")) synth_params.prompt_audio_path = json["reference_audio"].s();
            const bool has_reference_id = json.has("reference_id") && json["reference_id"].t() != crow::json::type::Null;
            if (json.has("voice") && has_reference_id) {
                const std::string a = json["voice"].s(), b = json["reference_id"].s();
                if (a != b) throw std::invalid_argument("conflicting voice and reference_id");
                synth_params.voice_id = a;
            } else if (json.has("voice")) synth_params.voice_id = json["voice"].s();
            else if (has_reference_id) synth_params.voice_id = json["reference_id"].s();
            if (json.has("trim_silence")) synth_params.trim_silence = checked_json_bool(json["trim_silence"], "trim_silence");

            // HTTP defaults to WAV regardless of the one-shot/file --rf64 CLI flag.
            std::string format = "wav";
            auto normalized_format = [](std::string value) {
                std::transform(value.begin(), value.end(), value.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                return value;
            };
            if (json.has("format") && json.has("response_format")) {
                if (json["format"].t() != crow::json::type::String || json["response_format"].t() != crow::json::type::String)
                    throw std::invalid_argument("format and response_format must be strings");
                const std::string a = normalized_format(json["format"].s());
                const std::string b = normalized_format(json["response_format"].s());
                if (a != b) throw std::invalid_argument("conflicting format and response_format");
                format = a;
            } else if (json.has("format")) {
                if (json["format"].t() != crow::json::type::String) throw std::invalid_argument("format must be a string");
                format = normalized_format(json["format"].s());
            } else if (json.has("response_format")) {
                if (json["response_format"].t() != crow::json::type::String) throw std::invalid_argument("response_format must be a string");
                format = normalized_format(json["response_format"].s());
            }
            if (format != "wav" && format != "pcm" && format != "rf64" && format != "mp3" && format != "opus")
                return crow::response(400, "Unsupported format. Supported formats: wav, pcm, rf64, mp3, opus");

            int mp3_bitrate = 128, opus_bitrate = 32;
            auto valid_mp3_bitrate = [](int v) { return v == 64 || v == 128 || v == 192; };
            auto valid_opus_bitrate = [](int v) { return v == -1000 || v == 24 || v == 32 || v == 48 || v == 64; };
            auto valid_mp3_rate = [](int v) {
                switch (v) {
                    case 8000: case 11025: case 12000: case 16000: case 22050: case 24000:
                    case 32000: case 44100: case 48000: return true;
                    default: return false;
                }
            };
            if (json.has("mp3_bitrate")) {
                if (format != "mp3") throw std::invalid_argument("mp3_bitrate is valid only when format=mp3");
                mp3_bitrate = checked_json_i32(json["mp3_bitrate"]);
                if (!valid_mp3_bitrate(mp3_bitrate)) throw std::invalid_argument("mp3_bitrate must be 64, 128, or 192 kbps");
            }
            if (json.has("opus_bitrate")) {
                if (format != "opus") throw std::invalid_argument("opus_bitrate is valid only when format=opus");
                opus_bitrate = checked_json_i32(json["opus_bitrate"]);
                if (!valid_opus_bitrate(opus_bitrate)) throw std::invalid_argument("opus_bitrate must be -1000 (auto), 24, 32, 48, or 64 kbps");
            }
            if (format == "opus") {
                if (synth_params.output_sample_rate != 0 && synth_params.output_sample_rate != 48000)
                    throw std::invalid_argument("Opus output uses a 48000 Hz clock; sample_rate must be omitted/0 or 48000");
                synth_params.output_sample_rate = 48000;
            } else if (format == "mp3" && synth_params.output_sample_rate != 0 &&
                       !valid_mp3_rate(synth_params.output_sample_rate)) {
                throw std::invalid_argument("MP3 sample_rate must be one of 8000,11025,12000,16000,22050,24000,32000,44100,48000");
            }
            synth_params.output_rf64 = format == "rf64";

            std::string validation_error;
            if (!validate_pipeline_params(synth_params, validation_error, true)) return crow::response(400, validation_error);
            request_validated = true;

            if (!synth_params.voice_id.empty() && synth_params.prompt_audio_path.empty()) {
                std::lock_guard<std::mutex> storage_lock(voice_storage_mutex);
                s2::VoiceProfileManager mgr;
                mgr.set_storage_dir(synth_params.voice_storage_dir);
                if (!mgr.exists(synth_params.voice_id)) return crow::response(404, "Voice not found: " + synth_params.voice_id);
            }

            int32_t effective_rate = 0;
            std::string wav_path;
            {
                auto lease = pipeline_pool.acquire();
                effective_rate = lease->output_sample_rate(synth_params);
                if (format == "mp3" && !valid_mp3_rate(effective_rate))
                    return crow::response(400, "Effective MP3 sample rate is not representable by MPEG Layer III");
                if (!lease->synthesize_to_file(synth_params, wav_path))
                    return crow::response(500, "Synthesis failed (check requested voice/reference and server log)");
            } // Release scarce model/codec worker before file parsing or FFmpeg.
            ScopedTempPath temp_wav{wav_path};

            crow::response res;
            std::string payload;
            if (format == "mp3" || format == "opus") {
                std::string encode_error;
                const int bitrate = format == "mp3" ? mp3_bitrate : opus_bitrate;
                if (!encode_with_ffmpeg(ffmpeg_bin, wav_path, format, bitrate, effective_rate, payload, encode_error))
                    return crow::response(500, encode_error + ". Configure --ffmpeg or S2_FFMPEG if needed.");
                if (format == "mp3") {
                    res.set_header("Content-Type", "audio/mpeg");
                    res.set_header("Content-Disposition", "attachment; filename=\"audio.mp3\"");
                } else {
                    res.set_header("Content-Type", "audio/ogg; codecs=opus");
                    res.set_header("Content-Disposition", "attachment; filename=\"audio.opus\"");
                }
            } else {
                uint32_t parsed_rate = 0;
                bool parsed_rf64 = false;
                if (!read_generated_wave(wav_path, format == "pcm", payload, parsed_rate, parsed_rf64))
                    return crow::response(500, "Generated WAV/RF64 file is truncated or invalid");
                if (parsed_rate != static_cast<uint32_t>(effective_rate))
                    return crow::response(500, "Generated audio sample rate does not match the request");
                if (format == "rf64" && !parsed_rf64) return crow::response(500, "RF64 request produced RIFF output");
                if (format != "rf64" && parsed_rf64) return crow::response(500, "RIFF/PCM request unexpectedly produced RF64 output");
                if (format == "pcm") {
                    res.set_header("Content-Type", "audio/pcm");
                    res.set_header("X-Sample-Rate", std::to_string(parsed_rate));
                } else {
                    res.set_header("Content-Type", "audio/wav");
                    res.set_header("X-Wave-Container", parsed_rf64 ? "RF64" : "RIFF");
                    res.set_header("Content-Disposition", parsed_rf64 ? "attachment; filename=\"audio.rf64\"" : "attachment; filename=\"audio.wav\"");
                }
            }
            res.set_header("Content-Length", std::to_string(payload.size()));
            res.body = std::move(payload);
            return res;
        } catch (const std::bad_alloc &) {
            return crow::response(500, "Server ran out of memory while preparing the response");
        } catch (const std::exception & e) {
            if (request_validated) {
                std::cerr << "[HTTP] Synthesis internal error: " << e.what() << "\n";
                return crow::response(500, "Synthesis failed internally");
            }
            return crow::response(400, std::string("Invalid request: ") + e.what());
        } catch (...) {
            return crow::response(request_validated ? 500 : 400, request_validated ? "Synthesis failed" : "Invalid request");
        }
    };

    // Strict JSON normalization may throw before Crow sees the body (for example,
    // malformed UTF-16 surrogate escapes). Convert all client parse/type failures
    // into 4xx here instead of letting them escape the route callback.
    auto handle_synthesis_request = [&](const crow::request & req, ApiFlavor flavor) -> crow::response {
        if (auto blocked = enforce_http_security(req, true)) return std::move(*blocked);
        s2::server::AtomicPermit inflight(active_http_expensive, static_cast<size_t>(max_http_inflight));
        if (!inflight) return crow::response(503, "Too many simultaneous synthesis requests; retry later");
        if (req.body.size() > MAX_JSON_REQUEST_BYTES) {
            return crow::response(413, "JSON request body too large");
        }
        try {
            auto json = load_json_strict(req.body);
            if (!json) {
                return crow::response(400, "Invalid JSON");
            }
            return do_synthesize(json, flavor);
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
        return handle_synthesis_request(req, ApiFlavor::Fish);
    });

    CROW_ROUTE(app, "/v1/tts/batch")
    .methods("POST"_method)
    ([&](const crow::request & req) -> crow::response {
        if (auto blocked = enforce_http_security(req, true)) return std::move(*blocked);
        s2::server::AtomicPermit inflight(active_http_expensive, static_cast<size_t>(max_http_inflight));
        if (!inflight) return crow::response(503, "Too many simultaneous synthesis requests; retry later");
        if (req.body.size() > MAX_JSON_REQUEST_BYTES) return crow::response(413, "JSON request body too large");
        try {
            auto root = load_json_strict(req.body);
            if (!root || root.t() != crow::json::type::Object || !root.has("requests") ||
                root["requests"].t() != crow::json::type::List)
                return crow::response(400, "Batch body must be an object with a requests array");
            const auto & requests = root["requests"];
            if (requests.size() < 1 || requests.size() > 32)
                return crow::response(400, "Batch requests must contain 1..32 items");
            for (size_t i = 0; i < requests.size(); ++i)
                if (requests[i].t() != crow::json::type::Object)
                    return crow::response(400, "Each batch item must be a JSON object");

            const size_t count = requests.size();
            std::vector<crow::json::rvalue> items;
            std::vector<std::string> formats;
            items.reserve(count); formats.reserve(count);
            uint64_t requested_token_budget = 0;
            for (size_t i = 0; i < count; ++i) {
                items.emplace_back(requests[i]); // deep copy before worker threads

                // Batch is buffered and returns every item in one JSON document. Bound the
                // nominal generation budget before starting any worker so a tiny request body
                // cannot fan out into dozens of maximum-length generations at once. The
                // default 32 * 1024-token batch still fits exactly.
                int32_t item_max_tokens = params.gen.max_new_tokens;
                if (items.back().has("max_tokens") && items.back().has("max_new_tokens")) {
                    const int32_t a = checked_json_i32(items.back()["max_tokens"]);
                    const int32_t b = checked_json_fish_max_new_tokens(items.back()["max_new_tokens"]);
                    if (a != b) throw std::invalid_argument("conflicting max_tokens and max_new_tokens in batch item");
                    item_max_tokens = a;
                } else if (items.back().has("max_tokens")) {
                    item_max_tokens = checked_json_i32(items.back()["max_tokens"]);
                } else if (items.back().has("max_new_tokens")) {
                    item_max_tokens = checked_json_fish_max_new_tokens(items.back()["max_new_tokens"]);
                }
                if (item_max_tokens < 1 || item_max_tokens > 32768)
                    throw std::invalid_argument("batch item max_tokens/max_new_tokens must resolve to 1..32768");
                const uint64_t item_budget = static_cast<uint64_t>(item_max_tokens);
                if (requested_token_budget > MAX_BATCH_REQUESTED_TOKENS - item_budget)
                    return crow::response(413, "Batch aggregate generation budget exceeds 32768 requested tokens");
                requested_token_budget += item_budget;

                // Each buffered Fish item defaults to WAV; --rf64 is a file/one-shot default only.
                std::string format = "wav";
                if (items.back().has("format") && items.back()["format"].t() == crow::json::type::String) format = items.back()["format"].s();
                else if (items.back().has("response_format") && items.back()["response_format"].t() == crow::json::type::String) format = items.back()["response_format"].s();
                std::transform(format.begin(), format.end(), format.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                formats.push_back(std::move(format));
            }

            std::vector<std::unique_ptr<crow::response>> responses(count);
            std::atomic<size_t> batch_base64_bytes{0};
            std::mutex batch_wait_mutex;
            std::condition_variable batch_wait_cv;
            std::atomic<size_t> batch_remaining{count};
            auto finish_batch_item = [&]() noexcept {
                batch_remaining.fetch_sub(1u, std::memory_order_release);
                batch_wait_cv.notify_one();
            };
            for (size_t i = 0; i < count; ++i) {
                const bool queued = batch_tasks->submit([&, i] {
                    try {
                        auto response = std::make_unique<crow::response>(do_synthesize(items[i], ApiFlavor::Fish));
                        if (response->code >= 200 && response->code < 300) {
                            size_t encoded_bytes = 0;
                            bool reserved = base64_encoded_size(response->body.size(), encoded_bytes) && encoded_bytes <= MAX_BATCH_BASE64_BYTES;
                            size_t observed = batch_base64_bytes.load(std::memory_order_relaxed);
                            while (reserved) {
                                if (observed > MAX_BATCH_BASE64_BYTES - encoded_bytes) { reserved = false; break; }
                                if (batch_base64_bytes.compare_exchange_weak(observed, observed + encoded_bytes,
                                                                            std::memory_order_relaxed,
                                                                            std::memory_order_relaxed)) break;
                            }
                            if (!reserved) response = std::make_unique<crow::response>(413, "Batch aggregate base64 audio payload exceeds 128 MiB");
                        }
                        responses[i] = std::move(response);
                    } catch (...) {
                        // Keep this path allocation/iostream-free: finish_batch_item() must
                        // always run so the request cannot wait forever during low-memory
                        // or other exceptional worker failures. Details stay server-side.
                        std::fputs("[Batch] Worker failed while processing an item\n", stderr);
                    }
                    finish_batch_item();
                });
                if (!queued) {
                    // Do not let allocation failure here escape while already-submitted
                    // tasks still reference this handler's stack. Mark completion first
                    // even if the best-effort 503 response cannot be allocated.
                    try {
                        responses[i] = std::make_unique<crow::response>(503, "Batch worker queue is full; retry later");
                    } catch (...) {
                        std::fputs("[Batch] Could not allocate queue-full response\n", stderr);
                    }
                    finish_batch_item();
                }
            }
            {
                std::unique_lock<std::mutex> lock(batch_wait_mutex);
                batch_wait_cv.wait(lock, [&] { return batch_remaining.load(std::memory_order_acquire) == 0; });
            }

            crow::json::wvalue out;
            out["object"] = "tts.batch";
            out["count"] = static_cast<int64_t>(requests.size());
            std::vector<crow::json::wvalue> rows;
            rows.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                crow::json::wvalue row;
                if (!responses[i]) {
                    row["status"] = 500;
                    row["error"] = "batch worker produced no response";
                    rows.emplace_back(std::move(row));
                    continue;
                }
                crow::response & item = *responses[i];
                row["status"] = item.code;
                if (item.code >= 200 && item.code < 300) {
                    row["format"] = formats[i];
                    int32_t item_sample_rate = params.output_sample_rate > 0
                        ? params.output_sample_rate : pipeline.sample_rate();
                    if (items[i].has("sample_rate")) {
                        const int32_t requested_rate = checked_json_i32(items[i]["sample_rate"]);
                        if (requested_rate > 0) item_sample_rate = requested_rate;
                    }
                    if (formats[i] == "opus") item_sample_rate = 48000;
                    row["sample_rate"] = item_sample_rate;
                    size_t expected_base64_bytes = 0;
                    const bool size_ok = base64_encoded_size(item.body.size(), expected_base64_bytes);
                    std::string encoded = size_ok
                        ? s2::base64_encode(reinterpret_cast<const unsigned char *>(item.body.data()), item.body.size())
                        : std::string();
                    if (!size_ok || encoded.size() != expected_base64_bytes) {
                        row["status"] = 500;
                        row["error"] = "audio response could not be base64-encoded safely";
                    } else {
                        row["audio_base64"] = std::move(encoded);
                    }
                    // The JSON row now owns the encoded representation; release the raw
                    // binary response immediately instead of holding both copies until the
                    // complete batch object is serialized.
                    std::string().swap(item.body);
                } else {
                    row["error"] = item.body;
                }
                rows.emplace_back(std::move(row));
            }
            out["results"] = std::move(rows);
            return crow::response(200, out);
        } catch (const std::bad_alloc &) {
            return crow::response(500, "Server ran out of memory while processing batch");
        } catch (const std::system_error & e) {
            std::cerr << "[Batch] Runtime failure: " << e.what() << "\n";
            return crow::response(500, "Batch worker runtime failure");
        } catch (const std::exception & e) {
            return crow::response(400, std::string("Invalid batch request: ") + e.what());
        }
    });

    // ================================================================
    // Route legacy : POST /synthesize (compatibilite avec vos tests)
    // ================================================================
    CROW_ROUTE(app, "/synthesize")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        return handle_synthesis_request(req, ApiFlavor::Legacy);
    });

    // ================================================================
    // Route OpenAI compatible : POST /v1/audio/speech
    // ================================================================
    CROW_ROUTE(app, "/v1/audio/speech")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        return handle_synthesis_request(req, ApiFlavor::OpenAI);
    });

    // ================================================================
    // Health check & info
    // ================================================================
    CROW_ROUTE(app, "/v1/models")
    .methods("GET"_method)
    ([&](const crow::request & req) {
        if (auto blocked = enforce_http_security(req, false)) return std::move(*blocked);
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
    ([&](const crow::request & req) {
        if (auto blocked = enforce_http_security(req, false)) return std::move(*blocked);
        try {
            std::lock_guard<std::mutex> storage_lock(voice_storage_mutex);
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
            std::cerr << "[HTTP] Failed to list voices: " << e.what() << "\n";
            return crow::response(500, "Failed to list voices");
        }
    });

    // POST /v1/voices/<id> -- save a voice profile from a local reference audio.
    // Body JSON: { "transcript": "...", "audio_path": "/abs/path/to/ref.wav" }
    CROW_ROUTE(app, "/v1/voices/<string>")
    .methods("POST"_method)
    ([&](const crow::request& req, const std::string & voice_id) {
        if (auto blocked = enforce_http_security(req, true)) return std::move(*blocked);
        s2::server::AtomicPermit inflight(active_http_expensive, static_cast<size_t>(max_http_inflight));
        if (!inflight) return crow::response(503, "Too many simultaneous synthesis requests; retry later");
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

            std::vector<int32_t> codes;
            int32_t T_prompt = 0;
            int32_t num_codebooks = 0, codebook_size = 0, sample_rate = 0;
            {
                auto lease = pipeline_pool.acquire();
                if (!lease->encode_reference(vp, codes, T_prompt) || codes.empty() || T_prompt <= 0) {
                    return crow::response(400, "Failed to load/encode reference audio");
                }
                num_codebooks = lease->num_codebooks();
                codebook_size = lease->codebook_size();
                sample_rate = lease->sample_rate();
            }
            if (codes.size() != static_cast<size_t>(num_codebooks) * static_cast<size_t>(T_prompt))
                return crow::response(500, "Encoded voice has inconsistent dimensions");

            s2::VoiceProfile profile;
            profile.transcript = vp.prompt_text;
            profile.codes = std::move(codes);
            profile.T_prompt = T_prompt;
            profile.num_codebooks = num_codebooks;
            profile.codebook_size = codebook_size;
            profile.sample_rate = sample_rate;
            {
                std::lock_guard<std::mutex> storage_lock(voice_storage_mutex);
                s2::VoiceProfileManager mgr;
                mgr.set_storage_dir(params.voice_storage_dir);
                if (!mgr.save(voice_id, profile)) return crow::response(500, "Failed to save voice profile");
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
            if (!request_validated) return crow::response(400, std::string("Invalid voice request: ") + e.what());
            std::cerr << "[HTTP] Voice profile creation failed: " << e.what() << "\n";
            return crow::response(500, "Failed to create voice profile");
        } catch (const std::exception & e) {
            if (!request_validated) return crow::response(400, std::string("Invalid voice request: ") + e.what());
            std::cerr << "[HTTP] Voice profile creation failed: " << e.what() << "\n";
            return crow::response(500, "Failed to create voice profile");
        } catch (...) {
            return crow::response(request_validated ? 500 : 400,
                                  request_validated ? "Failed to create voice profile"
                                                    : "Invalid voice request");
        }
    });

    // GET /v1/voices/<id> -- get metadata for a single voice profile
    CROW_ROUTE(app, "/v1/voices/<string>")
    .methods("GET"_method)
    ([&](const crow::request & req, const std::string & voice_id) {
        if (auto blocked = enforce_http_security(req, false)) return std::move(*blocked);
        try {
            std::lock_guard<std::mutex> storage_lock(voice_storage_mutex);
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
            std::cerr << "[HTTP] Failed to load voice profile: " << e.what() << "\n";
            return crow::response(500, "Failed to load voice profile");
        }
    });

    // DELETE /v1/voices/<id> -- delete a saved voice profile
    CROW_ROUTE(app, "/v1/voices/<string>")
    .methods("DELETE"_method)
    ([&](const crow::request & req, const std::string & voice_id) {
        if (auto blocked = enforce_http_security(req, false)) return std::move(*blocked);
        try {
            std::lock_guard<std::mutex> storage_lock(voice_storage_mutex);
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
            std::cerr << "[HTTP] Failed to delete voice: " << e.what() << "\n";
            return crow::response(500, "Failed to delete voice");
        }
    });

    // ================================================================
    CROW_ROUTE(app, "/health")
    .methods("GET"_method)
    ([&](const crow::request & req) {
        if (auto blocked = enforce_http_security(req, false)) return std::move(*blocked);
        return crow::response(200, "OK");
    });

    // Fish Speech's documented local-server health endpoint. Keep /health as
    // the historical alias so existing s2.cpp deployments do not break.
    CROW_ROUTE(app, "/v1/health")
    .methods("GET"_method)
    ([&](const crow::request & req) {
        if (auto blocked = enforce_http_security(req, false)) return std::move(*blocked);
        crow::json::wvalue status;
        status["status"] = "ok";
        return crow::response(200, status);
    });

    CROW_ROUTE(app, "/")
    ([&](const crow::request & req) {
        if (auto blocked = enforce_http_security(req, false)) return std::move(*blocked);
        crow::json::wvalue info;
        info["status"] = "running";
        info["host"] = bind_host;
        info["port"] = port;
        info["endpoints"][0] = "/v1/tts";
        info["endpoints"][1] = "/v1/tts/batch";
        info["endpoints"][2] = "/synthesize";
        info["endpoints"][3] = "/v1/audio/speech";
        info["endpoints"][4] = "/v1/models";
        info["endpoints"][5] = "/v1/voices";
        info["endpoints"][6] = "/health";
        info["endpoints"][7] = "/v1/health";
        info["endpoints"][8] = "/v1/voices/<id>";
        info["endpoints"][9] = "/ws/tts";
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
    auto ws_send_text = [](crow::websocket::connection * conn,
                           const std::shared_ptr<WsConnectionState> & state,
                           const std::string & msg) -> bool {
        if (!conn || !state) return false;
        std::lock_guard<std::mutex> lock(state->send_mutex);
        if (!state->alive.load(std::memory_order_relaxed)) return false;
        try { conn->send_text(msg); return true; } catch (...) { return false; }
    };
    auto ws_send_binary = [](crow::websocket::connection * conn,
                             const std::shared_ptr<WsConnectionState> & state,
                             const std::string & msg) -> bool {
        if (!conn || !state) return false;
        std::lock_guard<std::mutex> lock(state->send_mutex);
        if (!state->alive.load(std::memory_order_relaxed)) return false;
        try { conn->send_binary(msg); return true; } catch (...) { return false; }
    };

    auto process_ws_message = [&](crow::websocket::connection * conn,
                                  const std::shared_ptr<WsConnectionState> & state,
                                  std::string data) {
        struct BusyReset {
            std::shared_ptr<WsConnectionState> state;
            ~BusyReset() { if (state) state->busy.store(false, std::memory_order_release); }
        } busy_reset{state};
        if (!state || !state->alive.load(std::memory_order_relaxed)) return;

        try {
            auto json = load_json_strict(data);
            if (!json || (!json.has("text") && !json.has("input"))) {
                ws_send_text(conn, state, "{\"error\": \"missing 'text' or 'input' field\"}");
                return;
            }

            s2::PipelineParams ws_params = params;
            // Fish defaults are route-local; do not mutate CLI/legacy defaults.
            if (!normalize_cli_explicit) ws_params.normalize_text = true;
            ws_params.reference_memory_cache = false;
            validate_fish_json_subset(json, false);
            apply_fish_fields(json, ws_params, true);
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
            if (json.has("segment")) ws_params.segment_sentences = checked_json_bool(json["segment"], "segment");
            if (json.has("temperature")) ws_params.gen.temperature = static_cast<float>(json["temperature"].d());
            if (json.has("top_p")) ws_params.gen.top_p = static_cast<float>(json["top_p"].d());
            if (json.has("top_k")) ws_params.gen.top_k = checked_json_i32(json["top_k"]);
            if (json.has("seed")) ws_params.gen.seed = checked_json_fish_seed(json["seed"]);
            if (json.has("repetition_penalty")) ws_params.gen.repetition_penalty = static_cast<float>(json["repetition_penalty"].d());
            if (json.has("repetition_window")) ws_params.gen.repetition_window = checked_json_i32(json["repetition_window"]);
            if (json.has("multi_turn_history")) ws_params.multi_turn_history = checked_json_i32(json["multi_turn_history"]);
            if (json.has("threads")) {
                const int32_t requested_threads = checked_json_i32(json["threads"]);
                if (requested_threads > params.gen.n_threads)
                    throw std::invalid_argument("threads cannot exceed the server --threads limit");
                ws_params.gen.n_threads = requested_threads;
            }
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
            if (json.has("max_seg_tokens")) ws_params.max_tokens_per_segment = checked_json_i32(json["max_seg_tokens"]);
            if (json.has("reference_audio")) ws_params.prompt_audio_path = json["reference_audio"].s();
            if (json.has("codec_chunk")) ws_params.codec_chunk_frames = checked_json_i32(json["codec_chunk"]);
            if (json.has("codec_overlap")) ws_params.codec_overlap_frames = checked_json_i32(json["codec_overlap"]);
            if (json.has("min_seg_chars")) ws_params.min_seg_chars = checked_json_i32(json["min_seg_chars"]);
            if (json.has("chunk_length")) ws_params.chunk_length = checked_json_i32(json["chunk_length"]);
            if (json.has("min_chunk_length")) ws_params.min_chunk_length = checked_json_i32(json["min_chunk_length"]);
            if (json.has("condition_on_previous_chunks"))
                ws_params.condition_on_previous_chunks = checked_json_bool(json["condition_on_previous_chunks"], "condition_on_previous_chunks");
            if (json.has("min_end_tokens")) ws_params.gen.min_tokens_before_end = checked_json_i32(json["min_end_tokens"]);
            if (json.has("ras_window")) ws_params.gen.ras_window_size = checked_json_i32(json["ras_window"]);
            if (json.has("ras_temp")) ws_params.gen.ras_high_temp = static_cast<float>(json["ras_temp"].d());
            if (json.has("ras_top_p")) ws_params.gen.ras_high_top_p = static_cast<float>(json["ras_top_p"].d());
            if (json.has("prompt_text")) ws_params.prompt_text = json["prompt_text"].s();
            const bool has_reference_id = json.has("reference_id") && json["reference_id"].t() != crow::json::type::Null;
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
            if (json.has("trim_silence")) ws_params.trim_silence = checked_json_bool(json["trim_silence"], "trim_silence");
            if (json.has("stream_stride")) ws_params.stream_decode_stride_frames = checked_json_i32(json["stream_stride"]);

            // Do not silently downgrade an explicit low-latency request to the
            // segment-buffered path. Speed/loudness/non-native resampling need
            // the complete segment today; a stateful incremental postprocessor
            // would be required to preserve an explicit frame cadence exactly.
            bool explicit_tight_latency = false;
            if (json.has("latency")) {
                std::string latency = json["latency"].s();
                std::transform(latency.begin(), latency.end(), latency.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                explicit_tight_latency = latency == "balanced" || latency == "low";
            }
            const bool explicit_frame_cadence = json.has("stream_stride") &&
                                                ws_params.stream_decode_stride_frames >= 0;
            const bool needs_segment_postprocess =
                std::fabs(ws_params.prosody_speed - 1.0f) > 1e-6f ||
                ws_params.normalize_loudness ||
                (ws_params.output_sample_rate > 0 && ws_params.output_sample_rate != pipeline.sample_rate());
            if (needs_segment_postprocess && (explicit_tight_latency || explicit_frame_cadence)) {
                throw std::invalid_argument(
                    "latency=balanced/low or stream_stride>=0 cannot be combined with prosody.speed != 1, "
                    "prosody.normalize_loudness=true, or a non-native sample_rate; these require whole-segment postprocessing");
            }

            std::string validation_error;
            if (!validate_pipeline_params(ws_params, validation_error, true)) {
                crow::json::wvalue err;
                err["error"] = validation_error;
                ws_send_text(conn, state, err.dump());
                return;
            }

            int32_t segment_count = 0;
            s2::StreamCallback cb = [&](const int16_t * pcm, size_t n_samples, bool is_last) -> bool {
                if (!state->alive.load(std::memory_order_relaxed)) return false;
                if ((n_samples > 0 && pcm == nullptr) ||
                    n_samples > (std::numeric_limits<size_t>::max() - 2u) / 2u) return false;
                const uint16_t flags = is_last ? 1u : 0u;
                std::string msg(2u + n_samples * 2u, '\0');
                msg[0] = static_cast<char>(flags & 0xffu);
                msg[1] = static_cast<char>((flags >> 8) & 0xffu);
                if (n_samples > 0) std::memcpy(msg.data() + 2, pcm, n_samples * 2u);
                return ws_send_binary(conn, state, msg);
            };

            bool ok = false;
            int32_t stream_sample_rate = 0;
            {
                auto lease = pipeline_pool.acquire(&state->alive);
                if (!lease) return; // disconnected while waiting for a model worker
                stream_sample_rate = lease->output_sample_rate(ws_params);
                s2::CancelCallback should_continue = [state]() -> bool {
                    return state->alive.load(std::memory_order_relaxed);
                };
                ok = lease->synthesize_streaming(ws_params, cb, &segment_count, should_continue);
            }
            if (!state->alive.load(std::memory_order_relaxed)) return;

            crow::json::wvalue done_msg;
            if (ok) {
                done_msg["done"] = true;
                done_msg["segments"] = segment_count;
                done_msg["sample_rate"] = stream_sample_rate;
            } else {
                done_msg["error"] = "synthesis failed (check requested voice/reference and server log)";
                done_msg["segments"] = segment_count;
            }
            ws_send_text(conn, state, done_msg.dump());
        } catch (const std::bad_alloc &) {
            std::fputs("[WS] Synthesis request ran out of memory\n", stderr);
            ws_send_text(conn, state, "{\"error\": \"server error\"}");
        } catch (const std::invalid_argument & e) {
            crow::json::wvalue err;
            err["error"] = std::string("invalid request: ") + e.what();
            ws_send_text(conn, state, err.dump());
        } catch (const std::out_of_range & e) {
            crow::json::wvalue err;
            err["error"] = std::string("invalid request: ") + e.what();
            ws_send_text(conn, state, err.dump());
        } catch (const std::exception & e) {
            std::cerr << "[WS] Internal synthesis failure: " << e.what() << "\n";
            ws_send_text(conn, state, "{\"error\": \"server error\"}");
        } catch (...) {
            std::fputs("[WS] Unknown internal synthesis failure\n", stderr);
            ws_send_text(conn, state, "{\"error\": \"server error\"}");
        }
    };

    CROW_WEBSOCKET_ROUTE(app, "/ws/tts")
    .onaccept([&](const crow::request & req, std::optional<crow::response> & res, void ** userdata) {
        if (auto blocked = enforce_http_security(req, false)) { res = std::move(*blocked); return; }
        if (!s2::server::try_increment_bounded(accepted_ws_connections,
                                               static_cast<size_t>(max_ws_connections))) {
            res = crow::response(503, "WebSocket connection limit reached; retry later");
            return;
        }
        *userdata = &ws_reserved_slot_sentinel;
    })
    .onopen([&](crow::websocket::connection& conn) {
        try {
            auto state = std::make_shared<WsConnectionState>();
            {
                // Keep the reservation sentinel intact until the map insertion
                // succeeds. mark_ws_closed() takes the same mutex, so failure and
                // close/error paths cannot leak or double-release the slot.
                std::lock_guard<std::mutex> lock(ws_state_mutex);
                if (conn.userdata() != &ws_reserved_slot_sentinel)
                    throw std::runtime_error("WebSocket reservation disappeared before onopen");
                const auto inserted = ws_states.emplace(&conn, state).second;
                if (!inserted) throw std::runtime_error("duplicate WebSocket connection state");
                state->counted_connection.store(true, std::memory_order_relaxed);
                conn.userdata(nullptr);
            }
            std::cout << "[WS] Client connected: " << conn.get_remote_ip() << "\n";
        } catch (const std::exception & e) {
            std::cerr << "[WS] Failed to initialize connection state: " << e.what() << "\n";
            mark_ws_closed(conn);
            try { conn.close("server unavailable"); } catch (...) {}
        } catch (...) {
            std::fputs("[WS] Failed to initialize connection state\n", stderr);
            mark_ws_closed(conn);
            try { conn.close("server unavailable"); } catch (...) {}
        }
    })
    .onclose([&](crow::websocket::connection& conn, const std::string& reason, uint16_t) {
        mark_ws_closed(conn);
        std::cout << "[WS] Client disconnected: " << reason << "\n";
    })
    .onmessage([&](crow::websocket::connection& conn,
                   const std::string& data,
                   bool is_binary) {
        std::shared_ptr<WsConnectionState> state;
        {
            std::lock_guard<std::mutex> lock(ws_state_mutex);
            auto it = ws_states.find(&conn);
            if (it != ws_states.end()) state = it->second;
        }
        if (!state || !state->alive.load(std::memory_order_relaxed)) return;
        if (!request_rate_limiter.allow()) {
            ws_send_text(&conn, state, "{\"error\": \"request rate limit exceeded; retry later\"}");
            return;
        }
        if (is_binary) {
            ws_send_text(&conn, state, "{\"error\": \"expected JSON text message\"}");
            return;
        }
        if (data.size() > MAX_JSON_REQUEST_BYTES) {
            ws_send_text(&conn, state, "{\"error\": \"JSON request body too large\"}");
            return;
        }
        bool expected = false;
        if (!state->busy.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            ws_send_text(&conn, state, "{\"error\": \"a synthesis is already active on this WebSocket connection\"}");
            return;
        }

        std::string owned_data;
        try { owned_data = data; }
        catch (...) {
            state->busy.store(false, std::memory_order_release);
            ws_send_text(&conn, state, "{\"error\": \"server could not queue request\"}");
            return;
        }
        crow::websocket::connection * conn_ptr = &conn;
        const bool queued = ws_tasks->submit([&, conn_ptr, state, owned_data = std::move(owned_data)]() mutable {
            process_ws_message(conn_ptr, state, std::move(owned_data));
        });
        if (!queued) {
            state->busy.store(false, std::memory_order_release);
            ws_send_text(&conn, state, "{\"error\": \"WebSocket synthesis queue is full\"}");
        }
    })
    .onerror([&](crow::websocket::connection& conn, const std::string& error_message) {
        mark_ws_closed(conn);
        std::cerr << "[WS] Connection error: " << error_message << "\n";
    });

    std::cout << "\nEndpoints:\n"
              << "  POST /v1/tts           (Fish Audio compatible)\n"
              << "  POST /v1/tts/batch     (batch; up to 32 items, one base64 audio output/item)\n"
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

    if (!server_loopback_bind) {
        std::cerr << "[Security] Remote exposure explicitly enabled with --allow-remote on " << bind_host
                  << "; Authorization: Bearer <S2_API_TOKEN> is required. Built-in limits are process-wide; TLS, per-IP rate limiting and ingress body limits still belong in a trusted reverse proxy.\n";
    } else if (server_auth_required) {
        std::cerr << "[Security] S2_API_TOKEN is set; Bearer authentication is required on loopback too.\n";
    }
    std::cout << "Server listening on " << bind_host;
    if (resolved_bind_host != bind_host) std::cout << " (" << resolved_bind_host << ")";
    std::cout << ":" << port << "...\n";
    try {
        app.bindaddr(resolved_bind_host).port(static_cast<uint16_t>(port)).multithreaded().run();
        mark_all_ws_closed();
        ws_tasks->shutdown();
        batch_tasks->shutdown();
    } catch (const std::exception & e) {
        mark_all_ws_closed();
        ws_tasks->shutdown();
        batch_tasks->shutdown();
        std::cerr << "Server startup/runtime error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        mark_all_ws_closed();
        ws_tasks->shutdown();
        batch_tasks->shutdown();
        std::cerr << "Server startup/runtime error: unknown exception\n";
        return 1;
    }
    return 0;
}
