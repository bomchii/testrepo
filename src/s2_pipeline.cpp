#include "../include/s2_pipeline.h"
#include "../include/s2_text.h"
#include "../third_party/filesystem.hpp"
#include "s2_utf8.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <cmath>
#include <cctype>
#include <functional>
#include <thread>
#include <gguf.h>

namespace fs = ghc::filesystem;

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <io.h>      // _commit, _fileno
#else
#  include <unistd.h>
#  include <fcntl.h>
#endif

namespace s2 {

namespace {

constexpr int32_t MAX_REFERENCE_AUDIO_SECONDS = 30;

static bool reference_audio_within_limit(const AudioData & audio, int32_t sample_rate) noexcept {
    if (sample_rate <= 0) return false;
    const uint64_t max_samples = static_cast<uint64_t>(sample_rate) *
                                 static_cast<uint64_t>(MAX_REFERENCE_AUDIO_SECONDS);
    return static_cast<uint64_t>(audio.samples.size()) <= max_samples;
}

static FILE * open_binary_output_utf8(const std::string & path) {
#ifdef _WIN32
    if (path.empty()) return nullptr;
    const std::wstring wp = fs::path(path).wstring();
    return _wfopen(wp.c_str(), L"wb");
#else
    return std::fopen(path.c_str(), "wb");
#endif
}

static FILE * open_binary_input_utf8(const std::string & path) {
#ifdef _WIN32
    if (path.empty()) return nullptr;
    const std::wstring wp = fs::path(path).wstring();
    return _wfopen(wp.c_str(), L"rb");
#else
    return std::fopen(path.c_str(), "rb");
#endif
}

static void remove_file_utf8(const std::string & path) noexcept {
    if (path.empty()) return;
#ifdef _WIN32
    // Cleanup helpers run from destructors and must never turn an allocation
    // failure in path conversion into std::terminate.
    try {
        const std::wstring wp = fs::path(path).wstring();
        if (!wp.empty()) (void)DeleteFileW(wp.c_str());
    } catch (...) {
    }
#else
    (void)std::remove(path.c_str());
#endif
}

static bool sync_binary_file(FILE * f) noexcept {
    if (!f || std::fflush(f) != 0) return false;
#ifdef _WIN32
    return _commit(_fileno(f)) == 0;
#else
    return ::fsync(fileno(f)) == 0;
#endif
}

static bool replace_file_atomic_utf8(const std::string & tmp_path,
                                     const std::string & final_path) {
#ifdef _WIN32
    const std::wstring wt = fs::path(tmp_path).wstring();
    const std::wstring wf = fs::path(final_path).wstring();
    return !wt.empty() && !wf.empty() &&
           MoveFileExW(wt.c_str(), wf.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    if (std::rename(tmp_path.c_str(), final_path.c_str()) != 0) return false;
    // Best-effort directory sync makes the rename durable on filesystems that
    // require parent metadata to be flushed separately.
    fs::path parent = fs::path(final_path).parent_path();
    if (parent.empty()) parent = fs::path(".");
    const int dfd = ::open(parent.string().c_str(), O_RDONLY);
    if (dfd >= 0) { (void)::fsync(dfd); ::close(dfd); }
    return true;
#endif
}

static std::string output_temp_path(const std::string & final_path) {
    const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
#ifdef _WIN32
    const unsigned long pid = GetCurrentProcessId();
#else
    const long pid = static_cast<long>(::getpid());
#endif
    return final_path + ".tmp." + std::to_string(pid) + "." +
           std::to_string(stamp) + "." + std::to_string(tid);
}

static bool rewind_binary_file(FILE * f) noexcept {
    if (!f) return false;
#ifdef _WIN32
    return _fseeki64(f, 0, SEEK_SET) == 0;
#else
    return fseeko(f, 0, SEEK_SET) == 0;
#endif
}

static bool read_bounded_text_utf8(const std::string & path, size_t max_bytes,
                                   std::string & out, bool & exists) {
    out.clear();
    exists = false;
    FILE * f = open_binary_input_utf8(path);
    if (!f) return true; // Absent/unreadable is handled by the optional-reference logic.
    exists = true;
    struct Guard { FILE * f; ~Guard(){ if (f) std::fclose(f); } } guard{f};
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return false;
    const __int64 end = _ftelli64(f);
    if (end < 0 || static_cast<uint64_t>(end) > static_cast<uint64_t>(max_bytes)) return false;
    if (_fseeki64(f, 0, SEEK_SET) != 0) return false;
#else
    if (fseeko(f, 0, SEEK_END) != 0) return false;
    const off_t end = ftello(f);
    if (end < 0 || static_cast<uint64_t>(end) > static_cast<uint64_t>(max_bytes)) return false;
    if (fseeko(f, 0, SEEK_SET) != 0) return false;
#endif
    try { out.resize(static_cast<size_t>(end)); } catch (...) { return false; }
    if (!out.empty() && std::fread(out.data(), 1, out.size(), f) != out.size()) { out.clear(); return false; }
    if (std::ferror(f)) { out.clear(); return false; }
    return true;
}

// Ensures an initialized model KV cache is released even if generation throws
// (e.g. std::bad_alloc) or a streaming callback aborts. `cleanup_now()` keeps
// the existing behavior of releasing VRAM before codec decode begins.
struct KvCacheScope {
    SlowARModel * model = nullptr;
    bool * initialized = nullptr;
    int32_t * max_len = nullptr;
    bool active = false;

    void cleanup_now() noexcept {
        if (!active) return;
        if (model) model->free_kv_cache();
        if (initialized) *initialized = false;
        if (max_len) *max_len = 0;
        active = false;
    }
    ~KvCacheScope() { cleanup_now(); }
};

} // namespace

Pipeline::Pipeline()  = default;
Pipeline::~Pipeline() = default;

static void build_wav_header(char * hdr, uint32_t n_samples, int32_t sample_rate,
                             int16_t n_channels = 1, int16_t bits = 16);

// ---------------------------------------------------------------------------
// TempWavFile -- incremental mono PCM16 WAV writer in %TEMP% (or /tmp).
//
// Escribe float32 -> int16 directamente a disco segmento a segmento.
// Nunca acumula mas de un segmento en RAM.
// La cabecera RIFF provisional se parchea al final; no se crea una segunda
// copia WAV, por lo que el pico de disco queda cerca de 1x el audio final.
// ---------------------------------------------------------------------------
struct TempWavFile {
    FILE*    fp          = nullptr;
    uint64_t total_samps = 0;   // int16 samples escritas en total
    int32_t  wav_sample_rate = 0;
    bool     finalized = false;
    std::string path;
#ifdef _WIN32
    std::wstring path_w;
#endif

    bool open(int32_t sample_rate) {
        if (sample_rate < 1000 || sample_rate > 768000) return false;
        wav_sample_rate = sample_rate;
#ifdef _WIN32
        wchar_t tmp_dir[MAX_PATH] = {};
        const DWORD dir_len = GetTempPathW(MAX_PATH, tmp_dir);
        if (dir_len == 0 || dir_len >= MAX_PATH) return false;

        // This helper stores a provisional WAV; keeping the
        // path returned by GetTempFileNameW also avoids a rename race/failure.
        wchar_t tmp_file[MAX_PATH] = {};
        if (GetTempFileNameW(tmp_dir, L"s2_", 0, tmp_file) == 0) return false;

        // GetTempFileNameW has already created the file. Any std::string/
        // std::wstring allocation below can throw, so keep the OS path in the
        // fixed buffer and remove the file on every failure path.
        try {
            const int wlen = static_cast<int>(wcslen(tmp_file));
            const int n = WideCharToMultiByte(CP_UTF8, 0, tmp_file, wlen,
                                              nullptr, 0, nullptr, nullptr);
            if (n <= 0) {
                DeleteFileW(tmp_file);
                return false;
            }
            path_w.assign(tmp_file);
            path.resize(static_cast<size_t>(n));
            if (WideCharToMultiByte(CP_UTF8, 0, tmp_file, wlen,
                                    path.data(), n, nullptr, nullptr) != n) {
                DeleteFileW(tmp_file);
                path.clear();
                path_w.clear();
                return false;
            }
            fp = _wfopen(tmp_file, L"w+b");
            if (!fp) {
                DeleteFileW(tmp_file);
                path.clear();
                path_w.clear();
                return false;
            }
        } catch (...) {
            DeleteFileW(tmp_file);
            path.clear();
            path_w.clear();
            return false;
        }
#else
        path = "/tmp/s2_XXXXXX.wav";
        // mkstemps para extension
        int fd = mkstemps(path.data(), 4);
        if (fd < 0) return false;
        fp = fdopen(fd, "w+b");
        if (!fp) {
            ::close(fd);
            ::unlink(path.c_str());
            path.clear();
            return false;
        }
#endif
        char hdr[44];
        build_wav_header(hdr, 0, wav_sample_rate);
        if (std::fwrite(hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) { cleanup(); return false; }
        return true;
    }

    bool open_at(const std::string & custom_path, int32_t sample_rate) {
        if (custom_path.empty() || sample_rate < 1000 || sample_rate > 768000) return false;
        wav_sample_rate = sample_rate;
        try {
            path = custom_path;
#ifdef _WIN32
            path_w = fs::path(custom_path).wstring();
#endif
            fp = open_binary_output_utf8(custom_path);
        } catch (...) {
            path.clear();
#ifdef _WIN32
            path_w.clear();
#endif
            fp = nullptr;
            return false;
        }
        if (!fp) { cleanup(); return false; }
        char hdr[44];
        build_wav_header(hdr, 0, wav_sample_rate);
        if (std::fwrite(hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) { cleanup(); return false; }
        return true;
    }

    // Escribe un segmento de audio float32 -> int16 a disco.
    // Solo este segmento necesita estar en RAM simultaneamente.
    bool write_segment(const std::vector<float> & samples) {
        if (!fp || samples.empty()) return false;
        // Every non-streaming caller ultimately emits classic RIFF/WAV, whose
        // data chunk is uint32-sized. Reject an impossible result *before*
        // allocating the temporary int16 conversion buffer.
        constexpr uint64_t MAX_WAV_I16_SAMPLES =
            (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - 36u) / sizeof(int16_t);
        if (total_samps > MAX_WAV_I16_SAMPLES ||
            samples.size() > static_cast<size_t>(MAX_WAV_I16_SAMPLES - total_samps)) {
            std::cerr << "Pipeline error: accumulated audio exceeds classic WAV/RIFF 4 GiB limit.\n";
            return false;
        }
        // Convertir float32 -> int16 en un buffer temporal del tamano del segmento
        std::vector<int16_t> pcm(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            const float raw = samples[i];
            if (!std::isfinite(raw)) {
                std::cerr << "Pipeline error: codec produced NaN/Inf PCM at sample " << i << ".\n";
                return false;
            }
            const float s = std::clamp(raw, -1.0f, 1.0f);
            pcm[i] = static_cast<int16_t>(s * 32767.0f);
        }
        size_t written = std::fwrite(pcm.data(), sizeof(int16_t), pcm.size(), fp);
        if (written != pcm.size()) return false;
        total_samps += static_cast<uint64_t>(written);
        return true;
    }

    bool finalize_wav() {
        if (!fp || finalized || wav_sample_rate < 1000 || wav_sample_rate > 768000 || total_samps == 0) return false;
        constexpr uint64_t MAX_WAV_I16_SAMPLES =
            (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - 36u) / sizeof(int16_t);
        if (total_samps > MAX_WAV_I16_SAMPLES) return false;
        if (std::fflush(fp) != 0) return false;
        if (!rewind_binary_file(fp)) return false;
        char hdr[44];
        build_wav_header(hdr, static_cast<uint32_t>(total_samps), wav_sample_rate);
        if (std::fwrite(hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) return false;
        if (!sync_binary_file(fp)) return false;
        finalized = true;
        return true;
    }

    bool close_file() noexcept {
        if (!fp) return true;
        FILE * f = fp;
        fp = nullptr;
        return std::fclose(f) == 0;
    }

    // Transfer ownership of the temporary path to the caller. On Windows keep
    // the UTF-16 path in sync so the destructor does not delete the released file.
    std::string release_path() {
        std::string out = path;
        path.clear();
#ifdef _WIN32
        path_w.clear();
#endif
        return out;
    }

    // Cierra y borra el archivo temporal
    void cleanup() {
        if (fp) { std::fclose(fp); fp = nullptr; }
#ifdef _WIN32
        if (!path_w.empty()) DeleteFileW(path_w.c_str());
        path_w.clear();
#else
        if (!path.empty()) ::unlink(path.c_str());
#endif
        path.clear();
        wav_sample_rate = 0;
        finalized = false;
        total_samps = 0;
    }

    ~TempWavFile() { cleanup(); }
};

// ---------------------------------------------------------------------------
// build_wav_header -- 44 bytes estandar PCM WAV
// ---------------------------------------------------------------------------
static void build_wav_header(char * hdr, uint32_t n_samples, int32_t sample_rate,
                              int16_t n_channels, int16_t bits) {
    uint32_t data_size  = n_samples * n_channels * (bits / 8);
    uint32_t file_size  = 36 + data_size;
    uint32_t byte_rate  = sample_rate * n_channels * (bits / 8);
    uint16_t block_align= n_channels * (bits / 8);
    uint16_t fmt_pcm    = 1;

    std::memcpy(hdr +  0, "RIFF",      4);
    std::memcpy(hdr +  4, &file_size,  4);
    std::memcpy(hdr +  8, "WAVE",      4);
    std::memcpy(hdr + 12, "fmt ",      4);
    uint32_t fmt_sz = 16;
    std::memcpy(hdr + 16, &fmt_sz,     4);
    std::memcpy(hdr + 20, &fmt_pcm,    2);
    std::memcpy(hdr + 22, &n_channels, 2);
    std::memcpy(hdr + 24, &sample_rate,4);
    std::memcpy(hdr + 28, &byte_rate,  4);
    std::memcpy(hdr + 32, &block_align,2);
    std::memcpy(hdr + 34, &bits,       2);
    std::memcpy(hdr + 36, "data",      4);
    std::memcpy(hdr + 40, &data_size,  4);
}

// ---------------------------------------------------------------------------
// split_sentences -- Unicode/speaker/tag-aware implementation lives in
// s2_text.cpp so it can be fuzzed/sanitized without linking the model backend.
// ---------------------------------------------------------------------------
static bool has_non_unicode_whitespace(const std::string & value) {
    return utf8::has_non_whitespace(value);
}

std::vector<std::string> Pipeline::split_sentences(const std::string & text,
                                                    int32_t min_chars) {
    std::vector<std::string> out;
    const auto segments = split_text_segments(text, min_chars);
    out.reserve(segments.size());
    for (const auto & seg : segments) out.push_back(seg.text);
    return out;
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
bool Pipeline::init(const PipelineParams & params) {
    try {
    std::cout << "--- Pipeline init ---" << std::endl;

    // init() is allowed to be retried. Clear request/runtime/backend state first
    // so a failed prior initialization cannot leak model/codec allocations, a
    // reference voice, or cached request data into the next attempt.
    initialized_ = false;
    kv_cache_initialized_ = false;
    kv_cache_max_len_ = 0;
    model_.free_kv_cache();
    model_.unload();
    codec_.unload();
    reference_loaded_ = false;
    reference_embedding_.clear();
    reference_text_.clear();
    active_voice_transcript_.clear();
    voice_cache_.clear();
    voice_cache_order_.clear();

    int model_gpu = params.vulkan_device;
    int codec_gpu = params.codec_vulkan_device;
    if (codec_gpu == -2) codec_gpu = model_gpu;

    std::cout << "GPU: model=" << model_gpu << " codec=" << codec_gpu << std::endl;

    // Tokenizer
    if (params.tokenizer_data && params.tokenizer_data_size > 0) {
        if (!tokenizer_.load_from_memory(params.tokenizer_data, params.tokenizer_data_size)) {
            std::cerr << "Pipeline error: embedded tokenizer parse failed.\n";
            return false;
        }
        std::cout << "Tokenizer: embedded (" << params.tokenizer_data_size << " B)\n";
    } else {
        if (!tokenizer_.load(params.tokenizer_path)) {
            std::cerr << "Pipeline error: tokenizer not found: " << params.tokenizer_path << "\n";
            return false;
        }
    }

    // Modelo
    if (!model_.load(params.model_path, model_gpu)) {
        std::cerr << "Pipeline error: model load failed.\n";
        return false;
    }
    std::cout << "Model loaded (" << model_.backend_name() << ").\n";

    // Codec
    std::string codec_path = params.codec_model_path.empty()
        ? params.model_path : params.codec_model_path;

    bool codec_ok = false;
    if (codec_gpu >= 0 && codec_.load(codec_path, codec_gpu)) {
        std::cout << "Codec loaded (" << codec_.backend_name() << ").\n";
        codec_ok = true;
    }
    if (!codec_ok) codec_.unload();
    if (!codec_ok && codec_.load(codec_path, -1)) {
        std::cout << "Codec loaded (" << codec_.backend_name() << ").\n";
        codec_ok = true;
    }
    if (!codec_ok) {
        std::cerr << "Pipeline error: codec load failed.\n";
        model_.unload();
        return false;
    }

    // The slow model generates exactly one semantic + N residual codebook ids
    // in model codebook space. A mismatched codec would otherwise reinterpret the
    // same buffer with a different row count/size and can read past it.
    const ModelHParams & model_hp = model_.hparams();
    if (codec_.num_codebooks() != model_hp.num_codebooks ||
        codec_.semantic_codebook_size() != model_hp.codebook_size ||
        codec_.residual_codebook_size() != model_hp.codebook_size) {
        std::cerr << "Pipeline error: model/codec VQ layout mismatch (model="
                  << model_hp.num_codebooks << "x" << model_hp.codebook_size
                  << ", codec=" << codec_.num_codebooks() << " codebooks, semantic="
                  << codec_.semantic_codebook_size() << ", residual="
                  << codec_.residual_codebook_size() << ").\n";
        codec_.unload();
        model_.unload();
        return false;
    }
    codec_path_ = codec_path;

    // Sincronizar hparams tokenizer <-> modelo
    {
        // Model metadata has already passed strict range/consistency checks.
        // Copy it verbatim: semantic_begin_id == 0 is valid for compatible
        // custom GGUFs and must not leave the tokenizer's default offset behind.
        const ModelHParams & hp = model_.hparams();
        TokenizerConfig    & tc = tokenizer_.config();
        tc.semantic_begin_id = hp.semantic_begin_id;
        tc.semantic_end_id   = hp.semantic_end_id;
        tc.num_codebooks     = hp.num_codebooks;
        tc.codebook_size     = hp.codebook_size;
        tc.vocab_size        = hp.vocab_size;
    }

    // Optional global reference. Voice cloning requires BOTH the audio and its
    // transcript; accepting only one silently falls back to the base voice.
    const fs::path base_path = params.base_dir.empty() ? fs::path(".") : fs::path(params.base_dir);
    const std::string ref_wav = (base_path / "reference.wav").string();
    const std::string ref_txt = (base_path / "reference.txt").string();
    bool have_ref_wav = false;
    bool have_ref_txt = false;
    if (std::FILE* f = open_binary_input_utf8(ref_wav)) {
        std::fclose(f);
        have_ref_wav = true;
    }
    {
        constexpr size_t MAX_REFERENCE_TEXT_BYTES = 1024u * 1024u;
        bool ref_txt_exists = false;
        if (!read_bounded_text_utf8(ref_txt, MAX_REFERENCE_TEXT_BYTES, reference_text_, ref_txt_exists)) {
            if (ref_txt_exists)
                std::cerr << "[Voice] reference.txt is unreadable or exceeds the 1 MiB limit; it will be ignored.\n";
            reference_text_.clear();
        }
        while (!reference_text_.empty() &&
               (reference_text_.back() == '\n' || reference_text_.back() == '\r')) {
            reference_text_.pop_back();
        }
        if (!reference_text_.empty() && !utf8::is_valid(reference_text_)) {
            std::cerr << "[Voice] reference.txt contains invalid UTF-8; it will be ignored.\n";
            reference_text_.clear();
        }
        have_ref_txt = has_non_unicode_whitespace(reference_text_);
        if (!have_ref_txt) reference_text_.clear();
    }

    if (have_ref_wav && have_ref_txt) {
        AudioData ra;
        std::vector<int32_t> rc;
        int32_t Tp = 0;
        if (load_audio(ref_wav, ra, codec_.sample_rate()) &&
            reference_audio_within_limit(ra, codec_.sample_rate()) &&
            ra.samples.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) &&
            codec_.encode(ra.samples.data(), static_cast<int32_t>(ra.samples.size()),
                          params.gen.n_threads, rc, Tp) &&
            Tp > 0 && rc.size() == static_cast<size_t>(model_.hparams().num_codebooks) * static_cast<size_t>(Tp)) {
            reference_embedding_.assign(reinterpret_cast<const char*>(rc.data()), rc.size() * sizeof(int32_t));
            reference_loaded_ = true;
            std::cout << "Global reference loaded: " << Tp << " frames.\n";
        } else {
            std::cerr << "[Voice] Failed to load/encode global reference; it will be ignored.\n";
            reference_text_.clear();
        }
    } else if (have_ref_wav || have_ref_txt) {
        std::cerr << "[Voice] Global reference ignored: reference.wav and non-empty reference.txt are both required.\n";
        reference_text_.clear();
    }

    initialized_ = true;
    std::cout << "--- Pipeline ready ---\n";
    return true;
    } catch (const std::bad_alloc &) {
        std::cerr << "Pipeline error: initialization ran out of memory.\n";
    } catch (const std::exception & e) {
        std::cerr << "Pipeline error: initialization exception: " << e.what() << "\n";
    } catch (...) {
        std::cerr << "Pipeline error: unknown initialization exception.\n";
    }
    initialized_ = false;
    kv_cache_initialized_ = false;
    model_.free_kv_cache();
    model_.unload();
    codec_.unload();
    reference_loaded_ = false;
    reference_embedding_.clear();
    reference_text_.clear();
    active_voice_transcript_.clear();
    voice_cache_.clear();
    voice_cache_order_.clear();
    return false;
}

bool Pipeline::warmup(const PipelineParams & params) {
    if (!initialized_) return false;
    const int32_t num_cb = model_.hparams().num_codebooks;
    const int32_t model_ctx = model_.hparams().context_length;
    if (num_cb <= 0 || model_ctx <= 1) return false;

    // Intentionally bypass get_ref_codes(): warmup must not load, cache or
    // mutate a real voice/reference. One deterministic semantic frame is enough
    // to exercise transformer prefill/step plumbing and the codec path.
    PromptTensor prompt = build_prompt(tokenizer_, "Warmup.", {}, nullptr, num_cb, 0);
    if (prompt.cols <= 0 || prompt.cols >= model_ctx) return false;
    const int32_t max_seq = prompt.cols + 1;
    if (!kv_cache_initialized_ || kv_cache_max_len_ < max_seq) {
        if (!model_.init_kv_cache(max_seq)) return false;
        kv_cache_initialized_ = true;
        kv_cache_max_len_ = max_seq;
    }
    model_.reset();
    KvCacheScope kv_guard{&model_, &kv_cache_initialized_, &kv_cache_max_len_, true};
    GenerateParams gp = params.gen;
    gp.max_new_tokens = 1;
    gp.min_tokens_before_end = 0;
    gp.seed = 1;
    gp.force_first_token = true;
    gp.repetition_penalty = 1.0f;
    gp.repetition_window = 0;
    gp.verbose = false;
    auto res = generate(model_, tokenizer_.config(), prompt, gp);
    kv_guard.cleanup_now();
    if (!res.success || res.n_frames != 1) return false;
    std::vector<float> audio;
    return codec_.decode_chunked(res.codes.data(), res.n_frames, params.gen.n_threads,
                                 audio, 1, 0) && !audio.empty();
}

// ---------------------------------------------------------------------------
// synthesize_segment -- genera audio float32 para un fragmento de texto.
// El llamador decide si lo guarda en RAM o lo vuelca a disco.
// ---------------------------------------------------------------------------
bool Pipeline::synthesize_segment(
        const PipelineParams       & params,
        const std::string          & text_segment,
        const std::vector<int32_t> & ref_codes,
        int32_t                      T_prompt,
        std::vector<float>         & audio_out,
        std::vector<int32_t>       * generated_codes_out,
        int32_t                    * generated_frames_out,
        std::vector<PromptHistoryTurn> * history) {

    const int32_t num_cb = model_.hparams().num_codebooks;

    // The reference transcript must come from the same source as ref_codes.
    // get_ref_codes() binds active_voice_transcript_ to explicit reference audio,
    // a persisted .s2voice profile, or the global reference.  Do not let an
    // unrelated per-request prompt_text override a saved/global voice transcript.
    const std::string & effective_prompt_text = active_voice_transcript_;

    const int32_t model_ctx = model_.hparams().context_length;
    if (model_ctx <= 0) return false;
    std::vector<PromptHistoryTurn> no_history;
    std::vector<PromptHistoryTurn> * history_work = history ? history : &no_history;
    PromptTensor prompt;
    for (;;) {
        prompt = build_prompt_with_history(
            tokenizer_, text_segment, ref_codes.empty() ? std::string{} : effective_prompt_text,
            ref_codes.empty() ? nullptr : ref_codes.data(), num_cb, T_prompt, *history_work);
        if (prompt.cols <= 0 || prompt.data.empty()) {
            std::cerr << "Pipeline error: invalid/empty prompt tensor.\n";
            return false;
        }
        if (prompt.cols < model_ctx) break;
        if (history_work->empty()) {
            std::cerr << "Pipeline error: prompt is too long for the model context ("
                      << prompt.cols << " >= " << model_ctx << ").\n";
            return false;
        }
        // Evict oldest conversational turn first; never drop the external voice
        // reference or the current text. Persist the eviction in request state so
        // auto multi-speaker history stays bounded by the real model context.
        history_work->erase(history_work->begin());
    }

    // KV cache: reutilizar si cabe.
    // En modo segmentado, limitar max_new_tokens al minimo necesario para el segmento
    // para evitar OOM en GPUs con VRAM ajustada (RTX 3050 4GB con modelo+codec en VRAM).
    int32_t seg_max_tokens = params.gen.max_new_tokens;
    if (params.segment_sentences && params.chunk_length == 0 && params.max_tokens_per_segment > 0 &&
        params.max_tokens_per_segment < seg_max_tokens) {
        seg_max_tokens = params.max_tokens_per_segment;
    }
    seg_max_tokens = std::min(seg_max_tokens, model_ctx - prompt.cols);
    if (seg_max_tokens <= 0) return false;
    const int32_t max_seq = prompt.cols + seg_max_tokens;
    if (!kv_cache_initialized_ || max_seq > kv_cache_max_len_) {
        std::cout << "[KV] Init cache max_seq=" << max_seq << "\n";
        if (!model_.init_kv_cache(max_seq)) {
            std::cerr << "Pipeline error: init_kv_cache failed.\n";
            return false;
        }
        kv_cache_initialized_ = true;
        kv_cache_max_len_     = max_seq;
    }
    model_.reset();
    KvCacheScope kv_guard{&model_, &kv_cache_initialized_, &kv_cache_max_len_, true};

    GenerateParams seg_gen = params.gen;
    seg_gen.max_new_tokens = seg_max_tokens;
    GenerateResult res = generate(model_, tokenizer_.config(), prompt, seg_gen);
    if (!res.success || res.n_frames == 0) {
        std::cerr << "Pipeline error: generate() failed or returned 0 frames.\n";
        return false;
    }
    // Do not expose generated VQ codes until the corresponding acoustic decode
    // has succeeded.  Otherwise a failed first segment could become the voice
    // anchor for later segments and leave request state inconsistent.

    // Liberar el KV cache inmediatamente despues de generate() para recuperar
    // VRAM antes de que decode_chunked() intente allocar sus activaciones.
    // The scope guard also covers exceptions/cancellation before this point.
    kv_guard.cleanup_now();

    if (!codec_.decode_chunked(res.codes.data(), res.n_frames,
                               params.gen.n_threads, audio_out,
                               params.codec_chunk_frames,
                               params.codec_overlap_frames)) {
        // decode_chunked() can fail for backend allocation/compute errors,
        // invalid codes, or non-finite backend output. Do not misreport every
        // failure as GPU OOM; the codec already logs the concrete cause.
        std::cerr << "Pipeline error: decode_chunked() failed.\n";
        std::cerr << "  For GPU allocation failures, try reducing --codec-chunk.\n";
        return false;
    }

    if (generated_codes_out) *generated_codes_out = std::move(res.codes);
    if (generated_frames_out) *generated_frames_out = res.n_frames;
    return true;
}

static void apply_output_gain(std::vector<float> & audio, float db) {
    if (audio.empty() || db == 0.0f) return;
    const float gain = std::pow(10.0f, db / 20.0f);
    for (float & sample : audio) {
        if (!std::isfinite(sample)) { sample = 0.0f; continue; }
        sample = std::clamp(sample * gain, -1.0f, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// postprocess_audio -- trim silence (otras operaciones pueden anadirse aqui)
// ---------------------------------------------------------------------------
void Pipeline::postprocess_audio(std::vector<float> & audio, const PipelineParams & params) const {
    if (params.trim_silence && !audio.empty()) {
        auto trimmed = audio_trim_trailing_silence(audio.data(), audio.size(), codec_.sample_rate());
        if (!trimmed.empty()) audio = std::move(trimmed);
    }
    // Fish prosody.volume is specified in decibels. Apply it after trimming so
    // gain never changes silence-boundary decisions.
    apply_output_gain(audio, params.prosody_volume_db);
}

// ---------------------------------------------------------------------------
// encode_reference -- API publica para que main.cpp encodee sin sintetizar.
// Delega en get_ref_codes y hereda toda la logica de VoiceCache.
// ---------------------------------------------------------------------------
bool Pipeline::encode_reference(const PipelineParams & params,
                                 std::vector<int32_t> & out_codes,
                                 int32_t              & out_T_prompt) {
    if (!initialized_) {
        std::cerr << "[Pipeline] encode_reference: pipeline not initialized.\n";
        return false;
    }
    if (!utf8::is_valid(params.prompt_text)) {
        std::cerr << "[Pipeline] encode_reference: prompt_text must be valid UTF-8.\n";
        return false;
    }
    return get_ref_codes(params, out_codes, out_T_prompt);
}

// ---------------------------------------------------------------------------
// get_ref_codes -- obtiene los codes de referencia de voz con cache LRU.
//
// Prioridad:
//   1. prompt_audio_path explicito del request -- cachea/encodea la referencia
//   2. voice_id persistido (.s2voice) -- se carga fresco desde disco
//   3. reference.wav + reference.txt globales cargados durante init()
// ---------------------------------------------------------------------------
bool Pipeline::get_ref_codes(const PipelineParams & params,
                              std::vector<int32_t> & out_codes,
                              int32_t              & out_T_prompt) {
    const int32_t num_cb = model_.hparams().num_codebooks;
    active_voice_transcript_.clear();
    out_codes.clear();
    out_T_prompt = 0;

    if (num_cb <= 0) {
        std::cerr << "[Voice] Invalid model codebook count.\n";
        return false;
    }

    auto save_profile_if_requested = [&](const std::vector<int32_t> & codes,
                                         int32_t T_prompt) -> bool {
        if (!params.save_voice) return true;
        if (params.voice_id.empty() || !has_non_unicode_whitespace(params.prompt_text) ||
            codes.empty() || T_prompt <= 0) {
            std::cerr << "[Voice] Saving a voice requires voice id, prompt transcript and valid reference codes.\n";
            return false;
        }
        voice_mgr_.set_storage_dir(params.voice_storage_dir);
        VoiceProfile profile;
        profile.transcript    = params.prompt_text;
        profile.codes         = codes;
        profile.num_codebooks = model_.hparams().num_codebooks;
        profile.T_prompt      = T_prompt;
        profile.sample_rate   = codec_.sample_rate();
        profile.codebook_size = model_.hparams().codebook_size;
        try {
            if (!voice_mgr_.save(params.voice_id, profile)) {
                std::cerr << "[Voice] Failed to save profile: " << params.voice_id << "\n";
                return false;
            }
        } catch (const std::exception & e) {
            std::cerr << "[Voice] Failed to save '" << params.voice_id << "': " << e.what() << "\n";
            return false;
        }
        std::cout << "[Voice] Saved profile: " << params.voice_id
                  << " -> " << voice_mgr_.storage_dir() << "\n";
        return true;
    };

    // 1. Explicit reference audio always wins (also makes --save-voice reliable
    // even when a global reference.wav was loaded during init).
    if (!params.prompt_audio_path.empty() && !has_non_unicode_whitespace(params.prompt_text)) {
        std::cerr << "[Voice] reference audio requires its prompt transcript (prompt_text).\n";
        return false;
    }
    if (!params.prompt_audio_path.empty()) {
        // Keep transcript provenance paired with the codes produced from this
        // explicit reference audio, including cache hits.
        active_voice_transcript_ = params.prompt_text;

        // Cache by path, but verify the file did not change in place. Without
        // this, overwriting ref.wav would keep returning codes for the old audio.
        uintmax_t source_size = 0;
        int64_t source_mtime_ns = 0;
        bool have_fingerprint = false;
        try {
            source_size = fs::file_size(params.prompt_audio_path);
            const auto ft = fs::last_write_time(params.prompt_audio_path);
            source_mtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                ft.time_since_epoch()).count();
            have_fingerprint = true;
        } catch (...) {
            // Let load_audio() below report the real read error. We simply avoid
            // trusting an unverifiable cache entry.
        }
        const bool fingerprint_before_ok = have_fingerprint;
        const uintmax_t source_size_before = source_size;
        const int64_t source_mtime_before_ns = source_mtime_ns;

        auto it = voice_cache_.find(params.prompt_audio_path);
        if (it != voice_cache_.end()) {
            const bool fresh = have_fingerprint && it->second.has_fingerprint &&
                               it->second.source_size == source_size &&
                               it->second.source_mtime_ns == source_mtime_ns;
            if (fresh) {
                std::cout << "[VoiceCache] HIT: " << params.prompt_audio_path << "\n";
                out_codes    = it->second.codes;
                out_T_prompt = it->second.T_prompt;

                // Real LRU: a hit becomes the newest entry.
                voice_cache_order_.erase(
                    std::remove(voice_cache_order_.begin(), voice_cache_order_.end(), params.prompt_audio_path),
                    voice_cache_order_.end());
                voice_cache_order_.push_back(params.prompt_audio_path);
                return save_profile_if_requested(out_codes, out_T_prompt);
            }
            std::cout << "[VoiceCache] STALE -- re-encoding: " << params.prompt_audio_path << "\n";
            voice_cache_.erase(it);
            voice_cache_order_.erase(
                std::remove(voice_cache_order_.begin(), voice_cache_order_.end(), params.prompt_audio_path),
                voice_cache_order_.end());
        }

        std::cout << "[VoiceCache] MISS -- encoding: " << params.prompt_audio_path << "\n";
        AudioData ra;
        if (!load_audio(params.prompt_audio_path, ra, codec_.sample_rate()) || ra.samples.empty()) {
            std::cerr << "[VoiceCache] Error loading audio: " << params.prompt_audio_path << "\n";
            return false;
        }
        if (!reference_audio_within_limit(ra, codec_.sample_rate())) {
            std::cerr << "[VoiceCache] Reference audio exceeds the 30 second safety limit. "
                         "Use a shorter 5-30 second cloning sample.\n";
            return false;
        }

        std::vector<int32_t> codes;
        int32_t T_prompt = 0;
        if (ra.samples.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            !codec_.encode(ra.samples.data(), static_cast<int32_t>(ra.samples.size()),
                           params.gen.n_threads, codes, T_prompt) ||
            codes.empty() || T_prompt <= 0 ||
            codes.size() != static_cast<size_t>(num_cb) * static_cast<size_t>(T_prompt)) {
            std::cerr << "[VoiceCache] Error encoding reference audio.\n";
            return false;
        }

        // Re-read metadata after encoding. Cache only when the file fingerprint
        // is identical before and after the read/encode. Otherwise a concurrent
        // replacement could pair OLD codes with the NEW file's mtime/size and
        // make the stale entry look fresh forever.
        uintmax_t source_size_after = 0;
        int64_t source_mtime_after_ns = 0;
        bool fingerprint_after_ok = false;
        try {
            source_size_after = fs::file_size(params.prompt_audio_path);
            const auto ft = fs::last_write_time(params.prompt_audio_path);
            source_mtime_after_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                ft.time_since_epoch()).count();
            fingerprint_after_ok = true;
        } catch (...) {
        }
        const bool cacheable = fingerprint_before_ok && fingerprint_after_ok &&
                               source_size_before == source_size_after &&
                               source_mtime_before_ns == source_mtime_after_ns;
        if (cacheable) {
            if (voice_cache_.size() >= VOICE_CACHE_MAX && !voice_cache_order_.empty()) {
                const std::string oldest = voice_cache_order_.front();
                std::cout << "[VoiceCache] Evicting: " << oldest << "\n";
                voice_cache_.erase(oldest);
                voice_cache_order_.erase(voice_cache_order_.begin());
            }
            VoiceCache entry;
            entry.codes            = codes;
            entry.T_prompt         = T_prompt;
            entry.source_size      = source_size_after;
            entry.source_mtime_ns  = source_mtime_after_ns;
            entry.has_fingerprint  = true;
            voice_cache_[params.prompt_audio_path] = std::move(entry);
            voice_cache_order_.push_back(params.prompt_audio_path);
            std::cout << "[VoiceCache] Cached (" << voice_cache_.size()
                      << "/" << VOICE_CACHE_MAX << "): " << params.prompt_audio_path << "\n";
        } else {
            std::cout << "[VoiceCache] Reference changed while encoding; not caching this result.\n";
        }

        out_codes    = std::move(codes);
        out_T_prompt = T_prompt;
        return save_profile_if_requested(out_codes, out_T_prompt);
    }

    // 2. Persisted voice profile.
    if (!params.voice_id.empty()) {
        voice_mgr_.set_storage_dir(params.voice_storage_dir);
        std::cout << "[Voice] Loading profile: " << params.voice_id << "\n";
        try {
            VoiceProfile profile = voice_mgr_.load(params.voice_id);
            if (!profile.is_compatible(num_cb, model_.hparams().codebook_size, codec_.sample_rate())) {
                std::cerr << "[Voice] Profile incompatible with current model/codec.\n";
                return false;
            }
            if (profile.T_prompt <= 0 ||
                profile.codes.size() != static_cast<size_t>(num_cb) * static_cast<size_t>(profile.T_prompt)) {
                std::cerr << "[Voice] Profile has inconsistent code dimensions.\n";
                return false;
            }
            active_voice_transcript_ = profile.transcript;
            out_codes    = std::move(profile.codes);
            out_T_prompt = profile.T_prompt;
            std::cout << "[Voice] Loaded: " << params.voice_id
                      << " (" << out_T_prompt << " frames)\n";
            return true;
        } catch (const std::exception & e) {
            std::cerr << "[Voice] Failed to load '" << params.voice_id << "': " << e.what() << "\n";
            return false;
        }
    }

    // 3. Global reference loaded at startup, if any.
    if (reference_loaded_) {
        active_voice_transcript_ = reference_text_;
        if (reference_embedding_.size() % sizeof(int32_t) != 0) return false;
        out_codes.resize(reference_embedding_.size() / sizeof(int32_t));
        std::memcpy(out_codes.data(), reference_embedding_.data(), reference_embedding_.size());
        if (out_codes.empty() || out_codes.size() % static_cast<size_t>(num_cb) != 0) {
            std::cerr << "[Voice] Global reference has invalid code dimensions.\n";
            out_codes.clear();
            return false;
        }
        const size_t frames = out_codes.size() / static_cast<size_t>(num_cb);
        if (frames > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            out_codes.clear();
            return false;
        }
        out_T_prompt = static_cast<int32_t>(frames);
        return true;
    }

    // 4. No reference requested.
    return true;
}

static bool validate_pipeline_text_payload(const PipelineParams & params, bool require_text) {
    if (!utf8::is_valid(params.text) || !utf8::is_valid(params.prompt_text)) {
        std::cerr << "Pipeline error: text/prompt_text must be valid UTF-8.\n";
        return false;
    }
    if (require_text && !utf8::has_non_whitespace(params.text)) {
        std::cerr << "Pipeline error: text must not be empty or whitespace-only.\n";
        return false;
    }
    if (!params.prompt_audio_path.empty() && !utf8::has_non_whitespace(params.prompt_text)) {
        std::cerr << "Pipeline error: reference audio requires a non-empty prompt transcript.\n";
        return false;
    }
    if ((params.chunk_length != 0 && (params.chunk_length < 100 || params.chunk_length > 300)) ||
        params.min_chunk_length < 0 || params.min_chunk_length > 100 ||
        (params.chunk_length == 0 && params.min_chunk_length != 0) ||
        !std::isfinite(params.prosody_volume_db) || params.prosody_volume_db < -20.0f || params.prosody_volume_db > 20.0f) {
        std::cerr << "Pipeline error: invalid long-form/prosody parameters.\n";
        return false;
    }
    return true;
}

// Fish Speech's long-form inference appends each generated assistant VQ turn
// back into the conversation. For single-speaker segmented requests with no
// reference, V7.4's request-local first-segment voice anchor remains the default.
// For multi-speaker input, a monovoice anchor would be wrong, so keep generated
// conversational VQ history and evict oldest turns only when model context forces it.
static bool request_is_chunked(const PipelineParams & params) noexcept {
    return params.segment_sentences || params.chunk_length > 0;
}

static std::vector<std::string> request_text_segments(const PipelineParams & params) {
    std::vector<std::string> out;
    if (params.chunk_length > 0) {
        auto chunks = split_text_chunks(params.text, params.chunk_length, params.min_chunk_length);
        out.reserve(chunks.size());
        for (auto & chunk : chunks) out.push_back(std::move(chunk.text));
        return out;
    }
    if (params.segment_sentences) {
        auto segments = split_text_segments(params.text, params.min_seg_chars);
        out.reserve(segments.size());
        for (auto & segment : segments) out.push_back(std::move(segment.text));
        return out;
    }
    out.push_back(params.text);
    return out;
}

static bool request_has_multiple_speakers(const PipelineParams & params) {
    const auto segments = split_text_segments(params.text, 0);
    return distinct_speaker_count(segments) > 1;
}

static int32_t request_history_limit(const PipelineParams & params,
                                     const std::vector<int32_t> & ref_codes) {
    const int32_t explicit_limit = std::max<int32_t>(0, params.multi_turn_history);
    if (explicit_limit > 0) return explicit_limit;

    // Fish condition_on_previous_chunks=false is an explicit opt-out of
    // automatic inter-chunk acoustic/VQ conditioning.  An explicit
    // --multi-turn-history remains authoritative because it is a separate,
    // user-requested conversation-history feature.
    if (request_is_chunked(params) && !params.condition_on_previous_chunks) return 0;

    if (request_is_chunked(params) && ref_codes.empty() && request_has_multiple_speakers(params)) {
        // Multi-speaker history stays bounded by the real model context, not by
        // total document length. Oldest turns are evicted in synthesize_segment().
        return -1;
    }
    if (params.chunk_length > 0 && params.condition_on_previous_chunks) {
        // Long-form mono/reference mode keeps only the most recent generated VQ
        // turn plus the stable voice reference/anchor. This bounds RAM/KV usage.
        return 1;
    }
    return 0;
}

static bool append_prompt_history(std::vector<PromptHistoryTurn> & history,
                                  int32_t limit,
                                  const std::string & text,
                                  std::vector<int32_t> codes,
                                  int32_t frames,
                                  int32_t num_codebooks) {
    if (limit == 0) return true;
    if (frames <= 0 || num_codebooks <= 0) return false;
    const uint64_t count = static_cast<uint64_t>(frames) * static_cast<uint64_t>(num_codebooks);
    if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        codes.size() != static_cast<size_t>(count)) return false;
    PromptHistoryTurn turn;
    turn.text = text;
    turn.codes = std::move(codes);
    turn.n_frames = frames;
    history.push_back(std::move(turn));
    if (limit > 0) {
        while (history.size() > static_cast<size_t>(limit)) history.erase(history.begin());
    }
    return true;
}

static uint64_t derive_segment_seed(uint64_t request_seed, size_t segment_index) noexcept {
    if (request_seed == 0) return 0;
    // SplitMix64 finalizer over a stable request-seed/index pair. Segment 0 is
    // intentionally derived too, so segmented generation never restarts the
    // same RNG stream for each sentence.
    uint64_t z = request_seed + 0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(segment_index) + 1ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= (z >> 31);
    return z == 0 ? 0xA0761D6478BD642FULL : z;
}

static bool adopt_generated_voice_anchor(
        const std::string    & transcript,
        std::vector<int32_t> & generated_codes,
        int32_t                generated_frames,
        int32_t                num_codebooks,
        std::vector<int32_t> & ref_codes,
        int32_t              & T_prompt,
        std::string          & active_voice_transcript) {
    if (generated_frames <= 0 || num_codebooks <= 0 ||
        !utf8::is_valid(transcript) || !utf8::has_non_whitespace(transcript)) {
        return false;
    }
    const uint64_t expected = static_cast<uint64_t>(num_codebooks) *
                              static_cast<uint64_t>(generated_frames);
    if (expected > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        generated_codes.size() != static_cast<size_t>(expected)) {
        return false;
    }

    ref_codes = std::move(generated_codes);
    T_prompt = generated_frames;
    active_voice_transcript = transcript;
    std::cout << "[Voice] No reference supplied; locked the first generated segment "
              << "as an in-request voice anchor (" << T_prompt << " frames).\n";
    return true;
}

// ---------------------------------------------------------------------------
// synthesize -- guarda a disco (usa TempWavFile para RAM minima)
// ---------------------------------------------------------------------------
bool Pipeline::synthesize(const PipelineParams & params) {
    if (!initialized_) { std::cerr << "Pipeline not initialized.\n"; return false; }
    if (!validate_pipeline_text_payload(params, true)) return false;

    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    if (!get_ref_codes(params, ref_codes, T_prompt)) return false;
    const int32_t history_limit = request_history_limit(params, ref_codes);
    std::vector<PromptHistoryTurn> prompt_history;
    bool auto_voice_anchor_pending = request_is_chunked(params) && params.condition_on_previous_chunks &&
        ref_codes.empty() && params.multi_turn_history == 0 && !request_has_multiple_speakers(params);

    // Stream directly into a staged WAV beside the requested output. This keeps
    // peak disk usage near 1x output size and preserves atomic replacement.
    const std::string final_path = params.output_path.empty() ? "out.wav" : params.output_path;
    const std::string staged_path = output_temp_path(final_path);
    struct StagedCleanup {
        std::string path;
        ~StagedCleanup() { if (!path.empty()) remove_file_utf8(path); }
    } staged_cleanup{staged_path};
    TempWavFile tmp;
    if (!tmp.open_at(staged_path, codec_.sample_rate())) {
        std::cerr << "Pipeline error: could not create staged WAV.\n";
        return false;
    }

    auto process_segment = [&](const std::string & seg, bool is_last, size_t segment_index) -> bool {
        std::vector<float> audio;
        std::vector<int32_t> generated_codes;
        int32_t generated_frames = 0;
        const bool capture_anchor = auto_voice_anchor_pending && !is_last;
        // Request history is only needed to condition a later segment. Never
        // allocate/copy final-segment VQ just to append state that is discarded
        // when this request returns.
        const bool capture_history = history_limit != 0 && !capture_anchor && !is_last;
        const bool capture_codes = capture_anchor || capture_history;
        PipelineParams seg_params = params;
        if (request_is_chunked(params)) seg_params.gen.seed = derive_segment_seed(params.gen.seed, segment_index);
        if (!synthesize_segment(seg_params, seg, ref_codes, T_prompt, audio,
                                capture_codes ? &generated_codes : nullptr,
                                capture_codes ? &generated_frames : nullptr,
                                history_limit != 0 ? &prompt_history : nullptr)) return false;
        if (capture_history) {
            if (!append_prompt_history(prompt_history, history_limit, seg, std::move(generated_codes),
                                       generated_frames, model_.hparams().num_codebooks)) {
                std::cerr << "Pipeline error: could not append conversational VQ history.\n";
                return false;
            }
        } else if (capture_anchor) {
            if (!adopt_generated_voice_anchor(seg, generated_codes, generated_frames,
                                              model_.hparams().num_codebooks, ref_codes, T_prompt,
                                              active_voice_transcript_)) {
                std::cerr << "Pipeline error: could not preserve the generated voice across segments.\n";
                return false;
            }
            auto_voice_anchor_pending = false;
        }
        if (is_last) postprocess_audio(audio, params);
        else apply_output_gain(audio, params.prosody_volume_db);
        return tmp.write_segment(audio);
        // 'audio' se destruye aqui -> RAM liberada antes del siguiente segmento
    };

    auto segs = request_text_segments(params);
    if (segs.empty()) return false;
    std::cout << "[Segment] " << segs.size() << (params.chunk_length > 0 ? " long-form chunks.\n" : " segments.\n");
    for (size_t i = 0; i < segs.size(); ++i) {
        if (!process_segment(segs[i], i + 1 == segs.size(), i)) {
            std::cerr << "Segment " << (i+1) << " failed -- aborting request.\n"; return false;
        }
    }
    if (tmp.total_samps == 0) {
        std::cerr << "Pipeline error: no audio generated.\n";
        return false;
    }

    if (!tmp.finalize_wav() || !tmp.close_file()) {
        std::cerr << "Pipeline error: could not finalize staged WAV.\n";
        return false;
    }
    if (!replace_file_atomic_utf8(staged_path, final_path)) {
        std::cerr << "Pipeline error: could not atomically replace " << final_path << "\n";
        return false;
    }
    (void)tmp.release_path();
    staged_cleanup.path.clear();

    std::cout << "Saved " << tmp.total_samps << " samples to " << final_path << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// synthesize_to_buffer -- para HTTP (Crow).
//
// Estrategia de RAM minima:
//   - Cada segmento se escribe directamente al TempWavFile (disco, %TEMP%)
//   - Al final se construye el output_buffer leyendo el archivo temporal
//   - Pico de RAM = segmento mas largo + output_buffer final
//   - output_buffer es inevitable porque Crow necesita el body completo
//     antes de enviar la respuesta HTTP. Para streaming verdadero habria
//     que usar chunked transfer encoding (mejora futura).
// ---------------------------------------------------------------------------
bool Pipeline::synthesize_to_buffer(const PipelineParams & params,
                                     std::vector<char>    & output_buffer) {
    output_buffer.clear();
    if (!initialized_) { std::cerr << "Pipeline not initialized.\n"; return false; }
    if (!validate_pipeline_text_payload(params, true)) return false;

    std::cout << "--- Synthesize ---\n"
              << "Text: " << params.text << "\n"
              << "Mode: " << (params.segment_sentences ? "segmentado" : "bloque completo") << "\n";

    auto t0 = std::chrono::steady_clock::now();

    // Referencia de voz -- con cache LRU para evitar re-encodear entre requests
    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    auto t_ref0 = std::chrono::steady_clock::now();
    if (!get_ref_codes(params, ref_codes, T_prompt)) return false;
    const int32_t history_limit = request_history_limit(params, ref_codes);
    std::vector<PromptHistoryTurn> prompt_history;
    bool auto_voice_anchor_pending = request_is_chunked(params) && params.condition_on_previous_chunks &&
        ref_codes.empty() && params.multi_turn_history == 0 && !request_has_multiple_speakers(params);
    auto t_ref1 = std::chrono::steady_clock::now();
    std::cout << "[T] Ref audio: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_ref1-t_ref0).count()
              << " ms" << (ref_codes.empty() ? " (sin referencia)" : " (listo)") << "\n";

    // Abrir un unico WAV temporal incremental (cabecera provisional)
    TempWavFile tmp;
    if (!tmp.open(codec_.sample_rate())) {
        std::cerr << "Pipeline error: GetTempFileName failed.\n";
        return false;
    }
    std::cout << "[TempWAV] " << tmp.path << "\n";

    // Funcion lambda: sintetiza un segmento y lo vuelca a disco inmediatamente
    uint64_t total_frames = 0;
    auto process_segment = [&](const std::string & seg, bool is_last, size_t segment_index) -> bool {
        auto ts = std::chrono::steady_clock::now();
        std::vector<float> audio; // solo este segmento en RAM
        std::vector<int32_t> generated_codes;
        int32_t generated_frames = 0;
        const bool capture_anchor = auto_voice_anchor_pending && !is_last;
        // Request history is only needed to condition a later segment. Never
        // allocate/copy final-segment VQ just to append state that is discarded
        // when this request returns.
        const bool capture_history = history_limit != 0 && !capture_anchor && !is_last;
        const bool capture_codes = capture_anchor || capture_history;
        PipelineParams seg_params = params;
        if (request_is_chunked(params)) seg_params.gen.seed = derive_segment_seed(params.gen.seed, segment_index);
        if (!synthesize_segment(seg_params, seg, ref_codes, T_prompt, audio,
                                capture_codes ? &generated_codes : nullptr,
                                capture_codes ? &generated_frames : nullptr,
                                history_limit != 0 ? &prompt_history : nullptr)) return false;
        if (capture_history) {
            if (!append_prompt_history(prompt_history, history_limit, seg, std::move(generated_codes),
                                       generated_frames, model_.hparams().num_codebooks)) {
                std::cerr << "Pipeline error: could not append conversational VQ history.\n";
                return false;
            }
        } else if (capture_anchor) {
            if (!adopt_generated_voice_anchor(seg, generated_codes, generated_frames,
                                              model_.hparams().num_codebooks, ref_codes, T_prompt,
                                              active_voice_transcript_)) {
                std::cerr << "Pipeline error: could not preserve the generated voice across segments.\n";
                return false;
            }
            auto_voice_anchor_pending = false;
        }
        if (is_last) postprocess_audio(audio, params);
        else apply_output_gain(audio, params.prosody_volume_db);
        float dur_s = audio.size() / (float)codec_.sample_rate();
        if (!tmp.write_segment(audio)) {
            std::cerr << "Error writing segment to disk.\n";
            return false;
        }
        // audio se destruye aqui -> RAM liberada
        auto te = std::chrono::steady_clock::now();
        float inf_s = std::chrono::duration_cast<std::chrono::milliseconds>(te-ts).count()/1000.f;
        std::cout << "  -> " << dur_s << "s audio / " << inf_s << "s inferencia ("
                  << (dur_s/std::max(inf_s,0.001f)) << "x RT)\n";
        total_frames++;
        return true;
    };

    auto segs = request_text_segments(params);
    if (segs.empty()) return false;
    std::cout << "[Segment] " << segs.size() << (params.chunk_length > 0 ? " long-form chunks.\n" : " segments.\n");
    for (size_t i = 0; i < segs.size(); ++i) {
        if (!process_segment(segs[i], i + 1 == segs.size(), i)) {
            std::cerr << "Segment " << (i+1) << " failed -- aborting request.\n"; return false;
        }
    }

    if (tmp.total_samps == 0) {
        std::cerr << "Pipeline error: no audio generated.\n";
        return false;
    }

    auto t1 = std::chrono::steady_clock::now();
    float total_audio_s = tmp.total_samps / (float)codec_.sample_rate();
    float total_inf_s   = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()/1000.f;
    std::cout << "[Timing] " << total_audio_s << "s audio total in "
              << total_inf_s << "s (" << (total_audio_s/std::max(total_inf_s,0.001f)) << "x RT)\n";

    // Finalize the same incremental WAV and read it only once into Crow's body.
    // Peak disk usage is one output file rather than PCM + WAV duplicates.
    if (!tmp.finalize_wav()) {
        std::cerr << "Pipeline error: could not finalize temporary WAV.\n";
        return false;
    }
    const uint64_t wav_bytes64 = 44u + tmp.total_samps * sizeof(int16_t);
    if (wav_bytes64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return false;
    std::vector<char> result(static_cast<size_t>(wav_bytes64));
    std::clearerr(tmp.fp);
    if (!rewind_binary_file(tmp.fp)) return false;
    const size_t read = std::fread(result.data(), 1, result.size(), tmp.fp);
    if (read != result.size() || std::ferror(tmp.fp)) return false;
    output_buffer = std::move(result);

    std::cout << "[Buffer] WAV ready: " << output_buffer.size() / 1024 << " KB\n";
    return true;
}

// ---------------------------------------------------------------------------
// synthesize_to_file -- escribe WAV completo a un archivo temporal y retorna
// la ruta. El llamador decide como servir/copiar el archivo y es responsable de
// borrarlo cuando termine.
//
// Ventaja sobre synthesize_to_buffer dentro del pipeline:
//   - La sintesis no acumula todos los segmentos float32 en RAM.
//   - El WAV se escribe por segmentos y su cabecera se parchea al final.
// ---------------------------------------------------------------------------
bool Pipeline::synthesize_to_file(const PipelineParams & params,
                                   std::string          & out_wav_path) {
    out_wav_path.clear();
    if (!initialized_) { std::cerr << "Pipeline not initialized.\n"; return false; }
    if (!validate_pipeline_text_payload(params, true)) return false;

    std::cout << "--- Synthesize to file ---\n"
              << "Text: " << params.text << "\n"
              << "Mode: " << (params.segment_sentences ? "segmentado" : "bloque") << "\n";

    auto t0 = std::chrono::steady_clock::now();

    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    if (!get_ref_codes(params, ref_codes, T_prompt)) return false;
    const int32_t history_limit = request_history_limit(params, ref_codes);
    std::vector<PromptHistoryTurn> prompt_history;
    bool auto_voice_anchor_pending = request_is_chunked(params) && params.condition_on_previous_chunks &&
        ref_codes.empty() && params.multi_turn_history == 0 && !request_has_multiple_speakers(params);

    // Escribir directamente al unico WAV temporal incremental
    TempWavFile tmp;
    if (!tmp.open(codec_.sample_rate())) {
        std::cerr << "Pipeline error: could not create temporary WAV.\n";
        return false;
    }

    auto process_seg = [&](const std::string & seg, bool is_last, size_t segment_index) -> bool {
        auto ts = std::chrono::steady_clock::now();
        std::vector<float> audio;
        std::vector<int32_t> generated_codes;
        int32_t generated_frames = 0;
        const bool capture_anchor = auto_voice_anchor_pending && !is_last;
        // Request history is only needed to condition a later segment. Never
        // allocate/copy final-segment VQ just to append state that is discarded
        // when this request returns.
        const bool capture_history = history_limit != 0 && !capture_anchor && !is_last;
        const bool capture_codes = capture_anchor || capture_history;
        PipelineParams seg_params = params;
        if (request_is_chunked(params)) seg_params.gen.seed = derive_segment_seed(params.gen.seed, segment_index);
        if (!synthesize_segment(seg_params, seg, ref_codes, T_prompt, audio,
                                capture_codes ? &generated_codes : nullptr,
                                capture_codes ? &generated_frames : nullptr,
                                history_limit != 0 ? &prompt_history : nullptr)) return false;
        if (capture_history) {
            if (!append_prompt_history(prompt_history, history_limit, seg, std::move(generated_codes),
                                       generated_frames, model_.hparams().num_codebooks)) {
                std::cerr << "Pipeline error: could not append conversational VQ history.\n";
                return false;
            }
        } else if (capture_anchor) {
            if (!adopt_generated_voice_anchor(seg, generated_codes, generated_frames,
                                              model_.hparams().num_codebooks, ref_codes, T_prompt,
                                              active_voice_transcript_)) {
                std::cerr << "Pipeline error: could not preserve the generated voice across segments.\n";
                return false;
            }
            auto_voice_anchor_pending = false;
        }
        if (is_last) postprocess_audio(audio, params);
        else apply_output_gain(audio, params.prosody_volume_db);
        float dur_s = audio.size() / (float)codec_.sample_rate();
        bool ok = tmp.write_segment(audio);
        // audio destruido aqui -- RAM liberada
        auto te = std::chrono::steady_clock::now();
        float inf_s = std::chrono::duration_cast<std::chrono::milliseconds>(te-ts).count()/1000.f;
        std::cout << "  -> " << dur_s << "s audio in " << inf_s << "s ("
                  << (dur_s/std::max(inf_s,0.001f)) << "x RT)\n";
        return ok;
    };

    auto segs = request_text_segments(params);
    if (segs.empty()) return false;
    std::cout << "[Segment] " << segs.size() << (params.chunk_length > 0 ? " long-form chunks.\n" : " segments.\n");
    for (size_t i = 0; i < segs.size(); ++i) {
        if (!process_seg(segs[i], i + 1 == segs.size(), i)) {
            std::cerr << "Segment " << (i+1) << " failed -- aborting request.\n"; return false;
        }
    }

    if (tmp.total_samps == 0) {
        std::cerr << "Pipeline error: no audio generated.\n";
        return false;
    }

    if (!tmp.finalize_wav() || !tmp.close_file()) {
        std::cerr << "Pipeline error: could not finalize temporary WAV.\n";
        return false;
    }

    auto t1 = std::chrono::steady_clock::now();
    float total_s   = tmp.total_samps / (float)codec_.sample_rate();
    float elapsed_s = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()/1000.f;
    std::cout << "[Timing] " << total_s << "s audio in " << elapsed_s << "s ("
              << (total_s/std::max(elapsed_s,0.001f)) << "x RT)\n";
    std::cout << "[File] WAV: " << tmp.path
              << " (" << (44 + tmp.total_samps*2)/1024 << " KB)\n";

    // Transferir ownership de la ruta -- el llamador borra el archivo
    out_wav_path = tmp.release_path();
    return true;
}

// ---------------------------------------------------------------------------
// float_to_int16 -- convierte muestras float32 [-1,1] a int16
// ---------------------------------------------------------------------------
void Pipeline::float_to_int16(const std::vector<float> & in,
                                std::vector<int16_t>      & out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        const float raw = in[i];
        const float s = std::isfinite(raw) ? std::clamp(raw, -1.0f, 1.0f) : 0.0f;
        out[i] = static_cast<int16_t>(s * 32767.0f);
    }
}

// ---------------------------------------------------------------------------
// synthesize_streaming -- genera audio y llama al callback por segmento.
//
// Con stream_decode_stride_frames > 0 (o auto=4):
//   Usa generate_streaming() para recibir frames uno a uno mientras el
//   transformer genera. Cada vez que se acumulan stride frames, decodifica
//   ese chunk con el codec y lo envia al cliente. Latencia hasta el primer
//   chunk: prefill + stride * ~11ms en lugar de esperar todos los tokens.
//
// Con stream_decode_stride_frames < 0:
//   Desactiva el stride: genera el segmento completo, lo decodifica y lo envia.
//   El valor 0 usa el modo automatico de 4 frames.
// ---------------------------------------------------------------------------
bool Pipeline::synthesize_streaming(const PipelineParams & params,
                                     StreamCallback         callback,
                                     int32_t *              segments_out,
                                     CancelCallback         should_continue) {
    if (segments_out) *segments_out = 0;
    if (!initialized_) { std::cerr << "Pipeline not initialized.\n"; return false; }
    if (!validate_pipeline_text_payload(params, true)) return false;
    if (!callback)      { std::cerr << "synthesize_streaming: null callback.\n"; return false; }
    if (should_continue && !should_continue()) return false;
    if (params.codec_chunk_frames < 0 || params.codec_overlap_frames < 0 ||
        params.stream_decode_stride_frames < -1) {
        std::cerr << "synthesize_streaming: invalid codec/streaming parameters.\n";
        return false;
    }

    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    if (!get_ref_codes(params, ref_codes, T_prompt)) return false;
    const int32_t history_limit = request_history_limit(params, ref_codes);
    std::vector<PromptHistoryTurn> prompt_history;
    bool auto_voice_anchor_pending = request_is_chunked(params) && params.condition_on_previous_chunks &&
        ref_codes.empty() && params.multi_turn_history == 0 && !request_has_multiple_speakers(params);

    // Stride: numero de frames a acumular antes de cada decode+send.
    // 0 -> auto (4 frames), matching CLI/API documentation. Negative disables.
    // Valores tipicos: 4 (baja latencia) a 32 (menor overhead).
    const int32_t stride = params.stream_decode_stride_frames == 0 ? 4 : params.stream_decode_stride_frames;
    const bool    use_stride = (stride > 0);

    // -----------------------------------------------------------------------
    // Helper: genera un segmento con decode en tiempo real (stride activo)
    // -----------------------------------------------------------------------
    auto stream_segment_stride = [&](const std::string & seg, bool is_last_seg, size_t segment_index) -> bool {
        // Construir el prompt usando la API de s2_prompt
        const int32_t num_cb = model_.hparams().num_codebooks;

        const int32_t model_ctx = model_.hparams().context_length;
        if (model_ctx <= 0) return false;
        PromptTensor pt;
        for (;;) {
            pt = build_prompt_with_history(
                tokenizer_, seg,
                ref_codes.empty() ? std::string{} : active_voice_transcript_,
                ref_codes.empty() ? nullptr : ref_codes.data(), num_cb, T_prompt, prompt_history);
            if (pt.cols <= 0 || pt.data.empty()) {
                std::cerr << "[Stream] build_prompt returned empty for: \"" << seg << "\"\n";
                return false;
            }
            if (pt.cols < model_ctx) break;
            if (prompt_history.empty()) {
                std::cerr << "[Stream] prompt is too long for model context.\n";
                return false;
            }
            prompt_history.erase(prompt_history.begin());
        }

        // Configurar parametros de generacion para este segmento
        GenerateParams gp = params.gen;
        if (request_is_chunked(params)) {
            if (params.segment_sentences && params.chunk_length == 0 &&
                params.max_tokens_per_segment > 0 && params.max_tokens_per_segment < gp.max_new_tokens) {
                gp.max_new_tokens = params.max_tokens_per_segment;
            }
            gp.seed = derive_segment_seed(params.gen.seed, segment_index);
        }
        gp.verbose = false;
        gp.max_new_tokens = std::min(gp.max_new_tokens, model_ctx - pt.cols);
        if (gp.max_new_tokens <= 0) return false;

        // Validate codec stride BEFORE allocating the KV cache. An invalid
        // request must not leave a large model cache allocated on an early return.
        const int32_t decode_stride = (stride > 0) ? stride : 4;
        const int32_t codec_max_frames = codec_.max_decode_frames();
        if (codec_max_frames <= 0 || decode_stride > codec_max_frames) {
            std::cerr << "[Stream] stream stride exceeds codec transformer block size.\n";
            return false;
        }

        // Inicializar KV cache si es necesario. Never allocate beyond the model's
        // advertised context length.
        const int32_t ctx_len = pt.cols + gp.max_new_tokens;
        if (!kv_cache_initialized_ || kv_cache_max_len_ < ctx_len) {
            if (!model_.init_kv_cache(ctx_len)) {
                std::cerr << "[Stream] init_kv_cache failed.\n";
                return false;
            }
            kv_cache_initialized_ = true;
            kv_cache_max_len_     = ctx_len;
        }
        // init_kv_cache() manages storage capacity; reset() clears logical n_past
        // from any previous generation/request before the streaming prefill.
        model_.reset();
        KvCacheScope kv_guard{&model_, &kv_cache_initialized_, &kv_cache_max_len_, true};

        // Keep generated code frames per codebook. Streaming output is committed
        // only once the decoder has enough right context for that boundary; the
        // still-unstable tail is held back and revisited on the next stride.
        const int32_t requested_history = params.codec_overlap_frames > 0
            ? params.codec_overlap_frames
            : std::max(0, codec_.streaming_history_frames());
        int32_t decode_window_limit = 0;
        if (params.codec_chunk_frames > 0) {
            decode_window_limit = std::min(params.codec_chunk_frames, codec_max_frames);
        } else {
            const int64_t desired = std::max<int64_t>(120,
                static_cast<int64_t>(requested_history) * 2 + 4);
            decode_window_limit = static_cast<int32_t>(std::min<int64_t>(codec_max_frames, desired));
        }
        if (decode_window_limit <= 0) return false;
        const int32_t history_target = std::min(requested_history,
                                                std::max(0, (decode_window_limit - 1) / 2));
        if (history_target < requested_history) {
            std::cerr << "[Stream] codec history reduced from " << requested_history
                      << " to " << history_target << " frames by decode-window limit "
                      << decode_window_limit << ".\n";
        }

        const int32_t samples_per_frame = codec_.samples_per_code_frame();
        if (samples_per_frame <= 0) return false;

        std::vector<std::vector<int32_t>> accumulated_codes(static_cast<size_t>(num_cb));
        for (auto & row : accumulated_codes) {
            row.reserve(static_cast<size_t>(gp.max_new_tokens));
        }
        int32_t committed_frames = 0;
        int32_t last_decode_frames = 0;
        bool    cb_ok = true;
        std::vector<float> deferred_audio;    // one-chunk lookbehind when final trim is requested
        int32_t deferred_frames = 0;
        bool    terminal_sent = false;

        auto emit_audio = [&](std::vector<float> & audio_chunk,
                              int32_t new_frames,
                              bool is_last_chunk,
                              bool trim_final) -> bool {
            if (trim_final) postprocess_audio(audio_chunk, params);
            else apply_output_gain(audio_chunk, params.prosody_volume_db);

            std::vector<int16_t> pcm;
            float_to_int16(audio_chunk, pcm);
            audio_chunk.clear();
            audio_chunk.shrink_to_fit();

            const float dur_s = pcm.size() / static_cast<float>(codec_.sample_rate());
            std::cout << "[Stream] Chunk " << new_frames << " stable frames ("
                      << dur_s << "s)" << (is_last_chunk ? " [LAST]" : "") << "\n";

            const bool ok = callback(pcm.data(), pcm.size(), is_last_chunk);
            if (ok && is_last_chunk) terminal_sent = true;
            return ok;
        };

        auto decode_available = [&](bool finalize_codec, bool terminal_chunk) -> bool {
            if (accumulated_codes.empty()) return false;
            const size_t available_sz = accumulated_codes[0].size();
            if (available_sz > static_cast<size_t>(std::numeric_limits<int32_t>::max())) return false;
            const int32_t frames_available = static_cast<int32_t>(available_sz);
            for (const auto & row : accumulated_codes) {
                if (row.size() != available_sz) return false;
            }
            if (frames_available <= 0) return true;

            const int32_t stable_frames = finalize_codec
                ? frames_available
                : std::max(0, frames_available - history_target);
            if (stable_frames <= committed_frames) return true;

            const int32_t window_start = std::max(0, committed_frames - history_target);
            const int32_t window_frames = frames_available - window_start;
            if (window_frames <= 0) return false;
            const uint64_t code_count64 = static_cast<uint64_t>(num_cb) * window_frames;
            if (code_count64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(int32_t))) {
                return false;
            }

            std::vector<int32_t> decode_codes(static_cast<size_t>(code_count64));
            for (int32_t cb = 0; cb < num_cb; ++cb) {
                const auto first = accumulated_codes[static_cast<size_t>(cb)].begin() + window_start;
                std::copy(first,
                          accumulated_codes[static_cast<size_t>(cb)].begin() + frames_available,
                          decode_codes.begin() + static_cast<size_t>(cb) * window_frames);
            }

            std::vector<float> decoded;
            if (!codec_.decode_chunked(decode_codes.data(), window_frames,
                                       params.gen.n_threads, decoded,
                                       params.codec_chunk_frames,
                                       params.codec_overlap_frames)) {
                std::cerr << "[Stream] decode_chunked failed for stable window ["
                          << window_start << ".." << frames_available << ").\n";
                return false;
            }

            const uint64_t begin64 = static_cast<uint64_t>(committed_frames - window_start) *
                                     static_cast<uint64_t>(samples_per_frame);
            const uint64_t end64 = static_cast<uint64_t>(stable_frames - window_start) *
                                   static_cast<uint64_t>(samples_per_frame);
            if (begin64 > end64 || end64 > decoded.size()) {
                std::cerr << "[Stream] decoder frame/sample mapping mismatch.\n";
                return false;
            }

            std::vector<float> audio_chunk(
                decoded.begin() + static_cast<std::ptrdiff_t>(begin64),
                decoded.begin() + static_cast<std::ptrdiff_t>(end64));
            const int32_t new_frames = stable_frames - committed_frames;

            bool ok = true;
            if (params.trim_silence) {
                // Never trim an internal stride. Delay one stable range so only
                // the actual request tail can be trimmed.
                if (!deferred_audio.empty()) {
                    ok = emit_audio(deferred_audio, deferred_frames, false, false);
                    deferred_frames = 0;
                }
                if (ok) {
                    if (terminal_chunk) {
                        ok = emit_audio(audio_chunk, new_frames, true, true);
                    } else {
                        deferred_audio = std::move(audio_chunk);
                        deferred_frames = new_frames;
                    }
                }
            } else {
                ok = emit_audio(audio_chunk, new_frames, terminal_chunk, false);
            }
            if (ok) committed_frames = stable_frames;
            return ok;
        };

        // FrameCallback for generate_streaming. Accumulate in native
        // (codebook,row) layout and attempt a stable commit every stride.
        auto on_frame = [&](const int32_t * codes, int32_t n_cb) -> bool {
            if (!cb_ok) return false;
            if (should_continue && !should_continue()) {
                cb_ok = false;
                return false;
            }
            if (!codes || n_cb != num_cb) {
                std::cerr << "[Stream] generator returned an invalid codebook frame.\n";
                cb_ok = false;
                return false;
            }
            for (int32_t cb = 0; cb < n_cb; ++cb) {
                if (codes[cb] < 0 || codes[cb] >= model_.hparams().codebook_size) {
                    std::cerr << "[Stream] generator returned an out-of-range codebook id.\n";
                    cb_ok = false;
                    return false;
                }
                accumulated_codes[static_cast<size_t>(cb)].push_back(codes[cb]);
            }

            if (accumulated_codes[0].size() >
                static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                std::cerr << "[Stream] generated frame count exceeds supported range.\n";
                cb_ok = false;
                return false;
            }
            const int32_t frames_available = static_cast<int32_t>(accumulated_codes[0].size());
            if (frames_available - last_decode_frames >= decode_stride) {
                cb_ok = decode_available(false, false);
                last_decode_frames = frames_available;
            }
            return cb_ok;
        };

        GenerateResult stream_res = generate_streaming(model_, tokenizer_.config(), pt, gp, on_frame);

        // Release model KV before the final codec flush. The scope guard also
        // guarantees cleanup if generation throws or the callback cancels.
        kv_guard.cleanup_now();
        if (!cb_ok) return false;
        if (!stream_res.success || stream_res.n_frames <= 0) {
            std::cerr << "[Stream] generation failed or produced no audio frames.\n";
            return false;
        }
        if (accumulated_codes.empty() ||
            accumulated_codes[0].size() != static_cast<size_t>(stream_res.n_frames)) {
            std::cerr << "[Stream] generated-frame accounting mismatch.\n";
            return false;
        }

        // Always finalize, including exact stride multiples. This releases the
        // right-edge holdback that deliberately was not committed earlier.
        cb_ok = decode_available(true, is_last_seg);

        // With trimming enabled the last decoded chunk is intentionally delayed
        // until we know whether this is the final segment. Internal segment tails
        // are emitted unchanged; only the true request tail is trimmed.
        if (cb_ok && params.trim_silence && !deferred_audio.empty()) {
            cb_ok = emit_audio(deferred_audio, deferred_frames, is_last_seg, is_last_seg);
            deferred_frames = 0;
        }

        if (cb_ok && is_last_seg && !terminal_sent) {
            // Exact multiples with trimming disabled have already flushed all
            // audio as non-terminal chunks. Honour the callback contract.
            cb_ok = callback(nullptr, 0, true);
            if (cb_ok) terminal_sent = true;
        }

        if (cb_ok && !is_last_seg && (history_limit != 0 || auto_voice_anchor_pending)) {
            const int32_t frames = stream_res.n_frames;
            const uint64_t count64 = static_cast<uint64_t>(num_cb) * static_cast<uint64_t>(frames);
            if (count64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) return false;
            std::vector<int32_t> generated_codes(static_cast<size_t>(count64));
            for (int32_t cb = 0; cb < num_cb; ++cb) {
                if (accumulated_codes[static_cast<size_t>(cb)].size() != static_cast<size_t>(frames)) return false;
                std::copy(accumulated_codes[static_cast<size_t>(cb)].begin(),
                          accumulated_codes[static_cast<size_t>(cb)].end(),
                          generated_codes.begin() + static_cast<size_t>(cb) * frames);
            }
            if (auto_voice_anchor_pending) {
                if (!adopt_generated_voice_anchor(seg, generated_codes, frames, num_cb,
                                                  ref_codes, T_prompt, active_voice_transcript_)) {
                    std::cerr << "[Stream] could not preserve the generated voice across segments.\n";
                    return false;
                }
                auto_voice_anchor_pending = false;
            } else if (history_limit != 0) {
                if (!append_prompt_history(prompt_history, history_limit, seg, std::move(generated_codes),
                                           frames, num_cb)) return false;
            }
        }

        return cb_ok;
    };

    // -----------------------------------------------------------------------
    // Helper: segmento completo (comportamiento previo, sin stride)
    // -----------------------------------------------------------------------
    auto stream_segment_full = [&](const std::string & seg, bool is_last, size_t segment_index) -> bool {
        if (should_continue && !should_continue()) return false;
        std::vector<float>   audio;
        std::vector<int16_t> pcm;
        std::vector<int32_t> generated_codes;
        int32_t generated_frames = 0;
        const bool capture_anchor = auto_voice_anchor_pending && !is_last;
        // Request history is only needed to condition a later segment. Never
        // allocate/copy final-segment VQ just to append state that is discarded
        // when this request returns.
        const bool capture_history = history_limit != 0 && !capture_anchor && !is_last;
        const bool capture_codes = capture_anchor || capture_history;

        PipelineParams seg_params = params;
        if (request_is_chunked(params)) seg_params.gen.seed = derive_segment_seed(params.gen.seed, segment_index);
        if (!synthesize_segment(seg_params, seg, ref_codes, T_prompt, audio,
                                capture_codes ? &generated_codes : nullptr,
                                capture_codes ? &generated_frames : nullptr,
                                history_limit != 0 ? &prompt_history : nullptr)) {
            std::cerr << "[Stream] synthesize_segment failed: \"" << seg << "\"\n";
            return false;
        }
        if (should_continue && !should_continue()) return false;
        if (capture_history) {
            if (!append_prompt_history(prompt_history, history_limit, seg, std::move(generated_codes),
                                       generated_frames, model_.hparams().num_codebooks)) return false;
        } else if (capture_anchor) {
            if (!adopt_generated_voice_anchor(seg, generated_codes, generated_frames,
                                              model_.hparams().num_codebooks, ref_codes, T_prompt,
                                              active_voice_transcript_)) {
                std::cerr << "[Stream] could not preserve the generated voice across segments.\n";
                return false;
            }
            auto_voice_anchor_pending = false;
        }

        // A segment boundary is internal audio. Trimming every sentence removes
        // natural pauses, so only trim the final segment of the request.
        if (is_last) postprocess_audio(audio, params);
        else apply_output_gain(audio, params.prosody_volume_db);
        float_to_int16(audio, pcm);
        audio.clear(); audio.shrink_to_fit();

        float dur_s = pcm.size() / (float)codec_.sample_rate();
        std::cout << "[Stream] Sending " << pcm.size() << " samples ("
                  << dur_s << "s)" << (is_last ? " [LAST]" : "") << "\n";

        return callback(pcm.data(), pcm.size(), is_last);
    };

    // -----------------------------------------------------------------------
    // Dispatch: con stride o sin stride
    // -----------------------------------------------------------------------
    auto segs = request_text_segments(params);
    if (segs.empty() || segs.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) return false;
    std::cout << "[Stream] " << segs.size() << " segments"
              << (use_stride ? " (stride=" + std::to_string(stride) + ")" : "") << ".\n";
    for (size_t i = 0; i < segs.size(); ++i) {
        if (should_continue && !should_continue()) {
            std::cout << "[Stream] Cancelled.\n";
            return false;
        }
        const bool is_last = (i + 1 == segs.size());
        const bool ok = use_stride ? stream_segment_stride(segs[i], is_last, i)
                                   : stream_segment_full(segs[i], is_last, i);
        if (!ok) { std::cout << "[Stream] Callback aborted.\n"; return false; }
    }
    if (segments_out) *segments_out = static_cast<int32_t>(segs.size());

    return true;
}

} // namespace s2
