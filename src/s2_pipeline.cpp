#include "../include/s2_pipeline.h"
#include "../third_party/filesystem.hpp"
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
#else
#  include <unistd.h>
#  include <fcntl.h>
#endif

namespace s2 {

namespace {

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
    const std::wstring wp = fs::path(path).wstring();
    if (!wp.empty()) (void)DeleteFileW(wp.c_str());
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
                                     const std::string & final_path) noexcept {
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

// ---------------------------------------------------------------------------
// TempPcmFile -- archivo temporal de PCM crudo en %TEMP% (o /tmp en Linux).
//
// Escribe float32 -> int16 directamente a disco segmento a segmento.
// Nunca acumula mas de un segmento en RAM.
// Al terminar, total_samples() devuelve el numero total de muestras escritas
// y el FILE* se puede rebobinar para leer y construir el WAV final.
// ---------------------------------------------------------------------------
struct TempPcmFile {
    FILE*    fp          = nullptr;
    uint64_t total_samps = 0;   // int16 samples escritas en total
    std::string path;
#ifdef _WIN32
    std::wstring path_w;
#endif

    bool open() {
#ifdef _WIN32
        wchar_t tmp_dir[MAX_PATH] = {};
        const DWORD dir_len = GetTempPathW(MAX_PATH, tmp_dir);
        if (dir_len == 0 || dir_len >= MAX_PATH) return false;

        // This helper stores raw PCM; the extension is irrelevant. Keeping the
        // path returned by GetTempFileNameW also avoids a rename race/failure.
        wchar_t tmp_file[MAX_PATH] = {};
        if (GetTempFileNameW(tmp_dir, L"s2_", 0, tmp_file) == 0) return false;

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
#else
        path = "/tmp/s2_XXXXXX.pcm";
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
        return true;
    }

    // Escribe un segmento de audio float32 -> int16 a disco.
    // Solo este segmento necesita estar en RAM simultaneamente.
    bool write_segment(const std::vector<float> & samples) {
        if (!fp || samples.empty()) return false;
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
        // Every non-streaming caller ultimately emits classic RIFF/WAV, whose
        // data chunk is uint32-sized. Stop before filling the disk with a temp
        // PCM file that can never be represented by the final container.
        constexpr uint64_t MAX_WAV_I16_SAMPLES =
            (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - 36u) / sizeof(int16_t);
        if (total_samps > MAX_WAV_I16_SAMPLES ||
            samples.size() > static_cast<size_t>(MAX_WAV_I16_SAMPLES - total_samps)) {
            std::cerr << "Pipeline error: accumulated audio exceeds classic WAV/RIFF 4 GiB limit.\n";
            return false;
        }
        size_t written = std::fwrite(pcm.data(), sizeof(int16_t), pcm.size(), fp);
        if (written != pcm.size()) return false;
        total_samps += static_cast<uint64_t>(written);
        return true;
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
    }

    ~TempPcmFile() { cleanup(); }
};

// ---------------------------------------------------------------------------
// build_wav_header -- 44 bytes estandar PCM WAV
// ---------------------------------------------------------------------------
static void build_wav_header(char * hdr, uint32_t n_samples, int32_t sample_rate,
                              int16_t n_channels = 1, int16_t bits = 16) {
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
// split_sentences -- divide texto en oraciones respetando abreviaturas
// ---------------------------------------------------------------------------
std::vector<std::string> Pipeline::split_sentences(const std::string & text,
                                                    int32_t min_chars) {
    std::vector<std::string> sentences;
    if (text.empty()) return sentences;

    static const std::vector<std::string> abbrevs = {
        "mr","mrs","ms","dr","prof","sr","sra","dra","ing","lic",
        "etc","vs","fig","dept","approx","jan","feb","mar","apr",
        "jun","jul","aug","sep","oct","nov","dec","ene","abr","ago","dic"
    };

    std::string current;
    current.reserve(256);

    auto flush = [&]() {
        size_t s = current.find_first_not_of(" \t\n\r");
        size_t e = current.find_last_not_of(" \t\n\r");
        if (s != std::string::npos) {
            const size_t seg_len = e - s + 1;
            if (min_chars <= 0 ||
                seg_len >= static_cast<size_t>(min_chars))
                sentences.push_back(current.substr(s, e - s + 1));
            // Si el segmento es demasiado corto, se fusiona con el siguiente
            // dejando current sin limpiar para acumular mas texto.
            else { current = current.substr(s, e - s + 1) + " "; return; }
        }
        current.clear();
    };

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        current += c;

        bool is_end = (c == '.' || c == '!' || c == '?');
        if (!is_end) continue;

        // Consumir comillas/parentesis de cierre
        size_t j = i + 1;
        while (j < text.size() && (text[j] == '"' || text[j] == '\'' ||
               text[j] == ')' || text[j] == ']'))
            current += text[j++];

        if (j >= text.size() || text[j] == ' ' || text[j] == '\n') {
            if (c == '.') {
                // Comprobar abreviatura
                size_t we = i, ws = we;
                while (ws > 0 && std::isalpha((unsigned char)text[ws-1])) --ws;
                std::string word = text.substr(ws, we - ws);
                std::transform(word.begin(), word.end(), word.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
                bool abbrev = (word.size() <= 1);
                for (auto & ab : abbrevs) if (word == ab) { abbrev = true; break; }
                if (abbrev) { i = j - 1; continue; }
            }
            i = j - 1;
            flush();
        }
    }
    flush();
    // If the final sentence was below min_chars, flush() intentionally left it
    // in current waiting for a following sentence. There is no following one,
    // so merge it backward instead of silently dropping it.
    size_t s = current.find_first_not_of(" \t\n\r");
    size_t e = current.find_last_not_of(" \t\n\r");
    if (s != std::string::npos) {
        std::string tail = current.substr(s, e - s + 1);
        if (!sentences.empty()) sentences.back() += " " + tail;
        else sentences.push_back(std::move(tail));
    }
    return sentences;
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
        const ModelHParams & hp = model_.hparams();
        TokenizerConfig    & tc = tokenizer_.config();
        if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
        if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
        if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
        if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
        if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
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
        have_ref_txt = std::any_of(reference_text_.begin(), reference_text_.end(),
                                   [](unsigned char c) { return !std::isspace(c); });
        if (!have_ref_txt) reference_text_.clear();
    }

    if (have_ref_wav && have_ref_txt) {
        AudioData ra;
        std::vector<int32_t> rc;
        int32_t Tp = 0;
        if (load_audio(ref_wav, ra, codec_.sample_rate()) &&
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

// ---------------------------------------------------------------------------
// synthesize_segment -- genera audio float32 para un fragmento de texto.
// El llamador decide si lo guarda en RAM o lo vuelca a disco.
// ---------------------------------------------------------------------------
bool Pipeline::synthesize_segment(
        const PipelineParams       & params,
        const std::string          & text_segment,
        const std::vector<int32_t> & ref_codes,
        int32_t                      T_prompt,
        std::vector<float>         & audio_out) {

    const int32_t num_cb = model_.hparams().num_codebooks;

    // Prefer per-request prompt_text (HTTP/CLI); fall back to global reference_text_
    // loaded from reference.txt at init. Using reference_text_ here was a bug that
    // caused voice cloning prompt text to be ignored in HTTP and CLI modes.
    const std::string & effective_prompt_text = !params.prompt_text.empty()
        ? params.prompt_text
        : (!active_voice_transcript_.empty() ? active_voice_transcript_ : reference_text_);

    PromptTensor prompt = build_prompt(
        tokenizer_, text_segment, ref_codes.empty() ? std::string{} : effective_prompt_text,
        ref_codes.empty() ? nullptr : ref_codes.data(),
        num_cb, T_prompt);
    if (prompt.cols <= 0 || prompt.data.empty()) {
        std::cerr << "Pipeline error: invalid/empty prompt tensor.\n";
        return false;
    }

    // KV cache: reutilizar si cabe.
    // En modo segmentado, limitar max_new_tokens al minimo necesario para el segmento
    // para evitar OOM en GPUs con VRAM ajustada (RTX 3050 4GB con modelo+codec en VRAM).
    int32_t seg_max_tokens = params.gen.max_new_tokens;
    if (params.max_tokens_per_segment > 0 && params.max_tokens_per_segment < seg_max_tokens) {
        seg_max_tokens = params.max_tokens_per_segment;
    }
    const int32_t model_ctx = model_.hparams().context_length;
    if (model_ctx <= 0 || prompt.cols >= model_ctx) {
        std::cerr << "Pipeline error: prompt is too long for the model context ("
                  << prompt.cols << " >= " << model_ctx << ").\n";
        return false;
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
    return true;
}

// ---------------------------------------------------------------------------
// postprocess_audio -- trim silence (otras operaciones pueden anadirse aqui)
// ---------------------------------------------------------------------------
void Pipeline::postprocess_audio(std::vector<float> & audio, const PipelineParams & params) const {
    if (params.trim_silence && !audio.empty()) {
        auto trimmed = audio_trim_trailing_silence(audio.data(), audio.size(), codec_.sample_rate());
        if (!trimmed.empty()) audio = std::move(trimmed);
    }
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
        if (params.voice_id.empty() || params.prompt_text.empty() || codes.empty() || T_prompt <= 0) {
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
    if (!params.prompt_audio_path.empty() && params.prompt_text.empty()) {
        std::cerr << "[Voice] reference audio requires its prompt transcript (prompt_text).\n";
        return false;
    }
    if (!params.prompt_audio_path.empty()) {
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

// ---------------------------------------------------------------------------
// synthesize -- guarda a disco (usa TempPcmFile para RAM minima)
// ---------------------------------------------------------------------------
bool Pipeline::synthesize(const PipelineParams & params) {
    if (!initialized_) { std::cerr << "Pipeline not initialized.\n"; return false; }

    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    if (!get_ref_codes(params, ref_codes, T_prompt)) return false;

    // Siempre usar TempPcmFile para save_audio: RAM = 1 segmento a la vez
    TempPcmFile tmp;
    if (!tmp.open()) {
        std::cerr << "Pipeline error: could not open temp file.\n";
        return false;
    }

    auto process_segment = [&](const std::string & seg) -> bool {
        std::vector<float> audio;
        if (!synthesize_segment(params, seg, ref_codes, T_prompt, audio)) return false;
        postprocess_audio(audio, params);
        return tmp.write_segment(audio);
        // 'audio' se destruye aqui -> RAM liberada antes del siguiente segmento
    };

    if (params.segment_sentences) {
        auto segs = split_sentences(params.text, params.min_seg_chars);
        std::cout << "[Segment] " << segs.size() << " sentences.\n";
        for (size_t i = 0; i < segs.size(); ++i) {
            std::cout << "[" << (i+1) << "/" << segs.size() << "] \"" << segs[i] << "\"\n";
            if (!process_segment(segs[i])) {
                std::cerr << "Segment " << (i+1) << " failed -- aborting request.\n";
                return false;
            }
        }
    } else {
        if (!process_segment(params.text)) return false;
    }
    if (tmp.total_samps == 0) {
        std::cerr << "Pipeline error: no audio generated.\n";
        return false;
    }

    // Construir WAV final desde el archivo temporal
    const std::string final_path = params.output_path.empty() ? "out.wav" : params.output_path;
    if (!tmp.fp || std::fflush(tmp.fp) != 0) {
        std::cerr << "Pipeline error: could not flush temporary PCM file.\n";
        return false;
    }
    std::clearerr(tmp.fp);
    if (!rewind_binary_file(tmp.fp)) {
        std::cerr << "Pipeline error: could not rewind temporary PCM file.\n";
        return false;
    }
    std::clearerr(tmp.fp);
    // Never truncate an existing valid output until the complete replacement
    // WAV has been written and synced. The temporary lives beside the target so
    // the final rename/replace stays on the same filesystem and is atomic.
    const std::string staged_path = output_temp_path(final_path);
    struct StagedCleanup {
        std::string path;
        ~StagedCleanup() { if (!path.empty()) remove_file_utf8(path); }
    } staged_cleanup{staged_path};
    FILE * out = open_binary_output_utf8(staged_path);
    if (!out) {
        std::cerr << "Pipeline error: could not create staged output for " << final_path << "\n";
        return false;
    }
    struct OutputGuard { FILE * f; ~OutputGuard() { if (f) std::fclose(f); } } out_guard{out};

    if (tmp.total_samps > (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - 36u) / sizeof(int16_t)) {
        std::cerr << "Pipeline error: output exceeds classic WAV/RIFF 4 GiB limit.\n";
        return false;
    }
    char hdr[44];
    build_wav_header(hdr, static_cast<uint32_t>(tmp.total_samps), codec_.sample_rate());
    if (std::fwrite(hdr, 1, sizeof(hdr), out) != sizeof(hdr)) return false;

    // Streamear PCM desde disco -> disco sin pasar por RAM. Sabemos el
    // tamano exacto esperado; copiar hasta EOF ocultaria un temporal truncado.
    char copy_buf[65536];
    uint64_t remaining_bytes = tmp.total_samps * sizeof(int16_t);
    while (remaining_bytes > 0) {
        const size_t want = static_cast<size_t>(std::min<uint64_t>(remaining_bytes, sizeof(copy_buf)));
        const size_t n = std::fread(copy_buf, 1, want, tmp.fp);
        if (n != want) {
            std::cerr << "Pipeline error: temporary PCM file ended before expected size.\n";
            return false;
        }
        if (std::fwrite(copy_buf, 1, n, out) != n) return false;
        remaining_bytes -= n;
    }
    if (std::ferror(tmp.fp) || !sync_binary_file(out)) return false;
    if (std::fclose(out) != 0) { out_guard.f = nullptr; return false; }
    out_guard.f = nullptr;
    if (!replace_file_atomic_utf8(staged_path, final_path)) {
        std::cerr << "Pipeline error: could not atomically replace " << final_path << "\n";
        return false;
    }
    staged_cleanup.path.clear();

    std::cout << "Saved " << tmp.total_samps << " samples to " << final_path << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// synthesize_to_buffer -- para HTTP (Crow).
//
// Estrategia de RAM minima:
//   - Cada segmento se escribe al TempPcmFile (disco, %TEMP%)
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

    std::cout << "--- Synthesize ---\n"
              << "Text: " << params.text << "\n"
              << "Mode: " << (params.segment_sentences ? "segmentado" : "bloque completo") << "\n";

    auto t0 = std::chrono::steady_clock::now();

    // Referencia de voz -- con cache LRU para evitar re-encodear entre requests
    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    auto t_ref0 = std::chrono::steady_clock::now();
    if (!get_ref_codes(params, ref_codes, T_prompt)) return false;
    auto t_ref1 = std::chrono::steady_clock::now();
    std::cout << "[T] Ref audio: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_ref1-t_ref0).count()
              << " ms" << (ref_codes.empty() ? " (sin referencia)" : " (listo)") << "\n";

    // Abrir archivo temporal para PCM crudo
    TempPcmFile tmp;
    if (!tmp.open()) {
        std::cerr << "Pipeline error: GetTempFileName failed.\n";
        return false;
    }
    std::cout << "[TempPCM] " << tmp.path << "\n";

    // Funcion lambda: sintetiza un segmento y lo vuelca a disco inmediatamente
    uint64_t total_frames = 0;
    auto process_segment = [&](const std::string & seg) -> bool {
        auto ts = std::chrono::steady_clock::now();
        std::vector<float> audio; // solo este segmento en RAM
        if (!synthesize_segment(params, seg, ref_codes, T_prompt, audio)) return false;
        postprocess_audio(audio, params);
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

    if (params.segment_sentences) {
        auto segs = split_sentences(params.text, params.min_seg_chars);
        std::cout << "[Segment] " << segs.size() << " sentences detected.\n";
        for (size_t i = 0; i < segs.size(); ++i) {
            std::cout << "[" << (i+1) << "/" << segs.size() << "] \""
                      << segs[i] << "\"\n";
            if (!process_segment(segs[i])) {
                std::cerr << "Segment " << (i+1) << " failed -- aborting request.\n";
                return false;
            }
        }
    } else {
        if (!process_segment(params.text)) return false;
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

    // Construir output_buffer: cabecera WAV (44B) + leer PCM desde disco
    // Pico de RAM aqui = WAV completo (inevitable para HTTP response body)
    const int32_t sr = codec_.sample_rate();
    if (tmp.total_samps > (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - 36u) / sizeof(int16_t)) {
        std::cerr << "Pipeline error: output exceeds classic WAV/RIFF 4 GiB limit.\n";
        return false;
    }
    const uint32_t data_bytes = static_cast<uint32_t>(tmp.total_samps * sizeof(int16_t));
    // Build into a local result and publish it only after every I/O step succeeds.
    // Callers that reuse a buffer therefore never observe a half-written WAV.
    std::vector<char> result(44ull + data_bytes);
    build_wav_header(result.data(), static_cast<uint32_t>(tmp.total_samps), sr);

    if (!tmp.fp || std::fflush(tmp.fp) != 0) {
        std::cerr << "Pipeline error: could not flush temporary PCM file.\n";
        return false;
    }
    std::clearerr(tmp.fp);
    if (!rewind_binary_file(tmp.fp)) {
        std::cerr << "Pipeline error: could not rewind temporary PCM file.\n";
        return false;
    }
    std::clearerr(tmp.fp);
    size_t read = std::fread(result.data() + 44, 1, data_bytes, tmp.fp);
    if (read != data_bytes) {
        std::cerr << "Pipeline error: read " << read << "/" << data_bytes << " bytes del temp.\n";
        return false;
    }
    if (std::ferror(tmp.fp)) return false;
    output_buffer = std::move(result);
    // tmp se destruye aqui -> archivo temporal borrado automaticamente

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
//   - PCM se vuelca por segmentos y el WAV final se arma por bloques.
// ---------------------------------------------------------------------------
bool Pipeline::synthesize_to_file(const PipelineParams & params,
                                   std::string          & out_wav_path) {
    out_wav_path.clear();
    if (!initialized_) { std::cerr << "Pipeline not initialized.\n"; return false; }

    std::cout << "--- Synthesize to file ---\n"
              << "Text: " << params.text << "\n"
              << "Mode: " << (params.segment_sentences ? "segmentado" : "bloque") << "\n";

    auto t0 = std::chrono::steady_clock::now();

    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    if (!get_ref_codes(params, ref_codes, T_prompt)) return false;

    // Escribir PCM crudo al TempPcmFile
    TempPcmFile tmp;
    if (!tmp.open()) {
        std::cerr << "Pipeline error: could not create archivo temporal PCM.\n";
        return false;
    }

    auto process_seg = [&](const std::string & seg) -> bool {
        auto ts = std::chrono::steady_clock::now();
        std::vector<float> audio;
        if (!synthesize_segment(params, seg, ref_codes, T_prompt, audio)) return false;
        postprocess_audio(audio, params);
        float dur_s = audio.size() / (float)codec_.sample_rate();
        bool ok = tmp.write_segment(audio);
        // audio destruido aqui -- RAM liberada
        auto te = std::chrono::steady_clock::now();
        float inf_s = std::chrono::duration_cast<std::chrono::milliseconds>(te-ts).count()/1000.f;
        std::cout << "  -> " << dur_s << "s audio in " << inf_s << "s ("
                  << (dur_s/std::max(inf_s,0.001f)) << "x RT)\n";
        return ok;
    };

    if (params.segment_sentences) {
        auto segs = split_sentences(params.text, params.min_seg_chars);
        std::cout << "[Segment] " << segs.size() << " sentences.\n";
        for (size_t i = 0; i < segs.size(); ++i) {
            std::cout << "[" << (i+1) << "/" << segs.size() << "] \"" << segs[i] << "\"\n";
            if (!process_seg(segs[i])) {
                std::cerr << "Segment " << (i+1) << " failed -- aborting request.\n";
                return false;
            }
        }
    } else {
        if (!process_seg(params.text)) return false;
    }

    if (tmp.total_samps == 0) {
        std::cerr << "Pipeline error: no audio generated.\n";
        return false;
    }

    // Construir el WAV final en un NUEVO archivo temporal (con cabecera).
    // El TempPcmFile solo tiene PCM crudo sin cabecera -- necesitamos un
    // archivo WAV completo para que Crow lo sirva con Content-Type correcto.
    TempPcmFile wav_tmp;
    if (!wav_tmp.open()) {
        std::cerr << "Pipeline error: could not create archivo temporal WAV.\n";
        return false;
    }

    // Escribir cabecera WAV
    if (tmp.total_samps > (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) - 36u) / sizeof(int16_t)) {
        std::cerr << "Pipeline error: output exceeds classic WAV/RIFF 4 GiB limit.\n";
        return false;
    }
    char hdr[44];
    build_wav_header(hdr, static_cast<uint32_t>(tmp.total_samps), codec_.sample_rate());
    if (std::fwrite(hdr, 1, 44, wav_tmp.fp) != 44) return false;

    // Copiar PCM desde tmp -> wav_tmp en bloques de 64KB (sin pasar por RAM)
    if (!tmp.fp || std::fflush(tmp.fp) != 0) {
        std::cerr << "Pipeline error: could not flush temporary PCM file.\n";
        return false;
    }
    std::clearerr(tmp.fp);
    if (!rewind_binary_file(tmp.fp)) {
        std::cerr << "Pipeline error: could not rewind temporary PCM file.\n";
        return false;
    }
    std::clearerr(tmp.fp);
    char copy_buf[65536];
    uint64_t remaining_bytes = tmp.total_samps * sizeof(int16_t);
    while (remaining_bytes > 0) {
        const size_t want = static_cast<size_t>(std::min<uint64_t>(remaining_bytes, sizeof(copy_buf)));
        const size_t n = std::fread(copy_buf, 1, want, tmp.fp);
        if (n != want) {
            std::cerr << "Pipeline error: temporary PCM file ended before expected size.\n";
            return false;
        }
        if (std::fwrite(copy_buf, 1, n, wav_tmp.fp) != n) return false;
        remaining_bytes -= n;
    }
    if (std::ferror(tmp.fp) || std::fflush(wav_tmp.fp) != 0) return false;

    auto t1 = std::chrono::steady_clock::now();
    float total_s   = tmp.total_samps / (float)codec_.sample_rate();
    float elapsed_s = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()/1000.f;
    std::cout << "[Timing] " << total_s << "s audio in " << elapsed_s << "s ("
              << (total_s/std::max(elapsed_s,0.001f)) << "x RT)\n";
    std::cout << "[File] WAV: " << wav_tmp.path
              << " (" << (44 + tmp.total_samps*2)/1024 << " KB)\n";

    // Transferir ownership de la ruta -- el llamador borra el archivo
    out_wav_path = wav_tmp.release_path();
    // wav_tmp.fp se cierra al destruir -- el archivo queda en disco
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
                                     int32_t *              segments_out) {
    if (segments_out) *segments_out = 0;
    if (!initialized_) { std::cerr << "Pipeline not initialized.\n"; return false; }
    if (!callback)      { std::cerr << "synthesize_streaming: null callback.\n"; return false; }
    if (params.codec_chunk_frames < 0 || params.codec_overlap_frames < 0 ||
        params.stream_decode_stride_frames < -1) {
        std::cerr << "synthesize_streaming: invalid codec/streaming parameters.\n";
        return false;
    }

    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    if (!get_ref_codes(params, ref_codes, T_prompt)) return false;

    // Stride: numero de frames a acumular antes de cada decode+send.
    // 0 -> auto (4 frames), matching CLI/API documentation. Negative disables.
    // Valores tipicos: 4 (baja latencia) a 32 (menor overhead).
    const int32_t stride = params.stream_decode_stride_frames == 0 ? 4 : params.stream_decode_stride_frames;
    const bool    use_stride = (stride > 0);

    // -----------------------------------------------------------------------
    // Helper: genera un segmento con decode en tiempo real (stride activo)
    // -----------------------------------------------------------------------
    auto stream_segment_stride = [&](const std::string & seg, bool is_last_seg) -> bool {
        // Construir el prompt usando la API de s2_prompt
        const int32_t num_cb = model_.hparams().num_codebooks;

        // Construir PromptTensor usando build_prompt de s2_prompt.h
        PromptTensor pt = build_prompt(
            tokenizer_,
            seg,
            ref_codes.empty() ? std::string{} : (!params.prompt_text.empty() ? params.prompt_text : (!active_voice_transcript_.empty() ? active_voice_transcript_ : reference_text_)),
            ref_codes.empty() ? nullptr : ref_codes.data(),
            num_cb,
            T_prompt
        );
        if (pt.cols == 0) {
            std::cerr << "[Stream] build_prompt returned empty for: \"" << seg << "\"\n";
            return false;
        }

        // Configurar parametros de generacion para este segmento
        GenerateParams gp = params.gen;
        gp.max_new_tokens = std::min(params.gen.max_new_tokens, params.max_tokens_per_segment);
        gp.verbose        = false;
        const int32_t model_ctx = model_.hparams().context_length;
        if (model_ctx <= 0 || pt.cols >= model_ctx) {
            std::cerr << "[Stream] prompt is too long for model context.\n";
            return false;
        }
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
        KvCacheScope kv_guard{&model_, &kv_cache_initialized_, &kv_cache_max_len_, true};

        // Buffer de acumulacion de frames para decode por stride
        // `codec_overlap` now also applies ACROSS streaming flushes. Previously
        // every stride was decoded as an unrelated clip, so overlap had no effect
        // at the most important boundary. Keep enough prior VQ frames to provide
        // context while staying within the codec's advertised block size.
        const int32_t history_target = std::max(0, std::min(params.codec_overlap_frames,
                                                            codec_max_frames - decode_stride));
        std::vector<int32_t> history_codes;   // (num_cb, history_frames)
        int32_t history_frames = 0;

        std::vector<int32_t> pending_codes;   // current flush, (num_cb, T) when flushed
        pending_codes.reserve(static_cast<size_t>(num_cb) * decode_stride * 2);
        int32_t pending_frames = 0;
        bool    cb_ok          = true;

        // Funcion interna: decodifica pending_codes y envia al callback WS
        auto flush_pending = [&](bool is_last_chunk) -> bool {
            if (pending_frames == 0) return true;
            if (pending_codes.size() != static_cast<size_t>(num_cb) * pending_frames) return false;

            const int32_t use_history = std::min(history_frames, history_target);
            const int32_t combined_frames = use_history + pending_frames;
            if (combined_frames <= 0 || combined_frames > codec_max_frames) return false;

            std::vector<int32_t> decode_codes(static_cast<size_t>(num_cb) * combined_frames);
            for (int32_t cb = 0; cb < num_cb; ++cb) {
                int32_t * dst = decode_codes.data() + static_cast<size_t>(cb) * combined_frames;
                if (use_history > 0) {
                    const int32_t hist_start = history_frames - use_history;
                    const int32_t * hsrc = history_codes.data() + static_cast<size_t>(cb) * history_frames + hist_start;
                    std::copy(hsrc, hsrc + use_history, dst);
                }
                const int32_t * csrc = pending_codes.data() + static_cast<size_t>(cb) * pending_frames;
                std::copy(csrc, csrc + pending_frames, dst + use_history);
            }

            std::vector<float> audio_chunk;
            if (!codec_.decode_chunked(decode_codes.data(), combined_frames,
                                        params.gen.n_threads,
                                        audio_chunk,
                                        params.codec_chunk_frames,
                                        params.codec_overlap_frames)) {
                std::cerr << "[Stream] decode_chunked failed on " << combined_frames
                          << " frames (" << use_history << " history + " << pending_frames << " new).\n";
                pending_codes.clear(); pending_frames = 0;
                return false;
            }

            // Remove only the audio corresponding to the repeated history. The
            // decoder is frame-synchronous; proportional arithmetic avoids relying
            // on a hard-coded 512-sample hop if a compatible codec changes it.
            if (use_history > 0 && !audio_chunk.empty()) {
                const size_t uf = static_cast<size_t>(use_history);
                const size_t cf = static_cast<size_t>(combined_frames);
                if (audio_chunk.size() > std::numeric_limits<size_t>::max() / uf) return false;
                const size_t discard = (audio_chunk.size() * uf) / cf;
                if (discard >= audio_chunk.size()) return false;
                audio_chunk.erase(audio_chunk.begin(), audio_chunk.begin() + static_cast<std::ptrdiff_t>(discard));
            }

            // Retain the tail for the next streaming flush before pending_codes is
            // cleared. This state is per request/segment and never enters VoiceCache.
            const int32_t keep = std::min(history_target, combined_frames);
            if (keep > 0) {
                std::vector<int32_t> next_history(static_cast<size_t>(num_cb) * keep);
                for (int32_t cb = 0; cb < num_cb; ++cb) {
                    const int32_t * src = decode_codes.data() + static_cast<size_t>(cb) * combined_frames + (combined_frames - keep);
                    std::copy(src, src + keep, next_history.data() + static_cast<size_t>(cb) * keep);
                }
                history_codes = std::move(next_history);
                history_frames = keep;
            }

            postprocess_audio(audio_chunk, params);

            std::vector<int16_t> pcm;
            float_to_int16(audio_chunk, pcm);
            audio_chunk.clear(); audio_chunk.shrink_to_fit();

            float dur_s = pcm.size() / (float)codec_.sample_rate();
            std::cout << "[Stream] Chunk " << pending_frames << " new frames"
                      << (use_history > 0 ? " + context" : "") << " ("
                      << dur_s << "s)" << (is_last_chunk ? " [LAST]" : "") << "\n";

            bool ok = callback(pcm.data(), pcm.size(), is_last_chunk);
            pending_codes.clear(); pending_frames = 0;
            return ok;
        };

        // FrameCallback para generate_streaming
        auto on_frame = [&](const int32_t * codes, int32_t n_cb) -> bool {
            if (!cb_ok) return false;
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
            }
            // Append en row-major: pending_codes[cb * pending_frames_capacity + t]
            // Para simplificar, usamos layout (T, num_cb) transpuesto luego.
            // En realidad decode_chunked espera (num_cb, T) row-major.
            // Acumulamos frame a frame transponiendo al vuelo:
            //   Para el frame actual t=pending_frames:
            //     pending_codes[cb * max_T + t] = codes[cb]
            // Pero max_T no es conocido. Usamos (pending_frames, num_cb) y
            // transponemos al hacer flush. Mas simple: guardamos como (T, num_cb)
            // y transponemos en flush.
            for (int32_t cb = 0; cb < n_cb; ++cb) {
                pending_codes.push_back(codes[cb]);
            }
            pending_frames++;

            if (pending_frames >= decode_stride) {
                // Transponer de (T, num_cb) a (num_cb, T) para decode_chunked
                std::vector<int32_t> transposed(static_cast<size_t>(num_cb) * pending_frames);
                for (int32_t t = 0; t < pending_frames; ++t)
                    for (int32_t cb = 0; cb < num_cb; ++cb)
                        transposed[static_cast<size_t>(cb) * pending_frames + t] =
                            pending_codes[static_cast<size_t>(t) * num_cb + cb];
                pending_codes = std::move(transposed);
                // flush -- is_last_chunk=false (habra mas frames)
                cb_ok = flush_pending(false);
                // Reiniciar acumulador en formato (T, num_cb) para proximo stride
                pending_codes.clear();
            }
            return cb_ok;
        };

        GenerateResult stream_res = generate_streaming(model_, tokenizer_.config(), pt, gp, on_frame);

        // Release model KV before codec decode/final flush. The scope guard also
        // guarantees cleanup if generation throws or the callback cancels.
        kv_guard.cleanup_now();
        if (!stream_res.success || stream_res.n_frames <= 0) {
            std::cerr << "[Stream] generation failed or produced no audio frames.\n";
            return false;
        }

        // Flush frames restantes (el ultimo chunk, marcado como is_last)
        if (pending_frames > 0 && cb_ok) {
            // Transponer residuo (T, num_cb) -> (num_cb, T)
            std::vector<int32_t> transposed(static_cast<size_t>(num_cb) * pending_frames);
            for (int32_t t = 0; t < pending_frames; ++t)
                for (int32_t cb = 0; cb < num_cb; ++cb)
                    transposed[static_cast<size_t>(cb) * pending_frames + t] =
                        pending_codes[static_cast<size_t>(t) * num_cb + cb];
            pending_codes = std::move(transposed);
            cb_ok = flush_pending(is_last_seg);
        } else if (cb_ok && is_last_seg) {
            // Exact multiples of decode_stride have already flushed all audio
            // chunks with is_last=false. Honour the StreamCallback contract by
            // emitting a zero-length terminal marker.
            cb_ok = callback(nullptr, 0, true);
        }

        return cb_ok;
    };

    // -----------------------------------------------------------------------
    // Helper: segmento completo (comportamiento previo, sin stride)
    // -----------------------------------------------------------------------
    auto stream_segment_full = [&](const std::string & seg, bool is_last) -> bool {
        std::vector<float>   audio;
        std::vector<int16_t> pcm;

        if (!synthesize_segment(params, seg, ref_codes, T_prompt, audio)) {
            std::cerr << "[Stream] synthesize_segment failed: \"" << seg << "\"\n";
            return false;
        }

        postprocess_audio(audio, params);
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
    if (params.segment_sentences) {
        auto segs = split_sentences(params.text, params.min_seg_chars);
        if (segs.empty() || segs.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) return false;
        std::cout << "[Stream] " << segs.size() << " segments"
                  << (use_stride ? " (stride=" + std::to_string(stride) + ")" : "") << ".\n";
        for (size_t i = 0; i < segs.size(); ++i) {
            bool is_last = (i == segs.size() - 1);
            std::cout << "[Stream " << (i+1) << "/" << segs.size()
                      << "] \"" << segs[i] << "\"\n";
            bool ok = use_stride
                ? stream_segment_stride(segs[i], is_last)
                : stream_segment_full(segs[i], is_last);
            if (!ok) {
                std::cout << "[Stream] Callback aborted.\n";
                return false;
            }
        }
        if (segments_out) *segments_out = static_cast<int32_t>(segs.size());
    } else {
        bool ok = use_stride
            ? stream_segment_stride(params.text, true)
            : stream_segment_full(params.text, true);
        if (!ok) return false;
        if (segments_out) *segments_out = 1;
    }

    return true;
}

} // namespace s2
