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
#include <gguf.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <unistd.h>
#endif

namespace s2 {

Pipeline::Pipeline()  = default;
Pipeline::~Pipeline() = default;

// ---------------------------------------------------------------------------
// TempPcmFile — archivo temporal de PCM crudo en %TEMP% (o /tmp en Linux).
//
// Escribe float32 → int16 directamente a disco segmento a segmento.
// Nunca acumula más de un segmento en RAM.
// Al terminar, total_samples() devuelve el número total de muestras escritas
// y el FILE* se puede rebobinar para leer y construir el WAV final.
// ---------------------------------------------------------------------------
struct TempPcmFile {
    FILE*    fp          = nullptr;
    uint64_t total_samps = 0;   // int16 samples escritas en total
    std::string path;

    bool open() {
#ifdef _WIN32
        wchar_t tmp_dir[MAX_PATH];
        GetTempPathW(MAX_PATH, tmp_dir);
        // GetTempFileNameW crea un archivo con extensión .tmp — lo renombramos
        // a .wav para que Crow sirva el MIME type correcto (audio/wav).
        wchar_t tmp_base[MAX_PATH];
        GetTempFileNameW(tmp_dir, L"s2_", 0, tmp_base);
        // Construir ruta .wav: reemplazar extensión .tmp por .wav
        wchar_t tmp_file[MAX_PATH];
        wcsncpy_s(tmp_file, tmp_base, MAX_PATH);
        wchar_t * ext = wcsrchr(tmp_file, L'.');
        if (ext) wcscpy_s(ext, 5, L".wav");
        // Renombrar el archivo que Windows ya creó
        _wrename(tmp_base, tmp_file);
        // Convertir a UTF-8 para almacenar
        int n = WideCharToMultiByte(CP_UTF8, 0, tmp_file, -1, nullptr, 0, nullptr, nullptr);
        path.resize(n - 1);
        WideCharToMultiByte(CP_UTF8, 0, tmp_file, -1, path.data(), n, nullptr, nullptr);
        fp = _wfopen(tmp_file, L"w+b");
#else
        path = "/tmp/s2_XXXXXX.pcm";
        // mkstemps para extensión
        int fd = mkstemps(path.data(), 4);
        if (fd < 0) return false;
        fp = fdopen(fd, "w+b");
#endif
        return fp != nullptr;
    }

    // Escribe un segmento de audio float32 → int16 a disco.
    // Solo este segmento necesita estar en RAM simultáneamente.
    bool write_segment(const std::vector<float> & samples) {
        if (!fp || samples.empty()) return false;
        // Convertir float32 → int16 en un buffer temporal del tamaño del segmento
        std::vector<int16_t> pcm(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            float s = std::max(-1.0f, std::min(1.0f, samples[i]));
            pcm[i]  = static_cast<int16_t>(s * 32767.0f);
        }
        size_t written = std::fwrite(pcm.data(), sizeof(int16_t), pcm.size(), fp);
        total_samps += written;
        return written == pcm.size();
    }

    // Cierra y borra el archivo temporal
    void cleanup() {
        if (fp) { std::fclose(fp); fp = nullptr; }
        if (!path.empty()) {
#ifdef _WIN32
            std::remove(path.c_str());
#else
            ::unlink(path.c_str());
#endif
            path.clear();
        }
    }

    ~TempPcmFile() { cleanup(); }
};

// ---------------------------------------------------------------------------
// build_wav_header — 44 bytes estándar PCM WAV
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
// split_sentences — divide texto en oraciones respetando abreviaturas
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
        if (s != std::string::npos && (e - s) >= 3) {
            int32_t seg_len = static_cast<int32_t>(e - s + 1);
            if (min_chars <= 0 || seg_len >= min_chars)
                sentences.push_back(current.substr(s, e - s + 1));
            // Si el segmento es demasiado corto, se fusiona con el siguiente
            // dejando current sin limpiar para acumular más texto.
            else { current = current.substr(s, e - s + 1) + " "; return; }
        }
        current.clear();
    };

    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        current += c;

        bool is_end = (c == '.' || c == '!' || c == '?');
        if (!is_end) continue;

        // Consumir comillas/paréntesis de cierre
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
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                bool abbrev = (word.size() <= 1);
                for (auto & ab : abbrevs) if (word == ab) { abbrev = true; break; }
                if (abbrev) { i = j - 1; continue; }
            }
            i = j - 1;
            flush();
        }
    }
    flush();
    return sentences;
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
bool Pipeline::init(const PipelineParams & params) {
    std::cout << "--- Pipeline init ---" << std::endl;

    int model_gpu = params.vulkan_device;
    int codec_gpu = params.codec_vulkan_device;
    if (codec_gpu < 0) codec_gpu = model_gpu;

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
    std::cout << "Model loaded (GPU " << model_gpu << ").\n";

    // Codec
    std::string codec_path = params.codec_model_path.empty()
        ? params.model_path : params.codec_model_path;

    bool codec_ok = false;
    if (codec_gpu >= 0 && codec_.load(codec_path, codec_gpu)) {
        std::cout << "Codec loaded (GPU " << codec_gpu << ").\n";
        codec_ok = true;
    }
    if (!codec_ok && codec_.load(codec_path, -1)) {
        std::cout << "Codec loaded (CPU).\n";
        codec_ok = true;
    }
    if (!codec_ok) {
        std::cerr << "Pipeline error: codec load failed.\n";
        return false;
    }
    codec_path_ = codec_path;

    // Sincronizar hparams tokenizer ↔ modelo
    {
        const ModelHParams & hp = model_.hparams();
        TokenizerConfig    & tc = tokenizer_.config();
        if (hp.semantic_begin_id > 0) tc.semantic_begin_id = hp.semantic_begin_id;
        if (hp.semantic_end_id   > 0) tc.semantic_end_id   = hp.semantic_end_id;
        if (hp.num_codebooks     > 0) tc.num_codebooks     = hp.num_codebooks;
        if (hp.codebook_size     > 0) tc.codebook_size     = hp.codebook_size;
        if (hp.vocab_size        > 0) tc.vocab_size        = hp.vocab_size;
    }

    // Referencia de voz (opcional)
    const std::string ref_wav = params.base_dir + "reference.wav";
    const std::string ref_txt = params.base_dir + "reference.txt";
    if (std::FILE* f = std::fopen(ref_wav.c_str(), "rb")) {
        std::fclose(f);
        AudioData ra;
        if (load_audio(ref_wav, ra, codec_.sample_rate())) {
            std::vector<int32_t> rc; int32_t Tp = 0;
            if (codec_.encode(ra.samples.data(), (int32_t)ra.samples.size(),
                              params.gen.n_threads, rc, Tp)) {
                reference_embedding_ = std::string((const char*)rc.data(), rc.size()*sizeof(int32_t));
                reference_loaded_    = true;
                std::cout << "Reference audio: " << Tp << " frames.\n";
            }
        }
    }
    if (std::FILE* f = std::fopen(ref_txt.c_str(), "r")) {
        std::fclose(f);
        std::ifstream tf(ref_txt);
        std::getline(tf, reference_text_);
        std::cout << "Reference text loaded.\n";
    }

    initialized_ = true;
    std::cout << "--- Pipeline ready ---\n";
    return true;
}

// ---------------------------------------------------------------------------
// synthesize_segment — genera audio float32 para un fragmento de texto.
// El llamador decide si lo guarda en RAM o lo vuelca a disco.
// ---------------------------------------------------------------------------
bool Pipeline::synthesize_segment(
        const PipelineParams       & params,
        const std::string          & text_segment,
        const std::vector<int32_t> & ref_codes,
        int32_t                      T_prompt,
        std::vector<float>         & audio_out) {

    const int32_t num_cb = model_.hparams().num_codebooks;

    PromptTensor prompt = build_prompt(
        tokenizer_, text_segment, reference_text_,
        ref_codes.empty() ? nullptr : ref_codes.data(),
        num_cb, T_prompt);

    // KV cache: reutilizar si cabe.
    // En modo segmentado, limitar max_new_tokens al mínimo necesario para el segmento
    // para evitar OOM en GPUs con VRAM ajustada (RTX 3050 4GB con modelo+codec en VRAM).
    int32_t seg_max_tokens = params.gen.max_new_tokens;
    if (params.max_tokens_per_segment > 0 && params.max_tokens_per_segment < seg_max_tokens) {
        seg_max_tokens = params.max_tokens_per_segment;
    }
    int32_t max_seq = prompt.cols + seg_max_tokens;
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

    GenerateParams seg_gen = params.gen;
    seg_gen.max_new_tokens = seg_max_tokens;
    GenerateResult res = generate(model_, tokenizer_.config(), prompt, seg_gen);
    if (res.n_frames == 0) {
        std::cerr << "Pipeline error: generate() returned 0 frames.\n";
        return false;
    }

    // Liberar el KV cache inmediatamente después de generate() para recuperar
    // VRAM antes de que decode_chunked() intente allocar sus activaciones.
    // El siguiente segmento lo reinicializará con init_kv_cache().
    model_.free_kv_cache();
    kv_cache_initialized_ = false;
    kv_cache_max_len_     = 0;

    if (!codec_.decode_chunked(res.codes.data(), res.n_frames,
                               params.gen.n_threads, audio_out,
                               params.codec_chunk_frames,
                               params.codec_overlap_frames)) {
        // El codec q4_k_m no es compatible con backend CPU (GGML_ASSERT F16 en ops.cpp).
        // Si falla en GPU es OOM — reportar directamente sin intentar fallback CPU.
        std::cerr << "Pipeline error: decode_chunked() failed (GPU OOM).\n";
        std::cerr << "  Try reducing --codec-chunk, or use --codec-vulkan -1 with an f16/f32 codec.\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// postprocess_audio — trim silence (otras operaciones pueden añadirse aquí)
// ---------------------------------------------------------------------------
void Pipeline::postprocess_audio(std::vector<float> & audio, const PipelineParams & params) const {
    if (params.trim_silence && !audio.empty()) {
        auto trimmed = audio_trim_trailing_silence(audio.data(), audio.size(), codec_.sample_rate());
        if (!trimmed.empty()) audio = std::move(trimmed);
    }
}

// ---------------------------------------------------------------------------
// encode_reference — API pública para que main.cpp encodee sin sintetizar.
// Delega en get_ref_codes y hereda toda la lógica de VoiceCache.
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
// get_ref_codes — obtiene los codes de referencia de voz con caché LRU.
//
// Prioridad:
//   1. reference.wav cargado al init (ya en reference_embedding_)
//   2. prompt_audio_path del request — busca en caché, encodea si no está
// ---------------------------------------------------------------------------
bool Pipeline::get_ref_codes(const PipelineParams & params,
                              std::vector<int32_t> & out_codes,
                              int32_t              & out_T_prompt) {
    const int32_t num_cb = model_.hparams().num_codebooks;

    // 1. Referencia global cargada al arrancar
    if (reference_loaded_) {
        out_codes.resize(reference_embedding_.size() / sizeof(int32_t));
        std::memcpy(out_codes.data(), reference_embedding_.data(), reference_embedding_.size());
        out_T_prompt = (int32_t)(out_codes.size() / num_cb);
        return true;
    }

    // 2. Voice profile persistido (--voice <id>) — si no hay prompt_audio_path
    if (params.prompt_audio_path.empty() && !params.voice_id.empty()) {
        voice_mgr_.set_storage_dir(params.voice_storage_dir);
        std::cout << "[Voice] Loading profile: " << params.voice_id << "\n";
        try {
            VoiceProfile profile = voice_mgr_.load(params.voice_id);
            const int32_t num_cb = model_.hparams().num_codebooks;
            if (!profile.is_compatible(num_cb, model_.hparams().codebook_size, codec_.sample_rate())) {
                std::cerr << "[Voice] Profile incompatible with current model/codec.\n";
                out_codes.clear(); out_T_prompt = 0;
                return true; // no es fatal — genera sin referencia
            }
            out_codes    = std::move(profile.codes);
            out_T_prompt = profile.T_prompt;
            std::cout << "[Voice] Loaded: " << params.voice_id
                      << " (" << out_T_prompt << " frames)\n";
            return true;
        } catch (const std::exception & e) {
            std::cerr << "[Voice] Failed to load '" << params.voice_id << "': " << e.what() << "\n";
            out_codes.clear(); out_T_prompt = 0;
            return true;
        }
    }

    // 3. Sin referencia por request → generar sin voz de referencia
    if (params.prompt_audio_path.empty()) {
        out_codes.clear();
        out_T_prompt = 0;
        return true;
    }

    // 4. Buscar en caché
    auto it = voice_cache_.find(params.prompt_audio_path);
    if (it != voice_cache_.end()) {
        std::cout << "[VoiceCache] HIT: " << params.prompt_audio_path << "\n";
        out_codes    = it->second.codes;
        out_T_prompt = it->second.T_prompt;
        return true;
    }

    // 5. Encodear y guardar en caché
    std::cout << "[VoiceCache] MISS — encoding: " << params.prompt_audio_path << "\n";
    AudioData ra;
    if (!load_audio(params.prompt_audio_path, ra, codec_.sample_rate())) {
        std::cerr << "[VoiceCache] Error loading audio: " << params.prompt_audio_path << "\n";
        out_codes.clear(); out_T_prompt = 0;
        return true; // no es fatal — genera sin referencia
    }

    std::vector<int32_t> codes;
    int32_t T_prompt = 0;
    if (!codec_.encode(ra.samples.data(), (int32_t)ra.samples.size(),
                       params.gen.n_threads, codes, T_prompt)) {
        std::cerr << "[VoiceCache] Error encoding reference audio.\n";
        out_codes.clear(); out_T_prompt = 0;
        return true;
    }

    // LRU: si el caché está lleno, borrar la entrada más antigua
    if (voice_cache_.size() >= VOICE_CACHE_MAX && !voice_cache_order_.empty()) {
        const std::string & oldest = voice_cache_order_.front();
        std::cout << "[VoiceCache] Evicting: " << oldest << "\n";
        voice_cache_.erase(oldest);
        voice_cache_order_.erase(voice_cache_order_.begin());
    }

    VoiceCache entry;
    entry.codes    = codes;
    entry.T_prompt = T_prompt;
    voice_cache_[params.prompt_audio_path] = std::move(entry);
    voice_cache_order_.push_back(params.prompt_audio_path);

    std::cout << "[VoiceCache] Cached (" << voice_cache_.size()
              << "/" << VOICE_CACHE_MAX << "): " << params.prompt_audio_path << "\n";

    // --save-voice: persistir el perfil codificado en disco
    if (params.save_voice && !params.voice_id.empty() && !codes.empty()) {
        voice_mgr_.set_storage_dir(params.voice_storage_dir);
        if (params.prompt_text.empty()) {
            std::cerr << "[Voice] --save-voice requires --prompt-text to store transcript.\n";
        } else {
            VoiceProfile profile;
            profile.transcript    = params.prompt_text;
            profile.codes         = codes;
            profile.num_codebooks = model_.hparams().num_codebooks;
            profile.T_prompt      = T_prompt;
            profile.sample_rate   = codec_.sample_rate();
            profile.codebook_size = model_.hparams().codebook_size;
            if (voice_mgr_.save(params.voice_id, profile)) {
                std::cout << "[Voice] Saved profile: " << params.voice_id
                          << " -> " << voice_mgr_.storage_dir() << "\n";
            } else {
                std::cerr << "[Voice] Failed to save profile: " << params.voice_id << "\n";
            }
        }
    }

    out_codes    = codes;
    out_T_prompt = T_prompt;
    return true;
}

// ---------------------------------------------------------------------------
// synthesize — guarda a disco (usa TempPcmFile para RAM mínima)
// ---------------------------------------------------------------------------
bool Pipeline::synthesize(const PipelineParams & params) {
    if (!initialized_) { std::cerr << "Pipeline not initialized.\n"; return false; }

    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    get_ref_codes(params, ref_codes, T_prompt);

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
        // 'audio' se destruye aquí → RAM liberada antes del siguiente segmento
    };

    if (params.segment_sentences) {
        auto segs = split_sentences(params.text, params.min_seg_chars);
        std::cout << "[Segment] " << segs.size() << " sentences.\n";
        for (size_t i = 0; i < segs.size(); ++i) {
            std::cout << "[" << (i+1) << "/" << segs.size() << "] \"" << segs[i] << "\"\n";
            if (!process_segment(segs[i]))
                std::cerr << "Segment " << (i+1) << " failed — continuing.\n";
        }
    } else {
        if (!process_segment(params.text)) return false;
    }

    // Construir WAV final desde el archivo temporal
    const std::string final_path = params.output_path.empty() ? "out.wav" : params.output_path;
    std::rewind(tmp.fp);
    std::ofstream out(final_path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Pipeline error: could not create " << final_path << "\n";
        return false;
    }

    char hdr[44];
    build_wav_header(hdr, (uint32_t)tmp.total_samps, codec_.sample_rate());
    out.write(hdr, 44);

    // Streamear PCM desde disco → disco sin pasar por RAM
    char copy_buf[65536];
    size_t n;
    while ((n = std::fread(copy_buf, 1, sizeof(copy_buf), tmp.fp)) > 0)
        out.write(copy_buf, (std::streamsize)n);

    std::cout << "Saved " << tmp.total_samps << " samples to " << final_path << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// synthesize_to_buffer — para HTTP (Crow).
//
// Estrategia de RAM mínima:
//   - Cada segmento se escribe al TempPcmFile (disco, %TEMP%)
//   - Al final se construye el output_buffer leyendo el archivo temporal
//   - Pico de RAM = segmento más largo + output_buffer final
//   - output_buffer es inevitable porque Crow necesita el body completo
//     antes de enviar la respuesta HTTP. Para streaming verdadero habría
//     que usar chunked transfer encoding (mejora futura).
// ---------------------------------------------------------------------------
bool Pipeline::synthesize_to_buffer(const PipelineParams & params,
                                     std::vector<char>    & output_buffer) {
    if (!initialized_) { std::cerr << "Pipeline not initialized.\n"; return false; }

    std::cout << "--- Synthesize ---\n"
              << "Text: " << params.text << "\n"
              << "Mode: " << (params.segment_sentences ? "segmentado" : "bloque completo") << "\n";

    auto t0 = std::chrono::steady_clock::now();

    // Referencia de voz — con caché LRU para evitar re-encodear entre requests
    const int32_t num_cb = model_.hparams().num_codebooks;
    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    auto t_ref0 = std::chrono::steady_clock::now();
    get_ref_codes(params, ref_codes, T_prompt);
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

    // Función lambda: sintetiza un segmento y lo vuelca a disco inmediatamente
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
        // audio se destruye aquí → RAM liberada
        auto te = std::chrono::steady_clock::now();
        float inf_s = std::chrono::duration_cast<std::chrono::milliseconds>(te-ts).count()/1000.f;
        std::cout << "  → " << dur_s << "s audio / " << inf_s << "s inferencia ("
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
            if (!process_segment(segs[i]))
                std::cerr << "Segment " << (i+1) << " failed — skipping.\n";
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
    // Pico de RAM aquí = WAV completo (inevitable para HTTP response body)
    const int32_t sr = codec_.sample_rate();
    uint32_t data_bytes = (uint32_t)(tmp.total_samps * sizeof(int16_t));
    output_buffer.resize(44 + data_bytes);

    build_wav_header(output_buffer.data(), (uint32_t)tmp.total_samps, sr);

    std::rewind(tmp.fp);
    size_t read = std::fread(output_buffer.data() + 44, 1, data_bytes, tmp.fp);
    if (read != data_bytes) {
        std::cerr << "Pipeline warning: read " << read << "/" << data_bytes << " bytes del temp.\n";
    }
    // tmp se destruye aquí → archivo temporal borrado automáticamente

    std::cout << "[Buffer] WAV ready: " << output_buffer.size() / 1024 << " KB\n";
    return true;
}

// ---------------------------------------------------------------------------
// synthesize_to_file — escribe WAV completo a un archivo temporal y retorna
// la ruta. Crow lo sirve con set_static_file_info() sin cargarlo en RAM.
//
// Ventaja sobre synthesize_to_buffer:
//   - Cero copias extra: PCM → %TEMP% → cliente directo vía sendfile del OS
//   - RAM pico = 1 segmento en float32, nada más
//   - Crow usa stream_threshold para no hacer timeout en archivos grandes
// ---------------------------------------------------------------------------
bool Pipeline::synthesize_to_file(const PipelineParams & params,
                                   std::string          & out_wav_path) {
    if (!initialized_) { std::cerr << "Pipeline not initialized.\n"; return false; }

    std::cout << "--- Synthesize to file ---\n"
              << "Text: " << params.text << "\n"
              << "Mode: " << (params.segment_sentences ? "segmentado" : "bloque") << "\n";

    auto t0 = std::chrono::steady_clock::now();

    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    get_ref_codes(params, ref_codes, T_prompt);

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
        // audio destruido aquí — RAM liberada
        auto te = std::chrono::steady_clock::now();
        float inf_s = std::chrono::duration_cast<std::chrono::milliseconds>(te-ts).count()/1000.f;
        std::cout << "  → " << dur_s << "s audio in " << inf_s << "s ("
                  << (dur_s/std::max(inf_s,0.001f)) << "x RT)\n";
        return ok;
    };

    if (params.segment_sentences) {
        auto segs = split_sentences(params.text, params.min_seg_chars);
        std::cout << "[Segment] " << segs.size() << " sentences.\n";
        for (size_t i = 0; i < segs.size(); ++i) {
            std::cout << "[" << (i+1) << "/" << segs.size() << "] \"" << segs[i] << "\"\n";
            if (!process_seg(segs[i]))
                std::cerr << "Segment " << (i+1) << " failed — skipping.\n";
        }
    } else {
        if (!process_seg(params.text)) return false;
    }

    if (tmp.total_samps == 0) {
        std::cerr << "Pipeline error: no audio generated.\n";
        return false;
    }

    // Construir el WAV final en un NUEVO archivo temporal (con cabecera).
    // El TempPcmFile solo tiene PCM crudo sin cabecera — necesitamos un
    // archivo WAV completo para que Crow lo sirva con Content-Type correcto.
    TempPcmFile wav_tmp;
    if (!wav_tmp.open()) {
        std::cerr << "Pipeline error: could not create archivo temporal WAV.\n";
        return false;
    }

    // Escribir cabecera WAV
    char hdr[44];
    build_wav_header(hdr, (uint32_t)tmp.total_samps, codec_.sample_rate());
    std::fwrite(hdr, 1, 44, wav_tmp.fp);

    // Copiar PCM desde tmp → wav_tmp en bloques de 64KB (sin pasar por RAM)
    std::rewind(tmp.fp);
    char copy_buf[65536];
    size_t n;
    while ((n = std::fread(copy_buf, 1, sizeof(copy_buf), tmp.fp)) > 0)
        std::fwrite(copy_buf, 1, n, wav_tmp.fp);
    std::fflush(wav_tmp.fp);

    auto t1 = std::chrono::steady_clock::now();
    float total_s   = tmp.total_samps / (float)codec_.sample_rate();
    float elapsed_s = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()/1000.f;
    std::cout << "[Timing] " << total_s << "s audio in " << elapsed_s << "s ("
              << (total_s/std::max(elapsed_s,0.001f)) << "x RT)\n";
    std::cout << "[File] WAV: " << wav_tmp.path
              << " (" << (44 + tmp.total_samps*2)/1024 << " KB)\n";

    // Transferir ownership de la ruta — el llamador borra el archivo
    out_wav_path = wav_tmp.path;
    wav_tmp.path.clear(); // evitar que el destructor lo borre
    // wav_tmp.fp se cierra al destruir — el archivo queda en disco
    return true;
}

// ---------------------------------------------------------------------------
// float_to_int16 — convierte muestras float32 [-1,1] a int16
// ---------------------------------------------------------------------------
void Pipeline::float_to_int16(const std::vector<float> & in,
                                std::vector<int16_t>      & out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        float s = std::max(-1.0f, std::min(1.0f, in[i]));
        out[i]  = static_cast<int16_t>(s * 32767.0f);
    }
}

// ---------------------------------------------------------------------------
// synthesize_streaming — genera audio y llama al callback por segmento.
//
// Con stream_decode_stride_frames > 0 (o auto=4):
//   Usa generate_streaming() para recibir frames uno a uno mientras el
//   transformer genera. Cada vez que se acumulan stride frames, decodifica
//   ese chunk con el codec y lo envía al cliente. Latencia hasta el primer
//   chunk: prefill + stride * ~11ms en lugar de esperar todos los tokens.
//
// Con stream_decode_stride_frames == 0 y segment=true:
//   Comportamiento previo: genera el segmento completo, decodifica todo y
//   envía. Útil si el codec tiene poca VRAM (un chunk de decode grande es
//   más eficiente que muchos chunks pequeños).
// ---------------------------------------------------------------------------
bool Pipeline::synthesize_streaming(const PipelineParams & params,
                                     StreamCallback         callback) {
    if (!initialized_) { std::cerr << "Pipeline not initialized.\n"; return false; }
    if (!callback)      { std::cerr << "synthesize_streaming: null callback.\n"; return false; }

    std::vector<int32_t> ref_codes; int32_t T_prompt = 0;
    get_ref_codes(params, ref_codes, T_prompt);

    // Stride: número de frames a acumular antes de cada decode+send.
    // 0 → desactivado (flujo segmento-completo).
    // Valores típicos: 4 (baja latencia) a 32 (menor overhead).
    const int32_t stride = params.stream_decode_stride_frames;
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
            params.prompt_text,
            ref_codes.empty() ? nullptr : ref_codes.data(),
            num_cb,
            T_prompt
        );
        if (pt.cols == 0) {
            std::cerr << "[Stream] build_prompt returned empty for: \"" << seg << "\"\n";
            return true;
        }

        // Configurar parámetros de generación para este segmento
        GenerateParams gp = params.gen;
        gp.max_new_tokens = params.max_tokens_per_segment;
        gp.verbose        = false;

        // Inicializar KV cache si es necesario
        if (!kv_cache_initialized_ || kv_cache_max_len_ < (pt.cols + gp.max_new_tokens)) {
            int32_t ctx_len = pt.cols + gp.max_new_tokens + 64;
            if (!model_.init_kv_cache(ctx_len)) {
                std::cerr << "[Stream] init_kv_cache failed.\n";
                return false;
            }
            kv_cache_initialized_ = true;
            kv_cache_max_len_     = ctx_len;
        }

        // Buffer de acumulación de frames para decode por stride
        const int32_t decode_stride = (stride > 0) ? stride : 4;
        std::vector<int32_t> pending_codes;   // acumulados, row-major (num_cb, T)
        pending_codes.reserve(static_cast<size_t>(num_cb) * decode_stride * 2);
        int32_t pending_frames = 0;
        bool    cb_ok          = true;

        // Función interna: decodifica pending_codes y envía al callback WS
        auto flush_pending = [&](bool is_last_chunk) -> bool {
            if (pending_frames == 0) return true;

            std::vector<float> audio_chunk;
            if (!codec_.decode_chunked(pending_codes.data(), pending_frames,
                                        params.gen.n_threads,
                                        audio_chunk,
                                        params.codec_chunk_frames,
                                        params.codec_overlap_frames)) {
                std::cerr << "[Stream] decode_chunked failed on chunk of "
                          << pending_frames << " frames.\n";
                pending_codes.clear(); pending_frames = 0;
                return true; // no fatal — seguir con el siguiente stride
            }

            postprocess_audio(audio_chunk, params);

            std::vector<int16_t> pcm;
            float_to_int16(audio_chunk, pcm);
            audio_chunk.clear(); audio_chunk.shrink_to_fit();

            float dur_s = pcm.size() / (float)codec_.sample_rate();
            std::cout << "[Stream] Chunk " << pending_frames << " frames ("
                      << dur_s << "s)" << (is_last_chunk ? " [LAST]" : "") << "\n";

            bool ok = callback(pcm.data(), pcm.size(), is_last_chunk);
            pending_codes.clear(); pending_frames = 0;
            return ok;
        };

        // FrameCallback para generate_streaming
        auto on_frame = [&](const int32_t * codes, int32_t n_cb) -> bool {
            if (!cb_ok) return false;
            // Append en row-major: pending_codes[cb * pending_frames_capacity + t]
            // Para simplificar, usamos layout (T, num_cb) transpuesto luego.
            // En realidad decode_chunked espera (num_cb, T) row-major.
            // Acumulamos frame a frame transponiendo al vuelo:
            //   Para el frame actual t=pending_frames:
            //     pending_codes[cb * max_T + t] = codes[cb]
            // Pero max_T no es conocido. Usamos (pending_frames, num_cb) y
            // transponemos al hacer flush. Más simple: guardamos como (T, num_cb)
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
                // flush — is_last_chunk=false (habrá más frames)
                cb_ok = flush_pending(false);
                // Reiniciar acumulador en formato (T, num_cb) para próximo stride
                pending_codes.clear();
            }
            return cb_ok;
        };

        generate_streaming(model_, tokenizer_.config(), pt, gp, on_frame);

        model_.free_kv_cache();
        kv_cache_initialized_ = false;
        kv_cache_max_len_     = 0;

        // Flush frames restantes (el último chunk, marcado como is_last)
        if (pending_frames > 0 && cb_ok) {
            // Transponer residuo (T, num_cb) → (num_cb, T)
            std::vector<int32_t> transposed(static_cast<size_t>(num_cb) * pending_frames);
            for (int32_t t = 0; t < pending_frames; ++t)
                for (int32_t cb = 0; cb < num_cb; ++cb)
                    transposed[static_cast<size_t>(cb) * pending_frames + t] =
                        pending_codes[static_cast<size_t>(t) * num_cb + cb];
            pending_codes = std::move(transposed);
            cb_ok = flush_pending(is_last_seg);
        } else if (cb_ok && is_last_seg) {
            // Sin residuo pero es el último seg: enviar señal de fin al callback
            // con 0 muestras (el protocolo WS espera is_last=true en algún punto)
            // Crow ya envía {"done":true} al salir del lambda — no hace falta.
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
            return true; // continuar con el siguiente segmento
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
    } else {
        bool ok = use_stride
            ? stream_segment_stride(params.text, true)
            : stream_segment_full(params.text, true);
        if (!ok) return false;
    }

    return true;
}

} // namespace s2
