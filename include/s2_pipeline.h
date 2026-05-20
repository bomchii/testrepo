#pragma once
// s2_pipeline.h — End-to-end TTS pipeline

#include "s2_audio.h"
#include "s2_codec.h"
#include "s2_generate.h"
#include "s2_model.h"
#include "s2_tokenizer.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace s2 {

struct PipelineParams {
    std::string model_path;
    std::string tokenizer_path;
    std::string codec_model_path;
    std::string base_dir;

    const char * tokenizer_data      = nullptr;
    size_t       tokenizer_data_size = 0;

    std::string text;
    std::string prompt_text;
    std::string prompt_audio_path;
    std::string output_path;

    GenerateParams gen;

    int32_t vulkan_device       = -1;
    int32_t codec_vulkan_device = -1;

    bool    segment_sentences  = false;
    int32_t codec_chunk_frames = 0;
};

struct VoiceCache {
    std::vector<int32_t> codes;
    int32_t              T_prompt = 0;
};

// Callback para synthesize_streaming().
// Se llama una vez por segmento de audio generado.
// Parámetros:
//   pcm_int16  — puntero a los samples int16 del segmento
//   n_samples  — número de samples
//   is_last    — true si es el último segmento del texto
// Retorna false para abortar la generación.
using StreamCallback = std::function<bool(
    const int16_t * pcm_int16,
    size_t          n_samples,
    bool            is_last)>;

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    bool init(const PipelineParams & params);

    // HTTP clásico: todo el audio en un archivo temporal, Crow lo sirve con sendfile.
    bool synthesize_to_file(const PipelineParams & params, std::string & out_wav_path);

    // HTTP clásico: todo el audio en un buffer (para clientes simples).
    bool synthesize_to_buffer(const PipelineParams & params, std::vector<char> & output_buffer);

    // Guardar a archivo con ruta explícita.
    bool synthesize(const PipelineParams & params);

    // WebSocket / streaming: llama al callback una vez por segmento de oración.
    // El callback recibe PCM int16 listo para enviar por WebSocket.
    // Si segment_sentences=false, el callback se llama una sola vez con todo el audio.
    bool synthesize_streaming(const PipelineParams & params, StreamCallback callback);

    int32_t sample_rate() const { return codec_.sample_rate(); }

private:
    static std::vector<std::string> split_sentences(const std::string & text);

    bool synthesize_segment(
        const PipelineParams       & params,
        const std::string          & text_segment,
        const std::vector<int32_t> & ref_codes,
        int32_t                      T_prompt,
        std::vector<float>         & audio_out);

    bool get_ref_codes(const PipelineParams & params,
                       std::vector<int32_t> & out_codes,
                       int32_t              & out_T_prompt);

    // Convierte float32 → int16 (clipping a [-1,1])
    static void float_to_int16(const std::vector<float> & in, std::vector<int16_t> & out);

private:
    Tokenizer   tokenizer_;
    SlowARModel model_;
    AudioCodec  codec_;
    bool        initialized_          = false;
    bool        kv_cache_initialized_ = false;
    int32_t     kv_cache_max_len_     = 0;

    bool        reference_loaded_     = false;
    std::string reference_embedding_;
    std::string reference_text_;

    static constexpr size_t VOICE_CACHE_MAX = 8;
    std::unordered_map<std::string, VoiceCache> voice_cache_;
    std::vector<std::string>                    voice_cache_order_;
};

} // namespace s2
