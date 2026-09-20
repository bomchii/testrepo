#pragma once
// s2_pipeline.h — End-to-end TTS pipeline

#include "s2_audio.h"
#include "s2_codec.h"
#include "s2_generate.h"
#include "s2_model.h"
#include "s2_tokenizer.h"
#include "s2_voice.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace s2 {

struct InlineReference {
    std::vector<uint8_t> audio;
    std::string text;
};

struct PipelineParams {
    std::string model_path;
    std::string tokenizer_path;
    std::string codec_model_path;
    std::string base_dir;

    const char * tokenizer_data      = nullptr;
    size_t       tokenizer_data_size = 0;

    std::string text;
    std::string prompt_text;       // transcript of the reference audio
    std::string prompt_audio_path;
    std::string output_path;       // explicit output WAV path (--output)

    GenerateParams gen;

    int32_t vulkan_device       = -1;
    // -2 = inherit transformer device (default), -1 = force CPU, >=0 = GPU device.
    int32_t codec_vulkan_device = -2;

    bool    segment_sentences      = false;
    int32_t codec_chunk_frames     = 0;
    int32_t codec_overlap_frames   = 0;
    int32_t max_tokens_per_segment = 300;
    int32_t min_seg_chars          = 0;

    // Fish-compatible long-form chunking. 0 keeps V7.4/V7.5 native behavior.
    // Non-zero chunk_length is measured in visible Unicode characters.
    int32_t chunk_length           = 0;
    int32_t min_chunk_length       = 0;
    bool    condition_on_previous_chunks = true;

    // Fish-compatible postprocessing controls.
    float   prosody_volume_db      = 0.0f;
    float   prosody_speed          = 1.0f;
    bool    normalize_loudness     = false;
    bool    normalize_text         = false;
    int32_t output_sample_rate     = 0; // 0 = codec-native
    bool    output_rf64            = false;

    // Inline Fish reference audio. Saved voice/reference_id has higher priority.
    std::vector<InlineReference> inline_references;
    bool    reference_memory_cache = true;

    // Conversational VQ history. 0 preserves V7.4 for ordinary mono-speaker
    // requests; multi-speaker requests without an external reference retain at
    // least one prior turn to avoid applying one monovoice anchor to everyone.
    int32_t multi_turn_history     = 0;

    // Optional startup warmup; isolated from real voice/reference state.
    bool warmup                    = false;

    // Trailing-silence trimming applied after synthesis
    bool trim_silence = false;

    // Voice profile persistence
    std::string voice_id;                      // --voice <id>: load saved profile
    bool        save_voice        = false;     // --save-voice: save after encoding
    std::string voice_storage_dir = "./voices";// --voice-dir <path>

    // Streaming decode cadence (frames between codec calls); 0 = auto
    int32_t stream_decode_stride_frames = 0;
};

struct VoiceCache {
    std::vector<int32_t> codes;
    int32_t              T_prompt = 0;
    uintmax_t             source_size = 0;
    int64_t               source_mtime_ns = 0;
    uint64_t              source_hash = 0;
    bool                  has_fingerprint = false;
    std::string           transcript;
};

// Callback para synthesize_streaming().
// Con stride activo puede llamarse varias veces por segmento de texto; con
// stride desactivado se llama una vez por segmento.
// Parámetros:
//   pcm_int16  — puntero a los samples int16 del segmento
//   n_samples  — número de samples
//   is_last    — true si es el último segmento del texto
// Retorna false para abortar la generación.
using StreamCallback = std::function<bool(
    const int16_t * pcm_int16,
    size_t          n_samples,
    bool            is_last)>;

// Optional cancellation probe for streaming. Return false to stop generation.
// With stride streaming this is checked for every semantic frame; with stride
// disabled it is checked at segment boundaries because the non-streaming model
// path does not expose frame-level cancellation.
using CancelCallback = std::function<bool()>;

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    Pipeline(const Pipeline &) = delete;
    Pipeline & operator=(const Pipeline &) = delete;
    Pipeline(Pipeline &&) = delete;
    Pipeline & operator=(Pipeline &&) = delete;

    bool init(const PipelineParams & params);
    bool warmup(const PipelineParams & params);

    // HTTP clásico: todo el audio en un archivo temporal; el llamador toma ownership de la ruta.
    bool synthesize_to_file(const PipelineParams & params, std::string & out_wav_path);

    // HTTP clásico: todo el audio en un buffer (para clientes simples).
    bool synthesize_to_buffer(const PipelineParams & params, std::vector<char> & output_buffer);

    // Guardar a archivo con ruta explícita (--output).
    bool synthesize(const PipelineParams & params);

    // WebSocket / streaming: emite uno o mas chunks PCM por segmento de texto.
    bool synthesize_streaming(const PipelineParams & params, StreamCallback callback,
                              int32_t * segments_out = nullptr,
                              CancelCallback should_continue = {});

    int32_t sample_rate()    const { return codec_.sample_rate(); }
    int32_t output_sample_rate(const PipelineParams & p) const {
        return p.output_sample_rate > 0 ? p.output_sample_rate : codec_.sample_rate();
    }
    int32_t num_codebooks()  const { return model_.hparams().num_codebooks; }
    int32_t codebook_size()  const { return model_.hparams().codebook_size; }

    // Encode a reference audio and return the codes + T_prompt.
    // Used by the HTTP POST /v1/voices/<id> endpoint.
    // Does NOT run TTS generation.
    bool encode_reference(const PipelineParams & params,
                          std::vector<int32_t> & out_codes,
                          int32_t              & out_T_prompt);

private:
    static std::vector<std::string> split_sentences(const std::string & text,
                                                     int32_t min_chars = 0);

    bool synthesize_segment(
        const PipelineParams       & params,
        const std::string          & text_segment,
        const std::vector<int32_t> & ref_codes,
        int32_t                      T_prompt,
        std::vector<float>         & audio_out,
        std::vector<int32_t>       * generated_codes_out = nullptr,
        int32_t                    * generated_frames_out = nullptr,
        std::vector<PromptHistoryTurn> * history = nullptr);

    bool get_ref_codes(const PipelineParams & params,
                       std::vector<int32_t> & out_codes,
                       int32_t              & out_T_prompt);

    // Convierte float32 → int16 (clipping a [-1,1])
    static void float_to_int16(const std::vector<float> & in, std::vector<int16_t> & out);

    // Apply post-processing to an audio buffer.
    void postprocess_audio(std::vector<float> & audio, const PipelineParams & params, bool trim_tail = true) const;

private:
    Tokenizer   tokenizer_;
    SlowARModel model_;
    AudioCodec  codec_;
    bool        initialized_          = false;
    bool        kv_cache_initialized_ = false;
    int32_t     kv_cache_max_len_     = 0;

    std::string codec_path_;

    bool        reference_loaded_ = false;
    std::string reference_embedding_;
    std::string reference_text_;
    std::string active_voice_transcript_; // transcript loaded from a persisted .s2voice

    static constexpr size_t VOICE_CACHE_MAX = 8;
    std::unordered_map<std::string, VoiceCache> voice_cache_;
    std::vector<std::string>                    voice_cache_order_;
    std::unordered_map<std::string, VoiceCache> inline_voice_cache_;
    std::vector<std::string>                    inline_voice_cache_order_;

    VoiceProfileManager voice_mgr_;
};

} // namespace s2
