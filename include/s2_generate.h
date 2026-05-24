#pragma once
// s2_generate.h — Autoregressive generation loop
//
// Port of generate() from ggml_pure.py.
// Combines Slow-AR prefill + step-by-step generation + Fast-AR decode.

#include "s2_model.h"
#include "s2_sampler.h"
#include "s2_tokenizer.h"
#include "s2_prompt.h"

#include <cstdint>
#include <vector>
#include <functional>

namespace s2 {

struct GenerateParams {
    int32_t max_new_tokens          = 512;
    float   temperature             = 0.7f;
    float   top_p                   = 0.7f;
    int32_t top_k                   = 30;
    int32_t min_tokens_before_end   = 64;
    int32_t n_threads               = 4;
    bool    verbose                 = true;

    // RAS (Repetition Aware Sampling): previene loops de tokens repetidos.
    // Si el token actual ya aparece en la ventana reciente, se remuestrea
    // con temperatura alta para salir del bucle.
    int32_t ras_window_size         = 10;   // tokens recientes a vigilar
    float   ras_high_temp           = 1.0f; // temperatura del remuestreo
    float   ras_high_top_p          = 0.9f; // top_p del remuestreo
};

// Generate VQ codes autoregressively.
// Returns flattened (num_codebooks, T_generated) codes in row-major order.
struct GenerateResult {
    std::vector<int32_t> codes;
    int32_t num_codebooks = 0;
    int32_t n_frames      = 0;
};

GenerateResult generate(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params
);

// Streaming variant: calls frame_cb(codes_ptr, num_codebooks) once per generated
// frame, in the order the frames are produced. If frame_cb returns false, generation
// stops early (treated as EOS). Use this when you want to pipeline decoding with
// generation instead of waiting for all frames.
//
// codes_ptr points to a temporary buffer valid only for the duration of the call.
// Copy the codes if you need them past the callback return.
using FrameCallback = std::function<bool(const int32_t * codes, int32_t num_codebooks)>;

GenerateResult generate_streaming(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params,
    FrameCallback frame_cb
);

} // namespace s2
