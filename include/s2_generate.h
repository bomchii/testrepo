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
    // Legacy Fish early_stop_threshold compatibility. Fish used fractional
    // completion across multiple samples; this engine generates one sample per
    // Pipeline request. -1 (local disabled sentinel) and 1.0 (all samples
    // finished / neutral Fish value) are accepted; effectful 0..1 values are
    // rejected rather than assigned an invented single-sample meaning.
    float   early_stop_threshold     = -1.0f;
    int32_t n_threads               = 4;
    bool    verbose                 = true;

    // Reproducibility. 0 = random source, non-zero = deterministic request seed.
    uint64_t seed                    = 0;

    // Explicit repetition penalty, separate from RAS. 1.0 disables it.
    float   repetition_penalty      = 1.0f;
    int32_t repetition_window       = 64;

    // Internal warmup knob: block EOS for the first decision even if the
    // effective budget is exactly one frame. Not exposed as a sampling flag.
    bool    force_first_token       = false;

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
    bool    success       = false; // false on validation/compute failure
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
