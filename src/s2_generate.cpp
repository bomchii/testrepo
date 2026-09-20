#include "../include/s2_generate.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>

namespace s2 {

GenerateResult generate(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params
) {
    GenerateResult out;
    out.num_codebooks = model.hparams().num_codebooks;
    if (out.num_codebooks <= 0) out.num_codebooks = 1;

    const int32_t vocab_size   = model.hparams().vocab_size;
    const int32_t sem_begin    = model.hparams().semantic_begin_id;
    const int32_t sem_end      = model.hparams().semantic_end_id;
    const int32_t codebook_size = model.hparams().codebook_size;
    const int32_t im_end_id    = config.im_end_id;
    const int32_t num_cb       = out.num_codebooks;

    if (params.max_new_tokens <= 0 || params.max_new_tokens > 32768 ||
        params.n_threads <= 0 || params.n_threads > 256 ||
        !std::isfinite(params.temperature) || params.temperature < 0.0f || params.temperature > 10.0f ||
        !std::isfinite(params.top_p) || params.top_p <= 0.0f || params.top_p > 1.0f ||
        params.top_k < 0 || params.top_k > 1000000 ||
        params.min_tokens_before_end < 0 ||
        !std::isfinite(params.early_stop_threshold) || (params.early_stop_threshold != -1.0f && params.early_stop_threshold != 1.0f) ||
        params.ras_window_size < 0 || params.ras_window_size > 32768 ||
        !std::isfinite(params.repetition_penalty) || params.repetition_penalty < 1.0f || params.repetition_penalty > 10.0f ||
        params.repetition_window < 0 || params.repetition_window > 32768 ||
        !std::isfinite(params.ras_high_temp) || params.ras_high_temp < 0.0f || params.ras_high_temp > 10.0f ||
        !std::isfinite(params.ras_high_top_p) || params.ras_high_top_p <= 0.0f || params.ras_high_top_p > 1.0f ||
        vocab_size <= 0 || codebook_size <= 0 ||
        sem_begin < 0 || sem_begin >= vocab_size || sem_end < sem_begin || sem_end >= vocab_size ||
        im_end_id < 0 || im_end_id >= vocab_size ||
        prompt.rows != num_cb + 1 || prompt.cols <= 0 ||
        prompt.data.size() != static_cast<size_t>(prompt.rows) * static_cast<size_t>(prompt.cols)) {
        std::cerr << "[Generate] Invalid generation parameters/model/prompt dimensions.\n";
        return out;
    }

    // EOS must become eligible before the generation budget is exhausted.
    // Segment/context clamping can reduce max_new_tokens after request-level
    // validation, so clamp again here against the effective budget.
    const int32_t min_tokens_before_end =
        std::min(params.min_tokens_before_end, std::max(0, params.max_new_tokens - 1));
    if (params.verbose && min_tokens_before_end != params.min_tokens_before_end) {
        std::cout << "[Generate] min_end_tokens clamped from " << params.min_tokens_before_end
                  << " to " << min_tokens_before_end << " for max_new_tokens="
                  << params.max_new_tokens << ".\n";
    }

    // Build semantic mask: -inf everywhere except [sem_begin, sem_end] and im_end
    std::vector<float> sem_mask(vocab_size, -std::numeric_limits<float>::infinity());
    for (int32_t i = sem_begin; i <= sem_end && i < vocab_size; ++i) {
        sem_mask[i] = 0.0f;
    }
    if (im_end_id >= 0 && im_end_id < vocab_size) {
        sem_mask[im_end_id] = 0.0f;
    }

    // Prefill
    // eval_cached expects time-major: flat_tokens[t * (num_cb+1) + cb]
    // but prompt.data is codebook-major: data[cb * cols + t]
    // Transpose (num_cb+1, T) → (T, num_cb+1)
    const int32_t rows = prompt.rows;  // num_codebooks + 1
    const int32_t cols = prompt.cols;  // T
    std::vector<int32_t> prompt_tm(static_cast<size_t>(rows) * cols);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t c = 0; c < cols; ++c) {
            prompt_tm[static_cast<size_t>(c) * rows + r] = prompt.data[static_cast<size_t>(r) * cols + c];
        }
    }

    StepResult state;
    if (params.verbose) {
        std::cout << "[Generate] Prefilling " << prompt.cols << " tokens..." << std::endl;
    }
    if (!model.prefill(prompt_tm, prompt.cols, params.n_threads, state) ||
        state.logits.size() < static_cast<size_t>(vocab_size)) {
        std::cerr << "[Generate] Prefill failed or returned invalid logits." << std::endl;
        return out;
    }

    SamplerRng rng(params.seed);
    int32_t step = 0;
    std::vector<int32_t> repetition_last_seen(static_cast<size_t>(vocab_size), -1);

    auto build_biased = [&](const std::vector<float> & logits, bool block_im_end) {
        std::vector<float> biased(static_cast<size_t>(vocab_size));
        for (int32_t i = 0; i < vocab_size; ++i) {
            float value = logits[static_cast<size_t>(i)];
            if (params.repetition_penalty != 1.0f && params.repetition_window > 0 &&
                repetition_last_seen[static_cast<size_t>(i)] >= 0 &&
                step - repetition_last_seen[static_cast<size_t>(i)] <= params.repetition_window &&
                std::isfinite(value)) {
                if (value > 0.0f) value /= params.repetition_penalty;
                else if (value < 0.0f) value *= params.repetition_penalty;
            }
            biased[static_cast<size_t>(i)] = value + sem_mask[static_cast<size_t>(i)];
        }
        if (block_im_end && im_end_id >= 0 && im_end_id < vocab_size) {
            biased[static_cast<size_t>(im_end_id)] = -std::numeric_limits<float>::infinity();
        }
        return biased;
    };


    // Apply semantic mask/repetition penalty and sample from the request-local RNG.
    auto apply_mask_and_sample = [&](const std::vector<float> & logits,
                                     bool block_im_end) -> int32_t {
        std::vector<float> biased = build_biased(logits, block_im_end);
        SamplerParams sparams;
        sparams.temperature = params.temperature;
        sparams.top_p       = params.top_p;
        sparams.top_k       = params.top_k;
        const int32_t force_id = block_im_end ? -1 : im_end_id;
        return sample_token(biased.data(), vocab_size, sparams, force_id, &rng);
    };

    // Sample first main_token. Warmup may force one semantic frame even when
    // the effective budget is exactly one token.
    bool block_end = params.force_first_token || (min_tokens_before_end > 0);
    int32_t main_token = apply_mask_and_sample(state.logits, block_end);
    if (main_token < 0) {
        std::cerr << "[Generate] Sampling failed (non-finite logits).\n";
        return out;
    }

    // Pre-allocate codes array
    out.codes.resize(static_cast<size_t>(num_cb) * params.max_new_tokens, 0);
    out.n_frames = 0;

    std::vector<float> fast_logits;

    SamplerParams sparams;
    sparams.temperature = params.temperature;
    sparams.top_p       = params.top_p;
    sparams.top_k       = params.top_k;

    // RAS state — parámetros vienen de GenerateParams (configurables por CLI/JSON)
    std::vector<int32_t> ras_window;
    const int32_t ras_window_size = params.ras_window_size;
    const float ras_high_temp     = params.ras_high_temp;
    const float ras_high_top_p    = params.ras_high_top_p;

    if (params.verbose) {
        std::cout << "[Generate] Generating (max " << params.max_new_tokens << " tokens)..." << std::endl;
    }

    while (main_token != im_end_id && step < params.max_new_tokens) {
        // RAS check
        if (!ras_window.empty() &&
            std::find(ras_window.begin(), ras_window.end(), main_token) != ras_window.end() &&
            main_token >= sem_begin && main_token <= sem_end)
        {
            const bool ras_block_end = (params.force_first_token && step == 0) ||
                                       (step < min_tokens_before_end);
            std::vector<float> biased = build_biased(state.logits, ras_block_end);
            SamplerParams ras_sparams;
            ras_sparams.temperature = ras_high_temp;
            ras_sparams.top_p       = ras_high_top_p;
            ras_sparams.top_k       = params.top_k;
            const int32_t ras_force_id = ras_block_end ? -1 : im_end_id;
            main_token = sample_token(biased.data(), vocab_size, ras_sparams, ras_force_id, &rng);
            if (main_token < 0) {
                std::cerr << "[Generate] RAS sampling failed.\n";
                out.codes.clear();
                out.n_frames = 0;
                return out;
            }
        }

        // RAS can legitimately resample to EOS.  Stop immediately rather than
        // turning EOS into a clamped (and bogus) semantic codebook frame.
        if (main_token == im_end_id) {
            break;
        }
        if (main_token < sem_begin || main_token > sem_end) {
            std::cerr << "[Generate] Sampler returned a non-semantic token: " << main_token << "\n";
            out.codes.clear();
            out.n_frames = 0;
            return out;
        }

        // Record semantic history for the explicit repetition penalty.
        repetition_last_seen[static_cast<size_t>(main_token)] = step;

        // Update RAS window only with semantic tokens.
        if (ras_window_size > 0) {
            ras_window.push_back(main_token);
            if (static_cast<int32_t>(ras_window.size()) > ras_window_size) {
                ras_window.erase(ras_window.begin());
            }
        } else {
            ras_window.clear();
        }

        // Metadata validation guarantees the semantic span is exactly one
        // codebook, so this conversion must be in range without clamping.
        const int32_t sem_code = main_token - sem_begin;

        // Build codebook prefix for fast decoder
        // codebooks_cb[0] = sem_code (codebook-space index)
        std::vector<int32_t> codebooks_cb;
        codebooks_cb.reserve(num_cb);
        codebooks_cb.push_back(sem_code);

        // Fast AR: generate remaining num_cb-1 codebooks
        for (int32_t cb_idx = 1; cb_idx < num_cb; ++cb_idx) {
            // prefix = codebooks_cb[0..cb_idx-1]
            if (!model.fast_decode(state.hidden, codebooks_cb, params.n_threads, fast_logits)) {
                std::cerr << "[Generate] fast_decode failed at cb " << cb_idx << std::endl;
                out.codes.clear();
                out.n_frames = 0;
                return out;
            }
            if (fast_logits.size() != static_cast<size_t>(codebook_size)) {
                std::cerr << "[Generate] Fast decoder returned " << fast_logits.size()
                          << " logits; expected codebook_size=" << codebook_size << std::endl;
                out.codes.clear();
                out.n_frames = 0;
                return out;
            }
            int32_t cb_token = sample_token(fast_logits.data(), codebook_size, sparams, -1, &rng);
            if (cb_token < 0 || cb_token >= codebook_size) {
                std::cerr << "[Generate] Fast-decoder sampling failed/out of range at cb " << cb_idx << std::endl;
                out.codes.clear();
                out.n_frames = 0;
                return out;
            }
            codebooks_cb.push_back(cb_token);
        }

        // Store frame: codes[cb * n_frames_capacity + step] = codebooks_cb[cb]
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            out.codes[static_cast<size_t>(cb) * params.max_new_tokens + step] = codebooks_cb[cb];
        }
        out.n_frames++;
        step++;

        if (params.verbose && step % 50 == 0) {
            std::cout << "\r[Generate] " << step << " / " << params.max_new_tokens << " tokens..." << std::flush;
        }

        // The final permitted frame is already complete. Do not execute an
        // extra model step/sampling pass that can only fail spuriously because
        // there will be no next iteration to consume it.
        if (step >= params.max_new_tokens) break;

        // Build step_input: [main_token (vocab space), codebooks_cb[0..num_cb-1] (codebook space)]
        std::vector<int32_t> step_input(num_cb + 1);
        step_input[0] = main_token;
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            step_input[cb + 1] = codebooks_cb[cb];
        }

        if (!model.step(step_input, params.n_threads, state) ||
            state.logits.size() < static_cast<size_t>(vocab_size)) {
            std::cerr << "[Generate] step() failed or returned invalid logits at step " << step << std::endl;
            out.codes.clear();
            out.n_frames = 0;
            return out;
        }

        // Apply semantic mask and sample next main token
        bool block_next_end = (step < min_tokens_before_end);
        main_token = apply_mask_and_sample(state.logits, block_next_end);
        if (main_token < 0) {
            std::cerr << "[Generate] Sampling failed at step " << step << ".\n";
            out.codes.clear();
            out.n_frames = 0;
            return out;
        }
    }

    if (params.verbose) {
        std::cout << std::endl;
        std::cout << "[Generate] Done: " << out.n_frames << " frames generated." << std::endl;
    }

    // Compact codes from (num_cb, max_tokens) stride to (num_cb, n_frames) row-major
    const int32_t n_frames = out.n_frames;
    if (n_frames < params.max_new_tokens) {
        std::vector<int32_t> compacted(static_cast<size_t>(num_cb) * n_frames);
        for (int32_t cb = 0; cb < num_cb; ++cb) {
            for (int32_t t = 0; t < n_frames; ++t) {
                compacted[static_cast<size_t>(cb) * n_frames + t] =
                    out.codes[static_cast<size_t>(cb) * params.max_new_tokens + t];
            }
        }
        out.codes = std::move(compacted);
    } else {
        out.codes.resize(static_cast<size_t>(num_cb) * n_frames);
    }

    out.success = true;
    return out;
}


// ---------------------------------------------------------------------------
// generate_streaming — same loop as generate(), but fires frame_cb per frame
// instead of accumulating into a flat codes array.
// ---------------------------------------------------------------------------
GenerateResult generate_streaming(
    SlowARModel & model,
    const TokenizerConfig & config,
    const PromptTensor & prompt,
    const GenerateParams & params,
    FrameCallback frame_cb
) {
    GenerateResult out;
    out.num_codebooks = model.hparams().num_codebooks;
    if (out.num_codebooks <= 0) out.num_codebooks = 1;

    const int32_t vocab_size    = model.hparams().vocab_size;
    const int32_t sem_begin     = model.hparams().semantic_begin_id;
    const int32_t sem_end       = model.hparams().semantic_end_id;
    const int32_t codebook_size = model.hparams().codebook_size;
    const int32_t im_end_id     = config.im_end_id;
    const int32_t num_cb        = out.num_codebooks;

    if (params.max_new_tokens <= 0 || params.max_new_tokens > 32768 ||
        params.n_threads <= 0 || params.n_threads > 256 || !frame_cb ||
        !std::isfinite(params.temperature) || params.temperature < 0.0f || params.temperature > 10.0f ||
        !std::isfinite(params.top_p) || params.top_p <= 0.0f || params.top_p > 1.0f ||
        params.top_k < 0 || params.top_k > 1000000 ||
        params.min_tokens_before_end < 0 ||
        !std::isfinite(params.early_stop_threshold) || (params.early_stop_threshold != -1.0f && params.early_stop_threshold != 1.0f) ||
        params.ras_window_size < 0 || params.ras_window_size > 32768 ||
        !std::isfinite(params.repetition_penalty) || params.repetition_penalty < 1.0f || params.repetition_penalty > 10.0f ||
        params.repetition_window < 0 || params.repetition_window > 32768 ||
        !std::isfinite(params.ras_high_temp) || params.ras_high_temp < 0.0f || params.ras_high_temp > 10.0f ||
        !std::isfinite(params.ras_high_top_p) || params.ras_high_top_p <= 0.0f || params.ras_high_top_p > 1.0f ||
        vocab_size <= 0 || codebook_size <= 0 ||
        sem_begin < 0 || sem_begin >= vocab_size || sem_end < sem_begin || sem_end >= vocab_size ||
        im_end_id < 0 || im_end_id >= vocab_size ||
        prompt.rows != num_cb + 1 || prompt.cols <= 0 ||
        prompt.data.size() != static_cast<size_t>(prompt.rows) * static_cast<size_t>(prompt.cols)) {
        std::cerr << "[GenerateStream] Invalid generation parameters/model/prompt dimensions.\n";
        return out;
    }

    const int32_t min_tokens_before_end =
        std::min(params.min_tokens_before_end, std::max(0, params.max_new_tokens - 1));
    if (params.verbose && min_tokens_before_end != params.min_tokens_before_end) {
        std::cout << "[GenerateStream] min_end_tokens clamped from " << params.min_tokens_before_end
                  << " to " << min_tokens_before_end << " for max_new_tokens="
                  << params.max_new_tokens << ".\n";
    }

    std::vector<float> sem_mask(vocab_size, -std::numeric_limits<float>::infinity());
    for (int32_t i = sem_begin; i <= sem_end && i < vocab_size; ++i) sem_mask[i] = 0.0f;
    if (im_end_id >= 0 && im_end_id < vocab_size) sem_mask[im_end_id] = 0.0f;

    // Transpose prompt to time-major
    const int32_t rows = prompt.rows;
    const int32_t cols = prompt.cols;
    std::vector<int32_t> prompt_tm(static_cast<size_t>(rows) * cols);
    for (int32_t r = 0; r < rows; ++r)
        for (int32_t c = 0; c < cols; ++c)
            prompt_tm[static_cast<size_t>(c) * rows + r] = prompt.data[static_cast<size_t>(r) * cols + c];

    StepResult state;
    if (params.verbose) std::cout << "[GenerateStream] Prefilling " << cols << " tokens...\n";
    if (!model.prefill(prompt_tm, cols, params.n_threads, state) ||
        state.logits.size() < static_cast<size_t>(vocab_size)) {
        std::cerr << "[GenerateStream] Prefill failed or returned invalid logits.\n";
        return out;
    }

    SamplerRng rng(params.seed);
    int32_t step = 0;
    std::vector<int32_t> repetition_last_seen(static_cast<size_t>(vocab_size), -1);

    auto build_biased = [&](const std::vector<float> & logits, bool block_im_end) {
        std::vector<float> biased(static_cast<size_t>(vocab_size));
        for (int32_t i = 0; i < vocab_size; ++i) {
            float value = logits[static_cast<size_t>(i)];
            if (params.repetition_penalty != 1.0f && params.repetition_window > 0 &&
                repetition_last_seen[static_cast<size_t>(i)] >= 0 &&
                step - repetition_last_seen[static_cast<size_t>(i)] <= params.repetition_window &&
                std::isfinite(value)) {
                if (value > 0.0f) value /= params.repetition_penalty;
                else if (value < 0.0f) value *= params.repetition_penalty;
            }
            biased[static_cast<size_t>(i)] = value + sem_mask[static_cast<size_t>(i)];
        }
        if (block_im_end && im_end_id >= 0 && im_end_id < vocab_size)
            biased[static_cast<size_t>(im_end_id)] = -std::numeric_limits<float>::infinity();
        return biased;
    };


    auto apply_mask_and_sample = [&](const std::vector<float> & logits, bool block_im_end) -> int32_t {
        std::vector<float> biased = build_biased(logits, block_im_end);
        SamplerParams sp;
        sp.temperature = params.temperature; sp.top_p = params.top_p; sp.top_k = params.top_k;
        const int32_t force_id = block_im_end ? -1 : im_end_id;
        return sample_token(biased.data(), vocab_size, sp, force_id, &rng);
    };

    bool block_end = params.force_first_token || (min_tokens_before_end > 0);
    int32_t main_token = apply_mask_and_sample(state.logits, block_end);
    if (main_token < 0) {
        std::cerr << "[GenerateStream] Sampling failed (non-finite logits).\n";
        return out;
    }

    std::vector<float> fast_logits;
    SamplerParams sparams;
    sparams.temperature = params.temperature;
    sparams.top_p       = params.top_p;
    sparams.top_k       = params.top_k;

    std::vector<int32_t> ras_window;
    const int32_t ras_window_size = params.ras_window_size;
    const float   ras_high_temp   = params.ras_high_temp;
    const float   ras_high_top_p  = params.ras_high_top_p;

    // Per-frame codes buffer reused across iterations (avoids allocation per frame)
    std::vector<int32_t> frame_codes(num_cb);

    if (params.verbose) std::cout << "[GenerateStream] Generating...\n";

    bool internal_ok = true;
    while (main_token != im_end_id && step < params.max_new_tokens) {
        // RAS check
        if (!ras_window.empty() &&
            std::find(ras_window.begin(), ras_window.end(), main_token) != ras_window.end() &&
            main_token >= sem_begin && main_token <= sem_end)
        {
            const bool ras_block_end = (params.force_first_token && step == 0) ||
                                       (step < min_tokens_before_end);
            std::vector<float> biased = build_biased(state.logits, ras_block_end);
            SamplerParams ras_sp;
            ras_sp.temperature = ras_high_temp; ras_sp.top_p = ras_high_top_p; ras_sp.top_k = params.top_k;
            main_token = sample_token(biased.data(), vocab_size, ras_sp,
                                      ras_block_end ? -1 : im_end_id, &rng);
            if (main_token < 0) {
                std::cerr << "[GenerateStream] RAS sampling failed.\n";
                internal_ok = false;
                break;
            }
        }

        if (main_token == im_end_id) {
            break;
        }
        if (main_token < sem_begin || main_token > sem_end) {
            std::cerr << "[GenerateStream] Sampler returned a non-semantic token: " << main_token << "\n";
            internal_ok = false;
            break;
        }

        // Record semantic history for the explicit repetition penalty.
        repetition_last_seen[static_cast<size_t>(main_token)] = step;

        if (ras_window_size > 0) {
            ras_window.push_back(main_token);
            if (static_cast<int32_t>(ras_window.size()) > ras_window_size) ras_window.erase(ras_window.begin());
        } else {
            ras_window.clear();
        }

        const int32_t sem_code = main_token - sem_begin;

        std::vector<int32_t> codebooks_cb;
        codebooks_cb.reserve(num_cb);
        codebooks_cb.push_back(sem_code);

        for (int32_t cb_idx = 1; cb_idx < num_cb; ++cb_idx) {
            if (!model.fast_decode(state.hidden, codebooks_cb, params.n_threads, fast_logits)) {
                std::cerr << "[GenerateStream] fast_decode failed at cb " << cb_idx << "\n";
                internal_ok = false;
                break;
            }
            if (fast_logits.size() != static_cast<size_t>(codebook_size)) {
                std::cerr << "[GenerateStream] Fast decoder returned " << fast_logits.size()
                          << " logits; expected codebook_size=" << codebook_size << "\n";
                internal_ok = false;
                break;
            }
            const int32_t cb_token = sample_token(fast_logits.data(), codebook_size, sparams, -1, &rng);
            if (cb_token < 0 || cb_token >= codebook_size) {
                std::cerr << "[GenerateStream] Fast-decoder sampling failed/out of range at cb " << cb_idx << "\n";
                internal_ok = false;
                break;
            }
            codebooks_cb.push_back(cb_token);
        }

        if (!internal_ok) break;

        // Fire the per-frame callback immediately
        for (int32_t cb = 0; cb < num_cb; ++cb) frame_codes[cb] = codebooks_cb[cb];
        if (!frame_cb(frame_codes.data(), num_cb)) break;   // caller requested stop
        out.n_frames++;
        step++;

        if (params.verbose && step % 50 == 0)
            std::cout << "\r[GenerateStream] " << step << " tokens..." << std::flush;

        if (step >= params.max_new_tokens) break;

        std::vector<int32_t> step_input(num_cb + 1);
        step_input[0] = main_token;
        for (int32_t cb = 0; cb < num_cb; ++cb) step_input[cb + 1] = codebooks_cb[cb];

        if (!model.step(step_input, params.n_threads, state) ||
            state.logits.size() < static_cast<size_t>(vocab_size)) {
            std::cerr << "[GenerateStream] step() failed or returned invalid logits at step " << step << "\n";
            internal_ok = false;
            break;
        }

        bool block_next_end = (step < min_tokens_before_end);
        main_token = apply_mask_and_sample(state.logits, block_next_end);
        if (main_token < 0) {
            std::cerr << "[GenerateStream] Sampling failed at step " << step << ".\n";
            internal_ok = false;
            break;
        }
    }

    if (params.verbose) std::cout << "\n[GenerateStream] Done: " << out.n_frames << " frames.\n";

    // Out.codes is empty — the caller consumed frames via callback.
    // n_frames is set so callers can report totals.
    out.success = internal_ok;
    return out;
}

} // namespace s2
