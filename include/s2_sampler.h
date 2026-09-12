#pragma once
// s2_sampler.h — deterministic top-k / top-p / temperature sampling with RAS

#include <cstdint>
#include <vector>

namespace s2 {

struct SamplerParams {
    float   temperature = 0.7f;
    float   top_p       = 0.7f;
    int32_t top_k       = 30;
};

// Small fully-specified RNG.  A non-zero seed is reproducible across supported
// standard libraries because token selection does not depend on
// std::uniform_*_distribution or std::discrete_distribution mappings.
class SamplerRng {
public:
    explicit SamplerRng(uint64_t seed = 0);
    uint64_t next_u64() noexcept;
    double next_unit() noexcept; // [0, 1)

private:
    uint64_t state_ = 0;
};

// Sample a single token from logits using top-k + top-p + temperature.
// always_include_id: if >= 0 and has a finite logit, this token is guaranteed
// to survive both top-k and top-p truncation (used to ensure EOS is reachable).
// rng==nullptr creates an independent random stream for this call; generation
// code passes one request-local RNG through every sampling decision.
int32_t sample_token(const float * logits, int32_t vocab_size, const SamplerParams & params,
                     int32_t always_include_id = -1, SamplerRng * rng = nullptr);

// Repetition Aware Sampling (RAS):
// Tracks a window of recent tokens, resamples with high temp if repeating.
class RASSampler {
public:
    RASSampler(int32_t window_size = 10,
               float high_temp = 1.0f,
               float high_top_p = 0.9f);

    int32_t sample(const float * logits, int32_t vocab_size,
                   const SamplerParams & params,
                   int32_t sem_begin, int32_t sem_end);

    void reset();

private:
    int32_t window_size_;
    float   high_temp_;
    float   high_top_p_;
    std::vector<int32_t> window_;
    SamplerRng rng_;
};

} // namespace s2
