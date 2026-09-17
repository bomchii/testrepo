#include "../include/s2_sampler.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <chrono>

namespace s2 {

static uint64_t random_seed64() {
    std::random_device rd;
    uint64_t seed = static_cast<uint64_t>(rd()) << 32;
    seed ^= static_cast<uint64_t>(rd());
    seed ^= static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    if (seed == 0) seed = 0x9E3779B97F4A7C15ull;
    return seed;
}

SamplerRng::SamplerRng(uint64_t seed) : state_(seed == 0 ? random_seed64() : seed) {}

uint64_t SamplerRng::next_u64() noexcept {
    // SplitMix64: compact, deterministic, and sufficient for sampling logits.
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double SamplerRng::next_unit() noexcept {
    // Exact mapping from the top 53 random bits to IEEE-754 [0,1).
    return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
}

// Stable softmax. Returns false if the logits do not contain a usable finite
// distribution (for example every candidate is NaN/-Inf).
static bool apply_softmax(std::vector<float> & logits, float temp = 1.0f) {
    if (logits.empty()) return false;

    size_t pos_inf = 0;
    for (float v : logits) if (std::isinf(v) && v > 0.0f) ++pos_inf;
    if (pos_inf > 0) {
        const float p = 1.0f / static_cast<float>(pos_inf);
        for (float & v : logits) v = (std::isinf(v) && v > 0.0f) ? p : 0.0f;
        return true;
    }

    float max_val = -std::numeric_limits<float>::infinity();
    for (float v : logits) {
        if (std::isfinite(v) && v > max_val) max_val = v;
    }
    if (!std::isfinite(max_val)) return false;

    if (!std::isfinite(temp) || temp <= 0.0f) {
        bool emitted = false;
        for (float & v : logits) {
            if (!emitted && std::isfinite(v) && v == max_val) {
                v = 1.0f;
                emitted = true;
            } else {
                v = 0.0f;
            }
        }
        return emitted;
    }

    double sum = 0.0;
    for (float & v : logits) {
        if (!std::isfinite(v)) {
            v = 0.0f;
            continue;
        }
        const double e = std::exp((static_cast<double>(v) - max_val) / temp);
        v = std::isfinite(e) ? static_cast<float>(e) : 0.0f;
        sum += v;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) return false;
    for (float & v : logits) v = static_cast<float>(v / sum);
    return true;
}

int32_t sample_token(const float * logits, int32_t vocab_size, const SamplerParams & params,
                     int32_t always_include_id, SamplerRng * rng) {
    if (!logits || vocab_size <= 0) return -1;

    std::vector<std::pair<float, int32_t>> items;
    items.reserve(static_cast<size_t>(vocab_size));
    for (int32_t i = 0; i < vocab_size; ++i) {
        const float v = std::isnan(logits[i])
            ? -std::numeric_limits<float>::infinity()
            : logits[i];
        items.push_back({v, i});
    }

    // Stable total order: descending logit, then ascending token ID. Equal
    // logits therefore never depend on std::sort implementation details.
    const auto better = [](const auto & a, const auto & b) {
        if (a.first > b.first) return true;
        if (a.first < b.first) return false;
        return a.second < b.second;
    };
    std::sort(items.begin(), items.end(), better);

    int32_t k = params.top_k > 0 ? std::min(params.top_k, vocab_size) : vocab_size;
    if (k <= 0) return -1;
    items.resize(static_cast<size_t>(k));

    if (always_include_id >= 0 && always_include_id < vocab_size &&
        !std::isnan(logits[always_include_id]) &&
        logits[always_include_id] > -std::numeric_limits<float>::infinity()) {
        bool found = false;
        for (const auto & it : items) {
            if (it.second == always_include_id) { found = true; break; }
        }
        if (!found) items.push_back({logits[always_include_id], always_include_id});
    }
    std::sort(items.begin(), items.end(), better);

    const int32_t n = static_cast<int32_t>(items.size());
    std::vector<float> probs(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) probs[static_cast<size_t>(i)] = items[static_cast<size_t>(i)].first;

    if (!apply_softmax(probs, params.temperature)) {
        if (always_include_id >= 0 && always_include_id < vocab_size) return always_include_id;
        return -1;
    }

    const float top_p = std::isfinite(params.top_p)
        ? std::clamp(params.top_p, 0.0f, 1.0f)
        : 1.0f;

    int32_t always_pos = -1;
    if (always_include_id >= 0) {
        for (int32_t i = 0; i < n; ++i) {
            if (items[static_cast<size_t>(i)].second == always_include_id) { always_pos = i; break; }
        }
    }

    float cumsum = 0.0f;
    int32_t p_idx = 0;
    while (p_idx < n) {
        cumsum += probs[static_cast<size_t>(p_idx)];
        ++p_idx;
        if (cumsum >= top_p) break;
    }
    if (p_idx == 0) p_idx = 1;

    const bool append_forced = (always_pos >= p_idx);
    std::pair<float, int32_t> forced_item{};
    float forced_prob = 0.0f;
    if (append_forced) {
        forced_item = items[static_cast<size_t>(always_pos)];
        forced_prob = probs[static_cast<size_t>(always_pos)];
    }
    items.resize(static_cast<size_t>(p_idx));
    probs.resize(static_cast<size_t>(p_idx));
    if (append_forced) {
        items.push_back(forced_item);
        probs.push_back(forced_prob);
    }

    double sum_p = 0.0;
    for (float p : probs) sum_p += p;
    if (!(sum_p > 0.0) || !std::isfinite(sum_p)) {
        return always_include_id >= 0 && always_include_id < vocab_size
            ? always_include_id : items.front().second;
    }

    // Deterministic categorical mapping: one explicitly defined U[0,sum)
    // threshold and a left-to-right cumulative scan. Do not use
    // std::discrete_distribution, whose mapping is implementation-dependent.
    SamplerRng local_rng;
    SamplerRng & use_rng = rng ? *rng : local_rng;
    const double target = use_rng.next_unit() * sum_p;
    double cumulative = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += static_cast<double>(probs[i]);
        if (target < cumulative || i + 1 == probs.size()) return items[i].second;
    }
    return items.back().second;
}

RASSampler::RASSampler(int32_t window_size, float high_temp, float high_top_p)
    : window_size_(std::max<int32_t>(0, window_size)),
      high_temp_(std::isfinite(high_temp) && high_temp >= 0.0f ? high_temp : 1.0f),
      high_top_p_(std::isfinite(high_top_p) ? std::clamp(high_top_p, 0.0f, 1.0f) : 0.9f),
      rng_(0) {}

int32_t RASSampler::sample(const float * logits, int32_t vocab_size,
               const SamplerParams & params,
               int32_t sem_begin, int32_t sem_end) {
    int32_t token = sample_token(logits, vocab_size, params, -1, &rng_);
    if (token < 0) return token;

    if (!window_.empty() && token >= sem_begin && token <= sem_end) {
        if (std::find(window_.begin(), window_.end(), token) != window_.end()) {
            SamplerParams high_params = params;
            high_params.temperature = high_temp_;
            high_params.top_p = high_top_p_;
            token = sample_token(logits, vocab_size, high_params, -1, &rng_);
            if (token < 0) return token;
        }
    }

    if (token >= sem_begin && token <= sem_end) {
        if (window_size_ > 0) {
            window_.push_back(token);
            if (static_cast<int32_t>(window_.size()) > window_size_) {
                window_.erase(window_.begin());
            }
        }
    } else {
        window_.clear();
    }
    return token;
}

void RASSampler::reset() { window_.clear(); }

} // namespace s2
