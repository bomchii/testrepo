#include "../include/s2_sampler.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

namespace s2 {

// Stable softmax. Returns false if the logits do not contain a usable finite
// distribution (for example every candidate is NaN/-Inf).
static bool apply_softmax(std::vector<float> & logits, float temp = 1.0f) {
    if (logits.empty()) return false;

    // +Inf can occur after a numerical overflow. If present, make the
    // distribution uniform over only the +Inf maxima rather than producing
    // Inf-Inf => NaN below.
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

    // temperature == 0 means greedy sampling.
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

// Sample a single token from logits
int32_t sample_token(const float * logits, int32_t vocab_size, const SamplerParams & params,
                     int32_t always_include_id) {
    if (!logits || vocab_size <= 0) return -1;

    std::vector<std::pair<float, int32_t>> items;
    items.reserve(static_cast<size_t>(vocab_size));
    for (int32_t i = 0; i < vocab_size; ++i) {
        // NaN must not reach std::sort's comparator or the probability math.
        const float v = std::isnan(logits[i])
            ? -std::numeric_limits<float>::infinity()
            : logits[i];
        items.push_back({v, i});
    }

    // Sort descending by logit
    std::sort(items.begin(), items.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });

    // Top-K
    int32_t k = params.top_k > 0 ? std::min(params.top_k, vocab_size) : vocab_size;
    if (k <= 0) return -1;
    items.resize(static_cast<size_t>(k));

    // Force-include always_include_id after top-k if it was excluded.
    if (always_include_id >= 0 && always_include_id < vocab_size &&
        !std::isnan(logits[always_include_id]) &&
        logits[always_include_id] > -std::numeric_limits<float>::infinity())
    {
        bool found = false;
        for (const auto & it : items) {
            if (it.second == always_include_id) { found = true; break; }
        }
        if (!found) items.push_back({logits[always_include_id], always_include_id});
    }

    std::sort(items.begin(), items.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });

    int32_t n = static_cast<int32_t>(items.size());
    std::vector<float> probs(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) probs[static_cast<size_t>(i)] = items[static_cast<size_t>(i)].first;

    if (!apply_softmax(probs, params.temperature)) {
        // Prefer the explicitly forced token (normally EOS) as a safe escape
        // if all model logits became invalid; otherwise use the best candidate.
        if (always_include_id >= 0 && always_include_id < vocab_size) return always_include_id;
        return -1;
    }

    // Top-P. Clamp here because sample_token is a public helper and can be used
    // independently of the higher-level parameter validation.
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

    // Keep the nucleus itself and, if requested, the forced token (normally
    // EOS) as one additional candidate. Expanding p_idx to always_pos + 1
    // would accidentally admit every low-probability token between the
    // nucleus cutoff and EOS, defeating top-p when EOS is ranked low.
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
    for (float & p : probs) p = static_cast<float>(p / sum_p);

    thread_local std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
    const int32_t sampled_idx = dist(gen);
    return items[static_cast<size_t>(sampled_idx)].second;
}

RASSampler::RASSampler(int32_t window_size, float high_temp, float high_top_p)
    : window_size_(std::max<int32_t>(0, window_size)),
      high_temp_(std::isfinite(high_temp) && high_temp >= 0.0f ? high_temp : 1.0f),
      high_top_p_(std::isfinite(high_top_p) ? std::clamp(high_top_p, 0.0f, 1.0f) : 0.9f)
{
}

int32_t RASSampler::sample(const float * logits, int32_t vocab_size,
               const SamplerParams & params,
               int32_t sem_begin, int32_t sem_end) {
    int32_t token = sample_token(logits, vocab_size, params);
    if (token < 0) return token;

    if (!window_.empty() && token >= sem_begin && token <= sem_end) {
        if (std::find(window_.begin(), window_.end(), token) != window_.end()) {
            SamplerParams high_params = params;
            high_params.temperature = high_temp_;
            high_params.top_p = high_top_p_;
            token = sample_token(logits, vocab_size, high_params);
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

void RASSampler::reset() {
    window_.clear();
}

} // namespace s2
