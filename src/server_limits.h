#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <mutex>

namespace s2::server {

class TokenBucketRateLimiter {
public:
    TokenBucketRateLimiter(size_t per_minute, size_t burst)
        : capacity_(static_cast<double>(std::max<size_t>(1u, burst))),
          refill_per_second_(static_cast<double>(std::max<size_t>(1u, per_minute)) / 60.0),
          tokens_(capacity_), last_(std::chrono::steady_clock::now()) {}

    bool allow() noexcept {
        try {
            const auto now = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> lock(mutex_);
            const double elapsed = std::chrono::duration<double>(now - last_).count();
            if (elapsed > 0.0) {
                tokens_ = std::min(capacity_, tokens_ + elapsed * refill_per_second_);
                last_ = now;
            }
            if (tokens_ < 1.0) return false;
            tokens_ -= 1.0;
            return true;
        } catch (...) {
            return false;
        }
    }

private:
    const double capacity_;
    const double refill_per_second_;
    double tokens_;
    std::chrono::steady_clock::time_point last_;
    std::mutex mutex_;
};

class AtomicPermit {
public:
    AtomicPermit(std::atomic<size_t> & counter, size_t limit) noexcept : counter_(&counter) {
        acquired_ = try_increment_bounded(counter, limit);
    }
    AtomicPermit(const AtomicPermit &) = delete;
    AtomicPermit & operator=(const AtomicPermit &) = delete;
    ~AtomicPermit() {
        if (acquired_) counter_->fetch_sub(1u, std::memory_order_acq_rel);
    }
    explicit operator bool() const noexcept { return acquired_; }

    static bool try_increment_bounded(std::atomic<size_t> & counter, size_t limit) noexcept {
        size_t observed = counter.load(std::memory_order_relaxed);
        while (observed < limit) {
            if (counter.compare_exchange_weak(observed, observed + 1u,
                                              std::memory_order_acq_rel,
                                              std::memory_order_relaxed)) {
                return true;
            }
        }
        return false;
    }

private:
    std::atomic<size_t> * counter_ = nullptr;
    bool acquired_ = false;
};

inline bool try_increment_bounded(std::atomic<size_t> & counter, size_t limit) noexcept {
    return AtomicPermit::try_increment_bounded(counter, limit);
}

} // namespace s2::server
