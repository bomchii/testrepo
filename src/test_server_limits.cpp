#include "server_limits.h"

#include <atomic>
#include <iostream>

int main() {
    int failures = 0;
    auto ok = [&](bool cond, const char * name) {
        if (!cond) { std::cerr << "FAIL: " << name << "\n"; ++failures; }
    };

    {
        s2::server::TokenBucketRateLimiter limiter(60, 3);
        ok(limiter.allow(), "token bucket first token");
        ok(limiter.allow(), "token bucket second token");
        ok(limiter.allow(), "token bucket third token");
        ok(!limiter.allow(), "token bucket rejects above burst");
    }

    {
        std::atomic<size_t> active{0};
        {
            s2::server::AtomicPermit a(active, 2);
            s2::server::AtomicPermit b(active, 2);
            s2::server::AtomicPermit c(active, 2);
            ok(static_cast<bool>(a), "permit a acquired");
            ok(static_cast<bool>(b), "permit b acquired");
            ok(!static_cast<bool>(c), "permit c rejected at limit");
            ok(active.load() == 2, "permit counter capped");
        }
        ok(active.load() == 0, "permit counter released by RAII");
    }

    {
        std::atomic<size_t> counter{0};
        ok(s2::server::try_increment_bounded(counter, 1), "bounded increment first succeeds");
        ok(!s2::server::try_increment_bounded(counter, 1), "bounded increment rejects overflow");
        ok(counter.load() == 1, "bounded increment leaves exact count");
        counter.fetch_sub(1);
        ok(counter.load() == 0, "bounded counter can be released");
    }

    if (failures) return 1;
    std::cout << "SERVER_LIMITS_TEST_PASS\n";
    return 0;
}
