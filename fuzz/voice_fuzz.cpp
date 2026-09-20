#include "../include/s2_voice.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    // Voice profiles cap the transcript at 1 MiB and codes at 512 MiB. A 1 MiB
    // fuzz input is sufficient to exercise all header/length validation paths
    // without turning each iteration into a disk-amplification test.
    if (size > (1u << 20)) return 0;
#ifdef _WIN32
    (void)data;
    (void)size;
    return 0; // The hosted fuzz job is Linux; keep the target portable to compile.
#else
    char path[] = "/tmp/s2voice-fuzz-XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) return 0;
    size_t written = 0;
    while (written < size) {
        const ssize_t n = ::write(fd, data + written, size - written);
        if (n <= 0) break;
        written += static_cast<size_t>(n);
    }
    ::close(fd);
    if (written == size) {
        try { (void)s2::VoiceProfile::load(path); }
        catch (...) { /* malformed profiles are expected */ }
    }
    ::unlink(path);
    return 0;
#endif
}
