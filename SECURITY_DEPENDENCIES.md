# Dependency notes

Last checked: **2026-09-20**.

A few dependencies in this repo are just vendored headers or files downloaded directly by CI, so GitHub cannot always identify them the same way it can identify packages from npm, Cargo, pip, and similar ecosystems. This file is simply the place where we keep track of the ones that matter most for input parsing, networking, and release builds.

The versions here are **pins, not endorsements forever**. If upstream publishes a security fix or an important compatibility change, it still needs a manual review before we update it. That is intentional for now: this project is experimental, and dependency changes can affect CUDA, ROCm, Vulkan, packaging, or older systems in ways that are easy to miss.

| Dependency | Where it comes from | Version used here | Notes |
|---|---|---:|---|
| dr_wav | `third_party/dr_wav.h` | 0.14.5 | Parses untrusted reference audio. Upstream master identifies itself as 0.14.6/TBD. There were 2026 fuzz reports around 0.14.5 metadata/parser paths. s2 rejects undersized RIFF `fmt ` chunks before dr_wav, does not request dr_wav metadata objects, caps decoded duration/size, and fuzzes the normal decode path under ASan+UBSan. CVE-2026-71261 concerns a 32-bit W64 metadata path; the public s2 targets are 64-bit and s2 does not opt into metadata parsing. Keep watching upstream rather than treating 0.14.5 as permanently safe. |
| dr_mp3 | `third_party/dr_mp3.h` | 0.7.4 | Used for MP3 reference audio. The audio fuzzer exercises this path with sanitizers. No newer tagged dr_mp3 release was identified during this review. |
| nlohmann/json | `third_party/json.hpp` | 3.12.0 | Current upstream release when this was checked. The server also applies its own request-size/type checks before using parsed fields. |
| ghc::filesystem | `third_party/filesystem.hpp` | 1.5.14 | Upstream latest was 1.5.16 at review time. Newer releases include correctness/robustness fixes, but changing this large vendored header should be tested across the supported platforms instead of being bumped casually. |
| Crow | commit `ae0fef0ee67eec897e401321b99b6dd7cfbdc155` | CMake reports 1.3.4 | HTTP/WebSocket layer. CI fetches this exact commit. Crow v1.3.3 fixed a 2026 pre-auth multipart DoS; s2 does not expose multipart routes. Keep watching Crow releases before changing the pin. |
| standalone Asio | tag `asio-1-30-2`, archive SHA-256 `755bd7f85a4b269c67ae0ea254907c078d408cce8e1a352ad2ed664d233780e8` | 1.30.2 | Network runtime used by Crow. CI verifies the archive hash on the portable Linux path. Upstream was 1.38.2 at review time. The older pin is kept for compatibility and should be upgraded deliberately, with the hosted matrix watching for regressions. |

## How updates are handled right now

- Dependency and GitHub Actions updates are reviewed manually.
- Pull requests still run GitHub Dependency Review for ecosystems GitHub can recognize.
- Raw vendored C/C++ headers are not assumed to be covered by package-manager alerts; that is why they are listed here.
- Parsers that handle user-controlled bytes should stay behind the project's size/format checks and sanitizer-backed fuzz tests.
- If upstream publishes a security advisory, we check whether it actually applies to the way s2 uses that dependency instead of assuming either “affected” or “not affected” from the package name alone.
- Version bumps need the normal backend/packaging validation. “Latest” by itself is not enough reason to risk breaking a working CUDA/ROCm/Vulkan/macOS build.
- The GGML tree stays on the project's existing lineage unless there is a specific reason to change it.

`tools/check_security_invariants.py` keeps the expected pins and parser hardening hooks from drifting accidentally.

## Upstream links

- dr_libs / dr_wav: https://github.com/mackron/dr_libs/blob/master/dr_wav.h
- dr_wav malformed-`fmt` fuzz report: https://github.com/mackron/dr_libs/issues/300
- dr_wav additional fuzz findings: https://github.com/mackron/dr_libs/issues/305
- CVE-2026-71261 / GHSA-cfw7-vmp5-wg69: https://github.com/advisories/GHSA-cfw7-vmp5-wg69
- nlohmann/json releases: https://github.com/nlohmann/json/releases
- nlohmann/json security page: https://github.com/nlohmann/json/security
- ghc::filesystem releases: https://github.com/gulrak/filesystem/releases
- Crow releases: https://github.com/CrowCpp/Crow/releases
- Asio upstream: https://github.com/chriskohlhoff/asio
