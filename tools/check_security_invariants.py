#!/usr/bin/env python3
"""Keep the project's security-related CI and runtime assumptions from drifting."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []

def need(cond, msg):
    if not cond:
        errors.append(msg)

def has(path, token):
    p = ROOT / path
    need(p.is_file(), f"missing {path}")
    if p.is_file():
        txt = p.read_text(encoding="utf-8", errors="replace")
        need(token in txt, f"{path} missing {token!r}")
        return txt
    return ""

build = has('.github/workflows/build-all-backends.yml', 'Attest exact release executables')
security = has('.github/workflows/security.yml', 'CodeQL C/C++')
readme = (ROOT / 'README.md').read_text(encoding='utf-8')
main = (ROOT / 'src/main.cpp').read_text(encoding='utf-8')
audio_src = (ROOT / 'src/s2_audio.cpp').read_text(encoding='utf-8', errors='replace')

# Every external action in every workflow must be immutable.
for wf in sorted((ROOT / '.github/workflows').glob('*.yml')):
    text = wf.read_text(encoding='utf-8')
    for m in re.finditer(r'^\s*uses:\s+([^\s#]+)', text, re.M):
        ref = m.group(1)
        if ref.startswith('./'):
            continue
        need('@' in ref, f'{wf.name}: action without ref: {ref}')
        if '@' in ref:
            rev = ref.rsplit('@', 1)[1]
            need(bool(re.fullmatch(r'[0-9a-f]{40}', rev)),
                 f'{wf.name}: external action must use full 40-hex SHA: {ref}')

# Checkout must not leave a write/read token in git config for later build steps.
for wf in sorted((ROOT / '.github/workflows').glob('*.yml')):
    text = wf.read_text(encoding='utf-8')
    checkout_count = text.count('uses: actions/checkout@')
    persist_false_count = text.count('persist-credentials: false')
    need(persist_false_count >= checkout_count,
         f'{wf.name}: every checkout must set persist-credentials: false')

# Exact audited action pins.
for token in (
    'github/codeql-action/init@1c5b675653bb5c22dbe9b12b556ec555138e09fd # v4.38.1',
    'github/codeql-action/analyze@1c5b675653bb5c22dbe9b12b556ec555138e09fd # v4.38.1',
    'actions/dependency-review-action@a1d282b36b6f3519aa1f3fc636f609c47dddb294 # v5.0.0',
): need(token in security, f'security workflow missing immutable action pin {token}')
need('build-mode: none' in security, 'CodeQL C/C++ must keep explicit build-mode none')
need('config-file: ./.github/codeql-config.yml' in security, 'CodeQL must use the scoped first-party config')
need((ROOT / '.github/codeql-config.yml').is_file(), 'missing .github/codeql-config.yml')
need('queries: security-extended' in security, 'CodeQL must include security-extended queries')
need('-fsanitize=fuzzer,address,undefined' in security, 'fuzz smoke must use libFuzzer + ASan + UBSan')
need('build-fuzz/voice_fuzz' in security, 'voice-profile fuzz target must run')
need('build-fuzz/test_server_limits' in security, 'server limit regression test must run')
for f in ('fuzz/base64_fuzz.cpp', 'fuzz/json_fuzz.cpp', 'fuzz/audio_fuzz.cpp', 'fuzz/voice_fuzz.cpp'):
    need((ROOT / f).is_file(), f'missing fuzz target {f}')

# Release provenance must cover exactly the ten public executables and not add assets.
need('actions/attest@1e69f48acb82d1966a394da916b4c1698aa569d6 # v4.2.2' in build,
     'release must use pinned actions/attest v4.2.2')
for perm in ('id-token: write', 'attestations: write', 'artifact-metadata: write'):
    need(perm in build, f'release missing attestation permission {perm}')
release_names = (
    's2-windows-cpu-x86-64.exe','s2-windows-vulkan-x86-64.exe','s2-windows-cuda-x86-64.exe','s2-windows-amd-x86-64.exe',
    's2-linux-cpu-x86-64','s2-linux-vulkan-x86-64','s2-linux-cuda-x86-64','s2-linux-amd-x86-64',
    's2-macos-metal-arm64','s2-macos-cpu-x86-64',
)
attest = build[build.index('Attest exact release executables'):build.index('Publish GitHub Release assets')]
for name in release_names:
    need(name in attest, f'attestation list missing {name}')

# Manual dependency maintenance and vendored inventory. A formal vulnerability-policy
# file is intentionally deferred while the project is still small/experimental.
security_policy = ROOT / ('SECURITY' + '.md')
need(not security_policy.exists(), 'formal security policy file should remain absent for now')
need(not (ROOT / '.github/dependabot.yml').exists(),
     'Dependabot must remain disabled while dependency updates are manually reviewed')
for doc in ('README.md', 'SECURITY_DEPENDENCIES.md'):
    txt = (ROOT / doc).read_text(encoding='utf-8', errors='replace')
    need('Dependabot' not in txt and 'dependabot' not in txt,
         f'{doc} must document the manual dependency-update policy without Dependabot')
inv = has('SECURITY_DEPENDENCIES.md', 'dr_wav')
for token in ('0.14.5', '0.7.4', '3.12.0', '1.5.14', '1.3.4', '1.30.2'):
    need(token in inv, f'vendored dependency inventory missing version {token}')
need('2026-09-20' in inv and 'CVE-2026-71261' in inv,
     'dependency inventory must record the current review date and dr_wav applicability review')
voice_src = (ROOT / 'src/s2_voice.cpp').read_text(encoding='utf-8', errors='replace')
need('O_EXCL' in voice_src, 'POSIX voice temporary files must use exclusive creation')
need('O_NOFOLLOW' in voice_src, 'POSIX voice temporary files should refuse symlink traversal where available')

source_markers = {
    'third_party/dr_wav.h': 'dr_wav - v0.14.5',
    'third_party/dr_mp3.h': 'dr_mp3 - v0.7.4',
    'third_party/json.hpp': 'version 3.12.0',
    'third_party/filesystem.hpp': '#define GHC_FILESYSTEM_VERSION 10514L',
}
for path, marker in source_markers.items():
    txt = (ROOT / path).read_text(encoding='utf-8', errors='replace')
    need(marker in txt, f'{path} no longer matches security dependency inventory marker {marker!r}')

linux_deps = (ROOT / 'tools/ci/prepare-linux-deps.sh').read_text(encoding='utf-8', errors='replace')
need('ae0fef0ee67eec897e401321b99b6dd7cfbdc155' in linux_deps, 'Crow 1.3.4 immutable commit pin changed')
need('755bd7f85a4b269c67ae0ea254907c078d408cce8e1a352ad2ed664d233780e8' in linux_deps, 'Asio 1.30.2 archive checksum pin changed')

# Runtime controls and docs must remain aligned.
for flag in ('--request-rate', '--request-burst', '--max-http-inflight', '--max-ws-connections'):
    need(flag in main, f'main missing {flag}')
    need(flag in readme, f'README missing {flag}')
need('#include "server_limits.h"' in main, 'main must use tested server limit helpers')
need((ROOT / 'src/server_limits.h').is_file(), 'missing server limit helper header')
need((ROOT / 'src/test_server_limits.cpp').is_file(), 'missing server limit regression test')
need('TokenBucketRateLimiter' in (ROOT / 'src/server_limits.h').read_text(encoding='utf-8'), 'process-wide request rate limiter missing')
need('AtomicPermit' in main, 'HTTP in-flight limiter missing')
need('WebSocket connection limit reached' in main, 'WebSocket connection cap missing')
need('ws_reserved_slot_sentinel' in main and '.onerror(' in main, 'WebSocket reservation/error cleanup hardening missing')
need('counted_connection.exchange(false' in main, 'WebSocket close/error paths must release connection reservation exactly once')
need(main.count('auto mark_ws_closed =') == 1, 'WebSocket close accounting helper must be defined exactly once')
need('--allow-remote' in main and 'S2_API_TOKEN' in main, 'remote opt-in/auth hardening missing')
need('Purely local use requires no token' in readme, 'README must preserve zero-config localhost behavior')
need('per-client/IP' in readme or 'per-IP' in readme, 'README must distinguish process-wide limiter from edge per-IP limiting')
need('gh attestation verify' in readme, 'README must document release attestation verification')
need('std::memcmp(h, "fmt ", 4) == 0 && chunk_size < 16ull' in audio_src,
     'memory RIFF fmt minimum pre-validation missing')
need('std::memcmp(chunk, "fmt ", 4) == 0 && chunk_size < 16ull' in audio_src,
     'file RIFF fmt minimum pre-validation missing')

if errors:
    for e in errors:
        print('SECURITY_INVARIANT_FAIL:', e, file=sys.stderr)
    raise SystemExit(1)
print('SECURITY_INVARIANTS_PASS workflows=2 fuzz_targets=4 release_assets=10 tracked_deps=6')
