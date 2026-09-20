#!/usr/bin/env python3
"""Keep README/--help in sync with the real CLI, routes, and synthesis JSON surface."""
from pathlib import Path
import re
import json

ROOT = Path(__file__).resolve().parents[1]
main = (ROOT / "src/main.cpp").read_text(encoding="utf-8")
readme = (ROOT / "README.md").read_text(encoding="utf-8")


def fail(msg: str) -> None:
    raise SystemExit("DOCS_API_SYNC_FAIL: " + msg)


def must(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)

# The help text is a raw string in the parser. Restrict checks to that block so
# an option/route mentioned only in implementation comments cannot satisfy docs.
help_start = main.index('R"S2HELP(')
help_end = main.index(')S2HELP";', help_start)
help_text = main[help_start:help_end]

# Every parsed CLI spelling must remain present in both user-facing references.
parser_start = main.index('// --- Parse des arguments ---')
parser_end = main.index('} else if (arg == "--help" || arg == "-h")', parser_start)
parser_region = main[parser_start:parser_end]
options = set(re.findall(r'arg\s*==\s*"(-{1,2}[^"\\]+)"', parser_region))
options.update({'-h', '--help'})
for opt in sorted(options):
    must(opt in help_text, f"--help is missing parsed option {opt}")
    must(opt in readme, f"README is missing parsed option {opt}")

# Route list comes from actual Crow registrations, not a hand-maintained count.
routes = set(re.findall(r'CROW_ROUTE\(app,\s*"([^"]+)"\)', main))
routes.update(re.findall(r'CROW_WEBSOCKET_ROUTE\(app,\s*"([^"]+)"\)', main))
for route in sorted(routes):
    doc_route = route.replace("<string>", "<id>")
    must(doc_route in help_text, f"--help is missing server route {doc_route} (Crow: {route})")
    must(doc_route in readme, f"README is missing server route {doc_route} (Crow: {route})")

# Derive the top-level JSON surface from the actual HTTP, WS, and compatibility
# validator code so a future parser field cannot be added without documentation.
http_start = main.index("auto do_synthesize")
http_end = main.index("auto handle_synthesis_request", http_start)
http_block = main[http_start:http_end]
ws_handler_start = main.index("auto process_ws_message")
ws_start = main.index("auto json = load_json_strict(data);", ws_handler_start)
ws_end = main.index("std::string validation_error;", ws_start)
ws_block = main[ws_start:ws_end]
must(ws_handler_start < main.index("CROW_WEBSOCKET_ROUTE"),
     "WebSocket synthesis handler must live outside Crow onmessage so I/O stays responsive")
validator_start = main.index("static void validate_fish_json_subset")
validator_end = main.index("static crow::json::rvalue load_json_strict", validator_start)
validator_block = main[validator_start:validator_end]
actual_top_level_fields = set(re.findall(r'json\.has\("([^"]+)"\)', http_block + ws_block + validator_block))
for field in sorted(actual_top_level_fields):
    must(field in readme, f"README is missing actual JSON field {field}")
    must(field in help_text, f"--help is missing actual JSON field {field}")

common_fields = {
    'text', 'input', 'segment', 'reference_audio', 'prompt_text', 'voice',
    'reference_id', 'temperature', 'top_p', 'top_k', 'seed',
    'repetition_penalty', 'repetition_window', 'multi_turn_history', 'threads',
    'max_tokens', 'max_new_tokens', 'max_seg_tokens', 'min_end_tokens',
    'ras_window', 'ras_temp', 'ras_top_p', 'codec_chunk', 'codec_overlap',
    'min_seg_chars', 'chunk_length', 'min_chunk_length',
    'condition_on_previous_chunks', 'prosody', 'latency', 'trim_silence',
    'streaming', 'references', 'early_stop_threshold', 'normalize', 'sample_rate',
    'use_memory_cache',
}
for field in sorted(common_fields):
    must(field in help_text, f"--help is missing synthesis JSON field {field}")
    must(field in readme, f"README is missing synthesis JSON field {field}")

for field in ('format', 'response_format', 'stream_stride', 'mp3_bitrate', 'opus_bitrate'):
    must(field in help_text, f"--help is missing route-specific JSON field {field}")
    must(field in readme, f"README is missing route-specific JSON field {field}")

# Implemented compatibility fields must remain visible in both user references.
for field in (
    'references', 'early_stop_threshold', 'normalize', 'sample_rate',
    'mp3_bitrate', 'opus_bitrate', 'use_memory_cache', 'normalize_loudness',
):
    must(field in readme, f"README does not explain Fish field {field}")
    must(field in help_text, f"--help does not explain Fish field {field}")

# The two route-specific no-op traps must remain errors and be documented.
must('stream_stride is WebSocket-only' in main,
     'HTTP stream_stride must fail instead of being a silent no-op')
must('encoded-audio bitrate fields are HTTP-only' in main,
     'WebSocket format/bitrate fields must fail instead of being silently ignored')
must('stream_stride' in readme and 'rejected on HTTP' in readme,
     'README must explain HTTP stream_stride rejection')
must('WebSocket always returns framed PCM' in readme and '`format`/`response_format`' in readme and 'are rejected there' in readme,
     'README must explain WS format rejection')
must('websocket_max_payload' in main,
     'server must configure the Crow WebSocket frame payload limit')
must('Crow 1.3.4' in readme and 'complete reassembled message' in readme and 'fragmented messages' in readme,
     'README must document Crow 1.3.4 cumulative WebSocket payload enforcement')
must('Crow 1.3.4' in help_text and 'complete reassembled' in help_text and 'fragmented messages' in help_text,
     '--help must document Crow 1.3.4 cumulative WebSocket payload enforcement')
must('s2-pro-local' in readme and 'instructions' in readme and 'stream_format' in readme,
     'README must document the implemented OpenAI model/unsupported-field subset')
must('s2-pro-local' in help_text and 'instructions' in help_text and 'stream_format' in help_text,
     '--help must document the implemented OpenAI model/unsupported-field subset')
must('buffered HTTP still defaults to WAV' in readme,
     'README must distinguish one-shot --rf64 from buffered HTTP default WAV')
must('Buffered HTTP defaults to WAV even when CLI --rf64 is set' in help_text,
     '--help must distinguish one-shot --rf64 from buffered HTTP default WAV')
must('normalize` defaults to `true`' in readme and '`use_memory_cache` defaults to `off`' in readme,
     'README must state Fish normalize/cache defaults')
must('Fish requests default normalize=true' in help_text and 'use_memory_cache defaults off' in help_text,
     '--help must state Fish normalize/cache defaults')

# All inline curl JSON examples must remain valid JSON. This catches quoting edits
# that look fine in Markdown/help but fail when copied into a shell.
for label, text in (("README", readme), ("--help", help_text)):
    payloads = re.findall(r"(?:-d|--data-raw)\s+'(\{[^'\n]*\})'", text)
    for payload in payloads:
        try:
            json.loads(payload)
        except json.JSONDecodeError as exc:
            fail(f"{label} has invalid inline curl JSON: {payload!r}: {exc}")

# Every fixed URL used by curl examples must point at a registered route. Dynamic
# voice examples are normalized to Crow's /v1/voices/<string> route.
def known_route(path: str) -> bool:
    if path in routes:
        return True
    if path.startswith('/v1/voices/') and '/v1/voices/<string>' in routes:
        return True
    return False

for label, text in (("README", readme), ("--help", help_text)):
    for line in text.splitlines():
        if "curl" not in line:
            continue
        for path in re.findall(r'https?://127\.0\.0\.1:\d+([^\s"\']*)', line):
            # Trim Markdown/shell punctuation that can trail a URL token.
            path = path.rstrip(')`\\') or '/'
            must(known_route(path), f"{label} curl example uses unregistered route {path}")

# Check the practical startup/cURL paths the README is meant to teach.
release_names = (
    's2-windows-cpu-x86-64.exe','s2-windows-vulkan-x86-64.exe','s2-windows-cuda-x86-64.exe','s2-windows-amd-x86-64.exe',
    's2-linux-cpu-x86-64','s2-linux-vulkan-x86-64','s2-linux-cuda-x86-64','s2-linux-amd-x86-64',
    's2-macos-metal-arm64','s2-macos-cpu-x86-64',
)
for token in release_names:
    must(token in readme, f"README is missing release executable {token}")
    must(token in help_text, f"--help is missing release executable {token}")

startup_pairs = (
    ('s2-windows-cpu-x86-64.exe --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v -1 --port 8080', 'Windows CPU'),
    ('s2-windows-vulkan-x86-64.exe --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --codec-vulkan 0 --port 8080', 'Windows Vulkan'),
    ('s2-windows-cuda-x86-64.exe --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --port 8080', 'Windows CUDA'),
    ('s2-windows-amd-x86-64.exe --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --port 8080', 'Windows AMD'),
    ('./s2-linux-cpu-x86-64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v -1 --port 8080', 'Linux CPU'),
    ('./s2-linux-vulkan-x86-64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --codec-vulkan 0 --port 8080', 'Linux Vulkan'),
    ('./s2-linux-cuda-x86-64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --port 8080', 'Linux CUDA'),
    ('./s2-linux-amd-x86-64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --port 8080', 'Linux AMD'),
    ('./s2-macos-cpu-x86-64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v -1 --port 8080', 'macOS CPU'),
    ('./s2-macos-metal-arm64 --model s2-pro-q4_k_m-transformer-only.gguf --model-codec s2-pro-q4_k_m-codec-only.gguf -v 0 --port 8080', 'macOS Metal'),
)
for command, backend in startup_pairs:
    must(command in help_text, f"--help is missing copy-paste {backend} server start")
must('There is no --server flag' in help_text,
     '--help must state that server mode does not need a --server flag')
must('There is no `--server` option' in readme or 'do **not** need a `--server` flag' in readme,
     'README must state that server mode does not need a --server flag')
for token in (
    'curl http://127.0.0.1:8080/v1/health',
    '"format":"wav"', '"format":"pcm"', '"response_format":"wav"',
    '/synthesize', '/v1/voices/narrator', '"reference_id":"narrator"',
    '"chunk_length":300', 'curl.exe', '--data-binary',
):
    must(token in readme, f"README is missing practical server/cURL example token {token}")


for token in ('--runtime-info', '--clean-runtime'):
    must(token in readme, f"README is missing CUDA launcher command {token}")
    must(token in help_text, f"--help is missing CUDA launcher command {token}")

workflow = (ROOT / '.github/workflows/build-all-backends.yml').read_text(encoding='utf-8')
for token in (
    'glibc 2.17 or newer', 'glibc 2.28 or newer', 'manylinux2014',
    'Rocky Linux 8', '**580+** under CUDA minor-version compatibility', 'R595',
    'Apple Silicon (arm64)',
):
    must(token in readme, f"README is missing Linux portability note {token}")
for token in (
    's2-linux-cpu-x86-64', 's2-linux-vulkan-x86-64', 's2-linux-cuda-x86-64', 's2-linux-amd-x86-64',
    'quay.io/pypa/manylinux2014_x86_64', 'nvidia/cuda:13.2.0-devel-rockylinux8',
):
    must(token in workflow, f"workflow is missing Linux release token {token}")
must('cp build-metal/s2-metal release-metal/s2-macos-metal-arm64' in workflow,
     'Metal release must publish the canonical single executable')
must('test -x release-metal/s2-macos-metal-arm64' in workflow,
     'Metal release must verify executable mode')
active_workflow_lines = '\n'.join(line for line in workflow.splitlines() if not line.lstrip().startswith('#'))
must('--keepParent' not in active_workflow_lines and 's2-macos-metal.zip' not in active_workflow_lines,
     'Metal packaging must not wrap the one-file release in a ZIP')
print(
    f"DOCS_API_SYNC_PASS options={len(options)} routes={len(routes)} "
    f"json_fields={len(common_fields)+5}"
)
