# s2.cpp

Run **Fish Audio S2 Pro** locally from a small C++/GGML runtime.

`s2.cpp` can generate speech from the command line, run a local HTTP/WebSocket server, clone a voice from reference audio, save reusable `.s2voice` profiles, and use CPU, Vulkan, CUDA, ROCm/HIP, or Metal depending on the build you download.

This is still an alpha project. The main goal right now is to make S2 Pro practical to run locally without hiding the rough edges: different backends really do behave differently, model files are large, and some Fish/OpenAI API features are only partially compatible. Start locally, use the backend that matches your hardware, and test the exact setup you plan to keep using.

This fork started from [rodrigomatta/s2.cpp](https://github.com/rodrigomatta/s2.cpp). Most of the work here has gone into a usable local server, lower peak memory use, long-form synthesis, Unicode/text handling, voice reuse, streaming, and keeping the CPU/GPU backends isolated instead of trying to squeeze everything into one giant binary.

Fish Audio S2 Pro model weights use the **Fish Audio Research License**. See [LICENSE.md](LICENSE.md) before redistributing models or using them commercially.

## Contents

- [Features](#features)
- [Backends](#backends)
- [Models](#models)
- [Quick start](#quick-start)
- [Device selection](#device-selection)
- [Voice cloning](#voice-cloning)
- [Long text and low-memory synthesis](#long-text-and-low-memory-synthesis)
- [Text, languages, speakers, and segmentation](#text-languages-speakers-and-segmentation)
- [HTTP server](#http-server)
- [WebSocket streaming](#websocket-streaming)
- [CLI reference](#cli-reference)
- [CLI examples](#a-few-useful-cli-examples)
- [Concurrency](#concurrency)
- [Running it safely](#running-it-safely)
- [Build](#build)
- [Backend notes](#backend-notes)
- [What has been tested](#what-has-been-tested)
- [Known limitations](#known-limitations)
- [Project layout](#project-layout)
- [License](#license)

GitHub also shows its own outline from these headings, so you can use either one to jump around.

## Features

The short version:

- synthesize directly from the CLI to WAV/RF64
- run a local HTTP server with Fish-style, OpenAI-style, and legacy routes
- stream PCM over WebSocket
- clone a voice from WAV/MP3 reference audio
- save and reuse `.s2voice` profiles
- synthesize long text without keeping the whole output in RAM
- use `<|speaker:N|>` tags for multi-speaker text
- pass expression/style instructions such as `[whisper]` or `[professional broadcast tone]`
- handle UTF-8 text across CJK, Arabic, Hebrew, Cyrillic, Indic scripts, Armenian, Greek, Tibetan, emoji, and more
- choose a separate CPU, Vulkan, CUDA, AMD ROCm/HIP, or Metal build

There is intentionally **no** universal CPU+Vulkan+CUDA+AMD+Metal executable. Each backend is built on its own so one toolchain does not quietly break another.

## Backends

| Platform | Backend | Actions artifact | Executable |
|---|---|---|---|
| Windows x86-64 | CPU | `s2-windows-cpu-x86-64.exe` | `s2-windows-cpu-x86-64.exe` |
| Windows x86-64 | Vulkan | `s2-windows-vulkan-x86-64.exe` | `s2-windows-vulkan-x86-64.exe` |
| Windows x86-64 | CUDA | `s2-windows-cuda-x86-64.exe` | `s2-windows-cuda-x86-64.exe` |
| Windows x86-64 | AMD (ROCm/HIP) | `s2-windows-amd-x86-64.exe` | `s2-windows-amd-x86-64.exe` |
| Linux x86-64 | CPU | `s2-linux-cpu-x86-64` | `s2-linux-cpu-x86-64` |
| Linux x86-64 | Vulkan | `s2-linux-vulkan-x86-64` | `s2-linux-vulkan-x86-64` |
| Linux x86-64 | CUDA | `s2-linux-cuda-x86-64` | `s2-linux-cuda-x86-64` |
| Linux x86-64 | AMD (ROCm/HIP) | `s2-linux-amd-x86-64` | `s2-linux-amd-x86-64` |
| macOS arm64 | Metal | `s2-macos-metal-arm64` | `s2-macos-metal-arm64` |
| macOS Intel x86-64 | CPU | `s2-macos-cpu-x86-64` | `s2-macos-cpu-x86-64` |

Only one of `S2_VULKAN`, `S2_CUDA`, `S2_HIP`, or `S2_METAL` should be enabled in a build directory. CPU-only builds leave all four off.

The GitHub Actions workflow is `.github/workflows/build-all-backends.yml`. It builds ten release targets: Windows CPU/Vulkan/CUDA/AMD, Linux CPU/Vulkan/CUDA/AMD, macOS Apple-Silicon Metal, and macOS Intel CPU. Every Actions artifact and tagged-release asset contains exactly one public executable with the platform, backend, and architecture in its name.

Runtime requirements:

- Windows CPU: Windows x64 with AVX2. OpenMP may require the current Microsoft Visual C++ v14 x64 Redistributable.
- Windows Vulkan: Windows x64 with AVX2 and a Vulkan-capable GPU/driver, plus the same possible OpenMP runtime above. `vulkan-1.dll` comes from the driver.
- Windows AMD (ROCm/HIP): Windows 11 25H2 x64, a ROCm-7.14.1-supported AMD Radeon/Ryzen GPU, and AMD Adrenalin 26.6.4 (or another driver explicitly listed by AMD for ROCm 7.14.1). The release bundles the ROCm user-space runtime; the display driver is still supplied by AMD. Vulkan remains the fallback for AMD GPUs/Windows versions outside the official ROCm matrix.
- Windows CUDA: Windows x64 with AVX2, a **Turing-generation (compute capability 7.5) or newer** NVIDIA GPU, and a CUDA-13-compatible NVIDIA driver (**580+** under CUDA minor-version compatibility; **R595** or later is recommended for full CUDA 13.2 feature support). CUDA 13.x no longer builds Maxwell/Pascal/Volta targets. The CUDA runtime/cuBLAS payload is inside `s2-windows-cuda-x86-64.exe`; `nvcuda.dll` still comes from the driver.
- Linux AMD (ROCm/HIP): x86_64 Linux with a ROCm-7.14.1-supported AMD GPU/OS combination and the matching `amdgpu` driver. CI builds this target in an Ubuntu 22.04 container on a current Ubuntu 24.04 GitHub-hosted runner to preserve a glibc-2.35 userspace baseline without depending on GitHub's deprecated 22.04 runner label; AMD officially supports only the distributions/kernel combinations listed in its ROCm matrix. The single `s2-linux-amd-x86-64` executable embeds the ROCm 7.14.1 user-space runtime and extracts its verified payload to a private cache at startup. Vulkan remains the broader AMD fallback.
- Linux CPU: x86_64, AVX2, and glibc 2.17 or newer. The single executable embeds `libstdc++.so.6`, `libgcc_s.so.1`, and `libgomp.so.1` when OpenMP needs them; the extracted core uses an exact `$ORIGIN` runtime search path.
- Linux Vulkan: x86_64, AVX2, glibc 2.17 or newer, and a working Vulkan GPU driver/ICD. The single executable embeds the same GCC runtime sidecars plus its own Khronos `libvulkan.so.1` loader; the vendor GPU driver/ICD is still supplied by the system.
- Linux CUDA: x86_64, AVX2, glibc 2.28 or newer, a **Turing-generation (compute capability 7.5) or newer** NVIDIA GPU, and an NVIDIA driver compatible with CUDA 13.x (**580+** under CUDA minor-version compatibility; the CUDA 13.2 release branch is **R595**, recommended for full 13.2 feature support). CUDA 13.x dropped offline compilation/library support for Maxwell, Pascal, and Volta. CUDA Toolkit runtime/cuBLAS remain linked statically, while the single executable embeds the GCC runtime sidecars; `libcuda.so.1` still comes from the driver.
- Linux single-file payloads also embed Crow/Asio and GCC runtime license/notice files; the Vulkan payload includes the Vulkan Loader license, and the CUDA notice links the NVIDIA CUDA 13.2 EULA that governs its statically linked redistributable components.
- macOS Intel CPU: x86_64 Intel Mac, CPU-only, deployment target macOS 10.15. This compatibility build deliberately disables SSE4.2, AVX, AVX2, F16C, FMA, BMI2, and AVX-512 in GGML so it does not accidentally require a modern Intel CPU. It intentionally does not link Metal, Vulkan, or MoltenVK, so it is the release for older Intel Macs without Metal-capable GPUs. The tradeoff is lower CPU performance than an AVX2-tuned build.
- Metal: macOS on an Apple Silicon (arm64) Mac with Metal support. The `macos-15` GitHub-hosted runner used for this artifact is arm64; this is not a universal or Intel x64 binary.

The portable Linux executables target **glibc**, not musl. Alpine Linux is therefore not a native target of these binaries.

GitHub Actions uploads each release artifact directly as that one executable (no outer ZIP/TAR). On Linux or macOS, if your browser or download tool clears the executable bit, run `chmod +x <filename>` once after downloading.

### CUDA release layout

The Windows CUDA release is a single `s2-windows-cuda-x86-64.exe` file. It is a small native launcher that contains the real CUDA executable and the CUDA DLLs it needs.

At runtime it extracts those files to a hash-addressed cache under:

```text
%LOCALAPPDATA%\s2.cpp\runtime\cuda-<payload-sha256>\
```

The launcher verifies file sizes and hashes before starting the core executable. `nvcuda.dll` is not bundled because it comes from the NVIDIA driver.

Two launcher-only commands are available:

```powershell
s2-windows-cuda-x86-64.exe --runtime-info
s2-windows-cuda-x86-64.exe --clean-runtime
```

`--runtime-info` shows information about the embedded runtime. `--clean-runtime` removes inactive cached runtimes and leaves caches that are currently in use alone.

## Models

GGUF weights are not included in this repository.

### Split transformer + codec files

The [mach9243/s2-pro-gguf](https://huggingface.co/mach9243/s2-pro-gguf) collection provides transformer-only and codec-only files that can be paired together.

| Transformer | Codec | Approx. size |
|---|---|---:|
| `s2-pro-f16-transformer-only.gguf` | `s2-pro-f16-codec-only.gguf` | 9.1 GB + 1.4 GB |
| `s2-pro-q8_0-transformer-only.gguf` | `s2-pro-q8_0-codec-only.gguf` | 5.3 GB + 1.0 GB |
| `s2-pro-q4_k_m-transformer-only.gguf` | `s2-pro-q4_k_m-codec-only.gguf` | 2.8 GB + 0.95 GB |

Use `--model` for the transformer and `--model-codec` for the codec.

### Combined GGUFs

Combined Fish Speech GGUFs such as [rodrigomt/s2-pro-gguf](https://huggingface.co/rodrigomt/s2-pro-gguf) are also supported. Pass the same file to both `--model` and `--model-codec`.

The loader checks model/codec layout before synthesis. Two files loading successfully does not necessarily mean they are a compatible pair.

## Quick start

If you downloaded a release artifact, use the executable directly; the ten public artifacts no longer require extracting an outer ZIP/TAR. The examples below assume the two Q4_K_M model files are in the same directory as the executable; full paths work too.

You do **not** need a `--server` flag. Server mode is the default: give the executable the model files and do not pass `--output`. All ten release targets listen on `127.0.0.1:8080` by default.

### Windows CPU

```powershell
.\s2-windows-cpu-x86-64.exe `
  --model s2-pro-q4_k_m-transformer-only.gguf `
  --model-codec s2-pro-q4_k_m-codec-only.gguf `
  -v -1 `
  --port 8080
```

### Windows Vulkan

```powershell
.\s2-windows-vulkan-x86-64.exe `
  --model s2-pro-q4_k_m-transformer-only.gguf `
  --model-codec s2-pro-q4_k_m-codec-only.gguf `
  -v 0 `
  --codec-vulkan 0 `
  --port 8080
```

### Windows CUDA

```powershell
.\s2-windows-cuda-x86-64.exe `
  --model s2-pro-q4_k_m-transformer-only.gguf `
  --model-codec s2-pro-q4_k_m-codec-only.gguf `
  -v 0 `
  --port 8080
```

### Windows AMD (ROCm/HIP)

```powershell
.\s2-windows-amd-x86-64.exe `
  --model s2-pro-q4_k_m-transformer-only.gguf `
  --model-codec s2-pro-q4_k_m-codec-only.gguf `
  -v 0 `
  --port 8080
```

### Linux CPU

```bash
./s2-linux-cpu-x86-64 \
  --model s2-pro-q4_k_m-transformer-only.gguf \
  --model-codec s2-pro-q4_k_m-codec-only.gguf \
  -v -1 \
  --port 8080
```

### Linux Vulkan

```bash
./s2-linux-vulkan-x86-64 \
  --model s2-pro-q4_k_m-transformer-only.gguf \
  --model-codec s2-pro-q4_k_m-codec-only.gguf \
  -v 0 \
  --codec-vulkan 0 \
  --port 8080
```

### Linux CUDA

```bash
./s2-linux-cuda-x86-64 \
  --model s2-pro-q4_k_m-transformer-only.gguf \
  --model-codec s2-pro-q4_k_m-codec-only.gguf \
  -v 0 \
  --port 8080
```

### Linux AMD (ROCm/HIP)

```bash
./s2-linux-amd-x86-64 \
  --model s2-pro-q4_k_m-transformer-only.gguf \
  --model-codec s2-pro-q4_k_m-codec-only.gguf \
  -v 0 \
  --port 8080
```

### macOS Intel CPU

```bash
./s2-macos-cpu-x86-64 \
  --model s2-pro-q4_k_m-transformer-only.gguf \
  --model-codec s2-pro-q4_k_m-codec-only.gguf \
  -v -1 \
  --port 8080
```

### macOS Metal

```bash
./s2-macos-metal-arm64 \
  --model s2-pro-q4_k_m-transformer-only.gguf \
  --model-codec s2-pro-q4_k_m-codec-only.gguf \
  -v 0 \
  --port 8080
```

`-v -1` means CPU. `-v 0` means GPU 0 for the backend you downloaded. The codec follows the transformer device by default, so you normally do not need another device flag.

If you use a combined GGUF instead of separate transformer/codec files, pass the same file to both `--model` and `--model-codec`.

Once the server says it is listening, check it:

```bash
curl http://127.0.0.1:8080/v1/health
```

On Windows PowerShell, use the real curl executable:

```powershell
curl.exe http://127.0.0.1:8080/v1/health
```

Then make a WAV:

```bash
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from s2.cpp.","format":"wav"}' \
  -o output.wav
```

PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8080/v1/tts `
  -H "Content-Type: application/json" `
  -d '{"text":"Hello from s2.cpp.","format":"wav"}' `
  -o output.wav
```

That is enough for a basic server. The [HTTP server](#http-server) section below has long-form, PCM, saved-voice, reference-cloning and WebSocket examples.

### One-shot WAV instead of a server

Add `--text` and `--output`. `--output` switches the program to one-shot mode, writes PCM16 RIFF/WAV by default (or RF64 with `--rf64`), and exits.

```powershell
.\s2-windows-cuda-x86-64.exe `
  --model s2-pro-q4_k_m-transformer-only.gguf `
  --model-codec s2-pro-q4_k_m-codec-only.gguf `
  -v 0 `
  --text "Hello from s2.cpp." `
  --output output.wav
```

If `--output` is present and `--text` is omitted, text is read from stdin.

## Device selection

| Option | Meaning |
|---|---|
| `-v -1` | Transformer on CPU. This is the default. |
| `-v 0` or higher | Transformer on that GPU index for the compiled backend. |
| `--codec-vulkan -2` | Codec follows the transformer device. Default. |
| `--codec-vulkan -1` | Codec on CPU. |
| `--codec-vulkan 0` or higher | Codec on that GPU index. |

The `--vulkan` and `--codec-vulkan` names are historical. In CUDA builds they select CUDA devices, and in ROCm/HIP builds they select AMD GPU devices. Metal maps GPU selection to its single logical Metal device.

If GPU initialization fails and a CPU fallback is possible, the logs show the backend that ended up being used.

## Voice cloning

To keep the examples below short, `s2` means the executable for your backend: `.\s2-windows-cpu-x86-64.exe`, `.\s2-windows-vulkan-x86-64.exe`, `.\s2-windows-cuda-x86-64.exe`, `.\s2-windows-amd-x86-64.exe`, `./s2-linux-cpu-x86-64`, `./s2-linux-vulkan-x86-64`, `./s2-linux-cuda-x86-64`, `./s2-linux-amd-x86-64`, `./s2-macos-cpu-x86-64`, or `./s2-macos-metal-arm64`. These generic examples leave the default `-v -1`, so they also work with the CPU build. Add `-v 0` when using Vulkan, CUDA, ROCm/HIP, or Metal on the GPU.

Reference audio needs a transcript. `--prompt-audio` without a non-empty `--prompt-text` is rejected.

References are resampled to the codec rate and limited to **30 seconds** before encoding. In practice, a short clean single-speaker clip with an accurate transcript works better than feeding the model a long recording.

### Clone from a WAV/MP3 file

```bash
s2 \
  --model model.gguf --model-codec codec.gguf \
  --prompt-audio reference.wav \
  --prompt-text "Exact transcript of the reference audio." \
  --text "This uses the reference voice." \
  --output cloned.wav
```

### Save a voice and reuse it

```bash
# Save it once
s2 \
  --model model.gguf --model-codec codec.gguf \
  --prompt-audio reference.wav \
  --prompt-text "Exact transcript of the reference audio." \
  --voice narrator --save-voice

# Reuse it later
s2 \
  --model model.gguf --model-codec codec.gguf \
  --voice narrator \
  --text "A later request using the same voice." \
  --output narrator.wav
```

Profiles are stored in `voices/` next to the executable unless `--voice-dir` is used. Voice IDs may contain ASCII letters, digits, `_`, and `-`.

`.s2voice` files are validated when loaded and are written atomically.

## Long text and low-memory synthesis

For a book, article, or other large input, use `--chunk-length` or the HTTP `chunk_length` field.

```bash
cat book.txt | s2 \
  --model model.gguf \
  --model-codec codec.gguf \
  --output book.wav \
  --chunk-length 300 \
  --min-chunk-length 50 \
  --condition-on-previous-chunks
```

`chunk_length` uses visible Unicode characters, not UTF-8 bytes. The splitter avoids cutting through CJK text, Arabic/Hebrew marks, Indic viramas and conjuncts, emoji ZWJ sequences, flag pairs, balanced `[ ... ]` blocks, or valid `<|speaker:N|>` tags.

The long-form path is disk-backed. Only the current chunk needs to exist as float PCM in memory. WAV data is appended directly to a staged file and the RIFF header is fixed up at the end, so a long request no longer needs a raw PCM temporary file plus a second WAV copy.

`condition_on_previous_chunks` is on by default. It keeps a **bounded** amount of VQ/acoustic context between chunks so the voice does not restart from scratch, without letting history grow with the whole document. Use `--no-condition-on-previous-chunks` if you want each chunk to be independent.

`prosody.volume` / `--prosody-volume` applies `-20..20` dB after final silence trimming. `prosody.speed` / `--prosody-speed` supports `0.5..2.0` with a WSOLA-style time stretch for normal speech-length output; exceptionally short clips use a safe interpolation fallback. `prosody.normalize_loudness` / `--normalize-loudness` applies deterministic output loudness normalization before the final gain.

Classic RIFF/WAV has a roughly 4 GiB data limit. Use HTTP `format: "rf64"` or CLI `--rf64` for the RF64 container when that limit matters. RF64 keeps 64-bit sizes in its `ds64` chunk while preserving PCM16 audio.

## Text, languages, speakers, and segmentation

The tokenizer and public text paths require valid UTF-8.

Sentence splitting handles the usual ASCII terminators plus CJK punctuation, Unicode ellipsis, Arabic `؟`, Urdu `۔`, Devanagari danda `।`/`॥`, Armenian full stop `։`, and other supported boundaries. It keeps common abbreviations, initials, and decimal numbers together.

A few details matter for multilingual text:

- CJK does not need ASCII spaces between sentences.
- Combining marks do not count as visible characters for minimum-length decisions.
- Arabic harakat, Hebrew niqqud, Indic viramas, bidi controls, ZWJ, and ZWNJ are preserved.
- Armenian `՞` is not treated as a generic sentence terminator.
- Greek `;`/question-mark handling is contextual rather than a global semicolon rule.
- Balanced expression blocks such as `[whisper in small voice]` stay intact.
- Valid `<|speaker:N|>` tags survive segmentation and are repeated where needed. Malformed speaker tags remain literal text.

### Voice continuity without a reference

Fish S2 can choose a random timbre when no reference is supplied. With segmented synthesis, the first successful segment becomes a request-local VQ voice anchor and later segments reuse it. The anchor is discarded after the request.

If you need the same identity across different requests, use a saved `.s2voice` profile or reference audio.

For multi-speaker text, conversational VQ history is used instead of forcing one mono-voice anchor across every speaker.

## HTTP server

Server mode is the default; there is no `--server` switch. If you do **not** pass `--output`, `--list-voices`, or a save-only `--save-voice` command, `s2` loads the model/codec and starts the HTTP/WebSocket server.

The default address is:

```text
http://127.0.0.1:8080
```

CLI generation options become the server defaults. A JSON request can override the supported options for that request only. For example, starting the server with `--segment --chunk-length 300` makes those the defaults until a client sends different JSON values.

### Starting the downloaded release

Use the exact command for your backend from [Quick start](#quick-start). There is no `--server` option: if `--output`, `--list-voices`, and a save-only `--save-voice` are absent, the process starts the HTTP/WebSocket server.

PowerShell examples use the backtick (`` ` ``) for line continuation. Bash/zsh examples use `\`.

A server with long-form defaults and a different port:

```powershell
.\s2-windows-cuda-x86-64.exe `
  --model s2-pro-q4_k_m-transformer-only.gguf `
  --model-codec s2-pro-q4_k_m-codec-only.gguf `
  -v 0 `
  --segment `
  --chunk-length 300 `
  --min-chunk-length 50 `
  --port 8081
```

Local use stays zero-configuration: the default `127.0.0.1` bind does **not** require a token or any remote-access flag. Setting `S2_API_TOKEN` on loopback is optional and enables Bearer authentication if you want extra local protection. To expose s2 beyond localhost, you must opt in explicitly with `--allow-remote` **and** choose a non-loopback `--host`; remote binds additionally require `S2_API_TOKEN` (16..4096 visible non-space bytes), and HTTP/WebSocket clients must send `Authorization: Bearer <token>`. Browser-originated HTTP/WebSocket requests are accepted only when `Origin` is same-origin with `Host`; JSON POST routes require `Content-Type: application/json`. For LAN/public deployments, still put TLS, rate limiting, and a true pre-buffer HTTP body limit in a trusted reverse proxy.

Example LAN launch/client (Bash):

```bash
export S2_API_TOKEN='replace-with-a-long-random-token'
./s2-linux-cpu-x86-64 --model model.gguf --model-codec codec.gguf --allow-remote --host 0.0.0.0
curl http://192.168.1.10:8080/v1/health -H "Authorization: Bearer $S2_API_TOKEN"
```

### Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/v1/tts` | Fish-style synthesis endpoint |
| `POST` | `/v1/tts/batch` | Batch synthesis; returns one Base64-encoded complete audio output per batch item inside JSON |
| `POST` | `/v1/audio/speech` | OpenAI-style subset (`model`, `input`, local `voice`, `response_format`, `speed`) |
| `POST` | `/synthesize` | Legacy synthesis alias |
| `GET` | `/v1/models` | Local model entry |
| `GET` | `/v1/voices` | List saved voices |
| `POST` | `/v1/voices/<id>` | Create a saved voice from a server-local reference file |
| `GET` | `/v1/voices/<id>` | Read voice metadata |
| `DELETE` | `/v1/voices/<id>` | Delete a voice |
| `GET` | `/health` | Plain-text health check |
| `GET` | `/v1/health` | Fish-style JSON health check |
| `GET` | `/` | Basic service status |
| `WS` | `/ws/tts` | Incremental PCM streaming |

The Fish/OpenAI compatibility layer only covers behavior this server implements. Supported Fish fields are validated and applied rather than silently ignored.

HTTP supports WAV, raw PCM, RF64, MP3, and Ogg/Opus. WAV/PCM/RF64 are native. MP3/Opus are encoded by an external `ffmpeg` executable configured with `--ffmpeg` or `S2_FFMPEG`; if ffmpeg is unavailable, only MP3/Opus requests fail and the native formats remain self-contained.

### Check that the server is up

```bash
curl http://127.0.0.1:8080/
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/health
curl http://127.0.0.1:8080/v1/models
```

### Basic WAV request

```bash
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from s2.cpp.","format":"wav"}' \
  -o output.wav
```

### MP3, Opus, RF64, and batch requests

```bash
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"MP3 example.","format":"mp3","mp3_bitrate":128}' \
  -o output.mp3

curl -X POST http://127.0.0.1:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Large WAV container.","format":"rf64"}' \
  -o output.rf64.wav

curl -X POST http://127.0.0.1:8080/v1/tts/batch \
  -H 'Content-Type: application/json' \
  -d '{"requests":[{"text":"First.","format":"wav"},{"text":"Second.","format":"opus","opus_bitrate":64}]}'
```

`/v1/tts/batch` returns JSON because one batch can contain multiple independent synthesis results. For every successful item, `audio_base64` contains the **complete audio output bytes for that item**. Decode that Base64 value and, for `wav`, `rf64`, `mp3`, or `opus`, the result can be saved directly as the corresponding audio file (Opus is returned in an Ogg/Opus container). For `pcm`, the decoded value is raw mono PCM16 audio bytes rather than a self-describing container file. A batch with 10 requests therefore returns up to 10 independent audio outputs, in request order.

With `--workers N`, up to N items/requests can run model inference at once. Batching is request-level concurrency, not one tensor-batched transformer forward pass. Batch items run through one **global bounded worker pool** shared by all batch requests, so simultaneous batches cannot create an unbounded number of `std::async` threads. The 32-item request limit, aggregate token budget, and 128 MiB Base64-output budget still apply; queue saturation returns an item-level `503`.

### Windows PowerShell / `curl.exe`

PowerShell may map `curl` to a PowerShell command, so use `curl.exe` when you want the real curl program.

For simple ASCII JSON, a one-liner works:

```powershell
curl.exe -X POST http://127.0.0.1:8080/v1/tts -H "Content-Type: application/json" -d '{"text":"Hello from Windows.","format":"wav"}' -o output.wav
```

For multilingual text or a larger request, writing UTF-8 JSON first avoids shell quoting/encoding surprises:

```powershell
$bodyObj = @{
    text = "¡Hola! 你好。 مرحباً. नमस्ते।"
    format = "wav"
    segment = $true
    chunk_length = 300
    condition_on_previous_chunks = $true
}
$bodyJson = ConvertTo-Json -InputObject $bodyObj -Compress
$bodyPath = Join-Path $PWD "request.json"
[System.IO.File]::WriteAllText($bodyPath, $bodyJson, [System.Text.UTF8Encoding]::new($false))

curl.exe -X POST http://127.0.0.1:8080/v1/tts `
  -H "Content-Type: application/json; charset=utf-8" `
  --data-binary "@$bodyPath" `
  -o output.wav

Remove-Item $bodyPath
```

### Raw PCM

```bash
curl -D pcm-headers.txt \
  -X POST http://127.0.0.1:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Raw PCM request.","format":"pcm"}' \
  -o output.pcm
```

Raw PCM is mono signed 16-bit little-endian. The response header `X-Sample-Rate` contains the actual codec output rate; do not assume 44.1 kHz.

### OpenAI-style and legacy aliases

OpenAI `speed` is mapped to the same `prosody.speed` implementation used by the native/Fish routes. The `model` field is optional for this local subset, but when present it must be `s2-pro-local`, the same ID returned by `/v1/models`; built-in OpenAI model names are not silently remapped. `voice` selects a locally saved s2.cpp voice/reference ID. `instructions` and `stream_format` are not implemented and are rejected explicitly rather than ignored. `stream=true` is also rejected on this buffered HTTP subset; use `/ws/tts` for incremental PCM instead.

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"s2-pro-local","input":"OpenAI-style request.","response_format":"wav"}' \
  -o output.wav

curl -X POST http://127.0.0.1:8080/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Legacy endpoint.","format":"wav"}' \
  -o output.wav
```

### Long-form HTTP request

`chunk_length` is per request and uses visible Unicode characters. `max_new_tokens` is also applied per generated chunk and is still clamped to the real model context.

```bash
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"A long passage goes here...","format":"wav","segment":true,"chunk_length":300,"min_chunk_length":50,"condition_on_previous_chunks":true,"max_new_tokens":1024}' \
  -o long.wav
```

The API still has a 1 MiB limit for `text` in a single request. Long-form mode keeps inference memory/context bounded; it does not remove the request-size limit.

### Deterministic request

```bash
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Repeatable sampling request.","format":"wav","seed":123456,"temperature":0.7,"top_p":0.7,"top_k":30}' \
  -o seeded.wav
```

`seed: 0` or Fish-style `seed: null` uses a random seed. A nonzero `uint64` seed uses the deterministic sampler path; backend floating-point differences can still affect exact cross-backend output.

### Reference voice without saving a profile

`reference_audio` is a path on the **server machine**, not an uploaded file. The decoded reference is limited to 30 seconds after resampling, and `prompt_text` must match that reference.
On Windows JSON, use forward slashes (`C:/voices/ref.wav`) or escape backslashes (`C:\\voices\\ref.wav`).

```bash
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Clone this voice.","reference_audio":"/local/path/reference.wav","prompt_text":"Exact transcript."}' \
  -o cloned.wav
```

### Saved voice API

```bash
# Save
curl -X POST http://127.0.0.1:8080/v1/voices/narrator \
  -H 'Content-Type: application/json' \
  -d '{"audio_path":"/local/path/reference.wav","transcript":"Reference transcript."}'

# List
curl http://127.0.0.1:8080/v1/voices

# Inspect
curl http://127.0.0.1:8080/v1/voices/narrator

# Use the native field
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Use the saved narrator voice.","voice":"narrator","format":"wav"}' \
  -o narrator.wav

# Fish-style alias for the same saved voice
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Same saved voice.","reference_id":"narrator","format":"wav"}' \
  -o narrator-fish.wav

# Delete
curl -X DELETE http://127.0.0.1:8080/v1/voices/narrator
```

### Request fields

Most synthesis fields inherit the CLI value that was used to start the server. JSON only overrides that one request.

A normal request can stay small:

```json
{
  "text": "Hello from the local API.",
  "format": "wav",
  "segment": true,
  "chunk_length": 300,
  "condition_on_previous_chunks": true,
  "voice": "narrator"
}
```

<details>
<summary><strong>Complete HTTP/WebSocket synthesis field reference</strong></summary>

| JSON field | Where | Meaning |
|---|---|---|
| `text` / `input` | HTTP + WS | Text to synthesize. At least one is required. If both are sent they must match. Max 1 MiB. |
| `voice` / `reference_id` | HTTP + WS | Saved `.s2voice` ID, max 128 chars: ASCII letters, digits, `_`, `-`. If both are sent they must match. `reference_id: null` means no saved reference. |
| `reference_audio` | HTTP + WS | Server-local WAV/MP3 path (max 32768 bytes) for direct cloning. Requires `prompt_text`; decoded reference max 30 s. |
| `references` | HTTP + WS | Fish inline references. Each entry has canonical base64 `audio` plus non-empty UTF-8 `text`; max 8 entries and bounded decoded size. `reference_id`/`voice` takes priority. |
| `prompt_text` | HTTP + WS | Exact transcript for `reference_audio`, max 1 MiB. |
| `segment` | HTTP + WS | Override sentence segmentation for this request. |
| `temperature` | HTTP + WS | Sampling temperature, `0..10`; `0` is greedy. |
| `top_p` | HTTP + WS | Nucleus threshold `(0,1]`. |
| `top_k` | HTTP + WS | Top-k cutoff `0..1000000`. |
| `seed` | HTTP + WS | `0`/`null` = random; nonzero uint64 = deterministic request seed. |
| `repetition_penalty` | HTTP + WS | Explicit repetition penalty `1.0..10.0`; `1.0` disables it. |
| `repetition_window` | HTTP + WS | Recent-token window `0..32768`. |
| `multi_turn_history` | HTTP + WS | Explicit prior text→VQ turns to retain, `0..1024`. |
| `threads` | HTTP + WS | CPU thread count; `1..256` syntactically, but a request cannot exceed the server startup `--threads` value (default `4`). |
| `max_tokens` / `max_new_tokens` | HTTP + WS | Generation budget `1..32768`. If both are sent they must agree. Fish `max_new_tokens: 0` means no explicit Fish limit and is still context-clamped. |
| `max_seg_tokens` | HTTP + WS | Per-segment generation cap `1..32768` when sentence segmentation is active. |
| `min_end_tokens` | HTTP + WS | `0..32768`; minimum generated tokens before EOS and must remain below effective generation budget. |
| `early_stop_threshold` | HTTP + WS | Legacy compatibility: `-1` (local disabled sentinel) or `1.0` (neutral/all-finished) are accepted. Effectful fractional thresholds require true multi-sample tensor batching and are rejected rather than given invented single-sample EOS semantics. |
| `ras_window` | HTTP + WS | RAS recent-token window `0..32768`; `0` disables the window. |
| `ras_temp` | HTTP + WS | RAS resample temperature `0..10`. |
| `ras_top_p` | HTTP + WS | RAS resample top-p `(0,1]`. |
| `min_seg_chars` | HTTP + WS | Minimum visible characters used when merging sentence pieces, `0..1000000`. |
| `chunk_length` | HTTP + WS | Long-form chunk target: `0` off, otherwise `100..1000` visible Unicode characters. |
| `min_chunk_length` | HTTP + WS | Long-form minimum `0..100`; nonzero requires `chunk_length`. |
| `condition_on_previous_chunks` | HTTP + WS | Keep bounded automatic VQ/acoustic context between long-form chunks. |
| `prosody.volume` | HTTP + WS | Output gain `-20..20` dB, applied after final trim. |
| `prosody.speed` | HTTP + WS | WSOLA-style tempo change, `0.5..2.0`; `1.0` leaves duration unchanged. Very short clips use interpolation fallback. |
| `prosody.normalize_loudness` | HTTP + WS | Boolean deterministic loudness normalization before final volume gain. |
| `normalize` | HTTP + WS | Boolean local normalization that trims/collapses Unicode whitespace before segmentation/tokenization. It does not implement Fish cloud number/lexical normalization. |
| `sample_rate` | HTTP + WS | `0`/omitted = codec native rate; otherwise `8000..192000` for PCM/WAV/RF64. MP3 accepts only 8/11.025/12/16/22.05/24/32/44.1/48 kHz; Opus is 48 kHz. |
| `use_memory_cache` | HTTP + WS | Fish-compatible `"on"`/`"off"` or boolean control for reference encoding caches. |
| `latency` | HTTP + WS | HTTP accepts `normal` only because it is buffered. WebSocket accepts `normal`, `balanced`, or `low`; explicit values map to 8/4/2 codec-frame cadence unless `stream_stride` overrides it. |
| `codec_chunk` | HTTP + WS | Codec decode frame cap, `>=0`; `0` = automatic. |
| `codec_overlap` | HTTP + WS | Codec left-history override, `>=0`; `0` = automatic. |
| `trim_silence` | HTTP + WS | Trim only the real final trailing silence. |
| `streaming` | HTTP + WS | Route-consistency flag: HTTP accepts omitted/`false`; WS accepts omitted/`true`. Other route combinations are rejected. |
| `format` / `response_format` | HTTP only | `wav`, `pcm`, `rf64`, `mp3`, or `opus`; default `wav`. If both are sent they must agree. |
| `mp3_bitrate` | HTTP only | MP3 bitrate in kbps: `64`, `128`, or `192`; valid only with `format=mp3`. |
| `opus_bitrate` | HTTP only | Opus bitrate in kbps: `-1000` (automatic), `24`, `32`, `48`, or `64`; valid only with `format=opus`. |
| `stream_stride` | WS only | `-1` = segment-boundary streaming, `0` = automatic 4-frame cadence, `1..32768` = explicit frame cadence. |

For Fish-flavor HTTP and WebSocket requests, `normalize` defaults to `true` unless the server operator explicitly selected `--normalize`/`--no-normalize`; an explicit request field still overrides that value. `use_memory_cache` defaults to `off` for each Fish request and can be enabled explicitly with `"on"`/`true`. These Fish defaults do not silently change the historical CLI/legacy/OpenAI defaults.

WebSocket always returns framed PCM, so `format`/`response_format`, `mp3_bitrate`, and `opus_bitrate` are rejected there. Buffered HTTP does not use `stream_stride`, so that field is rejected on HTTP.

For WebSocket requests, `prosody.speed != 1`, `prosody.normalize_loudness=true`, or a non-native `sample_rate` require whole-segment postprocessing. Those fields remain supported, but an explicitly requested `latency=balanced/low` or `stream_stride>=0` is rejected in combination with them instead of silently degrading the requested frame cadence. `stream_stride=-1` remains compatible because it already means segment-boundary emission.

For compressed HTTP output, the encoder validates the actual container instead of returning WAV bytes under another name. MP3 accepts 64/128/192 kbps and sample rates 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, or 48000 Hz. Opus accepts `-1000` (automatic), 24/32/48/64 kbps and is emitted at 48000 Hz, as required by the Opus output path.

</details>

### HTTP status behavior

The main synthesis routes use these status classes:

- `200`: synthesis succeeded
- `400`: malformed JSON, wrong field type/range, incompatible aliases, unsupported format/semantic field, invalid voice ID, etc.
- `404`: a syntactically valid saved voice ID does not exist
- `401`: Bearer token missing/incorrect when `S2_API_TOKEN` is active
- `403`: browser `Origin` is not same-origin with the request `Host`
- `413`: JSON request body exceeds 8 MiB or a bounded batch output/generation budget
- `415`: JSON POST route was called without `Content-Type: application/json`
- `503`: bounded batch/streaming work queue is temporarily full
- `500`: model/codec/storage/runtime failure after request validation; internal exception/path details remain server-side

### Request-size note

JSON bodies/messages above 8 MiB are rejected, and `text`/`prompt_text` are each limited to 1 MiB.

Crow 1.3.4 has already buffered the HTTP body by the time the route-level 8 MiB check runs. If you expose the server outside localhost and need a real pre-buffer HTTP body limit, put a reverse proxy in front of it and enforce the limit there.

For WebSocket, Crow 1.3.4 enforces `websocket_max_payload` against the complete reassembled message, including fragmented messages. s2 also checks the delivered JSON message size before parsing it. WebSocket upgrades use the same Bearer-token and browser-Origin checks as HTTP. Use a trusted reverse proxy/gateway for TLS, rate limiting, and any stricter edge-level resource limits.

The server enables `CROW_ENFORCE_WS_SPEC`, so normal RFC 6455 client masking rules are enforced.

## WebSocket streaming

Connect to:

```text
ws://127.0.0.1:8080/ws/tts
```

For example with `websocat`:

```bash
websocat ws://127.0.0.1:8080/ws/tts
# Send one text message:
# {"text":"Streaming request.","segment":true,"stream_stride":0}
```

Binary messages use:

```text
[2-byte little-endian flags][PCM int16 little-endian samples]
```

`flags & 1` marks the final output boundary. When synthesis finishes, a JSON text message reports `done`, `segments`, and `sample_rate`.

`segments` counts text segments, not PCM packets.

Exact streaming decode keeps a bounded **left-history** window, so it does **not** re-decode the entire confirmed prefix on every update and does not wait for a future/right-edge holdback before committing new PCM. It still re-decodes the bounded left-context overlap because the current codec API is stateless between decode calls; a truly stateful/incremental codec decoder would be required to remove that remaining structural work. When `latency` is explicitly supplied, `normal`/`balanced`/`low` select 8/4/2 codec-frame cadence unless `stream_stride` is supplied. The generic `stream_stride=0` auto mode remains 4 frames.

Useful controls:

| Option / JSON field | Default | Meaning |
|---|---:|---|
| `--stream-decode-stride` / `stream_stride` | `0` | `0` = auto (4 frames), `-1` = no stride streaming, positive = explicit cadence |
| `--codec-chunk` / `codec_chunk` | `0` | `0` = automatic bounded window; positive = cap codec frames per decode |
| `--codec-overlap` / `codec_overlap` | `0` | `0` = automatic codec left history; positive = manual override |

Smaller codec windows can reduce peak memory, but going too small can hurt continuity at chunk boundaries.

If a WebSocket disconnects, stride streaming checks the connection on every semantic frame and stops promptly. With `--stream-decode-stride -1`, cancellation happens at text-segment boundaries because that path does not expose frame callbacks.

## CLI reference

All backend executables use the same CLI. The backend changes what the device selectors point to, not the option names.

The common modes are simple:

- `s2 --help`: show help
- `s2 --list-voices`: list saved voices without loading model/codec
- `s2 ... --output out.wav`: one-shot synthesis
- `s2 ... --voice ID --prompt-audio ref.wav --prompt-text "..." --save-voice`: save a voice
- `s2 [options]`: start the HTTP/WebSocket server

<details>
<summary><strong>Complete command-line option list</strong></summary>

### Models and devices

| Argument | Default | Meaning |
|---|---|---|
| `-m <path>`, `--model <path>` | `model.gguf` | Transformer/full GGUF |
| `--model-codec <path>` | `codec.gguf` | Codec/full GGUF |
| `-t <path>`, `--tokenizer <path>` | embedded in release builds when available | External tokenizer JSON |
| `-v <N>`, `--vulkan <N>` | `-1` | Transformer device: `-1` CPU, `0+` compiled GPU backend |
| `--codec-vulkan <N>` | `-2` | Codec: `-2` follow transformer, `-1` CPU, `0+` GPU |

### Server

| Argument | Default | Meaning |
|---|---|---|
| `-p <N>`, `--port <N>` | `8080` | TCP port `1..65535` |
| `--host <host>` | `127.0.0.1` | Server-only IPv4/IPv6 literal or resolvable DNS hostname; resolved once when the server starts |
| `--allow-remote` | off | Explicitly permits a non-loopback bind; remote binds also require `S2_API_TOKEN` |
| `--workers <N>` | `1` | Server-only independent model/codec/KV replicas, `1..16`; values >1 enable parallel inference with roughly proportional memory use |
| `--request-rate <N>` | `240` | Process-wide accepted HTTP/WS-message budget per minute, `1..100000` |
| `--request-burst <N>` | `60` | Process-wide token-bucket burst capacity, `1..10000` |
| `--max-http-inflight <N>` | `16` | Maximum simultaneous synthesis/voice-save HTTP requests, `1..4096`; excess work gets HTTP 503 |
| `--max-ws-connections <N>` | `64` | Maximum accepted WebSocket connections, `1..4096`; excess handshakes get HTTP 503 |
| `--ffmpeg <path>` | `S2_FFMPEG` or `ffmpeg` | Server-only ffmpeg executable used for HTTP MP3/Opus encoding |

Server/network options are not resolved or validated in one-shot `--output` or save-only `--save-voice` mode; those modes do not start Crow.

### Input/output

| Argument | Default | Meaning |
|---|---|---|
| `--text <text>` | empty | One-shot text, max 1 MiB |
| `-o <path>`, `--output <path>` | none | Write PCM16 RIFF/WAV (or RF64 with `--rf64`) and exit instead of starting the server |
| `--trim-silence` | off | Trim trailing silence at the real end of the request |
| `--no-trim-silence` | off | Explicitly disable trailing trim |
| `--prosody-speed <F>` | `1.0` | WSOLA-style tempo change, `0.5..2.0`; tiny clips use interpolation fallback |
| `--normalize` / `--no-normalize` | off | Trim/collapse Unicode whitespace before tokenization |
| `--normalize-loudness` / `--no-normalize-loudness` | off | Enable/disable deterministic output loudness normalization |
| `--sample-rate <N>` | `0` | `0` native or `8000..192000` output rate |
| `--rf64` / `--no-rf64` | off | Select RF64 vs classic RIFF/WAV for one-shot/file output; buffered HTTP still defaults to WAV unless the request asks for `format: "rf64"` |

### Voice/reference

| Argument | Default | Meaning |
|---|---|---|
| `-pa <path>`, `--prompt-audio <path>` | none | WAV/MP3 reference, max 30 s after resample; requires `--prompt-text` |
| `-pt <text>`, `--prompt-text <text>` | empty | Exact reference transcript, max 1 MiB |
| `--voice <id>` | none | Load a saved profile |
| `--save-voice` | off | Save the current explicit reference as `--voice` |
| `--voice-dir <path>` | `voices/` | Voice profile directory |
| `--list-voices` | off | List profiles and exit before model initialization |

### Generation and chunking

| Argument | Default | Meaning |
|---|---:|---|
| `-threads <N>`, `--threads <N>` | `4` | CPU threads, `1..256` |
| `--max-tokens <N>` | `1024` | Generation budget, `1..32768` |
| `--segment` | off | Unicode-aware sentence segmentation |
| `--max-seg-tokens <N>` | `300` | Segment budget, `1..32768` |
| `--min-seg-chars <N>` | `0` | Merge very short segments, `0..1000000` |
| `--chunk-length <N>` | `0` | Long-form chunks; `0` off, otherwise `100..1000` visible characters |
| `--min-chunk-length <N>` | `0` | Minimum long-form chunk, `0..100`; requires chunking |
| `--condition-on-previous-chunks` | on | Keep bounded automatic context between chunks |
| `--no-condition-on-previous-chunks` | off | Disable automatic chunk-to-chunk context |
| `--prosody-volume <dB>` | `0` | Output gain, `-20..20` dB |

### Sampling

| Argument | Default | Meaning |
|---|---:|---|
| `--temperature <F>`, `--temp <F>` | `0.7` | `0..10`; `0` = greedy |
| `--top-p <F>` | `0.7` | `(0,1]` |
| `--top-k <N>` | `30` | `0..1000000` |
| `--min-end-tokens <N>` | `64` | Minimum tokens before EOS, `0..32768`, less than `--max-tokens` |
| `--early-stop-threshold <F>` | `-1` | Legacy compatibility: `-1` disabled or `1.0` neutral; effectful fractional thresholds require true multi-sample tensor batching and are rejected |
| `--seed <uint64>` | `0` | `0` random; nonzero reproducible |
| `--repetition-penalty <F>` | `1.0` | `1.0..10.0`; `1.0` disables it |
| `--repetition-window <N>` | `64` | Recent-token window, `0..32768` |

### Conversation/startup

| Argument | Default | Meaning |
|---|---:|---|
| `--multi-turn-history <N>` | `0` | Explicit text→VQ turns to retain, `0..1024` |
| `--warmup` | off | Short isolated model+codec warmup at startup |

### RAS

| Argument | Default | Meaning |
|---|---:|---|
| `--ras-window <N>` | `10` | Recent-token repetition window, `0..32768`; `0` disables it |
| `--ras-temp <F>` | `1.0` | RAS temperature, `0..10` |
| `--ras-top-p <F>` | `0.9` | RAS top-p, `(0,1]` |

### Codec/streaming

| Argument | Default | Meaning |
|---|---:|---|
| `--codec-chunk <N>` | `0` | `0` automatic; positive values cap frames per codec decode |
| `--codec-overlap <N>` | `0` | `0` automatic history/holdback; positive = override |
| `--stream-decode-stride <N>` | `0` | `-1` off, `0` auto 4-frame cadence, positive = explicit cadence |

### Help

| Argument | Meaning |
|---|---|
| `-h`, `--help` | Print help and exit |

A few option interactions are worth knowing:

- `--prompt-audio` requires non-empty `--prompt-text`.
- Explicit `--prompt-audio` takes priority over a saved `--voice`.
- `--save-voice` needs `--voice`, `--prompt-audio`, and `--prompt-text`.
- `--min-end-tokens` must be lower than `--max-tokens`.
- `--codec-vulkan -2` follows the transformer device.
- `--host` accepts IPv4/IPv6 literals and DNS hostnames; names are resolved once at startup.
- Floating-point sampling options reject NaN/Inf.
- `--list-voices` does not load the model, codec, or GPU backend.
- `--output` writes PCM16 RIFF/WAV by default and RF64 when `--rf64` is enabled. Raw headerless PCM is only exposed by HTTP/WebSocket.

</details>

## A few useful CLI examples

Here too, `s2` is shorthand for the backend-specific executable from the Quick start.

Read a long file from stdin:

```bash
cat article.txt | s2 \
  --model model.gguf --model-codec codec.gguf \
  --segment --max-seg-tokens 300 --min-seg-chars 60 \
  --output article.wav
```

Transformer on GPU 0, codec on CPU:

```bash
s2 --model model.gguf --model-codec codec.gguf \
  -v 0 --codec-vulkan -1 \
  --text "GPU transformer, CPU codec." \
  --output mixed.wav
```

Lower codec peak memory while keeping automatic overlap:

```bash
s2 --model model.gguf --model-codec codec.gguf \
  -v 0 --codec-vulkan 0 \
  --codec-chunk 32 --codec-overlap 0 \
  --segment --max-seg-tokens 300 \
  --text "A longer synthesis request." \
  --output low-vram.wav
```

Run `s2 --help` for the same option reference directly from the binary.

## Concurrency

`--workers 1` keeps the old behavior: one model/codec/KV pipeline and one inference job at a time. `--workers N` creates N independent pipelines, so up to N inference jobs can run in parallel. That also means model/codec/KV memory is replicated, so RAM/VRAM use grows roughly with the worker count.

A request may lower its CPU `threads`, but it cannot raise that value above the server's startup `--threads` setting. The server also has process-wide request, in-flight HTTP, batch, and WebSocket limits so a burst of work does not turn into an unbounded number of threads or queued jobs.

The network-facing details are in [Running it safely](#running-it-safely).

## Running it safely

For normal local use, there is not much to configure. **Purely local use requires no token**: the server binds to `127.0.0.1` by default and works as-is. If you set `S2_API_TOKEN` yourself, Bearer authentication is also required on localhost.

Remote access is deliberately harder to enable by accident. A non-loopback bind needs both `--allow-remote` and `S2_API_TOKEN`. Browser requests also go through Host/Origin checks, JSON routes require `Content-Type: application/json`, and the server has process-wide request, HTTP in-flight, WebSocket, batch, input-size, generation, and reference-audio limits.

Those limits are useful guard rails, not an Internet edge. If you expose s2 to a LAN or the public Internet, put it behind a reverse proxy/gateway that handles TLS, **per-client/IP** rate and connection limits, access policy, and a request-body limit before Crow buffers the request.

The repository also runs a few automated checks around this:

- C/C++ CodeQL with the `security-extended` queries
- libFuzzer smoke tests under ASan/UBSan for Base64, JSON, audio, and `.s2voice` parsing
- Dependency Review on pull requests
- immutable SHA pins for third-party GitHub Actions
- artifact attestations for the same ten executables published in releases

The vendored/runtime dependencies that GitHub cannot reliably discover from package metadata are listed in [`SECURITY_DEPENDENCIES.md`](SECURITY_DEPENDENCIES.md). Updates are still reviewed manually because this project is experimental and a harmless-looking dependency bump can break CUDA, ROCm, Vulkan, packaging, or older systems.

If you downloaded a release from GitHub, you can verify that GitHub's workflow produced it:

```bash
gh attestation verify ./s2-linux-cpu-x86-64 --repo bomchii/testrepo
```

That verification is useful evidence about where the file came from. It does not replace normal code review, hashes, sandboxing, or testing on your own hardware.

## Build

If you are using a release artifact, you can skip this section.

You need:

- CMake 3.15+ (if you use the Visual Studio 2026 generator, use CMake 4.2+)
- a C++17 compiler
- the `ggml` submodule
- Crow and standalone Asio
- Vulkan SDK/runtime for Vulkan builds
- NVIDIA driver + CUDA toolkit for CUDA builds
- AMD ROCm/HIP SDK for ROCm builds
- Xcode command-line tools for Metal builds
- optional ffmpeg at runtime only when HTTP MP3/Opus output is requested

For Crow, either install a CMake package that provides `Crow::Crow`, or use the same header-only path supported by CI:

```bash
cmake -S . -B build-cpu \
  -DS2_CROW_INCLUDE_DIR=/path/to/Crow/include \
  -DS2_ASIO_INCLUDE_DIR=/path/to/asio/include
```

The release workflow currently pins an immutable Crow commit whose CMake version is 1.3.4, plus standalone Asio 1.30.2. The dependency versions used by the release build, plus notes on why they are pinned, live in [`SECURITY_DEPENDENCIES.md`](SECURITY_DEPENDENCIES.md).

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/mach92432/s2.cpp.git
cd s2.cpp
```

If you already cloned it without submodules:

```bash
git submodule update --init --recursive
```

### CPU

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DS2_VULKAN=OFF -DS2_CUDA=OFF -DS2_HIP=OFF -DS2_METAL=OFF
cmake --build build-cpu --parallel
```

### Vulkan

```bash
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release \
  -DS2_VULKAN=ON -DS2_CUDA=OFF -DS2_HIP=OFF -DS2_METAL=OFF
cmake --build build-vulkan --parallel
```

### CUDA

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DS2_CUDA=ON -DS2_VULKAN=OFF -DS2_HIP=OFF -DS2_METAL=OFF
cmake --build build-cuda --parallel
```

Do not force one CUDA architecture unless you have a specific deployment reason. GGML manages the supported architecture list.

### ROCm/HIP

```bash
cmake -S . -B build-rocm -DCMAKE_BUILD_TYPE=Release \
  -DS2_HIP=ON -DGGML_HIP=ON -DS2_VULKAN=OFF -DS2_CUDA=OFF -DS2_METAL=OFF
cmake --build build-rocm --parallel
```

On AMD hardware, keep the Vulkan build available as a fallback: ROCm supports a narrower GPU/OS matrix than Vulkan.

### Metal

```bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release \
  -DS2_METAL=ON -DS2_VULKAN=OFF -DS2_CUDA=OFF -DS2_HIP=OFF \
  -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build-metal --parallel
```

Local CMake builds keep backend-specific internal names (`s2-cpu`, `s2-vulkan`, CUDA/AMD core names, or `s2-metal`). CI gives the ten public release executables the canonical `s2-<platform>-<backend>-<arch>` names listed above; CUDA and AMD may use an internal core plus launcher/runtime container.

The Linux release jobs intentionally build against old ABI baselines instead of `ubuntu-latest`: CPU and Vulkan use a manylinux2014/glibc-2.17 baseline, while CUDA uses NVIDIA's CUDA 13.2 Rocky Linux 8 development image (glibc 2.28). CI checks the maximum referenced GLIBC symbol version before publishing. Linux Vulkan builds the pinned Khronos loader from source and embeds it in the single-file runtime payload; Linux CUDA uses GGML's static CUDA Toolkit linkage so CUDA user-space libraries do not become distro-dependent `.so` requirements. The public Linux artifact remains one executable and extracts only its verified runtime payload to cache.

Windows release builds use `/MT` for the normal MSVC CRT. OpenMP is separate: GGML may still require `VCOMP140.DLL`, so a current Microsoft Visual C++ v14 x64 Redistributable can still be needed. CI checks imports with `dumpbin` instead of assuming `/MT` makes every runtime static.

## Backend notes

A few implementation details are useful when debugging backend-specific problems:

- CUDA K-quant embedding fallback dequantizes only the rows actually used instead of expanding a full embedding table.
- Codec K-quant conversion is limited to operations that need it. Supported linear/attention tensors stay quantized.
- Metal uses an explicit finite F32 causal mask when the active backend is Metal, avoiding the unsupported `DIAG_MASK_INF` path.
- Codec/model fallback logs report the backend that initialized.
- Streaming decode keeps bounded causal left context with exact frame-to-sample geometry; it does not require a future/right-edge holdback. Offline chunked decode retains its own overlap handling.
- `--codec-overlap 0` means automatic codec-derived history, not “no overlap”.

## What has been tested

Before packaging the current tree, I run a fairly broad local test pass instead of relying only on “it compiled once”: 

- strict C++17 warning builds with GCC and Clang
- Clang Static Analyzer on model/codec/pipeline code
- ASan + UBSan regression tests for sampling, Unicode splitting, JSON surrogate handling, WAV I/O, prompt history, tokenizer behavior, and server limits
- multilingual/emoji/RTL tokenizer checks
- root CMake plus the generated Windows CMake paths
- YAML parsing and CI invariant checks
- CUDA launcher/bundler manifest tests
- README/`--help`/route/cURL synchronization checks
- checks for immutable Action pins, CodeQL, Dependency Review, fuzz targets, and release attestations
- libFuzzer + ASan + UBSan runs for Base64, JSON Unicode, audio, and `.s2voice` parsers
- clean ZIP extraction followed by byte/hash/permission comparison

That still does **not** replace the hosted matrix or real hardware. MSVC, CUDA, Vulkan, ROCm, Metal, drivers, ABI compatibility, and numerical behavior ultimately need the actual Windows/Linux/macOS jobs and the machines you care about.

## Known limitations

A few things are worth knowing before you build around this:

- The project is still alpha and the API may keep moving.
- Fish Audio/OpenAI compatibility is useful but not drop-in parity. Unsupported fields are rejected instead of being silently ignored where possible.
- Buffered HTTP audio is returned after that request finishes. Use `/ws/tts` when you want incremental PCM playback.
- Voice cloning quality depends a lot on the reference clip and how accurate its transcript is.
- Very aggressive quantization, very small codec history, or very small codec chunks can save memory at the cost of quality.
- GPU backends depend on drivers and vendor toolchains, so two machines with “the same GPU family” can still behave differently. Test the machine you actually plan to use.

## Project layout

If you want to poke around the source, the useful places are:

```text
include/                         Public C++ headers
src/                             Tokenizer, model, codec, generation, pipeline, server/CLI + limit tests
third_party/                     Header-only/support dependencies
ggml/                            GGML submodule
.github/workflows/               Backend build/release CI + the separate CodeQL/fuzz workflow
.github/codeql-config.yml        CodeQL scope for first-party C/C++ code
fuzz/                            libFuzzer targets for Base64, JSON, audio and .s2voice parsing
SECURITY_DEPENDENCIES.md         Notes on security-relevant vendored/runtime dependencies
patch-cmake.ps1                  Windows CPU/Vulkan CI CMake preparation
patch-cmake-cuda.ps1             Windows CUDA CI CMake preparation
tools/ci/prepare-linux-deps.sh       Pinned Linux Crow/Asio/Vulkan build inputs
tools/ci/build-linux-portable.sh     Portable Linux CPU/Vulkan/CUDA build + ABI checks
tools/ci/reclaim-linux-runner-space.sh  Frees unused host SDKs before large Docker builds
CMakeLists.txt                   Root CMake project
```

## License

Fish Audio S2 Pro model weights and related Fish Audio materials use the **Fish Audio Research License**. See [LICENSE.md](LICENSE.md) for the terms and attribution requirements.

Commercial licensing information is available from [Fish Audio](https://fish.audio/).

This repository is derived from the Fish Audio/S2 ecosystem and [rodrigomatta/s2.cpp](https://github.com/rodrigomatta/s2.cpp). Check the upstream and model licenses before redistribution or commercial deployment.
