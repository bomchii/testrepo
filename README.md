# s2.cpp

> **ALPHA — EXPERIMENTAL AND PROOF OF CONCEPT SOFTWARE**
> This is an early-stage, community-built inference engine. Expect rough edges, missing features, and breaking changes. Not production-ready.

**s2.cpp** — Fish Audio's S2 Pro Dual-AR text-to-speech model running locally via a pure C++/GGML inference engine with separate CPU, Vulkan, CUDA, and Metal backends. No Python runtime required after build.

> **Built on Fish Audio S2 Pro**
> The model weights are licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved.
> See [LICENSE.md](LICENSE.md) for full terms. Commercial use requires a separate license from Fish Audio — contact [business@fish.audio](mailto:business@fish.audio).

---

## What this is

This repository is a fork of https://github.com/rodrigomatta/s2.cpp which is currently a command-line script.

This version of s2.cpp is an API server version, compatible with the Fish Audio API endpoint /v1/tts . On good GPUs, it can achieve real-time performance while taking advantage of quantized models. If needed, VRAM savings can be used to run multiple services simultaneously.

The original project was tested primarily on NVIDIA RTX hardware with Vulkan. This fork also contains dedicated Windows CPU/CUDA/Vulkan and macOS Metal build workflows; those platform builds should be treated as experimental until their GitHub Actions jobs and target hardware tests pass.

This version of s2.cpp offers less flexibility and fewer features than the original s2.cpp version.

The two main areas of focus are reducing VRAM usage and maintaining the inference speed of the best configurations of the original model. This must be achieved while maintaining very high voice cloning quality and accurate intonation and tag reproduction. To achieve these two goals, our choices included using GPUs for the codec as well as loading the reference audio when the server was launched.

This repository contains:

- **`s2.cpp`** — a self-contained C++17 inference engine built on [ggml](https://github.com/ggml-org/ggml), handling tokenization, Dual-AR generation, audio codec encode/decode, and WAV output throw API similar with Fish Audio API
- **`tokenizer.json`** — Qwen3 BPE tokenizer with ByteLevel pre-tokenization
- GGUF model files are **not included** here — see [Model variants](#model-variants) below

The engine runs the full pipeline: text via API → tokens → Slow-AR transformer (with KV cache) → Fast-AR codebook decoder → audio codec → WAV file return via API.

---

## Model variants

To reduce VRAM usage, new GGUFs were created. The transformer-only GGUFs were created from the full GGUFs from rodrigomt in the table below. The codec-only GGUFs are adaptations of the original codec.pth file to GGUF and quantized formats.

GGUF files are available at [mach9243/s2-pro-gguf](https://huggingface.co/mach9243/s2-pro-gguf) on Hugging Face.

| File | Size | Notes |
|---|---|---|
| `s2-pro-f16-transformer-only.gguf` | 9.2 GB | Full precision — reference quality |
| `s2-pro-f16-codec-only.gguf` | 1.4 GB | Full precision — reference quality |
| `s2-pro-q8_0-transformer-only.gguf` | 5.4 GB | Near-lossless |
| `s2-pro-q8_0-codec-only.gguf` | 1.0 GB | Near-lossless |
| `s2-pro-q4_k_m-transformer-only.gguf` | 2.8 GB | Good quality/size balance |
| `s2-pro-q4_k_m-codec-only.gguf` | 1.0 GB | Good quality/size balance |

To use these models, you must use together one of the transformer models (--model) and one of the codec models (--model-codec). The VRAM used is the sum of the 2 models plus a few hundred MB.

In order to compare several samples generated with these model pairs, they were submitted to the HF page


Rodrigomt's original GGUF files remain functional if needed. Files are available at [rodrigomt/s2-pro-gguf](https://huggingface.co/rodrigomt/s2-pro-gguf) on Hugging Face.

| File | Size | Notes |
|---|---|---|
| `s2-pro-f16.gguf` | 9.3 GB | Full precision — reference quality 19+ GB VRAM |
| `s2-pro-q8_0.gguf` | 5.7 GB | Near-lossless — recommended for 12+ GB VRAM |
| `s2-pro-q6_k.gguf` | 4.8 GB | Good quality/size balance — recommended for 11+ GB VRAM |
| `s2-pro-q3_k.gguf` | 4.0 GB | Good quality/size balance — recommended for 8+ GB VRAM |



All variants include both the transformer weights and the audio codec in a single file.

---

## Requirements

### Build dependencies

- CMake ≥ 3.14
- C++17 compiler (GCC ≥ 10, Clang ≥ 11, MSVC 2019+)
- Crow
- For Vulkan GPU support: Vulkan SDK and `glslc`
- For CUDA GPU support: a CUDA Toolkit supported by your compiler/toolchain
- For Metal GPU support: macOS with Xcode command-line tools

```bash
# Ubuntu / Debian
sudo apt install cmake build-essential
```

# Vulkan (optional, recommended for GPU acceleration on AMD/Intel/Nvidia)

```bash
sudo apt install vulkan-tools libvulkan-dev glslc
```

### Runtime

No Python or PyTorch is required. Runtime dependencies depend on the selected backend and how GGML was linked. The GitHub Actions release jobs build one backend per artifact and package the backend-specific runtime files they require.

Windows release artifacts use the MSVC runtime dynamically, so the **Microsoft Visual C++ 2015–2022 Redistributable (x64)** must be installed unless it is already present. The Vulkan build also needs the Vulkan loader supplied by a working GPU driver; the CUDA build needs a compatible NVIDIA driver (its selected CUDA runtime DLLs are packaged with the artifact).

---

## Building

Clone including the GGML submodule:

```bash
git clone --recurse-submodules https://github.com/mach92432/s2.cpp.git
cd s2.cpp
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

### CPU only

```bash
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DS2_VULKAN=OFF -DS2_CUDA=OFF -DS2_METAL=OFF
cmake --build build-cpu --parallel
```

### With Vulkan GPU support

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DS2_VULKAN=ON
cmake --build build --parallel $(nproc)
```

The binary is produced at `build/s2`.

### With CUDA GPU support

```bash
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DS2_CUDA=ON
cmake --build build-cuda --parallel
```

### With Metal GPU support (macOS only)

```bash
cmake -B build-metal -DCMAKE_BUILD_TYPE=Release -DS2_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build-metal --parallel
```

Only one GPU backend should be enabled in a build directory. Prefer separate build directories (`build-cpu`, `build-vulkan`, `build-cuda`, `build-metal`) so cached CMake options cannot leak between artifacts.

---

## Usage

### Basic server launch for GPU Vulkan (ex: Nvidia)

Put model.gguf (or link) in s2.cpp directory

Put codec.gguf (or link) in s2.cpp directory

Only for cloning put reference.wav and reference.txt in s2.cpp directory

```bash
build/s2 -v 0 --codec-vulkan 0 --model-codec codec.gguf --port 8081
```

`--model model.gguf` to specify the path to a GGUF model (default model.gguf)
`--model-codec codec.gguf` to specify the path to a GGUF model for 'codec' processing only. By default, it's the model specified by '--model' or 'model.gguf'.
`-v 0` selects the first device of the GPU backend compiled into that executable (Vulkan/CUDA; Metal exposes device 0).
`--codec-vulkan 0` selects the first GPU device for the audio codec. The name is kept for compatibility, but it also works in CUDA builds; Metal maps any GPU selection to device 0. Omit it to inherit the transformer device, or pass `--codec-vulkan -1` to force the codec to CPU.
`--host 127.0.0.1` is the default bind address. Use `--host 0.0.0.0` only when LAN access is intentional.
`--port 8081` selects the listening port.
`--help` shows the complete option list.

### GPU inference via Vulkan with curl

```bash
curl -X POST http://localhost:8081/v1/tts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d "{\"text\":\"[emphasis] Bonjour Bob ! [pause] Je suis bien là, mais il semble y avoir une petite confusion : je ne suis pas Samantha. Je suis **Anna** (ou **Anaïs**, selon l'humeur du jour !). [laughing] Je suis parfaitement réveillée et prête à t'aider. Que souhaites-tu faire ?\",\"format\":\"wav\"}" \
  -o output.wav
```

The server binds to `127.0.0.1` by default. This matters because reference/voice-management endpoints can be given local audio paths. If you opt in to `--host 0.0.0.0`, place the service behind an access-control layer before using it on an untrusted LAN or the public Internet.

### Common options

| Flag | Default | Description |
|---|---:|---|
| `-v`, `--vulkan <N>` | `-1` | Transformer backend device. `-1` = CPU; `0+` = compiled GPU backend device. |
| `--codec-vulkan <N>` | `-2` (inherit) | Codec device. `-2` inherits transformer device, `-1` forces CPU, `0+` selects the compiled GPU backend device (Metal normalizes to 0). |
| `--host <IP>` | `127.0.0.1` | Server bind address. Use `0.0.0.0` only for intentional LAN exposure. |
| `--port <N>` | `8080` | HTTP/WebSocket listening port. |
| `--segment` | off | Split long text before generation. |
| `--codec-chunk <N>` | `0` (auto) | Codec frames per decode call. |
| `--codec-overlap <N>` | `0` | Context overlap used for codec chunk/stream boundaries. |
| `--stream-decode-stride <N>` | `0` (auto = 4) | Streaming decode stride; negative disables stride mode. |
| `--voice <id>` | — | Reuse a saved `.s2voice` profile. |
| `--save-voice` | off | Encode/save `--prompt-audio` + `--prompt-text` as `--voice` and exit unless `--output` is also supplied. |
| `--output <path>` | — | One-shot CLI synthesis to WAV; no server is started. |

Run `s2 --help` for the complete set of sampling, segmentation, RAS, reference-audio and generation-limit options.

---


## Benchmark

For speed, we don't recommend using the CPU for the codec. Using the CPU for the codec doubles the total processing time.

I suggest choosing transformer and codec quantized version that can fit in the allocated VRAM. Only a few hundred MB will be used extra during inference. It's possible to use two GPUs.

The audio generation speed is approximately 0.8x on an RTX3090 (RTF 1.3).

The speed is roughly the same regardless of the model.

The sound quality remains acceptable for the smallest model .

Voice cloning works correctly.

Tags may be less respected with high levels of quantization.

Generating short texts often results in artifacts at the end. Whenever possible, long texts should be split into segments of at least 90 characters.

---

## Architecture notes

S2 Pro uses a **Dual-AR** architecture:

- **Slow-AR** — a 36-layer Qwen3-based transformer (4.13B params) that processes the full token sequence with GQA (32 heads, 8 KV heads), RoPE at 1M base, QK norm, and a persistent KV cache
- **Fast-AR** — a 4-layer transformer (0.42B params) that autoregressively generates 10 acoustic codebook tokens from the Slow-AR hidden state for each semantic step
- **Audio codec** — a convolutional encoder/decoder with residual vector quantization (RVQ, 10 codebooks × 4096 entries) that converts between audio waveforms and discrete codes

Total: ~4.56B parameters.

---

## Implementation notes

The C++ engine (`src/`) is built entirely on [ggml](https://github.com/ggml-org/ggml) include in this code. Key design decisions:

- **Reference audio caching** — the optional global `reference.wav`/`reference.txt` pair is prepared at startup; per-request reference files are encoded on demand and cached by path plus file fingerprint.
- **Separate persistent `gallocr` allocators** for Slow-AR and Fast-AR — each path keeps its own compute buffer, avoiding memory re-planning per token
- **Temporary prefill allocator** — freed immediately after prefill, so the large compute buffer does not persist into the generation loop
- **Independent model/codec placement** — transformer and codec may use different devices. The codec defaults to inheriting the transformer device (`-2` internally); `--codec-vulkan -1` explicitly forces CPU.
- **posix_fadvise(DONTNEED)** after weight loading on supported POSIX systems — advises the kernel that model-file pages are no longer needed once weights have been copied to their backend buffers
- **Correct ByteLevel tokenization** — the GPT-2 byte-to-unicode table is applied before BPE, producing token IDs identical to the HuggingFace reference tokenizer

---

## Known limitations (alpha)

- HTTP WAV/PCM responses are returned after the requested synthesis finishes; low-latency incremental PCM is available separately through WebSocket `/ws/tts`.
- Inference is serialized through one shared pipeline/model instance, so concurrent HTTP/WebSocket requests queue rather than execute model inference in parallel.
- No batch inference
- Voice cloning quality depends heavily on reference audio length and SNR
- Windows CPU/Vulkan/CUDA and macOS Metal builds have CI jobs, but target-hardware behavior still needs validation on the GPUs/OS versions you intend to support.
- Only Nvidia GPUs were tested with Vulkan. Other Vulkan-compatible GPUs were not tested.

---

## License

The model weights and associated materials are licensed under the **Fish Audio Research License**. Key points:

- **Research and non-commercial use:** free, under the terms of this Agreement
- **Commercial use:** requires a separate written license from Fish Audio
- When distributing, you must include a copy of the license and the attribution notice
- Attribution: *"This model is licensed under the Fish Audio Research License, Copyright © 39 AI, INC. All Rights Reserved."*

Full license: [LICENSE.md](LICENSE.md)

Commercial licensing: [https://fish.audio](https://fish.audio) · [business@fish.audio](mailto:business@fish.audio)

The inference engine source code (`src/`) is a Derivative Work of the Fish Audio Materials as defined in the Agreement and is distributed under the same Fish Audio Research License terms.
