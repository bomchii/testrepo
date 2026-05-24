#include "s2_pipeline.h"
#include <crow.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <thread>
#include <chrono>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <unistd.h>
#  include <limits.h>
#endif

// tokenizer_data.h es generado por el workflow antes de compilar.
// Contiene: extern const char   tokenizer_json_data[];
//           extern const size_t tokenizer_json_size;
// Si no existe (build local sin el workflow), se usa el archivo en disco.
#if __has_include("tokenizer_data.h")
#  include "tokenizer_data.h"
#  define S2_TOKENIZER_EMBEDDED 1
#endif

// Devuelve el directorio donde vive el ejecutable, con separador final.
// Ej: "C:\\Users\\you\\s2\\"  o  "/home/you/s2/"
static std::string get_exe_dir() {
#ifdef _WIN32
    wchar_t buf[32768] = {};
    DWORD len = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(std::size(buf)));
    if (len == 0) return "";
    // Convertir UTF-16 → UTF-8
    int utf8_len = WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(len), nullptr, 0, nullptr, nullptr);
    std::string path(utf8_len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(len), path.data(), utf8_len, nullptr, nullptr);
    // Recortar hasta el último separador
    auto sep = path.find_last_of("/\\");
    return (sep == std::string::npos) ? "" : path.substr(0, sep + 1);
#else
    char buf[PATH_MAX] = {};
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return "";
    std::string path(buf, len);
    auto sep = path.rfind('/');
    return (sep == std::string::npos) ? "" : path.substr(0, sep + 1);
#endif
}

int main(int argc, char** argv) {
    const std::string exe_dir = get_exe_dir();

    s2::PipelineParams params;
    // -----------------------------------------------------------------------
    // Defaults conservadores — funcionan en cualquier PC sin GPU Vulkan.
    // El usuario activa GPU explícitamente con -v / --codec-vulkan.
    // -----------------------------------------------------------------------
    params.model_path           = exe_dir + "model.gguf";
    params.tokenizer_path       = exe_dir + "tokenizer.json";
    params.vulkan_device        = -1;   // CPU por defecto (seguro en cualquier PC)
    params.codec_vulkan_device  = -1;   // CPU por defecto
    params.gen.n_threads        = 4;
    params.gen.max_new_tokens   = 1024;
    params.gen.temperature      = 0.7f;
    params.gen.top_p            = 0.7f;
    params.gen.top_k            = 30;
    params.segment_sentences    = false; // OFF por defecto — el usuario activa con --segment
    params.codec_chunk_frames   = 0;     // 0 = automático (se calcula en runtime)
    params.codec_overlap_frames  = 0;     // 0 = sin overlap (recomendado con VRAM ajustada)
    params.min_seg_chars         = 0;     // 0 = sin filtro de longitud mínima
    // GenerateParams defaults (también en s2_generate.h)
    params.gen.temperature       = 0.7f;
    params.gen.top_p             = 0.7f;
    params.gen.top_k             = 30;
    params.gen.min_tokens_before_end = 64;
    params.gen.ras_window_size   = 10;
    params.gen.ras_high_temp     = 1.0f;
    params.gen.ras_high_top_p    = 0.9f;
    params.base_dir             = exe_dir;
    params.output_path          = "";     // vacío = usar archivo temporal Crow
    params.trim_silence         = false;
    params.voice_storage_dir    = exe_dir + "voices";
    params.stream_decode_stride_frames = 0;

    int port = 8080;
    bool list_voices = false;

    // --- Parse des arguments ---
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (arg == "--model-codec" && i + 1 < argc) {
            params.codec_model_path = argv[++i];
        } else if ((arg == "-t" || arg == "--tokenizer") && i + 1 < argc) {
            params.tokenizer_path = argv[++i];
        } else if ((arg == "-v" || arg == "--vulkan") && i + 1 < argc) {
            params.vulkan_device = std::stoi(argv[++i]);
        } else if (arg == "--codec-vulkan" && i + 1 < argc) {
            params.codec_vulkan_device = std::stoi(argv[++i]);
        } else if (arg == "--segment") {
            params.segment_sentences = true;
        } else if (arg == "--codec-chunk" && i + 1 < argc) {
            params.codec_chunk_frames = std::stoi(argv[++i]);
        } else if (arg == "--codec-overlap" && i + 1 < argc) {
            params.codec_overlap_frames = std::stoi(argv[++i]);
        } else if (arg == "--min-seg-chars" && i + 1 < argc) {
            params.min_seg_chars = std::stoi(argv[++i]);
        } else if ((arg == "--temperature" || arg == "--temp") && i + 1 < argc) {
            params.gen.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            params.gen.top_p = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            params.gen.top_k = std::stoi(argv[++i]);
        } else if (arg == "--min-end-tokens" && i + 1 < argc) {
            params.gen.min_tokens_before_end = std::stoi(argv[++i]);
        } else if (arg == "--ras-window" && i + 1 < argc) {
            params.gen.ras_window_size = std::stoi(argv[++i]);
        } else if (arg == "--ras-temp" && i + 1 < argc) {
            params.gen.ras_high_temp = std::stof(argv[++i]);
        } else if (arg == "--ras-top-p" && i + 1 < argc) {
            params.gen.ras_high_top_p = std::stof(argv[++i]);
        } else if (arg == "--max-seg-tokens" && i + 1 < argc) {
            params.max_tokens_per_segment = std::stoi(argv[++i]);
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if ((arg == "-threads" || arg == "--threads") && i + 1 < argc) {
            params.gen.n_threads = std::stoi(argv[++i]);
        } else if ((arg == "--max-tokens") && i + 1 < argc) {
            params.gen.max_new_tokens = std::stoi(argv[++i]);
        } else if ((arg == "--text") && i + 1 < argc) {
            params.text = argv[++i];
        } else if ((arg == "-pt" || arg == "--prompt-text") && i + 1 < argc) {
            params.prompt_text = argv[++i];
        } else if (arg == "--voice" && i + 1 < argc) {
            params.voice_id = argv[++i];
        } else if (arg == "--save-voice") {
            params.save_voice = true;
        } else if (arg == "--voice-dir" && i + 1 < argc) {
            params.voice_storage_dir = argv[++i];
        } else if (arg == "--list-voices") {
            list_voices = true;
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            params.output_path = argv[++i];
        } else if (arg == "--trim-silence") {
            params.trim_silence = true;
        } else if (arg == "--no-trim-silence") {
            params.trim_silence = false;
        } else if (arg == "--stream-decode-stride" && i + 1 < argc) {
            params.stream_decode_stride_frames = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout <<
R"(s2.exe — Fish Speech TTS server + CLI  (Windows / Vulkan)
HTTP+WebSocket server and CLI tool for local voice cloning with Fish Speech models.

QUICK START:
  Server — CPU (works everywhere):
    s2.exe --model s2-pro-q4_k_m-transformer-only.gguf \
           --model-codec s2-pro-q4_k_m-codec-only.gguf

  Server — RTX 3050 laptop (4 GB VRAM, iGPU on index 0):
    s2.exe --model s2-pro-q4_k_m-transformer-only.gguf \
           --model-codec s2-pro-q4_k_m-codec-only.gguf \
           -v 1 --codec-vulkan 1 --segment --codec-chunk 32 \
           --max-seg-tokens 300 --min-seg-chars 60 \
           --temperature 0.8 --top-p 0.8 --top-k 40

  CLI — synthesize once to a file (no server):
    s2.exe --model ... --model-codec ... -v 1 --codec-vulkan 1 \
           --prompt-audio ref.wav --prompt-text "Reference transcript." \
           --text "Hello, this is a cloned voice." \
           --trim-silence --output hello.wav

  Save a voice profile, then reuse it by name:
    s2.exe --model ... --model-codec ... -v 1 --codec-vulkan 1 \
           --prompt-audio ref.wav --prompt-text "Reference transcript." \
           --voice my_voice --save-voice --output /dev/null
    s2.exe --model ... --model-codec ... -v 1 --codec-vulkan 1 \
           --voice my_voice --text "Hello from saved voice." --output out.wav

  List saved voice profiles:
    s2.exe --list-voices

OPTIONS:

  Models:
    -m,  --model <path>          Path to the transformer GGUF model file.
                                 Default: model.gguf next to the executable.
         --model-codec <path>    Path to the codec GGUF model file.
                                 Default: codec.gguf next to the executable.
    -t,  --tokenizer <path>      Path to tokenizer.json.
                                 Default: tokenizer embedded inside the exe.

  GPU / device selection:
    -v,  --vulkan <N>            Vulkan device for the transformer model.
                                   -1  CPU (default — works on any machine)
                                    0  first GPU
                                    1  second GPU (use on laptops where
                                       index 0 is the Intel/AMD iGPU)
         --codec-vulkan <N>      Vulkan device for the codec model.
                                 Same index values as --vulkan.
                                 Note: q4_k_m codec only works on GPU; CPU
                                 fallback requires f16 or f32 codec weights.

  Server:
    -p,  --port <N>              HTTP port to listen on. Default: 8080.

  Reference audio (voice cloning):
    -pa, --prompt-audio <path>   Path to reference WAV/MP3 for voice cloning.
    -pt, --prompt-text <text>    Transcript of the reference audio.
                                 Required when using --save-voice.
                                 Helps the model align prosody to the reference.

  Voice profiles (save and reuse encoded reference voices):
         --voice <id>            Load a saved voice profile by name.
                                 Skips re-encoding the reference audio.
                                 Ignored if --prompt-audio is also given.
         --save-voice            Encode --prompt-audio and save it as a profile.
                                 Requires --voice <id>, --prompt-audio,
                                 and --prompt-text.
         --voice-dir <path>      Directory for .s2voice profile files.
                                 Default: voices/ next to the executable.
         --list-voices           List saved voice profiles and exit.

  CLI output (no HTTP server):
    -o,  --output <path>         Synthesize once, write WAV to <path>, exit.
                                 Combine with --text for the input text, or
                                 pipe text to stdin if --text is omitted.
         --text <text>           Input text for CLI mode (--output).
         --trim-silence          Trim trailing silence from the output WAV.
         --no-trim-silence       Keep trailing silence in the output (default).

  Generation limits:
         --threads <N>           CPU threads for CPU-bound ops. Default: 4.
         --max-tokens <N>        Max tokens per request (no --segment).
                                 Default: 1024. One token ≈ ~11 ms of audio.
         --max-seg-tokens <N>    Max tokens per sentence with --segment.
                                 Controls KV-cache size; lower = less VRAM.
                                 Default: 300 (~3.3 s per sentence).

  Segmentation (recommended for long texts or limited VRAM):
         --segment               Split text into sentences before generating.
                                 Each sentence uses its own KV cache, capping
                                 VRAM to the longest sentence, not the full text.
                                 Enable per-request: { "segment": true }
         --min-seg-chars <N>     Merge segments shorter than N characters with
                                 the next one. Prevents unnatural short clips.
                                 Default: 0 (no minimum). Recommended: 60-90.

  Codec chunking (advanced — tune for your VRAM budget):
         --codec-chunk <N>       Codec frames decoded per GPU call. Smaller =
                                 less peak VRAM, slightly more overhead.
                                 Default: 0 (auto, ~120 frames).
                                 RTX 3050 4 GB with transformer loaded: use 32.
         --codec-overlap <N>     Overlap frames between codec chunks.
                                 Smooths chunk boundaries but costs more VRAM
                                 (graph scales with chunk+overlap). Keep at 0
                                 on 4 GB GPUs with transformer in VRAM.
                                 Default: 0.

  Sampling:
         --temperature <F>       Sampling temperature. Higher = more expressive,
                                 less stable. Default: 0.7. Range: 0.1-1.5.
         --top-p <F>             Nucleus sampling threshold.
                                 Default: 0.7. Range: 0.1-1.0.
         --top-k <N>             Top-k candidates before applying top-p.
                                 Default: 30.
         --min-end-tokens <N>    Min tokens before EOS is allowed. Prevents
                                 empty output on short texts. Default: 64.

  RAS — Repetition Aware Sampling (anti-repetition):
    Detects when the model loops on a token and resamples it at higher
    temperature. Use if the output stutters or repeats syllables.
         --ras-window <N>        Recent-token window to watch. Default: 10.
         --ras-temp <F>          Temperature for the resample. Default: 1.0.
         --ras-top-p <F>         Top-p for the resample. Default: 0.9.

  Streaming (WebSocket /ws/tts):
         --stream-decode-stride <N>
                                 Codec decode cadence in frames. Lower values
                                 reduce first-chunk latency at the cost of more
                                 decode calls. 0 = auto (4 frames). Default: 0.

  Other:
    -h,  --help                  Show this help and exit.

HTTP ENDPOINTS:
  POST /v1/tts                   Fish Audio-compatible TTS endpoint.
  POST /v1/audio/speech          OpenAI-compatible TTS endpoint.
  POST /synthesize               Legacy endpoint.
  GET  /v1/models                List available models.
  GET  /health                   Health check.

  Request body (JSON) — all fields optional except "text":
    {
      "text":           "Text to synthesize.",
      "format":         "wav",
      "segment":        true,
      "prompt_text":    "Transcript of the reference audio.",
      "voice":          "my_voice",
      "temperature":    0.7,
      "top_p":          0.7,
      "top_k":          30,
      "min_end_tokens": 64,
      "ras_window":     10,
      "ras_temp":       1.0,
      "ras_top_p":      0.9,
      "codec_chunk":    32,
      "codec_overlap":  0,
      "min_seg_chars":  60,
      "trim_silence":   false,
      "stream_stride":  0
    }

WEBSOCKET ENDPOINT — /ws/tts  (streaming, minimum latency):
  Send JSON, receive binary PCM frames as each sentence finishes.
  Binary message format: [2-byte flags LE][PCM int16 LE, mono, 44100 Hz]
    flags bit 0 = is_last (1 on the final segment)
  Final message (text): {"done": true, "segments": N, "sample_rate": 44100}

  Python example:
    import asyncio, json, websockets

    async def tts():
        async with websockets.connect("ws://localhost:8080/ws/tts") as ws:
            await ws.send(json.dumps({
                "text": "Hello world. How are you today?",
                "segment": True,
                "voice": "my_voice"
            }))
            while True:
                msg = await ws.recv()
                if isinstance(msg, str):
                    print(json.loads(msg))   # {"done": true, "segments": 2, ...}
                    break
                pcm = msg[2:]               # strip 2-byte flags header
                # feed pcm (int16, mono, 44100 Hz) to your audio output

HTTP EXAMPLES:
  Basic synthesis:
    curl -X POST http://localhost:8080/v1/tts \
         -H "Content-Type: application/json" \
         -d "{"text": "Hello world.", "segment": true}" \
         --output audio.wav

  With a saved voice:
    curl -X POST http://localhost:8080/v1/tts \
         -H "Content-Type: application/json" \
         -d "{"text": "Hello.", "voice": "my_voice", "trim_silence": true}" \
         --output audio.wav

  Save a voice via HTTP:
    curl -X POST http://localhost:8080/v1/voices/my_voice \
         -H "Content-Type: application/json" \
         -d "{"audio_path": "C:/refs/speaker.wav", "transcript": "Reference text."}"

  List saved voices:
    curl http://localhost:8080/v1/voices

  Delete a voice:
    curl -X DELETE http://localhost:8080/v1/voices/my_voice
)";
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            std::cerr << "Use --help for usage information." << std::endl;
            return 1;
        }
    }

    // --list-voices: listar perfiles guardados y salir
    if (list_voices) {
        s2::VoiceProfileManager voice_mgr_tmp;
        voice_mgr_tmp.set_storage_dir(params.voice_storage_dir);
        std::vector<std::string> ids = voice_mgr_tmp.list();
        std::sort(ids.begin(), ids.end());
        if (ids.empty()) {
            std::cout << "No saved voice profiles in: " << params.voice_storage_dir << "\n";
        } else {
            std::cout << "Saved voice profiles (" << params.voice_storage_dir << "):\n";
            for (const std::string & id : ids) {
                std::cout << "  " << id << "\n";
            }
        }
        return 0;
    }

    // --save-voice validations
    if (params.save_voice) {
        if (params.voice_id.empty()) {
            std::cerr << "Error: --save-voice requires --voice <id>.\n";
            return 1;
        }
        if (params.prompt_audio_path.empty() || params.prompt_text.empty()) {
            std::cerr << "Error: --save-voice requires both --prompt-audio and --prompt-text.\n";
            return 1;
        }
    }

    auto gpu_str = [](int d) -> std::string {
        return d < 0 ? "CPU" : ("GPU " + std::to_string(d));
    };

    std::cout << "\nConfiguration:\n"
              << "  Model:         " << params.model_path << "\n"
              << "  Codec:         " << (params.codec_model_path.empty() ? params.model_path : params.codec_model_path) << "\n"
              << "  Model GPU:     " << gpu_str(params.vulkan_device) << "\n"
              << "  Codec GPU:     " << gpu_str(params.codec_vulkan_device) << "\n"
              << "  Port:          " << port << "\n"
              << "  CPU threads:   " << params.gen.n_threads << "\n"
              << "  Max tokens:    " << params.gen.max_new_tokens << "\n"
              << "  Seg tokens:    " << params.max_tokens_per_segment << " (per segment)\n"
              << "  Segmentation:  " << (params.segment_sentences ? "ON" : "OFF (use --segment to enable)") << "\n"
              << "  Codec chunk:   " << (params.codec_chunk_frames == 0 ? "auto" : std::to_string(params.codec_chunk_frames) + " frames") << "\n"
              << "  Codec overlap: " << params.codec_overlap_frames << " frames\n"
              << "  Min seg chars: " << (params.min_seg_chars == 0 ? "off" : std::to_string(params.min_seg_chars) + " chars") << "\n"
              << "  Temperature:   " << params.gen.temperature << "\n"
              << "  Top-p:        " << params.gen.top_p << "\n"
              << "  Top-k:        " << params.gen.top_k << "\n"
              << "  RAS window:    " << params.gen.ras_window_size << " tokens\n"
              << "  RAS temp:      " << params.gen.ras_high_temp << "\n";

    // --- Charger le modèle ---
    s2::Pipeline pipeline;

#ifdef S2_TOKENIZER_EMBEDDED
    // Tokenizer embebido: usar los bytes del array generado por el workflow.
    // No se necesita tokenizer.json en disco.
    params.tokenizer_data      = reinterpret_cast<const char*>(tokenizer_json_data);
    params.tokenizer_data_size = static_cast<size_t>(tokenizer_json_size);
    std::cout << "  Tokenizer:     [embedded in exe, "
              << tokenizer_json_size << " bytes]" << std::endl;
#else
    std::cout << "  Tokenizer:    " << params.tokenizer_path << std::endl;
#endif

    if (!pipeline.init(params)) {
        std::cerr << "Pipeline initialization failed." << std::endl;
        return 1;
    }

    // --- Modo CLI: --output path ---
    // Si se especificó --output, sintetizar una vez, guardar y salir (sin servidor HTTP).
    if (!params.output_path.empty()) {
        if (params.text.empty()) {
            // Leer stdin si no se pasó texto
            std::cout << "Reading text from stdin (Ctrl+D to finish)...\n";
            std::string line;
            while (std::getline(std::cin, line)) {
                if (!params.text.empty()) params.text += " ";
                params.text += line;
            }
        }
        if (params.text.empty()) {
            std::cerr << "Error: --output requires text. Pipe it or set --text.\n";
            return 1;
        }
        // segment_sentences ya configurado por --segment si el usuario lo pasó
        if (!pipeline.synthesize(params)) {
            std::cerr << "Synthesis failed.\n";
            return 1;
        }
        std::cout << "Done: " << params.output_path << "\n";
        return 0;
    }

    // --- Serveur HTTP ---
    crow::SimpleApp app;

    // Respuestas > 1MB se streamean automáticamente (sin timeout).
    // Los WAV de audio suelen ser varios MB — sin esto pueden cortar.
    app.stream_threshold(1024 * 1024); // 1 MB

    // ================================================================
    // Helper : traitement commun de synthèse
    // ================================================================
    auto do_synthesize = [&](const crow::json::rvalue& json) -> crow::response {
        s2::PipelineParams synth_params = params;

        // Compatibilité Fish Audio : "text" est le champ principal
        if (json.has("text")) {
            synth_params.text = json["text"].s();
        } else if (json.has("input")) {
            // Certaines implémentations utilisent "input"
            synth_params.text = json["input"].s();
        } else {
            return crow::response(400, "Missing 'text' field");
        }

        // Paramètres de génération (format s2.cpp natif)
        if (json.has("temperature")) synth_params.gen.temperature = json["temperature"].d();
        if (json.has("top_p"))       synth_params.gen.top_p = json["top_p"].d();
        if (json.has("top_k"))       synth_params.gen.top_k = json["top_k"].i();
        if (json.has("threads"))     synth_params.gen.n_threads = json["threads"].i();
        if (json.has("segment"))     synth_params.segment_sentences = json["segment"].b();
        if (json.has("codec_chunk"))   synth_params.codec_chunk_frames   = json["codec_chunk"].i();
        if (json.has("codec_overlap"))    synth_params.codec_overlap_frames        = json["codec_overlap"].i();
        if (json.has("min_seg_chars"))    synth_params.min_seg_chars               = json["min_seg_chars"].i();
        if (json.has("min_end_tokens"))   synth_params.gen.min_tokens_before_end   = json["min_end_tokens"].i();
        if (json.has("ras_window"))       synth_params.gen.ras_window_size         = json["ras_window"].i();
        if (json.has("ras_temp"))         synth_params.gen.ras_high_temp           = (float)json["ras_temp"].d();
        if (json.has("ras_top_p"))        synth_params.gen.ras_high_top_p          = (float)json["ras_top_p"].d();
        if (json.has("prompt_text"))      synth_params.prompt_text                  = json["prompt_text"].s();
        if (json.has("voice"))            synth_params.voice_id                     = json["voice"].s();
        if (json.has("trim_silence"))     synth_params.trim_silence                 = json["trim_silence"].b();
        if (json.has("stream_stride"))    synth_params.stream_decode_stride_frames  = json["stream_stride"].i();

        // Paramètres Fish Audio (ignorés gracieusement si non pertinents)
        // reference_id, chunk_length, normalize, format, mp3_bitrate, opus_bitrate, latency
        // On les accepte sans erreur pour la compatibilité

        // Déterminer le format de sortie
        std::string format = "wav";
        if (json.has("format")) {
            format = json["format"].s();
        }

        // Síntesis — escribe WAV a %TEMP%, Crow lo sirve directo sin cargarlo en RAM
        std::string wav_path;
        if (!pipeline.synthesize_to_file(synth_params, wav_path)) {
            return crow::response(500, "Synthesis failed");
        }

        crow::response res;

        if (format == "pcm") {
            // PCM crudo: cargar en buffer descartando cabecera WAV de 44 bytes.
            // Único caso donde sí cargamos en RAM (el cliente pide raw PCM).
            std::ifstream f(wav_path, std::ios::binary);
            f.seekg(44);
            std::string pcm_data((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
            std::remove(wav_path.c_str());
            res.set_header("Content-Type", "audio/pcm");
            res.set_header("X-Sample-Rate", std::to_string(pipeline.sample_rate()));
            res.body = std::move(pcm_data);
        } else {
            // WAV (default): Crow sirve el archivo temporal directamente.
            // set_static_file_info usa sendfile del OS — cero copias en RAM.
            // El archivo se borra después de que Crow lo envíe... pero Crow no
            // tiene callback de finalización, así que usamos un workaround:
            // ponemos la ruta en el header X-Temp-File y lo borramos en un
            // middleware de post-respuesta. Por ahora lo borramos con un pequeño
            // delay en un thread separado (simple y funcional).
            res.set_header("Content-Type", "audio/wav");
            res.set_static_file_info_unsafe(wav_path);

            // Borrar el archivo temporal después de ~5s (tiempo suficiente para
            // que Crow termine de enviarlo incluso con conexiones lentas)
            std::string path_copy = wav_path;
            std::thread([path_copy]() {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                std::remove(path_copy.c_str());
                std::cout << "[Cleanup] Deleted temp: " << path_copy << "\n";
            }).detach();
        }

        return res;
    };

    // ================================================================
    // Route Fish Audio compatible : POST /v1/tts
    // ================================================================
    CROW_ROUTE(app, "/v1/tts")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        auto json = crow::json::load(req.body);
        if (!json) {
            return crow::response(400, "Invalid JSON");
        }
        return do_synthesize(json);
    });

    // ================================================================
    // Route legacy : POST /synthesize (compatibilité avec vos tests)
    // ================================================================
    CROW_ROUTE(app, "/synthesize")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        auto json = crow::json::load(req.body);
        if (!json) {
            return crow::response(400, "Invalid JSON");
        }
        return do_synthesize(json);
    });

    // ================================================================
    // Route OpenAI compatible : POST /v1/audio/speech
    // ================================================================
    CROW_ROUTE(app, "/v1/audio/speech")
    .methods("POST"_method)
    ([&](const crow::request& req) {
        auto json = crow::json::load(req.body);
        if (!json) {
            return crow::response(400, "Invalid JSON");
        }
        return do_synthesize(json);
    });

    // ================================================================
    // Health check & info
    // ================================================================
    CROW_ROUTE(app, "/v1/models")
    .methods("GET"_method)
    ([&]() {
        crow::json::wvalue resp;
        resp["object"] = "list";
        crow::json::wvalue model;
        model["id"] = "s2-pro-local";
        model["object"] = "model";
        model["owned_by"] = "local";
        resp["data"][0] = std::move(model);
        return crow::response(200, resp);
    });

    // ================================================================
    // Voice profile management — GET/POST/DELETE /v1/voices
    // ================================================================

    // GET /v1/voices — list all saved voice profiles
    CROW_ROUTE(app, "/v1/voices")
    .methods("GET"_method)
    ([&]() {
        s2::VoiceProfileManager mgr;
        mgr.set_storage_dir(params.voice_storage_dir);
        std::vector<std::string> ids = mgr.list();
        std::sort(ids.begin(), ids.end());

        crow::json::wvalue resp;
        resp["object"] = "list";
        int idx = 0;
        for (const std::string & id : ids) {
            crow::json::wvalue item;
            item["id"]     = id;
            item["object"] = "voice";
            resp["data"][idx++] = std::move(item);
        }
        resp["count"] = (int)ids.size();
        return crow::response(200, resp);
    });

    // POST /v1/voices/<id> — save a voice profile from multipart (audio + transcript)
    // Body JSON: { "transcript": "...", "audio_path": "/abs/path/to/ref.wav" }
    // (uploading raw audio bytes via multipart is left for a future iteration;
    //  audio_path is sufficient for local use)
    CROW_ROUTE(app, "/v1/voices/<string>")
    .methods("POST"_method)
    ([&](const crow::request& req, const std::string & voice_id) {
        auto json = crow::json::load(req.body);
        if (!json) return crow::response(400, "Invalid JSON");

        if (!json.has("audio_path") || !json.has("transcript")) {
            return crow::response(400, "Required fields: audio_path, transcript");
        }

        s2::PipelineParams vp = params;
        vp.prompt_audio_path  = json["audio_path"].s();
        vp.prompt_text        = json["transcript"].s();
        vp.voice_id           = voice_id;
        vp.save_voice         = true;
        vp.voice_storage_dir  = params.voice_storage_dir;
        vp.text               = "__probe__";   // dummy — needed for init check only

        // Encode the reference audio and save the profile.
        // We reuse get_ref_codes via a minimal synthesize call that stops
        // after encoding (no actual TTS generation).
        std::vector<int32_t> codes;
        int32_t T_prompt = 0;
        if (!pipeline.encode_reference(vp, codes, T_prompt)) {
            return crow::response(500, "Failed to encode reference audio");
        }

        s2::VoiceProfileManager mgr;
        mgr.set_storage_dir(params.voice_storage_dir);

        s2::VoiceProfile profile;
        profile.transcript    = vp.prompt_text;
        profile.codes         = codes;
        profile.T_prompt      = T_prompt;
        profile.num_codebooks = pipeline.num_codebooks();
        profile.codebook_size = pipeline.codebook_size();
        profile.sample_rate   = pipeline.sample_rate();

        if (!mgr.save(voice_id, profile)) {
            return crow::response(500, "Failed to save voice profile");
        }

        crow::json::wvalue resp;
        resp["id"]       = voice_id;
        resp["object"]   = "voice";
        resp["T_prompt"] = T_prompt;
        resp["saved"]    = true;
        std::cout << "[Voice] Saved via HTTP: " << voice_id << " (" << T_prompt << " frames)\n";
        return crow::response(201, resp);
    });

    // GET /v1/voices/<id> — get metadata for a single voice profile
    CROW_ROUTE(app, "/v1/voices/<string>")
    .methods("GET"_method)
    ([&](const std::string & voice_id) {
        s2::VoiceProfileManager mgr;
        mgr.set_storage_dir(params.voice_storage_dir);
        try {
            s2::VoiceProfile profile = mgr.load(voice_id);
            crow::json::wvalue resp;
            resp["id"]             = voice_id;
            resp["object"]         = "voice";
            resp["transcript"]     = profile.transcript;
            resp["T_prompt"]       = profile.T_prompt;
            resp["num_codebooks"]  = profile.num_codebooks;
            resp["codebook_size"]  = profile.codebook_size;
            resp["sample_rate"]    = profile.sample_rate;
            return crow::response(200, resp);
        } catch (const std::exception & e) {
            crow::json::wvalue err;
            err["error"] = std::string("Voice not found: ") + e.what();
            return crow::response(404, err);
        }
    });

    // DELETE /v1/voices/<id> — delete a saved voice profile
    CROW_ROUTE(app, "/v1/voices/<string>")
    .methods("DELETE"_method)
    ([&](const std::string & voice_id) {
        s2::VoiceProfileManager mgr;
        mgr.set_storage_dir(params.voice_storage_dir);
        if (!mgr.remove(voice_id)) {
            crow::json::wvalue err;
            err["error"] = "Voice not found: " + voice_id;
            return crow::response(404, err);
        }
        crow::json::wvalue resp;
        resp["id"]     = voice_id;
        resp["deleted"] = true;
        std::cout << "[Voice] Deleted via HTTP: " << voice_id << "\n";
        return crow::response(200, resp);
    });

    // ================================================================
    CROW_ROUTE(app, "/health")
    .methods("GET"_method)
    ([]() {
        return crow::response(200, "OK");
    });

    CROW_ROUTE(app, "/")
    ([&port]() {
        crow::json::wvalue info;
        info["status"] = "running";
        info["port"] = port;
        info["endpoints"][0] = "/v1/tts";
        info["endpoints"][1] = "/synthesize";
        info["endpoints"][2] = "/v1/audio/speech";
        info["endpoints"][3] = "/v1/models";
        info["endpoints"][4] = "/v1/voices";
        info["endpoints"][5] = "/health";
        return crow::response(200, info);
    });

    // ================================================================
    // WebSocket /ws/tts — streaming real por segmento de oración.
    //
    // Protocolo (JSON sobre WebSocket):
    //
    //   Cliente → Servidor:
    //     { "text": "...", "segment": true, "reference_audio": "path" }
    //
    //   Servidor → Cliente (mensajes binarios):
    //     [2 bytes little-endian: flags] [PCM int16 LE, mono, 44100Hz]
    //     flags bit0 = is_last (1 si es el último segmento)
    //
    //   Servidor → Cliente (mensaje de texto al finalizar):
    //     { "done": true, "segments": N, "sample_rate": 44100 }
    //
    //   Servidor → Cliente (en caso de error):
    //     { "error": "descripción" }
    //
    // El cliente puede empezar a reproducir el primer mensaje binario
    // antes de que lleguen los siguientes — latencia = primera oración.
    // ================================================================
    CROW_WEBSOCKET_ROUTE(app, "/ws/tts")
    .onopen([](crow::websocket::connection& conn) {
        std::cout << "[WS] Client connected: " << conn.get_remote_ip() << "\n";
    })
    .onclose([](crow::websocket::connection& conn, const std::string& reason, uint16_t) {
        std::cout << "[WS] Client disconnected: " << reason << "\n";
    })
    .onmessage([&](crow::websocket::connection& conn,
                   const std::string& data,
                   bool is_binary) {
        if (is_binary) {
            conn.send_text("{\"error\": \"expected JSON text message\"}");
            return;
        }

        auto json = crow::json::load(data);
        if (!json || !json.has("text")) {
            conn.send_text("{\"error\": \"missing 'text' field\"}");
            return;
        }

        s2::PipelineParams ws_params = params;
        ws_params.text = json["text"].s();
        if (json.has("segment"))          ws_params.segment_sentences  = json["segment"].b();
        if (json.has("temperature"))      ws_params.gen.temperature    = json["temperature"].d();
        if (json.has("top_p"))            ws_params.gen.top_p          = json["top_p"].d();
        if (json.has("top_k"))            ws_params.gen.top_k          = json["top_k"].i();
        if (json.has("reference_audio"))  ws_params.prompt_audio_path  = json["reference_audio"].s();
        if (json.has("codec_chunk"))      ws_params.codec_chunk_frames   = json["codec_chunk"].i();
        if (json.has("codec_overlap"))    ws_params.codec_overlap_frames          = json["codec_overlap"].i();
        if (json.has("min_seg_chars"))    ws_params.min_seg_chars                 = json["min_seg_chars"].i();
        if (json.has("min_end_tokens"))   ws_params.gen.min_tokens_before_end     = json["min_end_tokens"].i();
        if (json.has("ras_window"))       ws_params.gen.ras_window_size           = json["ras_window"].i();
        if (json.has("ras_temp"))         ws_params.gen.ras_high_temp             = (float)json["ras_temp"].d();
        if (json.has("ras_top_p"))        ws_params.gen.ras_high_top_p            = (float)json["ras_top_p"].d();
        if (json.has("prompt_text"))      ws_params.prompt_text                     = json["prompt_text"].s();
        if (json.has("voice"))            ws_params.voice_id                        = json["voice"].s();
        if (json.has("trim_silence"))     ws_params.trim_silence                    = json["trim_silence"].b();
        if (json.has("stream_stride"))    ws_params.stream_decode_stride_frames     = json["stream_stride"].i();

        std::cout << "[WS] Sintetizando: \"" << ws_params.text << "\""
                  << (ws_params.segment_sentences ? " [segmentado]" : "") << "\n";

        int segment_count = 0;

        // El callback se llama por cada segmento de oración.
        // Enviamos: [2 bytes flags little-endian][PCM int16 LE]
        s2::StreamCallback cb = [&](const int16_t* pcm, size_t n_samples, bool is_last) -> bool {
            try {
                // Construir mensaje binario: 2 bytes de flags + PCM
                uint16_t flags = is_last ? 1u : 0u;
                std::string msg(2 + n_samples * 2, '\0');
                msg[0] = static_cast<char>(flags & 0xFF);
                msg[1] = static_cast<char>((flags >> 8) & 0xFF);
                std::memcpy(msg.data() + 2, pcm, n_samples * 2);
                conn.send_binary(msg);
                segment_count++;
                return true; // continuar generando
            } catch (...) {
                std::cerr << "[WS] Error sending segment — client disconnected?\n";
                return false; // abortar generación
            }
        };

        bool ok = pipeline.synthesize_streaming(ws_params, cb);

        // Mensaje final de texto
        crow::json::wvalue done_msg;
        if (ok) {
            done_msg["done"]        = true;
            done_msg["segments"]    = segment_count;
            done_msg["sample_rate"] = pipeline.sample_rate();
        } else {
            done_msg["error"]    = "synthesis failed";
            done_msg["segments"] = segment_count;
        }
        conn.send_text(done_msg.dump());
    });

    std::cout << "\nEndpoints:\n"
              << "  POST /v1/tts           (Fish Audio compatible)\n"
              << "  POST /synthesize       (legacy)\n"
              << "  POST /v1/audio/speech  (OpenAI compatible)\n"
              << "  GET  /v1/models\n"
              << "  GET  /health\n"
              << "  WS   /ws/tts           (streaming — minimum latency)\n\n";

    std::cout << "Server listening on port " << port << "...\n";
    app.port(port).multithreaded().run();
    return 0;
}
