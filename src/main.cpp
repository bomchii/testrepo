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

    int port = 8080;

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
        } else if (arg == "--help" || arg == "-h") {
            std::cout <<
R"(s2.exe — Fish Speech TTS inference server  (Windows / Vulkan)
A local HTTP+WebSocket server that converts text to speech using Fish Speech models.

QUICK START:
  CPU only (works everywhere):
    s2.exe --model model.gguf --model-codec codec.gguf

  Single GPU (desktop, plenty of VRAM):
    s2.exe --model model.gguf --model-codec codec.gguf -v 0 --codec-vulkan 0

  Laptop with iGPU + dGPU, limited VRAM (e.g. RTX 3050 4 GB):
    s2.exe --model model.gguf --model-codec codec.gguf \
           -v 1 --codec-vulkan 1 --segment --codec-chunk 32 \
           --max-seg-tokens 300 --min-seg-chars 60

OPTIONS:

  Models:
    -m,  --model <path>         Path to the transformer GGUF model file.
                                Default: model.gguf next to the executable.
         --model-codec <path>   Path to the codec GGUF model file.
                                Default: codec.gguf next to the executable.
    -t,  --tokenizer <path>     Path to tokenizer.json.
                                Default: version embedded inside the executable.

  GPU / device selection:
    -v,  --vulkan <N>           Vulkan device index for the transformer model.
                                  -1  CPU (default — works on any machine)
                                   0  first GPU
                                   1  second GPU (use this on laptops where
                                      index 0 is the Intel/AMD iGPU)
         --codec-vulkan <N>     Vulkan device index for the codec model.
                                Same values as --vulkan.
                                The codec and transformer can run on different
                                devices. If you get OOM on the codec, set this
                                to -1 to run it on CPU (only works with f16/f32
                                codec weights, NOT with q4_k_m quantization).

  Server:
    -p,  --port <N>             HTTP port to listen on. Default: 8080.

  Generation limits:
         --threads <N>          Number of CPU threads for CPU-bound ops.
                                Default: 4.
         --max-tokens <N>       Maximum tokens the model may generate per
                                request (whole text, no --segment).
                                Default: 1024.  One token ≈ one codec frame
                                ≈ ~11 ms of audio at 44100 Hz / hop 512.
         --max-seg-tokens <N>   Maximum tokens per sentence when --segment is
                                active. Controls KV-cache size per segment.
                                Lower values use less VRAM.
                                Default: 300 (~3.3 s of audio per sentence).

  Segmentation (recommended for long texts or limited VRAM):
         --segment              Split the input text into sentences before
                                generating. Each sentence is synthesized
                                independently, which caps VRAM usage to the
                                longest sentence instead of the full text.
                                Audio segments are concatenated into one WAV.
                                Can also be enabled per-request:
                                  { "text": "...", "segment": true }
         --min-seg-chars <N>    Minimum character length for a sentence segment.
                                Segments shorter than N are merged with the next
                                one before synthesis, preventing very short
                                clips that tend to sound unnatural or cut off.
                                Default: 0 (no minimum — keep all segments).
                                Recommended: 60-90 for most use cases.

  Codec chunking (advanced — tune for your VRAM budget):
         --codec-chunk <N>      Number of codec frames to decode per GPU call.
                                Each chunk is decoded independently. Smaller
                                values use less peak VRAM at the cost of
                                slightly more overhead.
                                Default: 0 (auto, ~120 frames per chunk).
                                With 4 GB VRAM and the transformer loaded,
                                use 32.
         --codec-overlap <N>    Number of frames of overlap between consecutive
                                codec chunks. Overlap smooths the audio at chunk
                                boundaries but increases the VRAM cost of every
                                chunk after the first (the graph grows with
                                chunk_len + overlap). On a 4 GB GPU with the
                                transformer already in VRAM, keep this at 0.
                                Default: 0.

  Sampling (controls voice variability and expressiveness):
         --temperature <F>      Sampling temperature for the transformer.
                                Higher = more expressive and varied, but less
                                stable. Lower = more monotone but consistent.
                                Default: 0.7.  Range: 0.1 – 1.5.
         --top-p <F>            Nucleus (top-p) sampling threshold.
                                The model samples only from the smallest set of
                                tokens whose cumulative probability exceeds p.
                                Default: 0.7.  Range: 0.1 – 1.0.
         --top-k <N>            Top-k sampling. Only the k most likely tokens
                                are kept as candidates before applying top-p.
                                Default: 30.
         --min-end-tokens <N>   Minimum tokens to generate before the model is
                                allowed to emit the end-of-sequence token.
                                Prevents the model from producing very short
                                or empty clips for short input texts.
                                Default: 64.

  RAS — Repetition Aware Sampling (anti-repetition):
    RAS detects when the model is looping (repeating the same semantic token
    recently seen) and resamples that token with a higher temperature to break
    the loop. Relevant when the model gets stuck repeating a syllable or sound.

         --ras-window <N>       Number of recent tokens to watch for repetition.
                                If the current token appears in this window,
                                RAS triggers a resample.
                                Default: 10.
         --ras-temp <F>         Temperature used when resampling a repeated token.
                                Should be higher than --temperature to escape loops.
                                Default: 1.0.
         --ras-top-p <F>        Top-p used when resampling a repeated token.
                                Default: 0.9.

  Other:
    -h,  --help                 Show this help and exit.

HTTP ENDPOINTS:
  POST /v1/tts                  Fish Audio-compatible synthesis endpoint.
  POST /v1/audio/speech         OpenAI-compatible synthesis endpoint.
  POST /synthesize              Legacy endpoint.
  GET  /v1/models               List available models.
  GET  /health                  Health check.

  Request body (JSON):
    {
      "text":            "Text to synthesize.",
      "format":          "wav",           // output format (wav only for now)
      "segment":         true,            // enable sentence segmentation
      "temperature":     0.7,
      "top_p":           0.7,
      "top_k":           30,
      "min_end_tokens":  64,
      "ras_window":      10,
      "ras_temp":        1.0,
      "ras_top_p":       0.9,
      "codec_chunk":     32,
      "codec_overlap":   0,
      "min_seg_chars":   60
    }

WEBSOCKET ENDPOINT — /ws/tts  (streaming, minimum latency):
  Send a JSON message with the text; receive binary audio frames as each
  sentence finishes, without waiting for the full synthesis to complete.
  Each binary message: [2-byte flags LE][PCM int16 LE, mono, 44100 Hz]
  A final JSON message {"done": true, "segments": N} signals completion.

  Python example:
    import asyncio, json, websockets

    async def tts():
        uri = "ws://localhost:8080/ws/tts"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "text": "Hello world. How are you?",
                "segment": True
            }))
            while True:
                msg = await ws.recv()
                if isinstance(msg, str):       # final status message
                    print(json.loads(msg))
                    break
                pcm = msg[2:]                  # skip 2-byte flags header
                # feed pcm (int16 mono 44100 Hz) to your audio output

HTTP EXAMPLE:
  curl -X POST http://localhost:8080/v1/tts \
       -H "Content-Type: application/json" \
       -d '{"text": "Hello world.", "segment": true}' \
       --output audio.wav
)";
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            std::cerr << "Use --help for usage information." << std::endl;
            return 1;
        }
    }

    auto gpu_str = [](int d) -> std::string {
        return d < 0 ? "CPU" : ("GPU " + std::to_string(d));
    };

    std::cout << "\nConfiguracion:\n"
              << "  Modelo:       " << params.model_path << "\n"
              << "  Codec:        " << (params.codec_model_path.empty() ? params.model_path : params.codec_model_path) << "\n"
              << "  Modelo GPU:   " << gpu_str(params.vulkan_device) << "\n"
              << "  Codec GPU:    " << gpu_str(params.codec_vulkan_device) << "\n"
              << "  Puerto:       " << port << "\n"
              << "  Hilos CPU:    " << params.gen.n_threads << "\n"
              << "  Max tokens:   " << params.gen.max_new_tokens << "\n"
              << "  Seg tokens:   " << params.max_tokens_per_segment << " (por segmento)\n"
              << "  Segmentacion: " << (params.segment_sentences ? "ON" : "OFF (usa --segment para activar)") << "\n"
              << "  Codec chunk:  " << (params.codec_chunk_frames == 0 ? "auto" : std::to_string(params.codec_chunk_frames) + " frames") << "\n"
              << "  Codec overlap:" << params.codec_overlap_frames << " frames\n"
              << "  Min seg chars:" << (params.min_seg_chars == 0 ? "off" : std::to_string(params.min_seg_chars) + " chars") << "\n"
              << "  Temperature:  " << params.gen.temperature << "\n"
              << "  Top-p:        " << params.gen.top_p << "\n"
              << "  Top-k:        " << params.gen.top_k << "\n"
              << "  RAS window:   " << params.gen.ras_window_size << " tokens\n"
              << "  RAS temp:     " << params.gen.ras_high_temp << "\n";

    // --- Charger le modèle ---
    s2::Pipeline pipeline;

#ifdef S2_TOKENIZER_EMBEDDED
    // Tokenizer embebido: usar los bytes del array generado por el workflow.
    // No se necesita tokenizer.json en disco.
    params.tokenizer_data      = reinterpret_cast<const char*>(tokenizer_json_data);
    params.tokenizer_data_size = static_cast<size_t>(tokenizer_json_size);
    std::cout << "  Tokenizer:    [embebido en el exe, "
              << tokenizer_json_size << " bytes]" << std::endl;
#else
    std::cout << "  Tokenizer:    " << params.tokenizer_path << std::endl;
#endif

    if (!pipeline.init(params)) {
        std::cerr << "Pipeline initialization failed." << std::endl;
        return 1;
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
                std::cout << "[Cleanup] Borrado temp: " << path_copy << "\n";
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
        info["endpoints"][4] = "/health";
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
        std::cout << "[WS] Cliente conectado: " << conn.get_remote_ip() << "\n";
    })
    .onclose([](crow::websocket::connection& conn, const std::string& reason, uint16_t) {
        std::cout << "[WS] Cliente desconectado: " << reason << "\n";
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
                std::cerr << "[WS] Error enviando segmento — cliente desconectado?\n";
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
              << "  WS   /ws/tts           (streaming por segmentos — latencia minima)\n\n";

    std::cout << "Servidor en puerto " << port << "...\n";
    app.port(port).multithreaded().run();
    return 0;
}
