#include "s2_pipeline.h"
#include <crow.h>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <thread>
#include <chrono>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
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
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if ((arg == "-threads" || arg == "--threads") && i + 1 < argc) {
            params.gen.n_threads = std::stoi(argv[++i]);
        } else if ((arg == "--max-tokens") && i + 1 < argc) {
            params.gen.max_new_tokens = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout <<
R"(s2.exe — Fish Speech TTS server (Windows + Vulkan)

USO BASICO (CPU, funciona en cualquier PC):
  s2.exe

USO CON GPU (RTX 3050 / cualquier GPU Vulkan):
  s2.exe -v 0 --codec-vulkan 0

PARA AUDIOS LARGOS CON POCA VRAM (4 GB):
  s2.exe -v 1 --codec-vulkan 1 --segment

OPCIONES:
  Modelos:
    -m,  --model <ruta>        Modelo principal GGUF
                               (default: model.gguf junto al exe)
         --model-codec <ruta>  Modelo codec separado GGUF (opcional)
    -t,  --tokenizer <ruta>    tokenizer.json
                               (default: embebido en el exe)

  GPU:
    -v,  --vulkan <N>          GPU Vulkan para el modelo principal
                                 -1 = CPU (default, funciona en todo)
                                  0 = primera GPU
                                  1 = segunda GPU (RTX en laptops con iGPU)
         --codec-vulkan <N>    GPU Vulkan para el codec (mismos valores)
                               Tip: si hay OOM, pon el codec en CPU (-1)

  Servidor:
    -p,  --port <N>            Puerto HTTP (default: 8080)

  Rendimiento:
         --threads <N>         Hilos CPU para operaciones en CPU (default: 4)
         --max-tokens <N>      Tokens máximos a generar por request (default: 1024)

  Memoria (opciones avanzadas, todas OFF por defecto):
         --segment             Divide el texto en oraciones antes de generar.
                               Limita el pico de VRAM al tamaño de la oración
                               más larga en vez de al texto completo.
                               Recomendado para textos largos con poca VRAM.
                               También se puede activar por request:
                                 { "text": "...", "segment": true }
         --codec-chunk <N>     Frames por chunk en el decode del codec.
                               0 = automático (recomendado).
                               Bajar si hay OOM en el codec; subir si hay
                               artefactos de audio en los cortes.

  Otros:
    -h,  --help                Muestra esta ayuda

ENDPOINTS HTTP:
  POST /v1/tts          (compatible Fish Audio)
  POST /v1/audio/speech (compatible OpenAI)
  POST /synthesize      (legacy)
  GET  /v1/models
  GET  /health

ENDPOINT WEBSOCKET (streaming — latencia minima):
  WS /ws/tts

  El cliente manda un JSON con el texto y recibe mensajes binarios
  con el audio PCM int16 por oración, sin esperar a que termine todo.
  Ideal para reproduccion en tiempo real.

  Ejemplo con Python:
    import websockets, asyncio, json
    async def tts():
        async with websockets.connect("ws://localhost:8080/ws/tts") as ws:
            await ws.send(json.dumps({"text": "Hola mundo", "segment": True}))
            while True:
                msg = await ws.recv()
                if isinstance(msg, str):  # {"done": true, ...}
                    break
                pcm_data = msg[2:]  # primeros 2 bytes son flags
                # reproducir pcm_data directamente (int16, mono, 44100Hz)

EJEMPLO HTTP:
  curl -X POST http://localhost:8080/v1/tts \
       -H "Content-Type: application/json" \
       -d '{"text": "Hola mundo", "segment": true}' \
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
              << "  Segmentacion: " << (params.segment_sentences ? "ON" : "OFF (usa --segment para activar)") << "\n"
              << "  Codec chunk:  " << (params.codec_chunk_frames == 0 ? "auto" : std::to_string(params.codec_chunk_frames) + " frames") << "\n";

    // --- Charger le modèle ---
    s2::Pipeline pipeline;

#ifdef S2_TOKENIZER_EMBEDDED
    // Tokenizer embebido: usar los bytes del array generado por el workflow.
    // No se necesita tokenizer.json en disco.
    params.tokenizer_data      = tokenizer_json_data;
    params.tokenizer_data_size = tokenizer_json_size;
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
        if (json.has("codec_chunk")) synth_params.codec_chunk_frames = json["codec_chunk"].i();

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
        if (json.has("codec_chunk"))      ws_params.codec_chunk_frames = json["codec_chunk"].i();

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
