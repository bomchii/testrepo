// Standalone faithful replica of the main.cpp argument parser.
// Tests every CLI flag, including edge cases that expose bugs:
//   - value-consuming flag given as the LAST arg (i+1 < argc fails silently)
//   - invalid numeric values (std::stoi/stof throw)
//   - unknown options
// Mirrors the exact parser structure from src/main.cpp.

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cassert>

struct GenParams {
    float temperature = 0.7f;
    float top_p = 0.7f;
    int top_k = 30;
    int min_tokens_before_end = 0;
    int ras_window_size = 10;
    float ras_high_temp = 1.0f;
    float ras_high_top_p = 1.0f;
    int n_threads = 4;
    int max_new_tokens = 1024;
};

struct Params {
    std::string model_path = "model.gguf";
    std::string codec_model_path = "codec.gguf";
    std::string tokenizer_path = "tokenizer.json";
    int vulkan_device = -1;
    int codec_vulkan_device = -1;
    bool segment_sentences = false;
    int codec_chunk_frames = 0;
    int codec_overlap_frames = 0;
    int min_seg_chars = 0;
    int max_tokens_per_segment = 0;
    std::string text;
    std::string prompt_audio_path;
    std::string prompt_text;
    std::string voice_id;
    bool save_voice = false;
    std::string voice_storage_dir = "voices";
    bool list_voices_flag = false;
    std::string output_path;
    bool trim_silence = false;
    int stream_decode_stride_frames = 0;
    GenParams gen;
};

// Returns: 0 = parsed OK (continue), 1 = error/exit, 2 = help/exit-0
int parse_args(int argc, char** argv, Params& params, int& port, bool& list_voices,
               std::string& err) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        try {
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
            } else if ((arg == "-pa" || arg == "--prompt-audio") && i + 1 < argc) {
                params.prompt_audio_path = argv[++i];
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
                return 2;
            } else {
                err = "Unknown option: " + arg;
                return 1;
            }
        } catch (const std::exception& e) {
            err = "Invalid value for " + arg + ": " + e.what();
            return 1;
        }
    }
    return 0;
}

// ---- Test framework ----
static int g_pass = 0, g_fail = 0;

void check(const std::string& name, bool cond) {
    if (cond) { g_pass++; }
    else { g_fail++; std::cout << "  FAIL: " << name << "\n"; }
}

template<class... A>
int run(std::vector<std::string> args, Params& p, int& port, bool& lv, std::string& err) {
    std::vector<char*> argv;
    argv.push_back((char*)"s2.exe");
    for (auto& a : args) argv.push_back((char*)a.c_str());
    return parse_args((int)argv.size(), argv.data(), p, port, lv, err);
}

int main() {
    std::cout << "=== CLI parser test harness ===\n";

    // 1. Every value-consuming flag parses correctly
    {
        Params p; int port = 8080; bool lv = false; std::string err;
        int rc = run({"-m","a.gguf","--model-codec","c.gguf","-t","tok.json",
                      "-v","1","--codec-vulkan","1","--codec-chunk","32",
                      "--codec-overlap","2","--min-seg-chars","60",
                      "--temperature","0.8","--top-p","0.85","--top-k","40",
                      "--min-end-tokens","5","--ras-window","12","--ras-temp","1.2",
                      "--ras-top-p","0.95","--max-seg-tokens","300","-p","9090",
                      "--threads","8","--max-tokens","2048","--text","hello",
                      "-pa","ref.wav","-pt","ref text","--voice","bob",
                      "--voice-dir","myvoices","-o","out.wav",
                      "--stream-decode-stride","16"}, p, port, lv, err);
        check("full valid set returns 0", rc == 0);
        check("model_path", p.model_path == "a.gguf");
        check("codec_model_path", p.codec_model_path == "c.gguf");
        check("tokenizer_path", p.tokenizer_path == "tok.json");
        check("vulkan_device", p.vulkan_device == 1);
        check("codec_vulkan_device", p.codec_vulkan_device == 1);
        check("codec_chunk_frames", p.codec_chunk_frames == 32);
        check("codec_overlap_frames", p.codec_overlap_frames == 2);
        check("min_seg_chars", p.min_seg_chars == 60);
        check("temperature", p.gen.temperature > 0.79f && p.gen.temperature < 0.81f);
        check("top_p", p.gen.top_p > 0.84f && p.gen.top_p < 0.86f);
        check("top_k", p.gen.top_k == 40);
        check("min_end_tokens", p.gen.min_tokens_before_end == 5);
        check("ras_window", p.gen.ras_window_size == 12);
        check("max_seg_tokens", p.max_tokens_per_segment == 300);
        check("port", port == 9090);
        check("threads", p.gen.n_threads == 8);
        check("max_tokens", p.gen.max_new_tokens == 2048);
        check("text", p.text == "hello");
        check("prompt_audio (-pa)", p.prompt_audio_path == "ref.wav");
        check("prompt_text (-pt)", p.prompt_text == "ref text");
        check("voice_id", p.voice_id == "bob");
        check("voice_dir", p.voice_storage_dir == "myvoices");
        check("output_path", p.output_path == "out.wav");
        check("stream_decode_stride", p.stream_decode_stride_frames == 16);
    }

    // 2. Boolean flags
    {
        Params p; int port = 8080; bool lv = false; std::string err;
        int rc = run({"--segment","--save-voice","--trim-silence"}, p, port, lv, err);
        check("bool flags rc 0", rc == 0);
        check("segment", p.segment_sentences);
        check("save_voice", p.save_voice);
        check("trim_silence on", p.trim_silence);
    }
    {
        Params p; int port = 8080; bool lv = false; std::string err;
        run({"--trim-silence","--no-trim-silence"}, p, port, lv, err);
        check("no-trim-silence overrides", !p.trim_silence);
    }
    {
        Params p; int port = 8080; bool lv = false; std::string err;
        int rc = run({"--list-voices"}, p, port, lv, err);
        check("list-voices sets flag", lv && rc == 0);
    }

    // 3. Help
    {
        Params p; int port = 8080; bool lv = false; std::string err;
        check("--help returns 2", run({"--help"}, p, port, lv, err) == 2);
        check("-h returns 2", run({"-h"}, p, port, lv, err) == 2);
    }

    // 4. Aliases match long forms
    {
        Params p1, p2; int port = 8080; bool lv = false; std::string err;
        run({"-m","x"}, p1, port, lv, err);
        run({"--model","x"}, p2, port, lv, err);
        check("-m == --model", p1.model_path == p2.model_path);

        Params p3, p4;
        run({"--temp","0.5"}, p3, port, lv, err);
        run({"--temperature","0.5"}, p4, port, lv, err);
        check("--temp == --temperature", p3.gen.temperature == p4.gen.temperature);
    }

    // 5. EDGE CASE: value-consuming flag as LAST arg (i+1 < argc is false)
    //    The parser falls through to the else branch -> "Unknown option".
    //    This is a real UX wart: "-m" alone reports "Unknown option: -m".
    {
        Params p; int port = 8080; bool lv = false; std::string err;
        int rc = run({"-m"}, p, port, lv, err);
        check("dangling -m returns error (1)", rc == 1);
        check("dangling -m error mentions -m", err.find("-m") != std::string::npos);
        std::cout << "  [info] dangling '-m' -> rc=" << rc << " err=\"" << err << "\"\n";
    }

    // 6. EDGE CASE: invalid numeric values now caught by try/catch
    {
        Params p; int port = 8080; bool lv = false; std::string err;
        int rc = run({"-v","notanumber"}, p, port, lv, err);
        check("invalid -v caught (rc 1)", rc == 1);
        std::cout << "  [info] '-v notanumber' -> rc=" << rc << " err=\"" << err << "\"\n";
    }
    {
        Params p; int port = 8080; bool lv = false; std::string err;
        int rc = run({"--temperature","abc"}, p, port, lv, err);
        check("invalid --temperature caught (rc 1)", rc == 1);
    }

    // 7. Unknown option
    {
        Params p; int port = 8080; bool lv = false; std::string err;
        int rc = run({"--frobnicate"}, p, port, lv, err);
        check("unknown option returns 1", rc == 1);
        check("unknown option named", err.find("--frobnicate") != std::string::npos);
    }

    // 8. Negative device index (CPU sentinel) is valid
    {
        Params p; int port = 8080; bool lv = false; std::string err;
        int rc = run({"-v","-1"}, p, port, lv, err);
        check("-v -1 (CPU) parses", rc == 0 && p.vulkan_device == -1);
    }

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
