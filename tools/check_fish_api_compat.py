#!/usr/bin/env python3
"""Static regression guard for the Fish Speech compatibility surface implemented by s2.cpp."""
from pathlib import Path
s = Path("src/main.cpp").read_text(encoding="utf-8")
pipeline = Path("src/s2_pipeline.cpp").read_text(encoding="utf-8")
gen = Path("src/s2_generate.cpp").read_text(encoding="utf-8")

def must(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit("FISH_API_COMPAT_FAIL: " + msg)

synth = s[s.index("auto do_synthesize"):s.index("auto handle_synthesis_request")]
ws_start = s.index("auto json = load_json_strict(data);", s.index("auto process_ws_message"))
ws = s[ws_start:s.index("std::string validation_error;", ws_start)]
for name, block in (("HTTP", synth), ("WS", ws)):
    must('json.has("max_new_tokens")' in block, f"{name} missing max_new_tokens alias")
    must('json.has("reference_id")' in block, f"{name} missing reference_id alias")
    must('conflicting max_tokens and max_new_tokens' in block, f"{name} must reject token-limit conflicts")
    must('conflicting voice and reference_id' in block, f"{name} must reject voice-id conflicts")
    must('checked_json_fish_seed(json["seed"])' in block, f"{name} must accept Fish null seed semantics")
    must('checked_json_fish_max_new_tokens(json["max_new_tokens"])' in block, f"{name} must accept Fish max_new_tokens=0 semantics")
    must('has_reference_id' in block, f"{name} must treat Fish reference_id=null as absent")
    must('conflicting text and input' in block, f"{name} must reject contradictory text/input aliases")
    must('apply_fish_fields(json,' in block, f"{name} must apply implemented Fish semantic fields")

must('conflicting format and response_format' in synth, "HTTP must reject contradictory output-format aliases")
must('format != "wav" && format != "pcm" && format != "rf64" && format != "mp3" && format != "opus"' in synth,
     "HTTP must support the five documented audio formats")
must('encode_with_ffmpeg' in synth and 'audio/mpeg' in synth and 'audio/ogg; codecs=opus' in synth,
     "MP3/Opus must be real encoded outputs, not renamed WAV")
must('mp3_bitrate must be 64, 128, or 192 kbps' in synth and
     'opus_bitrate must be -1000 (auto), 24, 32, 48, or 64 kbps' in synth,
     "encoded formats need Fish-compatible bitrate validation")
must('synth_params.output_rf64 = format == "rf64"' in synth and 'X-Wave-Container' in synth,
     "RF64 selection/response must be explicit")
must('parse_inline_references' in s and 'base64_decode' in s and 'supports at most 8 entries' in s,
     "inline Fish references must be decoded and bounded")
for token in ('early_stop_threshold', 'normalize_text', 'output_sample_rate', 'reference_memory_cache',
              'prosody_speed', 'normalize_loudness'):
    must(token in s or token in pipeline or token in gen, f"missing Fish semantic implementation {token}")
must('latency == "low" ? 2 : (latency == "balanced" ? 4 : 8)' in s,
     "normal/balanced/low latency modes must map to WebSocket decode cadence")
must('audio_time_stretch' in pipeline, "prosody.speed must use pitch-preserving time stretch")
must('threshold_wants_eos' not in gen and 'early_stop_threshold != 1.0f' in gen,
     "legacy early_stop_threshold must reject effectful fractional values instead of inventing EOS-probability semantics")
must('http_buffered && streaming.b()' in s and 'use /ws/tts for streaming audio' in s,
     "HTTP streaming=true must fail explicitly")
must('!http_buffered && !streaming.b()' in s and 'streaming=false is incompatible with /ws/tts' in s,
     "WebSocket streaming=false must fail explicitly")
must('stream_stride is WebSocket-only' in s, "HTTP stream_stride must fail explicitly")
must('encoded-audio bitrate fields are HTTP-only' in s, "WS encoded-format fields must fail explicitly")
must('value.t() == crow::json::type::Null' in s and 'return 0;' in s, "Fish seed=null must map to random seed")
must('return v == 0 ? 32768 : v;' in s, "Fish max_new_tokens=0 must remain bounded")
must('CROW_ROUTE(app, "/v1/tts/batch")' in s and 'audio_base64' in s,
     "batch synthesis route/result missing")
must('CROW_ROUTE(app, "/v1/health")' in s and 'status["status"] = "ok";' in s,
     "Fish health endpoint/shape missing")

must('synth_params.normalize_text = true;' in synth and 'synth_params.reference_memory_cache = false;' in synth,
     "Fish HTTP defaults must be normalize=true and memory cache off")
must('latency=balanced/low is supported by the local low-latency WebSocket path' in s,
     "buffered HTTP latency modes must not be silently ignored")
must('std::string format = "wav";' in synth and
     'synth_params.output_rf64 ? "rf64" : "wav"' not in synth,
     "HTTP default output must remain WAV regardless of one-shot --rf64")
must('std::string format = "wav";' in s[s.index('CROW_ROUTE(app, "/v1/tts/batch")'):],
     "batch items must default to WAV regardless of one-shot --rf64")
must('params.output_sample_rate > 0' in s[s.index('CROW_ROUTE(app, "/v1/tts/batch")'):] and
     '? params.output_sample_rate : pipeline.sample_rate()' in s[s.index('CROW_ROUTE(app, "/v1/tts/batch")'):],
     "batch sample_rate metadata must honor the server output-sample-rate default")
must('process_ws_message' in s and 'ws_tasks->submit' in s and 'state->busy.compare_exchange_strong' in s,
     "WebSocket synthesis must be offloaded from Crow I/O with one active request per connection")
must('const bool will_start_server = params.output_path.empty() && !params.save_voice;' in s and
     'if (will_start_server) {' in s and
     'if (!resolve_bind_host(bind_host, resolved_bind_host))' in s,
     "server-only host/port/workers/ffmpeg validation must not block one-shot/save-only modes")
must('json["model"].s() != "s2-pro-local"' in synth and
     'OpenAI instructions is not implemented' in synth and
     'OpenAI stream_format is not implemented' in synth,
     "OpenAI compatibility route must validate model and reject known unsupported fields")
must('is_mono_ogg_opus(payload)' in s and 'OpusHead' in s and 'channels == 1u' in s,
     "Opus output validation must verify a mono Opus identification header, not only OggS")

print("FISH_API_COMPAT_PASS aliases=6 semantic_fields=7 formats=5 batch=1 health=1 ws_async=1 openai_subset=1 opus_header=1")
