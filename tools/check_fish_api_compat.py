#!/usr/bin/env python3
"""Static regression guard for the Fish Speech compatibility subset we intentionally support."""
from pathlib import Path
s = Path("src/main.cpp").read_text(encoding="utf-8")

def must(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit("FISH_API_COMPAT_FAIL: " + msg)

synth = s[s.index("auto do_synthesize"):s.index("auto handle_synthesis_request")]
ws_start = s.index("auto json = load_json_strict(data);", s.index("CROW_WEBSOCKET_ROUTE"))
ws = s[ws_start:s.index("std::string validation_error;", ws_start)]
for name, block in (("HTTP", synth), ("WS", ws)):
    must('json.has("max_new_tokens")' in block, f"{name} missing max_new_tokens alias")
    must('json.has("reference_id")' in block, f"{name} missing reference_id alias")
    must('conflicting max_tokens and max_new_tokens' in block, f"{name} must reject token-limit conflicts")
    must('conflicting voice and reference_id' in block, f"{name} must reject voice-id conflicts")
    must('checked_json_fish_seed(json["seed"])' in block, f"{name} must accept Fish null seed semantics")
    must('checked_json_fish_max_new_tokens(json["max_new_tokens"])' in block,
         f"{name} must accept Fish max_new_tokens=0 semantics")
    must('has_reference_id' in block, f"{name} must treat Fish reference_id=null as absent")
    must('conflicting text and input' in block, f"{name} must reject contradictory text/input aliases")

must('conflicting format and response_format' in synth,
     "HTTP must reject contradictory output-format aliases")
must('validate_fish_json_subset(json, true);' in synth,
     "HTTP must validate unsupported Fish fields before synthesis")
for field in (
    "early_stop_threshold", "normalize", "sample_rate", "mp3_bitrate",
    "opus_bitrate", "use_memory_cache",
):
    must(f'"{field}"' in s, f"known unsupported Fish field {field} must fail explicitly")
for name, block in (("HTTP", synth), ("WS", ws)):
    for field in ("chunk_length", "min_chunk_length", "condition_on_previous_chunks", "prosody", "latency"):
        must(f'json.has("{field}")' in block, f"{name} missing supported Fish field {field}")
    must('prosody.volume must be between -20 and 20 dB' in block, f"{name} must validate Fish volume range")
    must('prosody.speed other than 1.0 is not implemented' in block, f"{name} must not fake speed control")
    must('latency modes balanced/low are not implemented; use normal' in block, f"{name} must not fake latency modes")
must("must not be ignored" in s, "unsupported Fish semantic fields need an explicit subset error")
must('validate_fish_json_subset(json, false);' in s,
     "WebSocket must validate Fish subset fields")
must("refs.size() != 0" in s and "inline Fish 'references' are not supported" in s,
     "non-empty inline Fish references must fail explicitly, not be ignored")
must('http_buffered && streaming.b()' in s and 'use /ws/tts for streaming audio' in s,
     "HTTP streaming=true must fail explicitly, not be ignored")
must('!http_buffered && !streaming.b()' in s and 'streaming=false is incompatible with /ws/tts' in s,
     "WebSocket streaming=false must fail explicitly, not be ignored")
must('http_buffered && json.has("stream_stride")' in s and 'stream_stride is WebSocket-only' in s,
     "HTTP stream_stride must fail explicitly instead of being ignored")
must('!http_buffered && (json.has("format") || json.has("response_format"))' in s and
     'format/response_format are HTTP-only' in s,
     "WebSocket format/response_format must fail explicitly instead of being ignored")
must('value.t() == crow::json::type::Null' in s and 'return 0;' in s,
     "Fish seed=null must map to random seed")
must('return v == 0 ? 32768 : v;' in s,
     "Fish max_new_tokens=0 must map to a finite context-clamped ceiling")
must('CROW_ROUTE(app, "/v1/health")' in s, "missing documented Fish Speech /v1/health route")
must('status["status"] = "ok";' in s, "/v1/health response must use Fish Speech status shape")
print("FISH_API_COMPAT_PASS aliases=6 supported_longform=5 explicit_unsupported=6 health=1")
