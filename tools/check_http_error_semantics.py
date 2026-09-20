#!/usr/bin/env python3
"""Static regression guard for request-validation vs execution error staging."""
from pathlib import Path

s = Path("src/main.cpp").read_text(encoding="utf-8")

def must(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit("HTTP_ERROR_SEMANTICS_FAIL: " + msg)

must(s.count("return handle_synthesis_request(req, ApiFlavor::") == 3,
     "all three synthesis HTTP routes must use the strict wrapper with an explicit API flavor")
helper = s[s.index("auto handle_synthesis_request"):s.index("// Route Fish Audio")]
must("load_json_strict(req.body)" in helper, "strict JSON normalization must live inside wrapper try/catch")
must("Invalid JSON request:" in helper, "wrapper must map strict JSON exceptions to client errors")

synth = s[s.index("auto do_synthesize"):s.index("auto handle_synthesis_request")]
must("bool request_validated = false;" in synth, "synthesis stage flag missing")
must("request_validated = true;" in synth, "synthesis stage transition missing")
must('if (request_validated)' in synth and 'Synthesis failed internally' in synth, "synthesis exceptions must keep validated/internal failures generic")
must('if (!mgr.exists(synth_params.voice_id))' in synth and
     'return crow::response(404, "Voice not found: " + synth_params.voice_id);' in synth,
     "missing saved voice must be 404 before pipeline execution")
must('synth_params.prompt_audio_path.empty()' in synth,
     "explicit reference audio must keep priority over saved voice preflight")

voice_start = s.index('// POST /v1/voices/<id>')
voice_end = s.index('// GET /v1/voices/<id>')
voice = s[voice_start:voice_end]
must("bool request_validated = false;" in voice, "voice stage flag missing")
must("request_validated = true;" in voice, "voice stage transition missing")
must('if (!request_validated)' in voice and 'Failed to create voice profile' in voice,
     "voice catches must preserve 4xx validation detail while sanitizing internal 5xx failures")
voice_impl = Path("src/s2_voice.cpp").read_text(encoding="utf-8")
must('!fs::exists(path) || !fs::is_regular_file(path)' in voice_impl,
     "voice DELETE must not remove a directory masquerading as <id>.s2voice")
must('non-loopback --host requires explicit --allow-remote' in s and
     '--allow-remote with a non-loopback --host requires S2_API_TOKEN' in s and
     'request_origin_allowed' in s and 'request_has_json_content_type' in s,
     "remote opt-in/auth/origin/content-type hardening missing")
must('.onaccept(' in s and 'enforce_http_security(req, false)' in s[s.index('CROW_WEBSOCKET_ROUTE'):],
     "WebSocket upgrade must enforce auth/origin policy")
must('std::async' not in s and 'batch_tasks->submit' in s and 'Batch worker queue is full' in s,
     "batch must use bounded global pool")
must('threads cannot exceed the server --threads limit' in s, "request CPU thread cap missing")
must('catch (const std::invalid_argument & e)' in s and
     'catch (const std::out_of_range & e)' in s and
     '[WS] Internal synthesis failure:' in s and
     'server error' in s,
     "WebSocket must expose request-validation details only for client errors and hide internal exception details")

print("HTTP_ERROR_SEMANTICS_PASS routes=3 staged_voice_errors=1 missing_voice_404=1 delete_regular_file=1 remote_optin_auth_origin=1 bounded_batch=1 thread_cap=1")
