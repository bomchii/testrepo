#!/usr/bin/env python3
"""Static regression guard for request-validation vs execution error staging."""
from pathlib import Path

s = Path("src/main.cpp").read_text(encoding="utf-8")

def must(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit("HTTP_ERROR_SEMANTICS_FAIL: " + msg)

must(s.count("return handle_synthesis_request(req);") == 3,
     "all three synthesis HTTP routes must use the strict wrapper")
helper = s[s.index("auto handle_synthesis_request"):s.index("// Route Fish Audio")]
must("load_json_strict(req.body)" in helper, "strict JSON normalization must live inside wrapper try/catch")
must("Invalid JSON request:" in helper, "wrapper must map strict JSON exceptions to client errors")

synth = s[s.index("auto do_synthesize"):s.index("auto handle_synthesis_request")]
must("bool request_validated = false;" in synth, "synthesis stage flag missing")
must("request_validated = true;" in synth, "synthesis stage transition missing")
must("request_validated ? 500 : 400" in synth, "synthesis exceptions must be phase-aware")
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
must(voice.count("request_validated ? 500 : 400") >= 3,
     "voice invalid_argument/std::exception/unknown catches must all be phase-aware")
print("HTTP_ERROR_SEMANTICS_PASS routes=3 staged_voice_errors=1 missing_voice_404=1")
