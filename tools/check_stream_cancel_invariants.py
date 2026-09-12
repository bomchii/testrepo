#!/usr/bin/env python3
from pathlib import Path
h=Path('include/s2_pipeline.h').read_text()
p=Path('src/s2_pipeline.cpp').read_text()
m=Path('src/main.cpp').read_text()

def need(c,msg):
    if not c: raise SystemExit('FAIL: '+msg)
need('using CancelCallback = std::function<bool()>;' in h, 'CancelCallback missing')
need('CancelCallback should_continue = {}' in h, 'public optional cancel probe missing')
need('if (should_continue && !should_continue())' in p, 'pipeline does not check cancellation')
need(p.count('should_continue && !should_continue()') >= 4, 'cancellation not checked at enough boundaries')
need('auto on_frame' in p and 'cb_ok = false;' in p, 'streaming frame callback cancel plumbing missing')
need('s2::CancelCallback should_continue = [alive]' in m, 'WebSocket alive signal not wired into pipeline')
need('synthesize_streaming(ws_params, cb, &segment_count, should_continue)' in m, 'WebSocket does not pass cancel callback')
print('STREAM_CANCEL_INVARIANTS_PASS frame_probe=1 ws_alive=1 fallback_segment_probe=1')
