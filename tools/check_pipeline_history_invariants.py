#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'src' / 's2_pipeline.cpp'
s = p.read_text(encoding='utf-8')

needle_capture = 'const bool capture_history = history_limit != 0 && !capture_anchor && !is_last;'
needle_anchor = 'const bool capture_anchor = auto_voice_anchor_pending && !is_last;'
needle_stride = 'if (cb_ok && !is_last_seg && (history_limit != 0 || auto_voice_anchor_pending)) {'

# Four non-stride segmented consumers: synthesize, synthesize_to_buffer,
# synthesize_to_file, and full-segment streaming.
if s.count(needle_capture) != 4:
    raise SystemExit(f'expected 4 final-segment history guards, got {s.count(needle_capture)}')
if s.count(needle_anchor) != 4:
    raise SystemExit(f'expected 4 final-segment anchor guards, got {s.count(needle_anchor)}')
if s.count(needle_stride) != 1:
    raise SystemExit('stride streaming final-segment VQ guard missing or duplicated')

# The full streaming callback must happen after the final-segment capture guard,
# so no final-segment VQ state is manufactured after terminal delivery.
stride_pos = s.index(needle_stride)
terminal_pos = s.index('if (cb_ok && is_last_seg && !terminal_sent)')
if not terminal_pos < stride_pos:
    raise SystemExit('stride history guard must remain after terminal handling and exclude final segment')

print('PIPELINE_HISTORY_INVARIANTS_PASS guarded_non_stride=4 guarded_stride=1')
