#!/usr/bin/env python3
from pathlib import Path
p=Path('src/s2_pipeline.cpp').read_text()
def need(c,msg):
    if not c: raise SystemExit('FAIL: '+msg)
need('MAX_REFERENCE_AUDIO_SECONDS = 30' in p, '30s reference limit missing')
need('reference_audio_within_limit' in p, 'reference duration helper missing')
need(p.count('load_audio_limited(') >= 2 and 'load_audio_from_memory_limited(' in p, 'reference paths must enforce duration while decoding')
need(p.count('reference_audio_within_limit(ra, codec_.sample_rate())') >= 2,
     'limit must cover global and explicit references')
need('Reference audio exceeds the 30 second safety limit' in p, 'explicit reference error missing')
print('REFERENCE_LIMITS_PASS max_seconds=30 early_decode=1 explicit=1 global=1 inline=1')
