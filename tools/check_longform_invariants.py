#!/usr/bin/env python3
from pathlib import Path
import re, sys

root=Path(__file__).resolve().parents[1]
pipeline=(root/'src/s2_pipeline.cpp').read_text(encoding='utf-8')
text=(root/'src/s2_text.cpp').read_text(encoding='utf-8')
main=(root/'src/main.cpp').read_text(encoding='utf-8')

def need(cond,msg):
    if not cond:
        print('LONGFORM_INVARIANT_FAIL:',msg,file=sys.stderr); sys.exit(1)

need('split_text_chunks(params.text, params.chunk_length, params.min_chunk_length)' in pipeline,
     'pipeline must dispatch Fish long-form through Unicode-aware chunker')
need('request_is_chunked(params) && !params.condition_on_previous_chunks) return 0;' in pipeline,
     'condition_on_previous_chunks=false must disable automatic history')
need('if (params.chunk_length > 0 && params.condition_on_previous_chunks)' in pipeline and 'return 1;' in pipeline,
     'long-form rolling history must stay bounded to one generated turn')
need('build_wav_header(hdr, 0, wav_sample_rate)' in pipeline and 'bool finalize_wav()' in pipeline,
     'incremental WAV must start provisional and finalize header')
need(pipeline.count('TempWavFile tmp;') >= 3,
     'disk-backed non-streaming routes must use incremental WAV writer')
need('PCM + WAV duplicates' in pipeline,
     'buffered route must document single-file peak disk behavior')
need('is_grapheme_linker' in text and 'is_regional_indicator' in text,
     'hard chunking must protect Indic linkers and RI flag pairs')
for field in ('chunk_length','min_chunk_length','condition_on_previous_chunks','prosody'):
    need(f'json.has("{field}")' in main, f'HTTP/WS parser missing {field}')
need('prosody.speed other than 1.0 is not implemented without pitch-preserving time stretch' in main,
     'unsupported speed must fail explicitly rather than pitch-shifting via resampling')
print('LONGFORM_INVARIANTS_PASS disk_backed=1 rolling_history=1 unicode_hard_split=1 fish_fields=4')
