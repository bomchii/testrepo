#!/usr/bin/env python3
"""Static guard for failure-atomic SlowARModel KV-cache initialization."""
from pathlib import Path
s = Path('src/s2_model.cpp').read_text(encoding='utf-8')
start = s.index('bool SlowARModel::init_kv_cache(int32_t max_seq_len)')
end = s.index('// ---------------------------------------------------------------------------\n// reset()', start)
f = s[start:end]
assign = 'max_seq_len_ = max_seq_len;'
alloc = 'kv_buf_ = ggml_backend_alloc_ctx_tensors(ctx_kv_, backend_);'
if f.count(assign) != 1:
    raise SystemExit(f'KV_INVARIANTS_FAIL: expected one capacity publish, got {f.count(assign)}')
if f.index(assign) < f.index(alloc):
    raise SystemExit('KV_INVARIANTS_FAIL: capacity is published before backend allocation')
if 'if (!kv_buf_)' not in f or 'free_kv_cache();' not in f[f.index('if (!kv_buf_)'):]:
    raise SystemExit('KV_INVARIANTS_FAIL: backend-allocation failure does not roll back cache state')
if 'if (dim <= 0) return false;' not in f:
    raise SystemExit('KV_INVARIANTS_FAIL: invalid zero/negative embedding dimension accepted')
print('MODEL_KV_INVARIANTS_PASS publish_after_alloc=1 rollback=1')
