#!/usr/bin/env python3
"""Static guards for Slow/Fast allocators, prefill lifetime, mmap release and KV-cache safety."""
from pathlib import Path
s = Path('src/s2_model.cpp').read_text(encoding='utf-8')
codec = Path('src/s2_codec.cpp').read_text(encoding='utf-8')
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
# Performance/lifetime invariants requested by the server design.  These are
# deliberately textual guards around code that already existed in the hosted-green
# audit12 baseline: keep Slow-AR and Fast-AR allocators separate, drop mapped
# weights from the host page cache after GPU upload, and use a temporary allocator
# for prefill so its large compute buffer does not persist into token generation.
required_model_tokens = {
    'separate Slow-AR gallocr': 'allocr_      = ggml_gallocr_new(',
    'separate Fast-AR gallocr': 'fast_allocr_ = ggml_gallocr_new(',
    'POSIX page-cache release': '::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);',
    'temporary prefill gallocr': 'ggml_gallocr_t prefill_allocr = ggml_gallocr_new(',
    'prefill allocator swap guard': 'GallocrSwapGuard allocator_guard(allocr_, prefill_allocr);',
}
for label, token in required_model_tokens.items():
    if token not in s:
        raise SystemExit(f'MODEL_KV_INVARIANTS_FAIL: missing {label}')
if s.count('ggml_gallocr_alloc_graph(fast_allocr_, gf)') < 1:
    raise SystemExit('MODEL_KV_INVARIANTS_FAIL: Fast-AR path is not using fast_allocr_')
if 'POSIX_FADV_DONTNEED' not in codec or 'posix_fadvise' not in codec:
    raise SystemExit('MODEL_KV_INVARIANTS_FAIL: codec GGUF page-cache release missing')
print('MODEL_MEMORY_INVARIANTS_PASS separate_allocators=1 fadvise_model=1 fadvise_codec=1 temporary_prefill=1')
