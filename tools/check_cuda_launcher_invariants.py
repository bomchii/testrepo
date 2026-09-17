#!/usr/bin/env python3
from pathlib import Path
s=Path('tools/cuda_launcher.cpp').read_text(encoding='utf-8')
checks={
 'exact_cache_allowlist': 'std::set<std::wstring> allowed={L".complete",L".active.lock"};' in s,
 'reject_reparse_files': 'FILE_ATTRIBUTE_REPARSE_POINT' in s and 'plain_file' in s,
 'reject_reparse_cache_dir': 'plain_directory(cache)' in s,
 'manifest_hash_check': 'sha256_file(p)!=x.sha' in s,
 'active_marker_before_unlock': 'Handle active=active_marker(cache);unlock_mutex(mutex);' in s,
 'inherited_handle_list': 'PROC_THREAD_ATTRIBUTE_HANDLE_LIST' in s,
 'driver_not_payload_rule': 's2-cuda-core.exe' in s,
 'amd_runtime_branch': 'S2_RUNTIME_AMD' in s and 's2-amd-core.exe' in s and 'amd-' in s,
 'recursive_cache_validation': 'recursive_directory_iterator' in s,
 'nested_payload_extraction': 'parent_path()' in s and 'create_directories' in s,
 'bounded_large_manifest': 'kMaxEntries = 16384' in s,
}
failed=[k for k,v in checks.items() if not v]
if failed: raise SystemExit('CUDA_LAUNCHER_INVARIANTS_FAIL: '+','.join(failed))
print('CUDA_LAUNCHER_INVARIANTS_PASS checks='+str(len(checks)))
