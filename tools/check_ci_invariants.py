#!/usr/bin/env python3
"""Static invariants for the four-backend GitHub Actions release workflow."""
from pathlib import Path
import re, sys
ROOT=Path(__file__).resolve().parents[1]
WF=ROOT/'.github/workflows/build-all-backends.yml'
text=WF.read_text(encoding='utf-8')
errors=[]
def req(token, why):
    if token not in text: errors.append(f'missing {why}: {token!r}')
def forbid(token, why):
    if token in text: errors.append(f'forbidden {why}: {token!r}')

if (ROOT/'.github/workflows/build-windows-vulkan.yml').exists(): errors.append('obsolete build-windows-vulkan.yml exists')
for job in ['build','build-cuda','build-cpu','build-metal','release']:
    if not re.search(rf'^  {re.escape(job)}:\s*$',text,re.M): errors.append(f'missing job {job}')
jobs_region=text[text.index('\njobs:')+6:]
job_names=re.findall(r'^  ([A-Za-z0-9_-]+):\s*$',jobs_region,re.M)
if job_names != ['build','build-cuda','build-cpu','build-metal','release']: errors.append(f'unexpected job order/names: {job_names}')
if text.count('runs-on: windows-2025-vs2026') != 4: errors.append('Windows runner must be used by Vulkan/CUDA/CPU/release jobs')
req('runs-on: macos-15','Metal runner')
req('actions/checkout@v7','checkout major')
req('actions/cache@v6','cache major')
req('actions/upload-artifact@v7','upload-artifact major')
req('actions/download-artifact@v8','download-artifact major')
req('softprops/action-gh-release@v3.0.3','release action pin')
req('Jimver/cuda-toolkit@v0.2.36','CUDA toolkit action pin')
req("cuda: '13.2.0'",'CUDA 13.2')
req('Visual Studio 18 2026','VS2026 generator')
req('-DGGML_METAL_EMBED_LIBRARY=ON','embedded Metal library')
req('needs: [build, build-cuda, build-cpu, build-metal]','release dependencies')
req('contents: write','release write permission')
req('dumpbin','PE import verification')
req('nvcuda.dll must never be bundled','driver DLL exclusion')
req('tools/cuda_bundle.py','native CUDA bundle creation')
forbid('--allow-unsupported-compiler','unsupported CUDA compiler bypass')
forbid('CMAKE_CUDA_ARCHITECTURES=86','fixed sm_86 architecture')
forbid('sm_86','fixed sm_86 architecture')

def source_list(path):
    s=(ROOT/path).read_text(encoding='utf-8')
    m=re.search(r'set\(S2_SOURCES\s*(.*?)\n\)', s, re.S)
    if not m:
        errors.append(f'{path}: S2_SOURCES block missing')
        return []
    return [line.strip() for line in m.group(1).splitlines() if line.strip().startswith('src/')]

root_sources=source_list('CMakeLists.txt')
for generated in ['patch-cmake.ps1','patch-cmake-cuda.ps1']:
    generated_sources=source_list(generated)
    if generated_sources != root_sources:
        errors.append(f'{generated}: S2_SOURCES diverges from root: {generated_sources!r} != {root_sources!r}')
if len(root_sources) != len(set(root_sources)):
    errors.append('CMakeLists.txt: duplicate entries in S2_SOURCES')
if 'src/main.cpp' not in root_sources or len(root_sources) != 12:
    errors.append(f'CMakeLists.txt: unexpected canonical S2_SOURCES set ({len(root_sources)} entries)')

for name in ['CMakeLists.txt','patch-cmake.ps1','patch-cmake-cuda.ps1']:
    s=(ROOT/name).read_text(encoding='utf-8')
    req_local=[]
    if 'cmake_minimum_required(VERSION 3.15)' not in s: req_local.append('CMake >=3.15')
    try:
        p_project=s.index('project(s2cpp')
        p_policy=s.index('cmake_policy(SET CMP0091 NEW)')
        p_runtime=s.index('CMAKE_MSVC_RUNTIME_LIBRARY')
        if not (p_policy < p_project and p_runtime < p_project): req_local.append('CMP0091 and /MT before project()')
    except ValueError: req_local.append('project/CMP0091/runtime marker missing')
    if req_local: errors.append(f'{name}: '+', '.join(req_local))

# Prefer CUDA Toolkit redistributables before falling back to System32.
# A runner may contain vendor DLLs that are not present on end-user systems.
try:
    p_container = text.index('- name: Build single-file CUDA container')
    p_cuda = text.index('$cudaDll = Join-Path $cudaBin $dep', p_container)
    p_sys = text.index('if (Test-Path (Join-Path $sys32 $dep)) { continue }', p_container)
    if p_cuda > p_sys:
        errors.append('CUDA dependency closure checks System32 before CUDA_PATH\\bin')
except ValueError:
    errors.append('CUDA dependency-resolution markers missing')

if errors:
    print('CI_INVARIANTS_FAIL', file=sys.stderr)
    for e in errors: print(' - '+e,file=sys.stderr)
    raise SystemExit(1)
print('CI_INVARIANTS_PASS jobs=5 backends=4')
