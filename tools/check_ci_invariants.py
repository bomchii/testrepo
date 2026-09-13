#!/usr/bin/env python3
"""Static invariants for the Windows/Linux/macOS backend GitHub Actions release workflow."""
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
expected=['build','build-cuda','build-cpu','build-linux-cpu','build-linux-vulkan','build-linux-cuda','build-metal','release']
for job in expected:
    if not re.search(rf'^  {re.escape(job)}:\s*$',text,re.M): errors.append(f'missing job {job}')
jobs_region=text[text.index('\njobs:')+6:]
job_names=re.findall(r'^  ([A-Za-z0-9_-]+):\s*$',jobs_region,re.M)
if job_names != expected: errors.append(f'unexpected job order/names: {job_names}')
if text.count('runs-on: windows-2025-vs2026') != 4: errors.append('Windows runner must be used by Vulkan/CUDA/CPU/release jobs')
if text.count('runs-on: ubuntu-24.04') != 3: errors.append('three Linux build jobs must run on explicit ubuntu-24.04 hosts')
req('runs-on: macos-15','Metal runner')
for token,label in [
    ('actions/checkout@v7','checkout major'),('actions/cache@v6','cache major'),
    ('actions/upload-artifact@v7','upload-artifact major'),('actions/download-artifact@v8','download-artifact major'),
    ('softprops/action-gh-release@v3.0.3','release action pin'),('Jimver/cuda-toolkit@v0.2.36','CUDA toolkit action pin')]: req(token,label)
req("FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: 'true'",'Node 24 Actions runtime')
req("VULKAN_VERSION: '1.4.357.0'",'pinned Vulkan SDK version')
req('ae0fef0ee67eec897e401321b99b6dd7cfbdc155','Crow 1.3.4 immutable security-release commit')
req('755bd7f85a4b269c67ae0ea254907c078d408cce8e1a352ad2ed664d233780e8','Asio 1.30.2 source archive checksum')
forbid('crow1.3.3','stale Crow 1.3.3 cache key')
forbid('refs/tags/v1.3.3','vulnerable Crow 1.3.3 source fetch')
req("VULKAN_SHA256: '81f474711e9042f4cd22b31b2f7a8870db2e428b21586fb43dd80150be97310d'",'Vulkan SDK checksum')
req('sdk.lunarg.com/sdk/download/$env:VULKAN_VERSION/windows/vulkansdk-windows-X64-$env:VULKAN_VERSION.exe','official LunarG Vulkan download')
forbid('humbletim/install-vulkan-sdk','legacy Vulkan installer action')
req("cuda: '13.2.0'",'Windows CUDA 13.2')
req('CudaToolkitDir=$cudaPath','explicit CUDA Toolkit path for MSBuild')
req("extras\\visual_studio_integration\\MSBuildExtensions",'CUDA Visual Studio integration source')
req('-T "cuda=$env:CUDA_PATH"','explicit CUDA Toolkit path for Visual Studio generator')
req('Probe CMake CUDA compiler before project build','standalone CUDA compiler probe')
if text.count('if-no-files-found: error') != 7: errors.append('all seven backend uploads must fail if artifact is missing')
req('Vulkan artifact must contain exactly one s2-vulkan.exe','exact Windows Vulkan artifact manifest')
req('CUDA artifact must contain exactly one s2-cuda.exe','exact Windows CUDA artifact manifest')
req('CPU artifact must contain exactly one s2-cpu.exe','exact Windows CPU artifact manifest')
req("$vulkan     = 'artifacts\\vulkan\\s2-vulkan.exe'",'release consumes exact Windows Vulkan executable')
req("$cuda      = 'artifacts\\cuda\\s2-cuda.exe'",'release consumes exact Windows CUDA executable')
req("$cpu       = 'artifacts\\cpu\\s2-cpu.exe'",'release consumes exact Windows CPU executable')
req('Visual Studio 18 2026','VS2026 generator')
req('-DGGML_METAL_EMBED_LIBRARY=ON','embedded Metal library')
req('macOS arm64 - Clang - Metal','Metal job marker')
req('needs: [build, build-cuda, build-cpu, build-linux-cpu, build-linux-vulkan, build-linux-cuda, build-metal]','release dependencies')
req('contents: write','release write permission')
req("if ($tag -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$')",'release tag filename safety')
req('dumpbin','PE import verification')
req('nvcuda.dll must never be bundled','driver DLL exclusion')
req('tools/cuda_bundle.py','native Windows CUDA bundle creation')
forbid('--allow-unsupported-compiler','unsupported CUDA compiler bypass')
forbid('CMAKE_CUDA_ARCHITECTURES=86','fixed sm_86 architecture')
forbid('sm_86','fixed sm_86 architecture')
if text.count('-DGGML_NATIVE=OFF') != 4: errors.append('Windows CPU/Vulkan/CUDA plus macOS Metal release configure steps must force GGML_NATIVE=OFF')
for patcher_name in ['patch-cmake.ps1','patch-cmake-cuda.ps1']:
    patcher=(ROOT/patcher_name).read_text(encoding='utf-8')
    for token,label in [
        ('ae0fef0ee67eec897e401321b99b6dd7cfbdc155','Crow 1.3.4 commit pin'),
        ('755bd7f85a4b269c67ae0ea254907c078d408cce8e1a352ad2ed664d233780e8','Asio checksum'),
        ('New-Item -ItemType Directory -Force -Path $asioExtracted','Asio extraction directory creation'),
        ('Assert-Sha256 -Path $asioTar','Asio checksum verification')]:
        if token not in patcher: errors.append(f'{patcher_name} missing {label}: {token!r}')


# Linux portability/release architecture.
for token,label in [
    ('quay.io/pypa/manylinux2014_x86_64','CPU/Vulkan glibc 2.17 build image'),
    ('nvidia/cuda:13.2.0-devel-rockylinux8','CUDA 13.2 Rocky 8 build image'),
    ('tools/ci/build-linux-portable.sh cpu','Linux CPU build script'),
    ('tools/ci/build-linux-portable.sh vulkan','Linux Vulkan build script'),
    ('tools/ci/build-linux-portable.sh cuda','Linux CUDA build script'),
    ('s2-linux-x86_64-cpu.tar.gz','Linux CPU archive'),
    ('s2-linux-x86_64-vulkan.tar.gz','Linux Vulkan archive'),
    ('s2-linux-x86_64-cuda.tar.gz','Linux CUDA archive'),
    ("$linuxCpu    = 'artifacts\\linux-cpu\\s2-linux-x86_64-cpu.tar.gz'",'release consumes Linux CPU archive'),
    ("$linuxVulkan = 'artifacts\\linux-vulkan\\s2-linux-x86_64-vulkan.tar.gz'",'release consumes Linux Vulkan archive'),
    ("$linuxCuda   = 'artifacts\\linux-cuda\\s2-linux-x86_64-cuda.tar.gz'",'release consumes Linux CUDA archive'),
]: req(token,label)
forbid('ubuntu-latest','moving Ubuntu alias in release workflow')
forbid('yum -y install pkgconfig','CentOS 7/EOL runtime package install')
if text.count('docker image inspect "$image"') != 3: errors.append('all three Linux jobs must log the pulled image digest')
req('Keep JavaScript Actions on the modern Ubuntu host','Node 24 stays outside old-glibc containers')
req('tools/ci/reclaim-linux-runner-space.sh','Linux runner disk cleanup')
if text.count('bash tools/ci/reclaim-linux-runner-space.sh') != 3: errors.append('all three Linux jobs must reclaim runner disk before Docker builds')
if text.count('quay.io/pypa/manylinux2014_x86_64') != 2: errors.append('CPU and Vulkan must each select the manylinux2014 x86_64 baseline')
if '| head -n1' in text: errors.append('Linux workflow diagnostics must avoid pipe-to-head under pipefail')
if re.search(r'\|\s*grep\s+-[A-Za-z]*q', text): errors.append('workflow uses producer | grep -q under pipefail; capture producer output first')

build_script=(ROOT/'tools/ci/build-linux-portable.sh').read_text(encoding='utf-8')
prep_script=(ROOT/'tools/ci/prepare-linux-deps.sh').read_text(encoding='utf-8')
cleanup_script=(ROOT/'tools/ci/reclaim-linux-runner-space.sh').read_text(encoding='utf-8')
for license_name in ['LICENSE-GPL-3.0.txt','LICENSE-GCC-Runtime-Library-Exception-3.1.txt']:
    if not (ROOT/'tools/ci'/license_name).is_file(): errors.append(f'missing Linux runtime license source: {license_name}')
for token in ['/usr/local/lib/android','/usr/local/.ghcup','/usr/share/dotnet','docker system prune -af','df -h /']:
    if token not in cleanup_script: errors.append(f'Linux runner cleanup missing expected safe cleanup/diagnostic: {token!r}')
if '/opt/hostedtoolcache' in cleanup_script and 'Do not remove' not in cleanup_script:
    errors.append('Linux runner cleanup must not remove /opt/hostedtoolcache')
if 'rm -rf -- /opt/hostedtoolcache' in cleanup_script or 'rm -rf /opt/hostedtoolcache' in cleanup_script:
    errors.append('Linux runner cleanup deletes /opt/hostedtoolcache')
for token,label in [
    ('ceiling=2.17','CPU/Vulkan GLIBC ceiling'),('[[ "$backend" == "cuda" ]] && ceiling=2.28','CUDA GLIBC ceiling'),
    ('-DGGML_STATIC=ON','static GGML/CUDA Toolkit release linkage'),
    ('libvulkan.so.1','bundled Vulkan loader'),('libgomp.so.1','bundled OpenMP runtime'),
    ("grep -Eq '^(libcudart|libcublas|libcublasLt|libnvJitLink)\\.so'",'dynamic CUDA Toolkit rejection'),
    ('-DGGML_NATIVE=OFF','host-independent CPU code generation'),('-DGGML_AVX512=OFF','AVX512 disabled for portability'),
    ('cmake==$want','pinned CMake 4.4.3 bootstrap'),('CMAKE_PREFIX_PATH="$vkprefix"','Vulkan-Headers package discovery'),
    ('-DLOADER_CODEGEN=OFF','Vulkan loader codegen disabled for pinned generated sources'),('-DBUILD_SHARED_LIBS=ON','Vulkan Loader shared-library build explicit'),
    ('-DSYSCONFDIR=/etc','portable Vulkan system config search path'),('-DFALLBACK_CONFIG_DIRS=/etc/xdg','portable Vulkan XDG config fallback'),('-DFALLBACK_DATA_DIRS=/usr/local/share:/usr/share','portable Vulkan data fallback'),
    ('CMAKE_BUILD_WITH_INSTALL_RPATH=ON','closed build/install RPATH'),('runtime_path','exact sidecar loader search path check'),('libcuda\\.so','driver-provided CUDA dependency check'),
    ('-DBUILD_WSI_XCB_SUPPORT=OFF','Vulkan XCB WSI disabled'),
    ('-DBUILD_WSI_XLIB_SUPPORT=OFF','Vulkan Xlib WSI disabled'),
    ('-DBUILD_WSI_XLIB_XRANDR_SUPPORT=OFF','Vulkan XRandR WSI disabled for pinned 1.4.357 loader'),
    ('-DBUILD_WSI_WAYLAND_SUPPORT=OFF','Vulkan Wayland WSI disabled'),
    ('-DBUILD_WSI_DIRECTFB_SUPPORT=OFF','Vulkan DirectFB WSI disabled'),
    ('check_dynamic_deps','sidecar dependency audit'),('Advanced Micro Devices X86-64','x86_64 ELF audit'),
    ('LICENSE-GPL-3.0.txt','GPLv3 runtime license in Linux artifacts'),
    ('LICENSE-GCC-Runtime-Library-Exception-3.1.txt','GCC runtime exception in Linux artifacts'),
    ('LICENSE-Vulkan-Loader.txt','Vulkan Loader license in Vulkan artifact'),
    ('LICENSE-Crow-BSD-3-Clause.txt','Crow BSD license in Linux artifacts'),
    ('LICENSE-Asio-Boost-1.0.txt','Asio Boost license in Linux artifacts'),
    ('THIRD_PARTY_NOTICES.txt','third-party notices in Linux artifacts'),
    ('gzip -n -9','deterministic gzip packaging'),('unexpected dynamic dependency','closed shared-library allowlist')]:
    if token not in build_script: errors.append(f'Linux build script missing {label}: {token!r}')
if '--sort=name' in build_script: errors.append('Linux build script uses GNU tar --sort=name, unavailable on CentOS/RHEL 7 tar 1.26')
if 'need pkg-config' in build_script: errors.append('compute-only Vulkan build must not require pkg-config after the pinned Loader conditional patch')
if '| head -n1' in build_script: errors.append('Linux build script uses a pipe-to-head selector under pipefail; use find -print -quit instead')
if re.search(r'\|\s*grep\s+-q', build_script): errors.append('Linux build script uses producer | grep -q under pipefail; capture producer output first')
for token,label in [('ae0fef0ee67eec897e401321b99b6dd7cfbdc155','Crow 1.3.4 immutable commit'),('755bd7f85a4b269c67ae0ea254907c078d408cce8e1a352ad2ed664d233780e8','Asio archive checksum'),('vulkan-sdk-1.4.357.0','pinned Vulkan source tag'),('v2026.3','pinned shaderc tag'),('e3b1eec08173d6b825cd3ac88c885a63b621504a','Vulkan-Headers commit pin'),('5f157b62e333c63260d05d81bf66faa216ab0fb8','Vulkan-Loader commit pin'),('2c8cae778eec0283b44acbe7ed1a386865d78799','shaderc commit pin'),('utils/git-sync-deps','shaderc known-good deps'),('unexpected Vulkan-Loader PkgConfig line','pinned Loader patch guard'),('BUILD_WSI_XCB_SUPPORT OR BUILD_WSI_XLIB_SUPPORT OR BUILD_WSI_DIRECTFB_SUPPORT','compute-only PkgConfig conditional')]:
    if token not in prep_script: errors.append(f'Linux dependency prep missing {label}: {token!r}')

# Keep canonical source lists synchronized across root and generated Windows CMake.
def source_list(path):
    s=(ROOT/path).read_text(encoding='utf-8')
    m=re.search(r'set\(S2_SOURCES\s*(.*?)\n\)', s, re.S)
    if not m:
        errors.append(f'{path}: S2_SOURCES block missing'); return []
    return [line.strip() for line in m.group(1).splitlines() if line.strip().startswith('src/')]
root_sources=source_list('CMakeLists.txt')
for generated in ['patch-cmake.ps1','patch-cmake-cuda.ps1']:
    generated_sources=source_list(generated)
    if generated_sources != root_sources: errors.append(f'{generated}: S2_SOURCES diverges from root')
if len(root_sources) != len(set(root_sources)): errors.append('CMakeLists.txt: duplicate entries in S2_SOURCES')
if 'src/main.cpp' not in root_sources or len(root_sources) != 12: errors.append(f'CMakeLists.txt: unexpected canonical S2_SOURCES set ({len(root_sources)} entries)')

# Root/Windows-generated CMake must publish the same backend-specific names.
for name in ['CMakeLists.txt','patch-cmake.ps1']:
    t=(ROOT/name).read_text(encoding='utf-8')
    for out in ['s2-vulkan','s2-cuda','s2-metal','s2-cpu']:
        if f'OUTPUT_NAME "{out}"' not in t: errors.append(f'{name}: missing OUTPUT_NAME {out}')
for name in ['CMakeLists.txt','patch-cmake.ps1','patch-cmake-cuda.ps1']:
    t=(ROOT/name).read_text(encoding='utf-8')
    req_local=[]
    if name.startswith('patch-cmake') and 'set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)' not in t: req_local.append('GGML_NATIVE forced OFF in generated release CMake')
    if 'cmake_minimum_required(VERSION 3.15)' not in t: req_local.append('CMake >=3.15')
    try:
        p_project=t.index('project(s2cpp'); p_policy=t.index('cmake_policy(SET CMP0091 NEW)'); p_runtime=t.index('CMAKE_MSVC_RUNTIME_LIBRARY')
        if not (p_policy < p_project and p_runtime < p_project): req_local.append('CMP0091 and /MT before project()')
    except ValueError: req_local.append('project/CMP0091/runtime marker missing')
    if req_local: errors.append(f'{name}: '+', '.join(req_local))

try:
    p_container=text.index('- name: Build single-file CUDA container')
    p_cuda=text.index('$cudaDll = Join-Path $cudaBin $dep', p_container)
    p_sys=text.index('if (Test-Path (Join-Path $sys32 $dep)) { continue }', p_container)
    if p_cuda > p_sys: errors.append('CUDA dependency closure checks System32 before CUDA_PATH\\bin')
except ValueError: errors.append('CUDA dependency-resolution markers missing')

if errors:
    print('CI_INVARIANTS_FAIL', file=sys.stderr)
    for e in errors: print(' - '+e,file=sys.stderr)
    raise SystemExit(1)
print('CI_INVARIANTS_PASS jobs=8 release_targets=7')
