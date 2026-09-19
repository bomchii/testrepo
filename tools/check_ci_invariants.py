#!/usr/bin/env python3
"""Fail-closed static invariants for the ten-backend release workflow.

These checks complement actionlint and real shell parsers. They encode observable
release/toolchain properties and deliberately avoid preserving historical
workarounds merely because they once existed.
"""
from __future__ import annotations
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / '.github/workflows/build-all-backends.yml'
text = WF.read_text(encoding='utf-8')
errors: list[str] = []


def req(token: str, why: str, haystack: str = text) -> None:
    if token not in haystack:
        errors.append(f'missing {why}: {token!r}')


def forbid(token: str, why: str, haystack: str = text) -> None:
    if token in haystack:
        errors.append(f'forbidden {why}: {token!r}')


if (ROOT / '.github/workflows/build-windows-vulkan.yml').exists():
    errors.append('obsolete build-windows-vulkan.yml exists')

expected = [
    'preflight', 'build', 'build-cuda', 'build-cpu', 'build-linux-cpu',
    'build-linux-vulkan', 'build-linux-cuda', 'build-metal', 'build-windows-amd',
    'build-linux-amd', 'build-macos-intel-cpu', 'release'
]
jobs_region = text[text.index('\njobs:') + len('\njobs:'):]
job_names = re.findall(r'^  ([A-Za-z0-9_-]+):\s*$', jobs_region, re.M)
if job_names != expected:
    errors.append(f'unexpected job order/names: {job_names}')

if text.count('runs-on: windows-2025-vs2026') != 5:
    errors.append('all four Windows backends plus release must use windows-2025-vs2026')
if text.count('runs-on: windows-2022') != 0:
    errors.append('windows-2022 is forbidden: current hosted-image routing has been observed to drift; ROCm uses the explicit VS2026 image too')
if text.count('runs-on: ubuntu-24.04') != 5:
    errors.append('preflight + portable CPU/Vulkan/CUDA + ROCm host must use ubuntu-24.04')
if text.count('runs-on: ubuntu-22.04') != 0:
    errors.append('deprecated ubuntu-22.04 hosted runner label must not be used')
if text.count('image: ubuntu:22.04') != 1:
    errors.append('Linux ROCm must preserve its Ubuntu 22.04 ABI in exactly one job container')
if len(re.findall(r'^    runs-on: macos-15$', text, re.M)) != 1:
    errors.append('Metal must use macos-15 exactly once')
if text.count('runs-on: macos-15-intel') != 1:
    errors.append('Intel CPU must use macos-15-intel exactly once')

# All ten release backends must be gated by the preflight job.
for job in ['build', 'build-cuda', 'build-cpu', 'build-linux-cpu', 'build-linux-vulkan', 'build-linux-cuda', 'build-metal', 'build-windows-amd', 'build-linux-amd', 'build-macos-intel-cpu']:
    m = re.search(rf'^  {re.escape(job)}:\s*$([\s\S]*?)(?=^  [A-Za-z0-9_-]+:\s*$)', text, re.M)
    if not m or 'needs: preflight' not in m.group(1):
        errors.append(f'{job} must need preflight')

for token, label in [
    ('actions/checkout@v7', 'checkout major'),
    ('actions/cache@v6', 'cache major'),
    ('actions/upload-artifact@v7', 'upload-artifact major'),
    ('actions/download-artifact@v8', 'download-artifact major'),
    ('softprops/action-gh-release@v3.0.3', 'release action pin'),
    ("FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: 'true'", 'Node 24 Actions runtime'),
]:
    req(token, label)
forbid('FORCE_JAVASCRIPT_ACTIONS_TO_NODE20', 'Node 20 override')
if text.count('if-no-files-found: error') != 10:
    errors.append('all ten backend uploads must use if-no-files-found: error')

# Successful backend artifacts are intentionally uploaded on every successful
# workflow run (push, PR, manual, or tag). Artifact names are unique per backend
# and GitHub scopes immutable artifact names to a single workflow run.
forbid('upload_artifacts:', 'manual artifact-upload input; uploads must be automatic')
forbid("inputs.upload_artifacts", 'conditional manual artifact upload gate')
forbid("startsWith(github.ref, 'refs/tags/') || (github.event_name == 'workflow_dispatch'", 'conditional success-artifact upload gate')
if text.count('uses: actions/upload-artifact@v7') != 20:
    errors.append('expected ten success artifact uploads plus ten diagnostic uploads')
if text.count('retention-days: 7') != 10:
    errors.append('only the ten failure diagnostics should force 7-day retention')
if 'retention-days: 3' in text:
    errors.append('successful artifacts must use repository/default retention, not forced 3-day expiry')

# GitHub will not trigger a workflow file larger than 500 KiB. Keep a hard guard
# with a little headroom so future backend additions cannot silently cross it.
workflow_bytes = WF.stat().st_size
if workflow_bytes > 480 * 1024:
    errors.append(f'workflow is too close to GitHub 500 KiB limit: {workflow_bytes} bytes')

# Preflight must catch the exact classes of syntax failures that escaped audit5.
for token, label in [
    ('version=1.7.12', 'actionlint 1.7.12 pin'),
    ('8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8', 'actionlint archive SHA-256'),
    ('--retry 5 --retry-all-errors', 'resilient actionlint download retries'),
    ('actionlint_${version}_linux_amd64.tar.gz', 'actionlint Linux amd64 asset name'),
    ('-shellcheck= -pyflakes=', 'actionlint external-linter disable flags'),
    ('tools/ci/extract-workflow-run-blocks.py', 'all run-command extractor'),
    ('System.Management.Automation.Language.Parser', 'real PowerShell parser'),
    ('bash -n "$file"', 'real Bash parser'),
    ('python3 tools/check_cli_parser_sync.py', 'CLI sync guard in preflight'),
    ('python3 tools/check_docs_api_sync.py', 'docs/API guard in preflight'),
    ('python3 tools/check_ci_invariants.py', 'CI invariant guard in preflight'),
]:
    req(token, label)
if not (ROOT / 'tools/ci/extract-workflow-run-blocks.py').is_file():
    errors.append('missing tools/ci/extract-workflow-run-blocks.py')
extractor = (ROOT / 'tools/ci/extract-workflow-run-blocks.py').read_text(encoding='utf-8')
req('m_inline = re.match', 'inline run-command extraction', extractor)
req('inline run step must declare shell: bash or pwsh', 'explicit shell guard for inline run commands', extractor)


# Workflow-level scheduling/diagnostic hygiene discovered during the 10-target audit.
for token, label in [
    ("group: ${{ github.workflow }}-${{ github.ref }}", 'workflow concurrency group'),
    ("cancel-in-progress: ${{ !startsWith(github.ref, 'refs/tags/') }}", 'do not cancel tagged releases'),
    ("actions/setup-python@v7", 'pinned Windows Python 3.12 setup for ROCm wheels'),
    ("python-version: '3.12'", 'ROCm wheel Python version'),
]:
    req(token, label)
forbid('.ci-logs/', 'hidden diagnostics directory excluded by upload-artifact defaults')
if text.count('path: ci-logs/') != 10:
    errors.append('all ten failure-diagnostic uploads must use visible ci-logs/')


# ROCm 7.14 wheel/toolchain/package invariants. Current upstream GGML HIP is
# shared-only and current multi-arch wheels expose canonical paths via rocm-sdk.
for token, label in [
    ('rocm[libraries,devel,device-gfx1030,device-gfx1100,device-gfx1101,device-gfx1102,device-gfx1103,device-gfx1150,device-gfx1151,device-gfx1152,device-gfx1153,device-gfx1200,device-gfx1201]==7.14.1', 'ROCm 7.14.1 selected device kernels'),
    ('name: Windows Server 2025 - AMD (ROCm/HIP 7.14.1)', 'explicit Windows ROCm hosted image'),
    ('image: ubuntu:22.04', 'Linux ROCm Ubuntu 22.04 ABI container'),
    ('Bootstrap Ubuntu 22.04 ROCm build container', 'Linux ROCm container bootstrap'),
    ("'cmake==4.4.3' ninja", 'Linux ROCm pinned CMake/Ninja'),
    ('rocm-sdk path --root', 'ROCm canonical root discovery'),
    ('rocm-sdk path --cmake', 'ROCm canonical CMake discovery'),
    ('rocm-sdk path --bin', 'ROCm canonical bin discovery'),
    ('HIP_DEVICE_LIB_PATH', 'ROCm device-library path'),
    ('-DCMAKE_HIP_COMPILER="$clang"', 'Windows ROCm HIP compiler'),
    ('-DGGML_HIP=ON -DGGML_STATIC=OFF', 'ROCm shared GGML requirement'),
    ('GGML HIP DLL was not produced', 'Windows ROCm GGML shared-library check'),
    ('Unresolved Windows ROCm dependency closure', 'Windows ROCm dependency closure audit'),
    ('tools/cuda_bundle.py --launcher $launcher --payload-dir release-rocm --core-name s2-amd-core.exe --output release-amd\\s2-windows-amd-x86-64.exe', 'single-file Windows AMD bundler'),
    ('archive: false', 'direct single-file artifact upload'),
    ('rocBLAS kernel library directory is missing', 'Windows ROCm rocBLAS kernel payload'),
    ("Where-Object { $_.Parent.Name -ieq $tree }", 'ROCm multi-arch kernel-directory discovery'),
    ('Ambiguous $tree kernel-library layout:', 'ROCm ambiguous kernel-layout rejection'),
    ("'rocm_kpack.dll'", 'Windows ROCm dynamically loaded kpack runtime'),
    ("'libhipblaslt.dll'", 'Windows ROCm hipBLASLt runtime'),
    ("'rocsolver.dll'", 'Windows ROCm rocSOLVER runtime'),
    ('libggml-hip was not packaged', 'Linux ROCm GGML shared-library check'),
    ('Linux ROCm package has unresolved ELF dependencies', 'Linux ROCm ELF closure audit'),
    ('above ROCm artifact ceiling GLIBC_$ceiling', 'Linux ROCm GLIBC 2.35 ceiling audit'),
    ('ROCm runtime symlink escapes package', 'Linux ROCm symlink containment audit'),
    ("'-DCMAKE_BUILD_RPATH=$ORIGIN'", 'Linux ROCm origin RPATH'),
]:
    req(token, label)
req('--no-cache-dir --index-url https://repo.amd.com/rocm/whl-multi-arch/', 'ROCm pip no-cache disk safeguard')
req('Windows ROCm build requires an x64 runner', 'Windows ROCm runner architecture verification')
forbid('device-all]==7.14.1', 'ROCm device-all disk-space hazard')
forbid('cp -aL "$ROCM_PATH/lib/."', 'ROCm symlink dereference/disk-space hazard')
forbid('cp -a "$ROCM_PATH/lib/."', 'ROCm full-SDK copy/disk-space hazard')
req('Build a fail-closed ELF dependency', 'bounded Linux ROCm runtime closure')
req('_rocm_sdk_core/lib/rocm_sysdeps/lib', 'ROCm wheel core sysdeps runtime path')
req('_rocm_sdk_libraries/lib', 'ROCm wheel libraries runtime path')
req('ROCM_CLOSURE_LD_PATH=', 'ROCm closure loader-path diagnostic')
req('is_rocm_sdk_path', 'ROCm multi-wheel-root dependency allowlist')
req('LD_LIBRARY_PATH="$scan_path" ldd', 'ROCm transitive closure SDK resolver')
req("'librocm_kpack.so*'", 'Linux ROCm dynamically loaded kpack runtime')
req('Expected exactly one %s kernel data directory', 'Linux ROCm kernel-data layout guard')
forbid('tar -tzf s2-linux-x86_64-rocm.tar.gz | head', 'pipefail/SIGPIPE-prone ROCm tar preview')

# The old-Intel macOS build exists specifically for machines without Metal and
# without modern x86 extensions. These explicit OFF values prevent GGML's
# defaults from silently raising the CPU baseline on the modern CI runner.
for token in [
    '-DCMAKE_OSX_ARCHITECTURES=x86_64', '-DCMAKE_OSX_DEPLOYMENT_TARGET=10.15',
    '-DGGML_SSE42=OFF', '-DGGML_AVX=OFF', '-DGGML_AVX2=OFF', '-DGGML_BMI2=OFF',
    '-DGGML_FMA=OFF', '-DGGML_F16C=OFF', '-DGGML_AVX512=OFF',
]:
    req(token, 'old-Intel macOS compatibility baseline')
req("grep -q 'minos 10.15'", 'macOS Intel deployment-target verification')
req("if grep -Eqi 'Metal|Vulkan|MoltenVK'", 'macOS Intel GPU-runtime exclusion')

# PowerShell interpolation: "$name:" is a parser error unless braced. Ignore
# valid scoped variables such as $env:NAME.
ps_sources = {
    'patch-cmake.ps1': (ROOT / 'patch-cmake.ps1').read_text(encoding='utf-8'),
    'patch-cmake-cuda.ps1': (ROOT / 'patch-cmake-cuda.ps1').read_text(encoding='utf-8'),
}
scopes = {'env', 'global', 'local', 'script', 'private', 'using', 'function', 'variable'}
for name, source in ps_sources.items():
    for match in re.finditer(r'\$([A-Za-z_][A-Za-z0-9_]*):', source):
        if match.group(1).lower() not in scopes:
            errors.append(f'{name}: unbraced PowerShell variable before colon: {match.group(0)!r}')

patch = ps_sources['patch-cmake.ps1']
patch_cuda = ps_sources['patch-cmake-cuda.ps1']
req('throw "SHA-256 mismatch for ${Path}: expected $Expected got $actual"', 'braced PowerShell SHA error', patch)
req('param([switch]$SkipVulkanPortabilityPatch)', 'Vulkan patch opt-out', patch)
req('[regex]::Matches($vkCmake, $vkPattern)', 'Vulkan patch exact-match validation', patch)
req('Vulkan portability patch postcondition failed', 'Vulkan patch postcondition', patch)
forbid('$newCmake', 'root CMake rewrite in dependency-prep script', patch)
forbid('CMakeLists.txt raiz reescrito', 'root CMake rewrite marker', patch)
req('& (Join-Path $PSScriptRoot "patch-cmake.ps1") -SkipVulkanPortabilityPatch', 'CUDA patch delegation', patch_cuda)
if 'set(S2_SOURCES' in patch_cuda:
    errors.append('patch-cmake-cuda.ps1 must delegate instead of carrying a second generated CMake copy')
for dep_token in [
    'ae0fef0ee67eec897e401321b99b6dd7cfbdc155',
    '755bd7f85a4b269c67ae0ea254907c078d408cce8e1a352ad2ed664d233780e8',
]:
    req(dep_token, 'pinned Windows header dependency', patch)

# Windows CUDA: no VS/MSBuild CUDA generator integration. Use Ninja, nvcc and
# the VS 2026 host compiler explicitly. CMake documents HOST_COMPILER as ignored
# by Visual Studio generators, which is why this is an architectural invariant.
cuda_region = text[text.index('  build-cuda:\n'):text.index('  build-cpu:\n')]
for token, label in [
    ('-G Ninja', 'Ninja CUDA generator'),
    ('CMAKE_CUDA_COMPILER=', 'explicit nvcc'),
    ('CMAKE_CUDA_HOST_COMPILER=', 'explicit cl host compiler'),
    ('CUDAToolkit_ROOT=', 'explicit CUDA toolkit root'),
    ('CMAKE_CUDA_RUNTIME_LIBRARY=Static', 'static cudart selection'),
    ('Launch-VsDevShell.ps1', 'VS 2026 developer environment'),
    ('CUDA_NINJA_PROBE_PASS', 'standalone Ninja CUDA probe'),
    ('build-cuda\\s2-cuda-core.exe', 'Ninja CUDA core output'),
    ('--runtime-info', 'final CUDA container self-validation'),
    ('payload_files:', 'CUDA runtime-info payload count validation'),
    ('build-cuda\\s2-cuda-launcher.exe', 'Ninja CUDA launcher output'),
    ('tools/cuda_bundle.py', 'single-file CUDA bundler'),
    ('nvcuda.dll must never be bundled', 'driver DLL exclusion'),
]:
    req(token, label, cuda_region)
for token, label in [
    ('Jimver/cuda-toolkit', 'third-party CUDA installer action'),
    ('-T "cuda=', 'Visual Studio CUDA toolset integration'),
    ('Visual Studio 18 2026" -A x64 -T', 'VS/MSBuild CUDA generator'),
    ('extras\\visual_studio_integration\\MSBuildExtensions', 'CUDA MSBuild integration dependency'),
]:
    forbid(token, label, cuda_region)
forbid('--allow-unsupported-compiler', 'unsupported CUDA compiler bypass')
forbid('CMAKE_CUDA_ARCHITECTURES=86', 'fixed sm_86 architecture')
forbid('sm_86', 'fixed sm_86 architecture')

# Exact CUDA 13.2 redistributable components/hashes selected for Windows.
for token in [
    "Key='cuda_cccl';        Version='13.2.27'; Sha='fb61bbca384ea9e5722f3b77cfdd635c916f33593d187562510ce14831aa629a'",
    "Key='cuda_crt';         Version='13.2.51'; Sha='02892bd6bc7ec832c5fc6fc794491803166ee8455bd87d52976c4b195b76c8e7'",
    "Key='cuda_cudart';      Version='13.2.51'; Sha='906b430b53e6396f7cbaa6874fbe7f9307a184e830b119a4ad3f58dc6e0165c6'",
    "Key='cuda_nvcc';        Version='13.2.51'; Sha='a6bacac79adb923e310eb39f2b86d4d9e756c896d035dfead50c69403eca4661'",
    "Key='libnvvm';          Version='13.2.51'; Sha='ef2a24cf40f9394e2f13df5a8a7315c65325103006a53692cd12b8c48431e372'",
    "Key='libnvptxcompiler'; Version='13.2.51'; Sha='19c12ad13d8d0a7ee2ad9f9896f18be3052684aa0b3e199ba57f344482dce45b'",
    "Key='libcublas';        Version='13.3.0.5'; Sha='438a88f07af3721eda15ed1f4d2b957ea810e29061b2a6292837f35520dbd212'",
    "Key='libnvjitlink';     Version='13.2.51'; Sha='b192c79a449efc6a6593be46bc56928281789a4bb0b8714321c8a1b1684ff74f'",
]:
    req(token, 'pinned Windows CUDA redistributable component', cuda_region)
for required in [
    "'bin\\nvcc.exe'", "'include\\cuda_runtime.h'", "'include\\cccl\\cub'", "'include\\cccl\\thrust'", "'include\\cccl\\cuda'",
    "'include\\cublas_v2.h'", "'lib\\x64\\cudart_static.lib'", "'lib\\x64\\cudadevrt.lib'",
    "'lib\\x64\\cuda.lib'", "'lib\\x64\\cublas.lib'", "'lib\\x64\\cublasLt.lib'",
    "'nvvm\\libdevice\\libdevice.10.bc'", "'nvvm\\bin\\cicc.exe'", "'bin\\ptxas.exe'",
    "'bin\\nvlink.exe'", "'bin\\fatbinary.exe'",
]:
    req(required, 'assembled CUDA toolkit validation', cuda_region)
forbid("'bin\\cicc.exe'", 'legacy incorrect CUDA cicc location', cuda_region)
req('CUDA_ASSEMBLY_FAILURE', 'early Windows CUDA assembly diagnostic capture', cuda_region)
req("Join-Path $cudaRoot 'bin\\x64'", 'CUDA 13.x Windows runtime DLL directory', cuda_region)
req("Join-Path $cudaBin 'x64'", 'CUDA payload bin\\x64 lookup', cuda_region)
req('Find-CudaRuntimeDll', 'CUDA payload dual runtime-directory resolver', cuda_region)

# Root CMake is authoritative and backend names must match artifacts.
cmake = (ROOT / 'CMakeLists.txt').read_text(encoding='utf-8')
for token, label in [
    ('cmake_policy(SET CMP0091 NEW)', 'CMP0091 policy'),
    ('CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded"', 'static MSVC runtime'),
    ('OUTPUT_NAME "s2-vulkan"', 'Windows/Linux Vulkan output'),
    ('OUTPUT_NAME "s2-cuda-core"', 'Windows CUDA core output'),
    ('OUTPUT_NAME "s2-cuda"', 'Linux CUDA output'),
    ('OUTPUT_NAME "s2-metal"', 'Metal output'),
    ('OUTPUT_NAME "s2-amd-core"', 'AMD/HIP core output'),
    ('add_executable(s2-amd-launcher tools/cuda_launcher.cpp)', 'Windows AMD launcher target'),
    ('OUTPUT_NAME "s2-cpu"', 'CPU output'),
    ('add_executable(s2-cuda-launcher tools/cuda_launcher.cpp)', 'Windows CUDA launcher target'),
    ('/arch:AVX2', 'Windows AVX2 baseline'),
]:
    req(token, label, cmake)
for token, label in [
    ('find_package(CUDAToolkit REQUIRED)', 'root CUDA toolkit discovery'),
    ('target_link_libraries(s2 PRIVATE CUDA::cudart_static)', 'direct CUDA runtime dependency'),
    ('find_package(hip REQUIRED)', 'root HIP package discovery'),
    ('target_link_libraries(s2 PRIVATE hip::host)', 'direct HIP host runtime dependency'),
]:
    req(token, label, cmake)
try:
    if not (cmake.index('cmake_policy(SET CMP0091 NEW)') < cmake.index('project(s2cpp') and
            cmake.index('CMAKE_MSVC_RUNTIME_LIBRARY') < cmake.index('project(s2cpp')):
        errors.append('CMP0091 and CMAKE_MSVC_RUNTIME_LIBRARY must precede project()')
except ValueError:
    errors.append('CMake project/policy/runtime markers missing')


main_cpp = (ROOT / 'src/main.cpp').read_text(encoding='utf-8')
for token, label in [
    ('#if defined(GGML_USE_HIP)', 'HIP runtime include branch'),
    ('#  include <hip/hip_runtime.h>', 'HIP runtime header'),
    ('hipGetDeviceCount(&hip_dev_count)', 'HIP device discovery'),
    ('No ROCm/HIP devices found', 'HIP fallback warning'),
]:
    req(token, label, main_cpp)
if main_cpp.find('#if defined(GGML_USE_HIP)') > main_cpp.find('#elif defined(GGML_USE_CUDA)'):
    errors.append('HIP runtime branch must take precedence over CUDA when both compatibility macros are defined')

model_cpp = (ROOT / 'src/s2_model.cpp').read_text(encoding='utf-8')
req('#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP)', 'NVIDIA CUDA workarounds excluded from HIP', model_cpp)
req('const char * gpu_backend_name = "ROCm/HIP";', 'ROCm model backend label', model_cpp)
codec_cpp = (ROOT / 'src/s2_codec.cpp').read_text(encoding='utf-8')
req('const char * gpu_backend_name = "ROCm/HIP";', 'ROCm codec backend label', codec_cpp)

req('/Applications/Xcode_16.4.app/Contents/Developer', 'pinned Xcode 16.4 for macOS reproducibility')
if text.count('Pin Xcode 16.4') != 2:
    errors.append('both macOS backend jobs must pin Xcode 16.4 explicitly')
req('test "$(uname -m)" = "arm64"', 'Metal runner architecture guard')
req('test "$(uname -m)" = "x86_64"', 'Intel macOS runner architecture guard')
if text.count('test -d /Applications/Xcode_16.4.app/Contents/Developer') != 2:
    errors.append('both macOS jobs must verify the pinned Xcode path before xcode-select')
if text.count('archive: false') != 10:
    errors.append('all ten success artifacts must use direct single-file upload (archive: false)')
if text.count('Fetch pinned header-only Crow and Asio') != 2:
    errors.append('both macOS jobs must prepare Crow/Asio natively in Bash')

# Metal publishes the executable itself: no wrapper ZIP/directory.
req('cp build-metal/s2-metal release-metal/s2-macos-metal-arm64', 'macOS Metal public executable name')
req('test -x release-metal/s2-macos-metal-arm64', 'macOS Metal executable verification')
forbid('--keepParent', 'legacy Metal ZIP keepParent packaging')
forbid('s2-macos-metal.zip', 'legacy Metal ZIP artifact')

# Linux portability and the exact audit5 Linux-CUDA regression.
build_script = (ROOT / 'tools/ci/build-linux-portable.sh').read_text(encoding='utf-8')
singlefile_script = (ROOT / 'tools/ci/make-linux-singlefile.sh').read_text(encoding='utf-8')
for token in ['__S2_EMBEDDED_RUNTIME_BELOW__', 'S2_PAYLOAD_SHA', '--runtime-info', '--clean-runtime']:
    if token not in singlefile_script:
        errors.append(f'Linux single-file packer missing invariant: {token!r}')
prep_script = (ROOT / 'tools/ci/prepare-linux-deps.sh').read_text(encoding='utf-8')
for token, label in [
    ('quay.io/pypa/manylinux2014_x86_64', 'CPU/Vulkan glibc 2.17 image'),
    ('nvidia/cuda:13.2.0-devel-rockylinux8', 'CUDA Rocky 8 image'),
    ('tools/ci/build-linux-portable.sh cpu', 'Linux CPU build'),
    ('tools/ci/build-linux-portable.sh vulkan', 'Linux Vulkan build'),
    ('tools/ci/build-linux-portable.sh cuda', 'Linux CUDA build'),
    ('s2-linux-cpu-x86-64', 'Linux CPU single-file artifact'),
    ('s2-linux-vulkan-x86-64', 'Linux Vulkan single-file artifact'),
    ('s2-linux-cuda-x86-64', 'Linux CUDA single-file artifact'),
]:
    req(token, label)
forbid('ubuntu-latest', 'moving Ubuntu release host')
if text.count('docker image inspect "$image"') != 3:
    errors.append('all three Linux jobs must log the pulled image digest')
for token, label in [
    ('g++ -std=c++17 -fopenmp', 'C++17/OpenMP compiler probe'),
    ('LINUX_CXX_TOOLCHAIN_PROBE_PASS', 'toolchain probe success marker'),
    ('-DCMAKE_EXE_LINKER_FLAGS=-Wl,--disable-new-dtags', 'old-dtags linker mode'),
    ('-DCMAKE_BUILD_RPATH=\\$ORIGIN', 'build RPATH'),
    ('-DCMAKE_INSTALL_RPATH=\\$ORIGIN', 'install RPATH'),
    ('libstdc++.so.6', 'bundled libstdc++'),
    ('libgcc_s.so.1', 'bundled libgcc'),
    ('libgomp.so.1', 'bundled OpenMP when needed'),
    ('runtime_path', 'exact sidecar RPATH audit'),
    ('ceiling=2.17', 'glibc 2.17 ceiling'),
    ('[[ "$backend" == "cuda" ]] && ceiling=2.28', 'glibc 2.28 CUDA ceiling'),
    ('check_dynamic_deps', 'ELF dependency allowlist'),
    ('Advanced Micro Devices X86-64', 'x86_64 ELF check'),
    ("grep -Eq '^(libcudart|libcublas|libcublasLt|libnvJitLink)\\.so'", 'shared CUDA Toolkit rejection'),
    ('tools/ci/make-linux-singlefile.sh', 'Linux single-file runtime packer'),
]:
    if token not in build_script:
        errors.append(f'Linux build script missing {label}: {token!r}')
active_build_lines = "\n".join(line for line in build_script.splitlines() if not line.lstrip().startswith('#'))
for token, label in [
    ('-static-libstdc++', 'static libstdc++ linker flag that failed on Rocky 8'),
    ('-static-libgcc', 'static libgcc linker flag that failed on Rocky 8'),
    ('--sort=name', 'GNU tar option unavailable on CentOS 7 baseline'),
]:
    if token in active_build_lines:
        errors.append(f'Linux build script contains forbidden {label}: {token!r}')
if re.search(r'\|\s*grep\s+-q', build_script):
    errors.append('Linux build script uses producer | grep -q under pipefail')
req('-DSHADERC_ENABLE_EXECUTABLES=ON', 'shaderc executable build enabled', build_script)
req('--target glslc_exe', 'shaderc glslc executable target', build_script)
req('build-linux-shaderc/glslc/glslc', 'deterministic glslc executable path', build_script)
req('asio-src/asio/LICENSE_1_0.txt', 'correct nested Asio license path', build_script)
if 'asio-src/LICENSE_1_0.txt' in build_script:
    errors.append('Linux build script still references the nonexistent root Asio license path')
req('export PATH="$glslc_dir:$PATH"', 'pinned Linux Vulkan glslc PATH', build_script)
req('-DVulkan_GLSLC_EXECUTABLE="$glslc"', 'explicit Linux Vulkan glslc CMake hint', build_script)
req('VULKAN_STALE_SHADER_HEADER_CLEARED', 'stale source Vulkan shader header cleanup', build_script)
req('rm -f "$stale_shader_header"', 'ephemeral stale Vulkan generated-header removal', build_script)
req('bash "$root/tools/ci/make-linux-singlefile.sh"', 'ZIP-safe Bash invocation of Linux single-file packer', build_script)
for token in [
    'vulkan-sdk-1.4.357.0', 'v2026.3',
    'e3b1eec08173d6b825cd3ac88c885a63b621504a',
    '5f157b62e333c63260d05d81bf66faa216ab0fb8',
    '2c8cae778eec0283b44acbe7ed1a386865d78799',
    'BUILD_WSI_XCB_SUPPORT OR BUILD_WSI_XLIB_SUPPORT OR BUILD_WSI_DIRECTFB_SUPPORT',
]:
    if token not in prep_script:
        errors.append(f'Linux dependency prep missing {token!r}')

# Failure diagnostics are part of CI usability: keep concise annotations/job summaries
# and preserve full captured logs as short-lived artifacts on failure.
diag = (ROOT / 'tools/ci/run-with-diagnostics.py').read_text(encoding='utf-8')
for token, label in [
    ('GITHUB_STEP_SUMMARY', 'GitHub job summary output'),
    ('::error title=', 'GitHub error annotation'),
    ('FAILURE DIGEST', 'concise failure digest'),
    ('First useful error context', 'first-error context extraction'),
    ('CMake Error', 'CMake error detection'),
    ('undefined reference', 'linker error detection'),
    ('cannot find -l', 'missing library detection'),
    ('nvcc fatal', 'CUDA compiler error detection'),
    ('HIP error', 'HIP runtime/compiler error detection'),
    ('HSA_STATUS_ERROR', 'HSA runtime error detection'),
    ('ROCm.*(?:missing|not found|failed)', 'ROCm setup error detection'),
]:
    if token not in diag:
        errors.append(f'diagnostic runner missing {label}: {token!r}')
for token in [
    'Windows Vulkan configure', 'Windows Vulkan build',
    'Windows CUDA compiler probe configure', 'Windows CUDA compiler probe build',
    'Windows CUDA configure', 'Windows CUDA build',
    'Windows CPU configure', 'Windows CPU build',
    'Linux CPU portable build', 'Linux Vulkan portable build', 'Linux CUDA portable build',
    'macOS Metal configure', 'macOS Metal build',
    'Windows ROCm compiler probe', 'Windows ROCm configure', 'Windows ROCm build',
    'Linux ROCm compiler probe', 'Linux ROCm configure', 'Linux ROCm build',
    'macOS Intel CPU configure', 'macOS Intel CPU build',
]:
    req(token, 'critical-stage diagnostic wrapper')
if text.count('Collect robust crash report') != 10:
    errors.append('all ten backend jobs must collect a robust crash report on failure')
if text.count('Upload concise failure diagnostics') != 10:
    errors.append('all ten backend jobs must upload captured diagnostic logs on failure')
crash = (ROOT / 'tools/ci/collect-crash-report.py').read_text(encoding='utf-8')
for source, label in [(diag, 'diagnostic runner'), (crash, 'crash report collector')]:
    if 'CMAKE_PROBE_MISS' not in source:
        errors.append(f'{label} must ignore nonfatal CMake feature-test misses')
for token, label in [
    ('GITHUB_RUN_ATTEMPT', 'run-attempt metadata'),
    ('RUNNER_ARCH', 'runner architecture metadata'),
    ('Disk bytes:', 'disk-capacity metadata'),
    ('--- TOOLCHAIN ---', 'toolchain inventory'),
    ('FIRST MATCH:', 'first-error extraction'),
    ('TAIL:', 'log tail extraction'),
    ('GITHUB_STEP_SUMMARY', 'crash report job summary'),
]:
    if token not in crash:
        errors.append(f'crash report collector missing {label}: {token!r}')


public_names = [
    's2-windows-cpu-x86-64.exe','s2-windows-vulkan-x86-64.exe','s2-windows-cuda-x86-64.exe','s2-windows-amd-x86-64.exe',
    's2-linux-cpu-x86-64','s2-linux-vulkan-x86-64','s2-linux-cuda-x86-64','s2-linux-amd-x86-64',
    's2-macos-metal-arm64','s2-macos-cpu-x86-64',
]
for name in public_names:
    req(name, 'canonical public executable name')
if text.count('uses: actions/upload-artifact@v7') < 20:
    errors.append('expected ten success uploads plus ten diagnostic uploads')

# Release fan-in and exact ten outputs.
req('needs: [build, build-cuda, build-cpu, build-windows-amd, build-linux-cpu, build-linux-vulkan, build-linux-cuda, build-linux-amd, build-metal, build-macos-intel-cpu]', 'release dependency fan-in')
req('contents: write', 'release write permission')
req('fail_on_unmatched_files: true', 'release must fail closed when any asset glob is missing')
for token in [
    'artifacts\\windows-vulkan\\s2-windows-vulkan-x86-64.exe',
    'artifacts\\windows-cuda\\s2-windows-cuda-x86-64.exe',
    'artifacts\\windows-cpu\\s2-windows-cpu-x86-64.exe',
    'artifacts\\windows-amd\\s2-windows-amd-x86-64.exe',
    'artifacts\\linux-cpu\\s2-linux-cpu-x86-64',
    'artifacts\\linux-vulkan\\s2-linux-vulkan-x86-64',
    'artifacts\\linux-cuda\\s2-linux-cuda-x86-64',
    'artifacts\\linux-amd\\s2-linux-amd-x86-64',
    'artifacts\\macos-metal\\s2-macos-metal-arm64',
    'artifacts\\macos-cpu\\s2-macos-cpu-x86-64',
]:
    req(token, 'exact release input')
req('Artifact must contain exactly one executable: ${dir}:', 'braced PowerShell release-directory interpolation')
forbid('s2-windows-rocm.zip', 'legacy Windows ROCm ZIP release')
forbid('s2-linux-x86_64-rocm.tar.gz', 'legacy Linux ROCm archive release')

if errors:
    print('CI_INVARIANTS_FAIL', file=sys.stderr)
    for error in errors:
        print(' - ' + error, file=sys.stderr)
    raise SystemExit(1)
print('CI_INVARIANTS_PASS jobs=12 release_targets=10')
