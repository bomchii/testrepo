#!/usr/bin/env bash
set -euo pipefail

backend="${1:-}"
case "$backend" in
  cpu|vulkan|cuda) ;;
  *) echo "usage: $0 cpu|vulkan|cuda" >&2; exit 2 ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
deps="$root/build-linux-deps"

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing required tool: $1" >&2; exit 1; }; }

# Keep the build-system version deterministic inside both manylinux2014 and
# Rocky 8. Vulkan-Loader 1.4.357 requires CMake >= 3.22.1; using the same
# CMake 4.4.3 as the GitHub Windows runner avoids image-dependent behavior.
ensure_cmake() {
  local want="4.4.3" have="" py=""
  if command -v cmake >/dev/null 2>&1; then
    have="$(cmake --version | sed -n '1s/^cmake version //p')"
  fi
  if [[ "$have" == "$want" ]]; then
    return 0
  fi
  # manylinux rotates the set of bundled CPython versions over time. Do not
  # couple the release to one cpXY directory; use any working manylinux Python
  # (or the distro python3 on Rocky) solely to install the pinned CMake wheel.
  local bundled_py=""
  if [[ -d /opt/python ]]; then
    bundled_py="$(find /opt/python -maxdepth 3 -path '*/bin/python' -print -quit 2>/dev/null || true)"
  fi
  for candidate in "$bundled_py" "$(command -v python3 || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then py="$candidate"; break; fi
  done
  [[ -n "$py" ]] || { echo "CMake $want required but no bootstrap Python was found" >&2; exit 1; }
  "$py" -m pip install --disable-pip-version-check --no-cache-dir "cmake==$want" >/dev/null
  local scripts_dir
  scripts_dir="$("$py" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
  [[ -n "$scripts_dir" && -d "$scripts_dir" ]] || { echo "cannot resolve Python scripts directory for CMake bootstrap" >&2; exit 1; }
  # /opt/python puts pip console scripts beside Python; distro Python commonly
  # puts them in /usr/local/bin. Put both locations first so the pinned wheel
  # wins over an older system CMake in either layout.
  export PATH="$scripts_dir:$(dirname "$py"):$PATH"
  have="$(cmake --version | sed -n '1s/^cmake version //p')"
  [[ "$have" == "$want" ]] || { echo "failed to activate CMake $want (got $have)" >&2; exit 1; }
}
ensure_cmake

for tool in cmake gcc g++ make readelf tar gzip find sort; do need "$tool"; done
python_bin="$(command -v python3 || true)"
if [[ -z "$python_bin" ]]; then
  python_bin="$(find /opt/python -maxdepth 4 -path '*/bin/python3' -print -quit 2>/dev/null || true)"
  if [[ -n "$python_bin" ]]; then
    export PATH="$(dirname "$python_bin"):$PATH"
  fi
fi

test -f "$deps/crow/include/crow.h"
test -f "$deps/asio/include/asio.hpp"
test -f src/tokenizer_data.cpp
test -f src/tokenizer_data.h

jobs="${S2_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
build="build-linux-$backend"
release="release-linux-$backend"
rm -rf "$build" "$release"
mkdir -p "$release"

# Portable release policy: do not tune code for the CI host, do not emit
# AVX-512, keep normal glibc dynamic, but make libstdc++/libgcc independent of
# the end-user distro. AVX2 remains the x86_64 performance baseline used by
# the existing Windows build as well.
common=(
  -DCMAKE_BUILD_TYPE=Release
  -DS2_CROW_INCLUDE_DIR="$deps/crow/include"
  -DS2_ASIO_INCLUDE_DIR="$deps/asio/include"
  -DBUILD_SHARED_LIBS=OFF
  -DGGML_STATIC=ON
  -DGGML_NATIVE=OFF
  -DGGML_AVX512=OFF
  -DCMAKE_EXE_LINKER_FLAGS=-static-libgcc\ -static-libstdc++
  -DCMAKE_BUILD_RPATH=\$ORIGIN
  -DCMAKE_INSTALL_RPATH=\$ORIGIN
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
)

if [[ "$backend" == "vulkan" ]]; then
  test -f "$deps/Vulkan-Headers/CMakeLists.txt"
  test -f "$deps/Vulkan-Loader/CMakeLists.txt"
  test -f "$deps/shaderc/CMakeLists.txt"

  vkprefix="$root/build-linux-vulkan-sdk"
  rm -rf "$vkprefix" "$root/build-linux-vulkan-headers" "$root/build-linux-vulkan-loader" "$root/build-linux-shaderc"

  cmake -S "$deps/Vulkan-Headers" -B "$root/build-linux-vulkan-headers" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$vkprefix"
  cmake --build "$root/build-linux-vulkan-headers" --target install --parallel "$jobs"

  cmake -S "$deps/Vulkan-Loader" -B "$root/build-linux-vulkan-loader" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$vkprefix" \
    -DCMAKE_PREFIX_PATH="$vkprefix" \
    -DBUILD_SHARED_LIBS=ON \
    -DVULKAN_HEADERS_INSTALL_DIR="$vkprefix" \
    -DSYSCONFDIR=/etc \
    -DFALLBACK_CONFIG_DIRS=/etc/xdg \
    -DFALLBACK_DATA_DIRS=/usr/local/share:/usr/share \
    -DLOADER_CODEGEN=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_WSI_XCB_SUPPORT=OFF \
    -DBUILD_WSI_XLIB_SUPPORT=OFF \
    -DBUILD_WSI_XLIB_XRANDR_SUPPORT=OFF \
    -DBUILD_WSI_WAYLAND_SUPPORT=OFF \
    -DBUILD_WSI_DIRECTFB_SUPPORT=OFF
  cmake --build "$root/build-linux-vulkan-loader" --target install --parallel "$jobs"

  cmake -S "$deps/shaderc" -B "$root/build-linux-shaderc" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSHADERC_SKIP_TESTS=ON \
    -DSHADERC_SKIP_EXAMPLES=ON \
    -DSHADERC_ENABLE_HLSL=OFF
  cmake --build "$root/build-linux-shaderc" --target glslc --parallel "$jobs"
  glslc="$(find "$root/build-linux-shaderc" -type f -name glslc -perm -111 -print -quit)"
  test -n "$glslc" && test -x "$glslc"


  libvulkan="$(find "$vkprefix" -type f \( -name 'libvulkan.so.1' -o -name 'libvulkan.so.1.*' \) -print -quit)"
  test -n "$libvulkan" && test -f "$libvulkan"
  cmake -S . -B "$build" "${common[@]}" \
    -DS2_VULKAN=ON -DS2_CUDA=OFF -DS2_METAL=OFF \
    -DVulkan_INCLUDE_DIR="$vkprefix/include" \
    -DVulkan_LIBRARY="$libvulkan" \
    -DVulkan_GLSLC_EXECUTABLE="$glslc"
else
  if [[ "$backend" == "cuda" ]]; then
    cmake -S . -B "$build" "${common[@]}" \
      -DS2_VULKAN=OFF -DS2_CUDA=ON -DS2_METAL=OFF
  else
    cmake -S . -B "$build" "${common[@]}" \
      -DS2_VULKAN=OFF -DS2_CUDA=OFF -DS2_METAL=OFF
  fi
fi

cmake --build "$build" --parallel "$jobs"
exe="$build/s2-$backend"
test -x "$exe"
cp "$exe" "$release/s2-$backend"
chmod +x "$release/s2-$backend"

# Linux releases statically include GCC runtime code (libstdc++/libgcc) and may
# also carry libgomp.so.1. Ship the GPLv3 + GCC Runtime Library Exception texts
# alongside the binaries instead of relying on a distro package to provide them.
cp "$root/tools/ci/LICENSE-GPL-3.0.txt" "$release/LICENSE-GPL-3.0.txt"
cp "$root/tools/ci/LICENSE-GCC-Runtime-Library-Exception-3.1.txt" \
   "$release/LICENSE-GCC-Runtime-Library-Exception-3.1.txt"

# OpenMP remains enabled for performance. Bundle libgomp beside the executable
# so users do not depend on the exact GCC runtime package installed by a distro.
# Capture readelf output first: an early quiet-grep exit under pipefail can
# otherwise SIGPIPE the producer and turn a successful check into exit 141.
exe_dynamic="$(readelf -d "$exe")"
if grep -q 'Shared library: \[libgomp.so.1\]' <<<"$exe_dynamic"; then
  libgomp="$(gcc -print-file-name=libgomp.so.1)"
  test -f "$libgomp"
  cp -L "$libgomp" "$release/libgomp.so.1"
fi

if [[ "$backend" == "vulkan" ]]; then
  cp -L "$libvulkan" "$release/libvulkan.so.1"
  # A relocatable loader must not retain the CI workspace/install prefix in its
  # compiled search paths. Standard ICD discovery stays /etc + XDG share dirs.
  if grep -aFq "$root/build-linux-vulkan-sdk" "$release/libvulkan.so.1"; then
    echo "bundled Vulkan loader leaks the CI install prefix into runtime search paths" >&2
    exit 1
  fi
  test -f "$deps/Vulkan-Loader/LICENSE.txt"
  cp "$deps/Vulkan-Loader/LICENSE.txt" "$release/LICENSE-Vulkan-Loader.txt"
fi

# Crow and standalone Asio are header-only dependencies whose code is compiled
# into every executable. Preserve their redistribution notices in each Linux
# archive, alongside the runtime licenses above.
test -f "$deps/crow-src/LICENSE"
test -f "$deps/asio-src/LICENSE_1_0.txt"
cp "$deps/crow-src/LICENSE" "$release/LICENSE-Crow-BSD-3-Clause.txt"
cp "$deps/asio-src/LICENSE_1_0.txt" "$release/LICENSE-Asio-Boost-1.0.txt"

cat > "$release/THIRD_PARTY_NOTICES.txt" <<EOF
This archive contains code from Crow 1.3.4 (BSD-3-Clause) and standalone
Asio 1.30.2 (Boost Software License 1.0). See the accompanying license files.

It may also contain GCC runtime code/libraries covered by GPLv3 together with
the GCC Runtime Library Exception 3.1. See the accompanying license files.
EOF
if [[ "$backend" == "vulkan" ]]; then
  cat >> "$release/THIRD_PARTY_NOTICES.txt" <<EOF
It also contains the Khronos Vulkan Loader 1.4.357, licensed under Apache-2.0.
See LICENSE-Vulkan-Loader.txt.
EOF
fi
if [[ "$backend" == "cuda" ]]; then
  cat >> "$release/THIRD_PARTY_NOTICES.txt" <<EOF
The CUDA build statically links redistributable NVIDIA CUDA Toolkit runtime/math
components. Their use and redistribution are governed by the NVIDIA CUDA 13.2
Toolkit EULA: https://docs.nvidia.com/cuda/archive/13.2.0/eula/
The NVIDIA display/compute driver and libcuda.so.1 are not redistributed here.
EOF
fi

# Static libstdc++ / libgcc are deliberate; dynamic glibc and system driver
# libraries remain external. CUDA Toolkit libraries must be static in the
# Linux CUDA release; libcuda.so.1 is supplied by the NVIDIA driver.
needed="$(readelf -d "$exe" | sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p')"
printf '%s\n' "$needed" | sort -u
if grep -Eq '^(libstdc\+\+\.so|libgcc_s\.so)' <<<"$needed"; then
  echo "portable release unexpectedly depends on dynamic libstdc++/libgcc" >&2
  exit 1
fi
if [[ "$backend" == "cpu" ]] && grep -Eq '^(libvulkan|libcuda|libcudart|libcublas)' <<<"$needed"; then
  echo "CPU release unexpectedly links a GPU runtime" >&2
  exit 1
fi
if [[ "$backend" == "vulkan" ]] && grep -Eq '^(libcuda|libcudart|libcublas)' <<<"$needed"; then
  echo "Vulkan release unexpectedly links CUDA" >&2
  exit 1
fi
if [[ "$backend" == "cuda" ]] && grep -Eq '^(libcudart|libcublas|libcublasLt|libnvJitLink)\.so' <<<"$needed"; then
  echo "CUDA Toolkit shared library leaked into portable release; GGML_STATIC should keep it static" >&2
  exit 1
fi
if [[ "$backend" == "cuda" ]] && ! grep -Eq '^libcuda\.so(\.1)?$' <<<"$needed"; then
  echo "CUDA release does not expose libcuda.so.1 as a driver-provided dependency" >&2
  exit 1
fi

# Fail closed on unexpected shared libraries. Check every ELF that will be
# published, not just the executable, so a bundled loader/OpenMP sidecar cannot
# silently pull in X11, Wayland, a newer C++ runtime, or another distro package.
check_dynamic_deps() {
  local file="$1" soname
  while IFS= read -r soname; do
    [[ -n "$soname" ]] || continue
    case "$soname" in
      libc.so.6|libm.so.6|libpthread.so.0|libdl.so.2|librt.so.1|libgomp.so.1) ;;
      libvulkan.so.1) [[ "$backend" == "vulkan" ]] || { echo "unexpected $soname for $backend in $file" >&2; exit 1; } ;;
      libcuda.so|libcuda.so.1) [[ "$backend" == "cuda" ]] || { echo "unexpected $soname for $backend in $file" >&2; exit 1; } ;;
      *) echo "unexpected dynamic dependency in $backend release ($file): $soname" >&2; exit 1 ;;
    esac
  done < <(readelf -d "$file" 2>/dev/null | sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p')
}

for elf in "$release"/*; do
  [[ -f "$elf" ]] || continue
  if readelf -h "$elf" >/dev/null 2>&1; then
    elf_header="$(LC_ALL=C readelf -h "$elf")"
    if ! grep -q 'Machine:.*Advanced Micro Devices X86-64' <<<"$elf_header"; then
      echo "release contains a non-x86_64 ELF: $elf" >&2
      exit 1
    fi
    check_dynamic_deps "$elf"
  fi
done

# Any sidecar shared objects we intentionally ship must be discoverable next to
# the executable without LD_LIBRARY_PATH or distro-specific install paths.
if [[ -n "$(find "$release" -maxdepth 1 -type f -name '*.so*' -print -quit)" ]]; then
  runtime_path="$(readelf -d "$exe" | sed -n 's/.*\(RUNPATH\|RPATH\).*Library runpath: \[\([^]]*\)\].*/\2/p; s/.*\(RUNPATH\|RPATH\).*Library rpath: \[\([^]]*\)\].*/\2/p')"
  if [[ "$runtime_path" != '$ORIGIN' ]]; then
    echo "portable release with bundled .so sidecars must use exactly \$ORIGIN RUNPATH/RPATH (got: ${runtime_path:-<none>})" >&2
    exit 1
  fi
fi

check_glibc_ceiling() {
  local file="$1" ceiling="$2" max
  max="$(readelf --version-info "$file" 2>/dev/null | grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' | sed 's/^GLIBC_//' | sort -V | tail -n1 || true)"
  [[ -n "$max" ]] || return 0
  if [[ "$(printf '%s\n%s\n' "$max" "$ceiling" | sort -V | tail -n1)" != "$ceiling" ]]; then
    echo "$file requires GLIBC_$max, above GLIBC_$ceiling portability ceiling" >&2
    exit 1
  fi
  echo "GLIBC_BASELINE_OK file=$(basename "$file") max=$max ceiling=$ceiling"
}

ceiling=2.17
[[ "$backend" == "cuda" ]] && ceiling=2.28
for f in "$release"/*; do
  [[ -f "$f" ]] || continue
  if readelf -h "$f" >/dev/null 2>&1; then
    check_glibc_ceiling "$f" "$ceiling"
  fi
done

# Verify the executable can locate its bundled sidecars inside the build
# container. CUDA --help may still need libcuda.so.1 from a real driver, so its
# structural ELF checks above are the driver-independent CI gate.
if [[ "$backend" != "cuda" ]]; then
  (cd "$release" && "./s2-$backend" --help >/dev/null)
fi

# Keep the release directory closed: only the executable, optional OpenMP,
# and (for Vulkan) our pinned loader may be published.
mapfile -t release_names < <(find "$release" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort)
for name in "${release_names[@]}"; do
  case "$name" in
    "s2-$backend"|libgomp.so.1|LICENSE-GPL-3.0.txt|LICENSE-GCC-Runtime-Library-Exception-3.1.txt|LICENSE-Crow-BSD-3-Clause.txt|LICENSE-Asio-Boost-1.0.txt|THIRD_PARTY_NOTICES.txt) ;;
    libvulkan.so.1|LICENSE-Vulkan-Loader.txt) [[ "$backend" == "vulkan" ]] || { echo "unexpected $name in $backend release" >&2; exit 1; } ;;
    *) echo "unexpected file in $backend release: $name" >&2; exit 1 ;;
  esac
done
[[ -f "$release/s2-$backend" ]] || { echo "missing s2-$backend" >&2; exit 1; }
if [[ "$backend" == "vulkan" ]]; then
  [[ -f "$release/libvulkan.so.1" ]] || { echo "missing bundled libvulkan.so.1" >&2; exit 1; }
fi

archive="s2-linux-x86_64-$backend.tar.gz"
tar_plain="${archive%.gz}"
rm -f "$archive" "$tar_plain"
# manylinux2014 is CentOS 7 based and may provide GNU tar 1.26, which does
# not support the newer directory-sorting option. release_names is already LC_ALL=C sorted above,
# so pass the files explicitly in that order and normalize metadata. gzip -n
# removes filename/timestamp metadata from the gzip header.
(
  cd "$release"
  tar --owner=0 --group=0 --numeric-owner --mtime='1970-01-01 00:00:00 UTC' \
      -cf "$root/$tar_plain" -- "${release_names[@]}"
)
gzip -n -9 "$tar_plain"
rm -rf "verify-linux-$backend"
mkdir "verify-linux-$backend"
tar -xzf "$archive" -C "verify-linux-$backend"
test -x "verify-linux-$backend/s2-$backend"

echo "LINUX_RELEASE_PASS backend=$backend glibc_ceiling=$ceiling archive=$archive"
