#!/usr/bin/env bash
set -euo pipefail

mode="${1:-common}"
case "$mode" in
  common|vulkan) ;;
  *) echo "usage: $0 [common|vulkan]" >&2; exit 2 ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
deps="$root/build-linux-deps"
mkdir -p "$deps"

fetch_tar() {
  local url="$1" out="$2" dir="$3" expected_sha256="${4:-}"
  rm -rf "$dir" "$out"
  curl -fL --retry 3 --retry-delay 2 "$url" -o "$out"
  if [[ -n "$expected_sha256" ]]; then
    printf '%s  %s\n' "$expected_sha256" "$out" | sha256sum -c -
  fi
  mkdir -p "$dir"
  tar -xzf "$out" -C "$dir" --strip-components=1
  rm -f "$out"
}

crow_commit="ae0fef0ee67eec897e401321b99b6dd7cfbdc155"
asio_sha256="755bd7f85a4b269c67ae0ea254907c078d408cce8e1a352ad2ed664d233780e8"

if [[ ! -f "$deps/crow/include/crow.h" ]]; then
  # Crow 1.3.4 fixes the fragmented-WebSocket payload-limit bypass and other
  # security issues. Use the immutable release commit rather than a movable tag.
  fetch_tar \
    "https://github.com/CrowCpp/Crow/archive/${crow_commit}.tar.gz" \
    "$deps/crow.tar.gz" "$deps/crow-src"
  grep -Eq 'VERSION[[:space:]]+1\.3\.4' "$deps/crow-src/CMakeLists.txt"
  mkdir -p "$deps/crow"
  cp -a "$deps/crow-src/include" "$deps/crow/"
fi

if [[ ! -f "$deps/asio/include/asio.hpp" ]]; then
  fetch_tar \
    "https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-30-2.tar.gz" \
    "$deps/asio.tar.gz" "$deps/asio-src" "$asio_sha256"
  mkdir -p "$deps/asio"
  cp -a "$deps/asio-src/asio/include" "$deps/asio/"
fi

# Crow is built without CROW_ENABLE_SSL. Keep the same HTTP/WebSocket-only
# Asio layout used by the Windows release builds so OpenSSL is not a runtime
# dependency of the portable binaries.
mkdir -p "$deps/asio/include/asio/ssl"
cat > "$deps/asio/include/asio/ssl.hpp" <<'STUB'
#pragma once
// SSL disabled: CROW_ENABLE_SSL is intentionally not defined.
STUB
for stub in context.hpp stream.hpp error.hpp rfc2818_verification.hpp verify_mode.hpp; do
  cat > "$deps/asio/include/asio/ssl/$stub" <<'STUB'
#pragma once
// SSL disabled.
STUB
done

python3 "$root/tools/embed_tokenizer.py" \
  "$root/tokenizer.json" "$root/src/tokenizer_data.cpp" "$root/src/tokenizer_data.h"

if [[ "$mode" == "vulkan" ]]; then
  vk_tag="vulkan-sdk-1.4.357.0"
  shaderc_tag="v2026.3"
  headers_commit="e3b1eec08173d6b825cd3ac88c885a63b621504a"
  loader_commit="5f157b62e333c63260d05d81bf66faa216ab0fb8"
  shaderc_commit="2c8cae778eec0283b44acbe7ed1a386865d78799"

  clone_verified() {
    local url="$1" tag="$2" expected="$3" dest="$4"
    rm -rf "$dest"
    git clone --quiet --depth 1 --branch "$tag" "$url" "$dest"
    local actual
    actual="$(git -C "$dest" rev-parse HEAD)"
    if [[ "$actual" != "$expected" ]]; then
      echo "pinned source mismatch for $url tag $tag: expected $expected got $actual" >&2
      exit 1
    fi
    echo "PIN_OK repo=$url tag=$tag commit=$actual"
  }

  clone_verified https://github.com/KhronosGroup/Vulkan-Headers.git "$vk_tag" "$headers_commit" "$deps/Vulkan-Headers"
  clone_verified https://github.com/KhronosGroup/Vulkan-Loader.git  "$vk_tag" "$loader_commit"  "$deps/Vulkan-Loader"
  clone_verified https://github.com/google/shaderc.git "$shaderc_tag" "$shaderc_commit" "$deps/shaderc"

  # This pinned Loader asks for PkgConfig unconditionally on Linux even though
  # it only uses it for XCB/Xlib/XRandR/DirectFB discovery. Our compute-only
  # loader disables every one of those WSI backends. Make that requirement
  # conditional so the glibc-2.17 build does not need CentOS 7/EOL yum repos.
  python3 - "$deps/Vulkan-Loader/CMakeLists.txt" <<'PY_PATCH'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text(encoding='utf-8')
needle = '    find_package(PkgConfig REQUIRED QUIET) # Use PkgConfig to find Linux system libraries\n'
replacement = (
    '    if(BUILD_WSI_XCB_SUPPORT OR BUILD_WSI_XLIB_SUPPORT OR BUILD_WSI_DIRECTFB_SUPPORT)\n'
    '        find_package(PkgConfig REQUIRED QUIET) # only needed by enabled WSI backends\n'
    '    endif()\n'
)
if s.count(needle) != 1:
    raise SystemExit('unexpected Vulkan-Loader PkgConfig line; pinned source changed')
p.write_text(s.replace(needle, replacement), encoding='utf-8')
PY_PATCH

  # shaderc pins tested glslang/SPIR-V revisions in known_good.json.
  (cd "$deps/shaderc" && python3 utils/git-sync-deps)
fi

test -f "$deps/crow/include/crow.h"
test -f "$deps/asio/include/asio.hpp"
test -f "$root/src/tokenizer_data.cpp"
test -f "$root/src/tokenizer_data.h"
if [[ "$mode" == "vulkan" ]]; then
  test -f "$deps/Vulkan-Headers/include/vulkan/vulkan.h"
  test -f "$deps/Vulkan-Loader/CMakeLists.txt"
  test -f "$deps/shaderc/glslc/CMakeLists.txt"
fi

echo "LINUX_DEPS_READY mode=$mode"
