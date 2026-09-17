#!/usr/bin/env bash
set -euo pipefail

payload_dir="${1:?payload directory required}"
core_name="${2:?core executable name required}"
output="${3:?output path required}"
cache_tag="${4:?cache tag required}"

[[ -d "$payload_dir" ]] || { echo "payload directory not found: $payload_dir" >&2; exit 2; }
[[ -x "$payload_dir/$core_name" ]] || { echo "payload core is not executable: $payload_dir/$core_name" >&2; exit 2; }
case "$cache_tag" in (*[!A-Za-z0-9._-]*|'') echo "unsafe cache tag: $cache_tag" >&2; exit 2;; esac

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
archive="$work/payload.tar.gz"
# Deterministic archive. GNU tar on the CI baselines accepts these options.
(
  cd "$payload_dir"
  find . -mindepth 1 -print0 | LC_ALL=C sort -z | \
    tar --null --no-recursion --owner=0 --group=0 --numeric-owner \
        --mtime='1970-01-01 00:00:00 UTC' -T - -cf "$work/payload.tar"
)
gzip -n -9 "$work/payload.tar"
payload_sha="$(sha256sum "$archive" | awk '{print $1}')"
payload_size="$(stat -c %s "$archive")"
mkdir -p "$(dirname "$output")"
cat > "$output" <<EOF_HEADER
#!/usr/bin/env bash
set -euo pipefail
S2_PAYLOAD_SHA='$payload_sha'
S2_PAYLOAD_SIZE='$payload_size'
S2_CORE_NAME='$core_name'
S2_CACHE_TAG='$cache_tag'
marker='__S2_EMBEDDED_RUNTIME_BELOW__'
self="\${BASH_SOURCE[0]}"
cache_base="\${XDG_CACHE_HOME:-\${HOME:-/tmp}/.cache}/s2.cpp/runtime"
cache="\$cache_base/\${S2_CACHE_TAG}-\${S2_PAYLOAD_SHA}"
complete="\$cache/.complete"
payload_line="\$(awk -v m="\$marker" '\$0==m {print NR+1; exit}' "\$self")"
[[ -n "\$payload_line" ]] || { echo 's2 launcher: embedded runtime marker missing' >&2; exit 111; }
verify_cache() {
  [[ -d "\$cache" && -x "\$cache/\$S2_CORE_NAME" && -f "\$complete" ]] || return 1
  [[ "\$(cat "\$complete" 2>/dev/null || true)" == "\$S2_PAYLOAD_SHA" ]] || return 1
}
if [[ "\${1:-}" == '--runtime-info' ]]; then
  printf 'bundle_sha256: %s\\ncache: %s\\npayload_size: %s\\ncache_valid: %s\\n' "\$S2_PAYLOAD_SHA" "\$cache" "\$S2_PAYLOAD_SIZE" "\$(verify_cache && echo yes || echo no)"
  exit 0
fi
if [[ "\${1:-}" == '--clean-runtime' ]]; then
  rm -rf -- "\$cache"
  echo 'runtime cache removed'
  exit 0
fi
mkdir -p "\$cache_base"
lock="\$cache.lock"
for _ in \$(seq 1 300); do
  if mkdir "\$lock" 2>/dev/null; then
    printf '%s\n' "\$\$" > "\$lock/pid"
    trap 'rm -f "\$lock/pid" 2>/dev/null || true; rmdir "\$lock" 2>/dev/null || true' EXIT
    if ! verify_cache; then
      tmp="\$cache.tmp.\$\$"
      rm -rf -- "\$tmp"
      mkdir -p "\$tmp"
      packed="\$tmp.payload.tar.gz"
      tail -n +"\$payload_line" "\$self" > "\$packed"
      actual="\$(sha256sum "\$packed" | awk '{print \$1}')"
      [[ "\$actual" == "\$S2_PAYLOAD_SHA" ]] || { rm -rf -- "\$tmp" "\$packed"; echo 's2 launcher: embedded runtime SHA-256 mismatch' >&2; exit 111; }
      tar -xzf "\$packed" -C "\$tmp"
      rm -f -- "\$packed"
      [[ -x "\$tmp/\$S2_CORE_NAME" ]] || { rm -rf -- "\$tmp"; echo 's2 launcher: core executable missing after extraction' >&2; exit 111; }
      printf '%s\\n' "\$S2_PAYLOAD_SHA" > "\$tmp/.complete"
      rm -rf -- "\$cache"
      mv "\$tmp" "\$cache"
    fi
    rm -f "\$lock/pid"
    rmdir "\$lock" || true
    trap - EXIT
    break
  fi
  if [[ -f "\$lock/pid" ]]; then
    owner="\$(cat "\$lock/pid" 2>/dev/null || true)"
    if [[ "\$owner" =~ ^[0-9]+$ ]] && ! kill -0 "\$owner" 2>/dev/null; then
      rm -rf -- "\$lock"
      continue
    fi
  fi
  sleep 0.1
done
verify_cache || { echo 's2 launcher: runtime cache unavailable' >&2; exit 111; }
export LD_LIBRARY_PATH="\$cache:\$cache/rocm/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "\$cache/\$S2_CORE_NAME" "\$@"
exit 111
__S2_EMBEDDED_RUNTIME_BELOW__
EOF_HEADER
cat "$archive" >> "$output"
chmod +x "$output"
# Verify the outer file has exactly one marker and reports its metadata without extraction.
[[ "$(grep -a -c '^__S2_EMBEDDED_RUNTIME_BELOW__$' "$output")" == 1 ]]
"$output" --runtime-info | grep -F "bundle_sha256: $payload_sha" >/dev/null
printf 'LINUX_SINGLEFILE_PASS output=%s payload_sha256=%s payload_bytes=%s\n' "$output" "$payload_sha" "$payload_size"
