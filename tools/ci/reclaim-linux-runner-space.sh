#!/usr/bin/env bash
set -euo pipefail

# GitHub's standard Ubuntu runners expose a relatively small writable SSD.
# Linux release builds run in Docker and CUDA/Vulkan toolchains are large, so
# remove only heavyweight host SDKs that these jobs never use. Do not remove
# /opt/hostedtoolcache: later JavaScript Actions still need the runner tooling.
echo 'Disk space before cleanup:'
df -h /
docker system df || true

for path in \
  /usr/local/lib/android \
  /usr/local/.ghcup \
  /usr/share/dotnet
do
  if [[ -e "$path" ]]; then
    echo "Removing unused host toolchain: $path"
    sudo rm -rf -- "$path"
  fi
done

sudo apt-get clean || true
sudo rm -rf /var/lib/apt/lists/*
docker system prune -af || true

echo 'Disk space after cleanup:'
df -h /
docker system df || true
