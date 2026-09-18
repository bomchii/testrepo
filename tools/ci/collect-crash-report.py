#!/usr/bin/env python3
"""Collect a compact, non-secret CI crash report after a GitHub Actions job fails."""
from __future__ import annotations
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CMAKE_PROBE_MISS = re.compile(r"^\s*--\s+(?:Performing Test|Looking for)\b.*-\s*(?:Failed|not found)\s*$", re.I)
ERR = re.compile(r"(?:CMake Error|fatal error|\berror:|\bFAILED:|undefined reference|cannot find -l|nvcc fatal|HIP error|HSA_STATUS_ERROR|Traceback|Exception|PowerShell parser rejected|SHA-256 mismatch|not found|failed)", re.I)
SAFE_ENV = [
    'GITHUB_WORKFLOW','GITHUB_JOB','GITHUB_RUN_ID','GITHUB_RUN_NUMBER','GITHUB_RUN_ATTEMPT',
    'GITHUB_EVENT_NAME','GITHUB_REF','GITHUB_SHA','GITHUB_REPOSITORY','GITHUB_WORKFLOW_SHA',
    'RUNNER_NAME','RUNNER_OS','RUNNER_ARCH','RUNNER_ENVIRONMENT','RUNNER_DEBUG',
    'ImageOS','ImageVersion','AGENT_TOOLSDIRECTORY','CUDA_PATH','ROCM_PATH','HIP_PATH','HIP_PLATFORM',
    'VULKAN_SDK','CMAKE_GENERATOR','CC','CXX','S2_MSVC_TOOLSET','VCToolsVersion','WindowsSDKVersion',
]

def clean(s: str) -> str:
    return ANSI.sub('', s).replace('\x00', '')

def memory_snapshot() -> list[str]:
    try:
        if sys.platform.startswith('linux'):
            vals = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                if ':' in line:
                    k, v = line.split(':', 1)
                    if k in {'MemTotal','MemAvailable','SwapTotal','SwapFree'}:
                        vals[k] = v.strip()
            return [f"{k}={vals.get(k,'?')}" for k in ['MemTotal','MemAvailable','SwapTotal','SwapFree']]
        if sys.platform == 'darwin':
            total = subprocess.run(['sysctl','-n','hw.memsize'], text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=5, check=False).stdout.strip()
            vm = subprocess.run(['vm_stat'], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=5, check=False).stdout.splitlines()[:8]
            return [f"hw.memsize={total}"] + [clean(x) for x in vm]
        if os.name == 'nt':
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return [f"memory_load_percent={m.dwMemoryLoad}", f"total_phys={m.ullTotalPhys}",
                        f"avail_phys={m.ullAvailPhys}", f"total_pagefile={m.ullTotalPageFile}",
                        f"avail_pagefile={m.ullAvailPageFile}"]
    except Exception as exc:
        return [f"memory probe failed: {type(exc).__name__}: {exc}"]
    return ['memory probe unavailable']


def command_version(cmd: list[str]) -> str:
    exe = shutil.which(cmd[0])
    if not exe:
        return f"{cmd[0]}: not found"
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           errors='replace', timeout=8, check=False)
        first = next((clean(x).strip() for x in p.stdout.splitlines() if x.strip()), '')
        return f"{cmd[0]}: {exe} | {first[:500]} | rc={p.returncode}"
    except Exception as exc:
        return f"{cmd[0]}: {exe} | version probe failed: {type(exc).__name__}: {exc}"

def first_error(lines: list[str]) -> tuple[int, str]:
    for i, line in enumerate(lines):
        cleaned = clean(line)
        if CMAKE_PROBE_MISS.search(cleaned):
            continue
        if ERR.search(cleaned):
            return i, cleaned.strip()
    if lines:
        return len(lines)-1, clean(lines[-1]).strip()
    return 0, 'No captured log line was available.'

def tail_context(path: Path, before: int = 8, after: int = 14, tail: int = 35) -> str:
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except Exception as exc:
        return f"[cannot read {path}: {exc}]"
    idx, headline = first_error(lines)
    start, end = max(0, idx-before), min(len(lines), idx+after+1)
    excerpt = lines[start:end]
    out = [f"FILE: {path.as_posix()} ({path.stat().st_size} bytes, {len(lines)} lines)", f"FIRST MATCH: {headline}", "CONTEXT:"]
    out.extend(clean(x) for x in excerpt)
    out.append("TAIL:")
    out.extend(clean(x) for x in lines[-tail:])
    return '\n'.join(out)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--logs', default='ci-logs')
    ap.add_argument('--output', default='ci-logs/job-crash-report.txt')
    ns = ap.parse_args()
    out = Path(ns.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = Path(ns.logs)
    now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    du = shutil.disk_usage(Path.cwd())
    run_url = ''
    if os.environ.get('GITHUB_SERVER_URL') and os.environ.get('GITHUB_REPOSITORY') and os.environ.get('GITHUB_RUN_ID'):
        run_url = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    parts = [
        '========== S2 CI CRASH REPORT ==========',
        f'Label: {ns.label}', f'UTC: {now}', f'Run URL: {run_url or "n/a"}',
        f'CWD: {Path.cwd()}', f'Python: {sys.version.replace(chr(10), " ")}',
        f'Platform: {platform.platform()}', f'Machine: {platform.machine()}', f'CPU count: {os.cpu_count()}',
        f'Disk bytes: total={du.total} used={du.used} free={du.free}',
        '', '--- MEMORY ---', *memory_snapshot(),
        '', '--- SAFE GITHUB/RUNNER ENV ---'
    ]
    for key in SAFE_ENV:
        val = os.environ.get(key)
        if val:
            parts.append(f'{key}={val}')
    parts += ['', '--- GIT ---']
    parts.append(command_version(['git','rev-parse','HEAD']))
    parts.append(command_version(['git','status','--short']))
    parts += ['', '--- TOOLCHAIN ---']
    probes = [
        ['cmake','--version'], ['ninja','--version'], ['git','--version'],
        ['python','--version'], ['python3','--version'], ['clang','--version'], ['clang++','--version'],
        ['gcc','--version'], ['g++','--version'], ['cl.exe'], ['nvcc','--version'], ['hipcc','--version'],
    ]
    parts.extend(command_version(x) for x in probes)
    parts += ['', '--- CAPTURED LOGS ---']
    candidates = []
    if logs.exists():
        candidates = sorted((p for p in logs.rglob('*') if p.is_file()), key=lambda p: p.as_posix())
    if not candidates:
        parts.append('No ci-logs files were present.')
    else:
        for p in candidates:
            # Bound the report itself: detailed context for text logs, metadata for binary/huge files.
            if p.stat().st_size > 20 * 1024 * 1024:
                parts.append(f'FILE: {p.as_posix()} ({p.stat().st_size} bytes) [too large for inline crash report]')
                continue
            parts += ['', tail_context(p)]
    parts += ['', '========== END S2 CI CRASH REPORT ==========']
    text = '\n'.join(parts) + '\n'
    out.write_text(text, encoding='utf-8', errors='replace')
    print(text)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8', errors='replace') as f:
            f.write(f"\n## 🧯 Crash report: {ns.label}\n\n")
            f.write(f"Run: `{os.environ.get('GITHUB_RUN_ID','n/a')}` · attempt `{os.environ.get('GITHUB_RUN_ATTEMPT','n/a')}` · `{os.environ.get('RUNNER_OS','?')}/{os.environ.get('RUNNER_ARCH','?')}`\n\n")
            if run_url:
                f.write(f"Run URL: {run_url}\n\n")
            f.write(f"Full crash report artifact: `{out.as_posix()}`\n")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
