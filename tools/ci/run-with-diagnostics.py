#!/usr/bin/env python3
"""Run a command, stream its output, and emit a concise GitHub Actions failure digest."""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
import time
import shlex
import shutil
import platform
from pathlib import Path

PATTERNS = [
    re.compile(p, re.I) for p in [
        r"\bCMake Error\b",
        r"\b(?:fatal )?error\s*(?:C\d+|LNK\d+|:)",
        r"\bMSB\d{4}\b.*\berror\b",
        r"\bnvcc fatal\b",
        r"\bHIP error\b",
        r"\bHSA_STATUS_ERROR\b",
        r"\brocBLAS(?: error| status| failure)\b",
        r"\bhip(?:cc|config).*\berror\b",
        r"clang(?:\+\+)?: error: cannot find ROCm device librar",
        r"\bROCm.*(?:missing|not found|failed)\b",
        r"\bFAILED:\s",
        r"\bundefined reference\b",
        r"\bcannot find -l\S+",
        r"\bld(?:\.exe)?: .*error",
        r"\bcollect2: error\b",
        r"\bNinja:.*build stopped\b",
        r"\bPowerShell parser rejected\b",
        r"\bSHA-256 mismatch\b",
        r"\bTraceback \(most recent call last\)",
        r"\bException\b",
        r"\bError:\s",
        r"\berror:\s",
        r"\bfailed\b",
    ]
]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def gh_escape(s: str) -> str:
    return s.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')


def find_error(lines: list[str]) -> int:
    clean = [ANSI.sub('', x) for x in lines]
    for pat in PATTERNS:
        for i, line in enumerate(clean):
            if pat.search(line):
                return i
    return max(0, len(lines) - 1)


def append_summary(label: str, rc: int, excerpt: list[str], log_path: Path, command: str = '', elapsed: float = 0.0) -> None:
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if not summary:
        return
    with open(summary, 'a', encoding='utf-8', errors='replace') as f:
        f.write(f"\n## ❌ {label} failed\n\n")
        f.write(f"**Exit code:** `{rc}`  \n**Elapsed:** `{elapsed:.2f}s`  \n**Runner:** `{os.environ.get('RUNNER_OS','?')}/{os.environ.get('RUNNER_ARCH','?')}`  \n**Full log:** `{log_path.as_posix()}`\n\n")
        if command:
            f.write("### Command\n\n```text\n" + command + "\n```\n\n")
        f.write("### First useful error context\n\n```text\n")
        for line in excerpt:
            f.write(ANSI.sub('', line).rstrip('\n') + '\n')
        f.write("```\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--before', type=int, default=12)
    ap.add_argument('--after', type=int, default=18)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    ns = ap.parse_args()
    cmd = ns.command
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        ap.error('missing command after --')

    log_path = Path(ns.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rendered_cmd = subprocess.list2cmdline(cmd) if os.name == 'nt' else shlex.join(cmd)
    print(f"::group::{ns.label}", flush=True)
    print('+ ' + rendered_cmd, flush=True)
    lines: list[str] = []
    try:
        with log_path.open('w', encoding='utf-8', errors='replace') as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, errors='replace', bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                lines.append(line)
            rc = proc.wait()
    except FileNotFoundError as exc:
        rc = 127
        line = f"runner error: {exc}\n"
        lines.append(line)
        log_path.write_text(line, encoding='utf-8')
        sys.stdout.write(line)
    finally:
        print('::endgroup::', flush=True)

    elapsed = time.monotonic() - started
    if rc == 0:
        print(f"DIAGNOSTIC_RUN_PASS label={ns.label} rc=0 seconds={elapsed:.2f} lines={len(lines)} bytes={log_path.stat().st_size if log_path.exists() else 0} log={log_path}")
        return 0

    idx = find_error(lines)
    start = max(0, idx - ns.before)
    end = min(len(lines), idx + ns.after + 1)
    excerpt = lines[start:end] or [f"Command exited with code {rc}\n"]
    headline = ANSI.sub('', lines[idx]).strip() if lines else f"Command exited with code {rc}"
    if len(headline) > 500:
        headline = headline[:497] + '...'

    print('\n========== FAILURE DIGEST ==========', file=sys.stderr)
    print(f"Stage: {ns.label}", file=sys.stderr)
    print(f"Command: {rendered_cmd}", file=sys.stderr)
    print(f"Working directory: {Path.cwd()}", file=sys.stderr)
    print(f"Exit code: {rc}", file=sys.stderr)
    if rc < 0:
        print(f"Terminated by signal: {-rc}", file=sys.stderr)
    print(f"Elapsed seconds: {elapsed:.2f}", file=sys.stderr)
    print(f"Runner: {os.environ.get('RUNNER_OS','?')}/{os.environ.get('RUNNER_ARCH','?')} name={os.environ.get('RUNNER_NAME','?')}", file=sys.stderr)
    print(f"Platform: {platform.platform()}", file=sys.stderr)
    try:
        du = shutil.disk_usage(Path.cwd())
        print(f"Disk free bytes: {du.free}", file=sys.stderr)
    except Exception:
        pass
    print(f"Captured lines: {len(lines)}", file=sys.stderr)
    print(f"Full log: {log_path} ({log_path.stat().st_size if log_path.exists() else 0} bytes)", file=sys.stderr)
    print('First useful error context:', file=sys.stderr)
    for line in excerpt:
        sys.stderr.write(ANSI.sub('', line))
    print('--- Last 25 log lines ---', file=sys.stderr)
    for line in lines[-25:]:
        sys.stderr.write(ANSI.sub('', line))
    print('========== END FAILURE DIGEST ======', file=sys.stderr)
    print(f"::error title={gh_escape(ns.label)} failed::{gh_escape(headline)}")
    append_summary(ns.label, rc, excerpt, log_path, rendered_cmd, elapsed)
    return rc

if __name__ == '__main__':
    raise SystemExit(main())
