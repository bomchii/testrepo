#!/usr/bin/env python3
"""Extract every GitHub Actions run command for real shell syntax checks.

This is intentionally not a YAML parser. actionlint validates workflow schema and
expressions; this helper preserves run-step shell text so bash -n and the
PowerShell parser can validate exactly what GitHub will hand to each shell.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

SHELL_ALIASES = {
    "bash": "bash",
    "sh": "bash",
    "pwsh": "pwsh",
    "powershell": "pwsh",
}


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def extract(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    current_step_indent = None
    current_shell = None
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        ind = indent_of(line)

        if re.match(r"^-\s+(name|uses|run):", line.lstrip()):
            # A list item at the steps level starts a new step. Nested list items
            # inside a run block are consumed below and never reach this branch.
            current_step_indent = ind
            current_shell = None

        m_shell = re.match(r"^\s*shell:\s*([^#\s]+)", line)
        if m_shell and current_step_indent is not None and ind > current_step_indent:
            raw = m_shell.group(1).split("{")[0].strip().lower()
            current_shell = SHELL_ALIASES.get(raw, raw)

        m_run = re.match(r"^(\s*)run:\s*\|[-+]?\s*(?:#.*)?$", line)
        if m_run:
            run_indent = len(m_run.group(1))
            if current_shell not in ("bash", "pwsh"):
                raise SystemExit(
                    f"{path}:{i+1}: run step must declare shell: bash or pwsh "
                    f"(got {current_shell!r})"
                )
            i += 1
            block = []
            content_indent = None
            while i < len(lines):
                candidate = lines[i]
                if candidate.strip() and indent_of(candidate) <= run_indent:
                    break
                if candidate.strip() and content_indent is None:
                    content_indent = indent_of(candidate)
                if content_indent is None:
                    block.append("")
                else:
                    # YAML literal indentation is at least content_indent. Preserve
                    # any extra indentation belonging to here-strings/heredocs.
                    block.append(candidate[content_indent:] if len(candidate) >= content_indent else "")
                i += 1
            out.append((current_shell, "\n".join(block) + "\n"))
            continue

        # Inline scalar run commands are easy to overlook because they do not
        # enter the literal-block branch above. Require an explicit supported
        # shell and feed the command to the same real shell parser. Current CI
        # intentionally keeps these commands unquoted YAML scalars so the text
        # after `run:` is exactly the command GitHub hands to the shell.
        m_inline = re.match(r"^\s*run:\s+(.+?)\s*$", line)
        if m_inline:
            if current_shell not in ("bash", "pwsh"):
                raise SystemExit(
                    f"{path}:{i+1}: inline run step must declare shell: bash or pwsh "
                    f"(got {current_shell!r})"
                )
            body = m_inline.group(1)
            if body.startswith(("'", '"')):
                raise SystemExit(
                    f"{path}:{i+1}: quoted inline run scalars are not supported by the "
                    "preflight extractor; use run: | so the real shell parser sees exact text"
                )
            out.append((current_shell, body + "\n"))
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow", type=Path)
    ap.add_argument("output_dir", type=Path)
    ns = ap.parse_args()
    blocks = extract(ns.workflow)
    if not blocks:
        raise SystemExit("no literal run blocks found")
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    for old in ns.output_dir.glob("*-run-*.*"):
        old.unlink()
    counts = {"bash": 0, "pwsh": 0}
    for shell, body in blocks:
        counts[shell] += 1
        ext = "sh" if shell == "bash" else "ps1"
        dest = ns.output_dir / f"{shell}-run-{counts[shell]:02d}.{ext}"
        dest.write_text(body, encoding="utf-8", newline="\n")
    print(f"RUN_BLOCK_EXTRACT_PASS total={len(blocks)} bash={counts['bash']} pwsh={counts['pwsh']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
