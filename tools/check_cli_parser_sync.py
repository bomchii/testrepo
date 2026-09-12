#!/usr/bin/env python3
"""Fail if the standalone CLI parser regression test drifts from src/main.cpp."""
from pathlib import Path
import re, sys

ROOT=Path(__file__).resolve().parents[1]

def parser_region(path: Path, marker: str) -> str:
    text=path.read_text(encoding='utf-8')
    start=text.index(marker)
    end=text.index('} else if (arg == "--help" || arg == "-h")', start)
    return text[start:end]

def options(region: str) -> set[str]:
    return set(re.findall(r'arg\s*==\s*"(-{1,2}[^"\\]+)"', region))

main=options(parser_region(ROOT/'src/main.cpp', '// --- Parse des arguments ---'))
test=options(parser_region(ROOT/'src/test_cli_parser.cpp', 'int parse_args('))
if main != test:
    print('CLI parser/test option drift detected', file=sys.stderr)
    print('missing in test:', sorted(main-test), file=sys.stderr)
    print('extra in test:', sorted(test-main), file=sys.stderr)
    raise SystemExit(1)
print(f'CLI_PARSER_SYNC_PASS options={len(main)+2} (including -h/--help)')
