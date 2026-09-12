#!/usr/bin/env python3
"""Generate the C++ Sentence_Break Extend/Format table from Unicode UCD data.

Usage:
  python tools/gen_sentence_break_ignore.py SentenceBreakProperty.txt \
      --output src/s2_sentence_break_ignore.inc

The input should be a version-pinned Unicode UCD auxiliary
SentenceBreakProperty.txt. Only Extend and Format are emitted. This keeps
runtime behavior independent of host locale/ICU and makes the source snapshot
auditable/reproducible.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re

LINE_RE = re.compile(r"^\s*([0-9A-Fa-f]+)(?:\.\.([0-9A-Fa-f]+))?\s*;\s*(Extend|Format)\b")

def parse(path: Path):
    points = {"Extend": set(), "Format": set()}
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(raw)
        if not m:
            continue
        lo = int(m.group(1), 16)
        hi = int(m.group(2) or m.group(1), 16)
        if hi < lo or hi > 0x10FFFF:
            raise ValueError(f"invalid UCD range: {raw}")
        points[m.group(3)].update(range(lo, hi + 1))
    if not points["Extend"] or not points["Format"]:
        raise ValueError("input does not contain both Extend and Format")
    if points["Extend"] & points["Format"]:
        raise ValueError("Sentence_Break properties unexpectedly overlap")
    return points

def merge(points):
    vals = sorted(points)
    out = []
    lo = hi = vals[0]
    for cp in vals[1:]:
        if cp == hi + 1:
            hi = cp
        else:
            out.append((lo, hi))
            lo = hi = cp
    out.append((lo, hi))
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expect-extend", type=int)
    ap.add_argument("--expect-format", type=int)
    args = ap.parse_args()
    p = parse(args.input)
    ne, nf = len(p["Extend"]), len(p["Format"])
    if args.expect_extend is not None and ne != args.expect_extend:
        raise SystemExit(f"Extend count {ne} != expected {args.expect_extend}")
    if args.expect_format is not None and nf != args.expect_format:
        raise SystemExit(f"Format count {nf} != expected {args.expect_format}")
    ranges = merge(p["Extend"] | p["Format"])
    lines = [
        "// Generated from Unicode SentenceBreakProperty.txt; do not hand-edit.",
        f"// Extend={ne}, Format={nf}, union={ne + nf} code points.",
        "static constexpr UnicodeRange kSentenceIgnoreRanges[] = {",
    ]
    lines += [f"    {{0x{lo:X}u, 0x{hi:X}u}}," for lo, hi in ranges]
    lines += ["};", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
