#!/usr/bin/env python3
"""Differential test for s2.cpp tokenizer against tokenizer.json semantics.

Reference implementation deliberately uses Python's regex Unicode properties and
unicodedata NFC rather than s2.cpp's generated tables.
"""
from __future__ import annotations

import json
import random
import regex
import heapq
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer.json"
DRIVER = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/data/test_v75_tokenizer")

j = json.loads(TOKENIZER.read_text(encoding="utf-8"))
pattern = j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
rx = regex.compile(pattern)
vocab = {k: int(v) for k, v in j["model"]["vocab"].items()}
merge_rank = {tuple(m): i for i, m in enumerate(j["model"]["merges"])}
added = [(x["content"], int(x["id"])) for x in j["added_tokens"]]

# GPT-2 ByteLevel byte -> unicode mapping.
bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}

class TrieNode:
    __slots__ = ("next", "tok")
    def __init__(self):
        self.next = {}
        self.tok = None

trie = TrieNode()
for content, tid in added:
    node = trie
    for ch in content:
        node = node.next.setdefault(ch, TrieNode())
    node.tok = (content, tid)


def bytelevel(s: str) -> str:
    return "".join(byte_encoder[b] for b in s.encode("utf-8"))


def bpe(piece: str) -> list[int]:
    bl = bytelevel(piece)
    if bl in vocab:
        return [vocab[bl]]
    values = list(bl)
    n = len(values)
    if n == 0:
        return []
    prev = [i - 1 for i in range(n)]
    nxt = [i + 1 for i in range(n)]
    nxt[-1] = -1
    alive = [True] * n
    version = [0] * n
    heap = []

    def push_pair(i: int):
        if i < 0 or not alive[i]:
            return
        j = nxt[i]
        if j < 0 or not alive[j]:
            return
        rank = merge_rank.get((values[i], values[j]))
        if rank is not None:
            # Original index is a stable left-to-right tiebreak for repeated
            # occurrences of the same merge rank.
            heapq.heappush(heap, (rank, i, version[i], j, version[j]))

    for i in range(n - 1):
        push_pair(i)

    while heap:
        rank, i, vi, j, vj = heapq.heappop(heap)
        if not alive[i] or not alive[j] or version[i] != vi or version[j] != vj or nxt[i] != j:
            continue
        if merge_rank.get((values[i], values[j])) != rank:
            continue
        left = prev[i]
        right = nxt[j]
        values[i] += values[j]
        version[i] += 1
        alive[j] = False
        version[j] += 1
        nxt[i] = right
        if right >= 0:
            prev[right] = i
        if left >= 0:
            push_pair(left)
        push_pair(i)

    out = []
    i = 0
    while i >= 0 and i < n:
        if alive[i]:
            try:
                out.append(vocab[values[i]])
            except KeyError as exc:
                raise AssertionError(f"missing BPE symbol {exc!r}") from exc
        i = nxt[i]
    return out


def encode_plain(s: str) -> list[int]:
    if not s:
        return []
    s = unicodedata.normalize("NFC", s)
    pieces = [m.group(0) for m in rx.finditer(s)]
    if "".join(pieces) != s:
        raise AssertionError(("regex did not cover input", repr(s), pieces))
    out = []
    for p in pieces:
        out.extend(bpe(p))
    return out


def encode_ref(s: str) -> list[int]:
    out = []
    plain_start = 0
    pos = 0
    while pos < len(s):
        node = trie
        best = None
        jpos = pos
        while jpos < len(s) and s[jpos] in node.next:
            node = node.next[s[jpos]]
            jpos += 1
            if node.tok is not None:
                best = (jpos, node.tok[1])
        if best is not None:
            out.extend(encode_plain(s[plain_start:pos]))
            out.append(best[1])
            pos = best[0]
            plain_start = pos
        else:
            pos += 1
    out.extend(encode_plain(s[plain_start:]))
    return out

fixed = [
    "Hello world!", "don't we're I'LL", "café", "cafe\u0301", "A\u030a", "Å",
    "مرحبا بالعالم؟", "مَرْحَبًا بِالْعَالَمِ؟", "فارسی خوب است۔", "اردو ٹھیک ہے۔",
    "שלום עוֹלָם", "שָׁלוֹם", "Привет, мир!", "Україна — це Європа.", "А. С. Пушкин",
    "नमस्ते दुनिया।", "हिन्दी भाषा", "বাংলা ভাষা।", "தமிழ் மொழி", "తెలుగు భాష",
    "ಕನ್ನಡ ಭಾಷೆ", "മലയാളം ഭാഷ", "සිංහල භාෂාව", "ภาษาไทย", "ພາສາລາວ",
    "မြန်မာဘာသာ", "ខ្មែរ", "བོད་ཡིག", "ქართული", "Հայերեն",
    "Ελληνικά; ερώτηση", "中文没有空格。下一句。", "日本語です。次です。", "한국어입니다.다음.",
    "Tiếng Việt", "tiếng Việt", "👩\u200d💻🙂🏳️\u200d🌈", "a\u200fb\u2067c\u2069",
    "<|speaker:0|>你好。<|speaker:1|>مرحبا؟", "[whisper in small voice] hello",
    "<think>reason</think>", "x<tool_call>y</tool_call>z", "<|semantic:0|><|semantic:4095|>",
    "\n\nHello\r\nworld\t ", "  punctuation!!!\n", "1234567890", "１２３", "٣٤٥", "१२३",
]

alphabets = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '.,!?;:-\n",
    "你好世界今天明天语言测试。，！？……",
    "こんにちは世界ですテスト。！？カタカナ",
    "안녕하세요세계테스트입니다.?!",
    "مرحباالعالمكيفحالكَُِّ؟،۔ فارسیاردو",
    "אבגדהוזחטיךכלםמןנסעףפץצקרשתְֱֲִֵֶַָֹֻּׁׂ",
    "абвгдежзийклмнопрстуфхцчшщыэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЭЮЯ .",
    "अआइईउऊकखगघचछजझटठडढतथदधनपफबभमयरलवशषसहािीुूेैोौंः्।॥",
    "অআইঈউকখগঘচছজঝটঠডঢতথদধনপফবভমযরলশষসহািীুূেৈোৌংঃ্।",
    "αβγδεζηθικλμνξοπρστυφχψωάέήίόύώ ;;.",
    "Հայերենաբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցուփքևօֆ՞։",
    "ཀཁགངཅཆཇཉཏཐདནཔཕབམཙཚཛཝཞཟའཡརལཤསཧཨ་།",
]
marks = ["", "\u0301", "\u0308", "\u064e", "\u0651", "\u093c", "\u094d", "\u200c", "\u200d", "\u200f"]
tags = ["", "<|speaker:0|>", "<|speaker:1|>", "[whisper]", "[低声说]", "<think>", "</think>"]

rng = random.Random(0x753)
cases = list(fixed)
while len(cases) < 622:
    alpha = rng.choice(alphabets)
    nchar = rng.randint(1, 90)
    s = "".join(rng.choice(alpha) for _ in range(nchar))
    if rng.random() < 0.4:
        p = rng.randrange(len(s) + 1)
        s = s[:p] + rng.choice(marks) + s[p:]
    if rng.random() < 0.25:
        p = rng.randrange(len(s) + 1)
        s = s[:p] + rng.choice(tags) + s[p:]
    cases.append(s)

payload = "".join(s.encode("utf-8").hex() + "\n" for s in cases)
proc = subprocess.run([str(DRIVER), str(TOKENIZER)], input=payload, stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE, text=True, check=False)
if proc.returncode != 0:
    sys.stderr.write(proc.stderr)
    raise SystemExit(f"driver failed rc={proc.returncode}")
lines = proc.stdout.splitlines()
if len(lines) != len(cases):
    raise SystemExit(f"driver line count mismatch: {len(lines)} != {len(cases)}")

mismatches = []
for i, (s, line) in enumerate(zip(cases, lines)):
    got = [] if line == "" else [int(x) for x in line.split(",")]
    ref = encode_ref(s)
    if got != ref:
        mismatches.append((i, s, ref, got))
        if len(mismatches) >= 8:
            break
if mismatches:
    for i, s, ref, got in mismatches:
        print("MISMATCH", i, repr(s))
        print("REF", ref)
        print("GOT", got)
    raise SystemExit(1)
print(f"TOKENIZER_DIFF_PASS {len(cases)}/{len(cases)} Unicode={unicodedata.unidata_version}")
