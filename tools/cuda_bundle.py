#!/usr/bin/env python3
"""Append a validated CUDA runtime payload to the native s2 CUDA launcher."""
from __future__ import annotations
import argparse, hashlib, os, struct, tempfile
from pathlib import Path

INDEX_MAGIC=b'S2CIDX01'
FOOTER_MAGIC=b'S2CEND01'
VERSION=1
MAX_ENTRIES=16384
DEVICE={"con","prn","aux","nul","clock$",*(f"com{i}" for i in range(1,10)),*(f"lpt{i}" for i in range(1,10))}

def valid_name(name:str)->bool:
    try:
        raw=name.encode('ascii')
    except UnicodeEncodeError:
        return False
    if not raw or len(raw)>1024 or name.startswith('/') or name.endswith('/'):
        return False
    if '\\' in name:
        return False
    parts=name.split('/')
    if any(not part or part in ('.','..') or len(part)>240 or part[-1] in '. ' for part in parts):
        return False
    for part in parts:
        if any(ord(c)<0x20 or ord(c)>0x7e or c in '<>:"\\|?*' for c in part):
            return False
        stem=part.split('.',1)[0].rstrip(' .').casefold()
        if stem in DEVICE:
            return False
    return True

def hfile(path:Path)->bytes:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.digest()

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--launcher',required=True,type=Path)
    ap.add_argument('--payload-dir',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--core-name',default='s2-cuda-core.exe')
    args=ap.parse_args()
    files=sorted((p for p in args.payload_dir.rglob('*') if p.is_file()),key=lambda p:p.relative_to(args.payload_dir).as_posix().casefold())
    if not files: raise SystemExit('empty payload')
    if len(files)>MAX_ENTRIES: raise SystemExit(f'too many payload files: {len(files)} > {MAX_ENTRIES}')
    if not valid_name(args.core_name): raise SystemExit(f'unsafe core name: {args.core_name!r}')
    if not any(p.relative_to(args.payload_dir).as_posix().casefold()==args.core_name.casefold() for p in files):
        raise SystemExit(f'payload must contain {args.core_name}')
    seen=set()
    for p in files:
        rel=p.relative_to(args.payload_dir).as_posix()
        if not valid_name(rel): raise SystemExit(f'unsafe payload filename: {rel!r}')
        k=rel.casefold()
        if k in seen: raise SystemExit(f'case-insensitive duplicate: {p.name}')
        seen.add(k)

    launcher=args.launcher.read_bytes()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=args.output.name+'.',suffix='.tmp',dir=str(args.output.parent))
    os.close(fd); tmp=Path(tmp)
    try:
        entries=[]
        with tmp.open('wb') as out:
            out.write(launcher)
            payload_start=out.tell()
            for p in files:
                off=out.tell(); size=0; h=hashlib.sha256()
                with p.open('rb') as f:
                    while True:
                        b=f.read(1<<20)
                        if not b: break
                        out.write(b); h.update(b); size+=len(b)
                entries.append((p.relative_to(args.payload_dir).as_posix(),off,size,h.digest()))
            payload_end=out.tell()
            index=bytearray(struct.pack('<8sIIQ',INDEX_MAGIC,VERSION,len(entries),payload_start))
            for name,off,size,digest in entries:
                nb=name.encode('utf-8')
                index += struct.pack('<HHQQ32s',len(nb),0,off,size,digest)+nb
            index_off=out.tell(); out.write(index)
            index_hash=hashlib.sha256(index).digest()
            footer=struct.pack('<8sIIQQQQ32s',FOOTER_MAGIC,VERSION,0,index_off,len(index),payload_start,payload_end,index_hash)
            out.write(footer)
            out.flush(); os.fsync(out.fileno())
        os.replace(tmp,args.output)
    finally:
        if tmp.exists(): tmp.unlink()
    print(f'RUNTIME_BUNDLE_ID={index_hash.hex()}')
    print(f'RUNTIME_BUNDLE_FILES={len(entries)}')
    print(f'RUNTIME_BUNDLE_SIZE={args.output.stat().st_size}')
    for name,off,size,digest in entries:
        print(f'  {name} size={size} sha256={digest.hex()} offset={off}')
    return 0
if __name__=='__main__': raise SystemExit(main())
