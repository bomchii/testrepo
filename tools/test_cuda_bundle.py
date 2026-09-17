#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, struct, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'tools/cuda_bundle.py'
spec=importlib.util.spec_from_file_location('cuda_bundle', SCRIPT)
cb=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(cb)

VALID=['s2-cuda-core.exe','cublas64_13.dll','cublasLt64_13.dll','nvJitLink_130_0.dll','zlibwapi.dll']
INVALID=['','.', '..','NUL','nul.dll','COM1.txt','COM1 .txt','a/../b','a\\b','bad:name.dll','bad<name.dll','bad>name.dll','bad"name.dll','bad|name.dll','bad?name.dll','bad*name.dll','trail.','trail ','café.dll','\x01bad.dll']
for n in VALID:
    assert cb.valid_name(n), n
for n in INVALID:
    assert not cb.valid_name(n), n

def run(payload:Path,out:Path):
    launcher=payload.parent/'launcher.exe'; launcher.write_bytes(b'MZ'+b'launcher-stub'*11)
    return subprocess.run([sys.executable,str(SCRIPT),'--launcher',str(launcher),'--payload-dir',str(payload),'--output',str(out)],text=True,capture_output=True)

with tempfile.TemporaryDirectory() as td0:
    td=Path(td0); payload=td/'payload'; payload.mkdir()
    (payload/'s2-cuda-core.exe').write_bytes(b'core'*17)
    (payload/'cublas64_13.dll').write_bytes(b'cublas'*19)
    out=td/'s2-cuda.exe'; r=run(payload,out)
    assert r.returncode==0, r.stderr+r.stdout
    data=out.read_bytes(); footer=data[-80:]
    magic,ver,res,index_off,index_size,pstart,pend,ih=struct.unpack('<8sIIQQQQ32s',footer)
    assert magic==cb.FOOTER_MAGIC and ver==1 and res==0
    assert pend==index_off and index_off+index_size==len(data)-80
    index=data[index_off:index_off+index_size]
    assert hashlib.sha256(index).digest()==ih
    imagic,iver,count,ips=struct.unpack_from('<8sIIQ',index,0)
    assert imagic==cb.INDEX_MAGIC and iver==1 and count==2 and ips==pstart

assert cb.MAX_ENTRIES == 16384

with tempfile.TemporaryDirectory() as td0:
    td=Path(td0); payload=td/'payload'; payload.mkdir()
    (payload/'s2-amd-core.exe').write_bytes(b'amd-core')
    nested=payload/'rocblas'/'library'/'gfx1100'; nested.mkdir(parents=True)
    (nested/'kernel.dat').write_bytes(b'kernel')
    launcher=td/'launcher.exe'; launcher.write_bytes(b'MZ'+b'launcher-stub'*11)
    out=td/'s2-amd.exe'
    r=subprocess.run([sys.executable,str(SCRIPT),'--launcher',str(launcher),'--payload-dir',str(payload),'--output',str(out),'--core-name','s2-amd-core.exe'],text=True,capture_output=True)
    assert r.returncode==0, r.stderr+r.stdout
    data=out.read_bytes(); footer=data[-80:]
    _,_,_,index_off,index_size,_,_,_=struct.unpack('<8sIIQQQQ32s',footer)
    index=data[index_off:index_off+index_size]
    assert b'rocblas/library/gfx1100/kernel.dat' in index

with tempfile.TemporaryDirectory() as td0:
    td=Path(td0); payload=td/'payload'; payload.mkdir()
    (payload/'s2-cuda-core.exe').write_bytes(b'x')
    (payload/'NUL.dll').write_bytes(b'x')
    r=run(payload,td/'out.exe')
    assert r.returncode!=0 and 'unsafe payload filename' in (r.stderr+r.stdout)

with tempfile.TemporaryDirectory() as td0:
    td=Path(td0); payload=td/'payload'; payload.mkdir()
    (payload/'s2-cuda-core.exe').write_bytes(b'x')
    (payload/'A.dll').write_bytes(b'x'); (payload/'a.dll').write_bytes(b'x')
    r=run(payload,td/'out.exe')
    assert r.returncode!=0 and 'case-insensitive duplicate' in (r.stderr+r.stdout)

print('CUDA_BUNDLE_TEST_PASS')
