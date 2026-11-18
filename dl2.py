import json, os, urllib.request, urllib.parse, sys
from concurrent.futures import ThreadPoolExecutor
TOK=os.environ.get("KAGGLE_KEY","<YOUR_KAGGLE_API_TOKEN>")
COMP="nfl-big-data-bowl-2026-prediction"
ROOT="data/raw"
files=json.load(open('data/raw/filelist.json'))
CHUNK=5_000_000
jobs=[]
for f in files:
    dest=os.path.join(ROOT,f['name']); sz=f['totalBytes']
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest)==sz: continue
    if sz==0:
        open(dest,'wb').close(); continue
    q=urllib.parse.quote(f['name'], safe='')
    url=f"https://www.kaggle.com/api/v1/competitions/data/download/{COMP}/{q}"
    n=max(1,(sz+CHUNK-1)//CHUNK)
    for i in range(n):
        s=i*CHUNK; e=min(s+CHUNK, sz)-1
        jobs.append((url, dest, s, e, sz))
jobs.sort(key=lambda j:-j[4])
print("segments:", len(jobs), flush=True)
def fetch(j):
    url,dest,s,e,sz=j
    part=f"{dest}.part{s}"
    if os.path.exists(part) and os.path.getsize(part)==e-s+1: return 0
    for attempt in range(6):
        try:
            r=urllib.request.Request(url, headers={"Authorization":f"Bearer {TOK}","Range":f"bytes={s}-{e}"})
            with urllib.request.urlopen(r, timeout=180) as resp:
                data=resp.read()
            if len(data)!=e-s+1: raise IOError(f"short {len(data)} vs {e-s+1}")
            open(part,'wb').write(data); return len(data)
        except Exception as ex:
            if attempt==5: print("FAIL",dest,s,ex,flush=True); raise
    return 0
done=0
with ThreadPoolExecutor(max_workers=64) as ex:
    for n in ex.map(fetch, jobs):
        done+=n
        if done % (50_000_000) < n: print(f"{done/1e6:.0f} MB", flush=True)
# assemble
for f in files:
    dest=os.path.join(ROOT,f['name']); sz=f['totalBytes']
    if sz==0 or (os.path.exists(dest) and os.path.getsize(dest)==sz): continue
    n=max(1,(sz+CHUNK-1)//CHUNK)
    with open(dest,'wb') as out:
        for i in range(n):
            out.write(open(f"{dest}.part{i*CHUNK}",'rb').read())
    for i in range(n): os.remove(f"{dest}.part{i*CHUNK}")
    assert os.path.getsize(dest)==sz, (dest, os.path.getsize(dest), sz)
    print("assembled", f['name'], flush=True)
print("ALL DONE")
