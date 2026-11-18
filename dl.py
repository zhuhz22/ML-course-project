import json, os, urllib.request, urllib.parse
TOK=os.environ.get("KAGGLE_KEY","<YOUR_KAGGLE_API_TOKEN>")
COMP="nfl-big-data-bowl-2026-prediction"
def api(url):
    r=urllib.request.Request(url, headers={"Authorization":f"Bearer {TOK}"})
    return json.load(urllib.request.urlopen(r, timeout=120))
files=[]
tok=None
while True:
    u=f"https://www.kaggle.com/api/v1/competitions/data/list/{COMP}"
    if tok: u+="?pageToken="+urllib.parse.quote(tok)
    d=api(u); files+=d['files']; tok=d.get('nextPageTokenNullable')
    if not tok: break
print(len(files),"files", flush=True)
ok=0
for f in files:
    name=f['name']; dest=os.path.join('data/raw',name)
    if os.path.exists(dest) and os.path.getsize(dest)==f['totalBytes']:
        ok+=1; continue
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    q=urllib.parse.quote(name, safe='')
    url=f"https://www.kaggle.com/api/v1/competitions/data/download/{COMP}/{q}"
    r=urllib.request.Request(url, headers={"Authorization":f"Bearer {TOK}"})
    with urllib.request.urlopen(r, timeout=600) as resp, open(dest,'wb') as fh:
        while True:
            b=resp.read(1<<20)
            if not b: break
            fh.write(b)
    print(name, os.path.getsize(dest), flush=True); ok+=1
print("DONE", ok)
