"""pe26: download CDLI thumbnail photographs (tn_photo, ~300 px) and record EXIF batch info.
Images stay in the scratchpad; only per-tablet metadata goes to data/pe26_ckpt.
usage: python3 pe26_download.py idlist.txt out_meta.json  (2 workers max)"""
import sys, os, json, time, subprocess
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/pe26/tn'
os.makedirs(SCR, exist_ok=True)
ids = [l.split()[0] for l in open(sys.argv[1]) if l.strip()]
out = sys.argv[2]
meta = json.load(open(out)) if os.path.exists(out) else {}
def get(p):
    if p in meta: return None
    f = f'{SCR}/{p}.jpg'
    if not os.path.exists(f):
        r = subprocess.run(['curl', '-sS', '-m', '60', '-o', f, '-w', '%{http_code}',
                            f'https://cdli.earth/dl/tn_photo/{p}.jpg'], capture_output=True, text=True)
        if r.stdout.strip() != '200':
            if os.path.exists(f): os.remove(f)
            return p, {'photo': False}
    try:
        im = Image.open(f); ex = im.getexif()
        return p, {'photo': True, 'size': im.size, 'make': str(ex.get(271, '')), 'model': str(ex.get(272, '')),
                   'software': str(ex.get(305, '')), 'date': str(ex.get(306, '')),
                   'icc': len(im.info.get('icc_profile', b'') or b''), 'bytes': os.path.getsize(f)}
    except Exception as e:
        return p, {'photo': False, 'err': str(e)}
with ThreadPoolExecutor(2) as ex:
    for i, r in enumerate(ex.map(get, ids)):
        if r: meta[r[0]] = r[1]
        if i % 100 == 0:
            json.dump(meta, open(out, 'w')); print(i, len(meta), flush=True)
json.dump(meta, open(out, 'w'))
print('done', len(meta), sum(v['photo'] for v in meta.values()))
