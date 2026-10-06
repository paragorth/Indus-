"""pe57 cycle 1b: capacities of 3D-scanned vessels (Zenodo 3dbigdataspace / Objaverse GLBs).
Each GLB is downloaded to a temp file, measured with pe57_meshvol.capacity, then deleted (disk is short).
Results cached in data/pe57_ckpt/mesh_<tag>.json.  Usage: pe57_cycle1_mesh.py <records.json> <tag> [max]"""
import json, os, sys, tempfile, urllib.request, time
sys.path.insert(0, os.path.dirname(__file__))
from pe57_meshvol import capacity, load_vertices
CK = os.path.join(os.path.dirname(__file__), '..', 'data', 'pe57_ckpt')
recs = json.load(open(sys.argv[1])); tag = sys.argv[2]; mx = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
out = os.path.join(CK, f'mesh_{tag}.json')
res = json.load(open(out)) if os.path.exists(out) else {}
n = 0
for rid, r in recs.items():
    if rid in res: continue
    if n >= mx: break
    g = [f for f in r['files'] if f[0].endswith('.glb')]
    if not g: continue
    n += 1
    url = g[0][2] if len(g[0]) > 2 else f"https://zenodo.org/records/{rid}/files/{g[0][0]}?download=1"
    if not url.endswith('/content') and 'api' in url: url = url + '/content'
    fd, tmp = tempfile.mkstemp(suffix='.glb'); os.close(fd)
    try:
        for t in range(3):
            try:
                urllib.request.urlretrieve(url, tmp); break
            except Exception as e:
                print('retry', rid, e, flush=True); time.sleep(5)
        c = capacity(load_vertices(tmp))
    except Exception as e:
        c = {'error': str(e)[:200]}
    finally:
        os.remove(tmp)
    res[rid] = {'title': r['title'], 'desc': r['desc'][:400], 'cap': c}
    json.dump(res, open(out, 'w'), indent=1)
    print(rid, r['title'][:50], None if not c else {k: round(v, 4) for k, v in c.items() if isinstance(v, float)}, flush=True)
