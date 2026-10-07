"""la82: verify every frozen Linear A prediction file against its stored/claimed sha256 (read-only)."""
import json, hashlib, os, itertools
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CLAIMS = {  # file -> claimed hash prefix (from .sha256 file or FINDINGS/loop text)
 'data/la22/la22_predictions.json': '37303998', 'data/la66_frozen_factors.json': '3127699d',
 'loops/la70_predictions.txt': '2d7c6c9b', 'loops/la72_predictions.txt': '6c493bb3',
 'data/la73_predictions.json': 'ff584786', 'data/la74_predictions.json': '526ce4fb',
 'data/la75_frozen_profiles.json': 'a9f83d76', 'data/la75_frozen_links.json': '6be5c184',
 'data/la76_values.json': '62b453ed', 'data/la77/frozen_direction.json': '8003f2d4',
 'data/la78_frozen_ranking.json': '8de78817', 'data/la79_frozen_predictions.json': '6f75f0d3',
 'data/la80_frozen_c1.json': 'dbece0ed', 'data/la80_frozen_c2.json': '841fd69b', 'data/la80_frozen_c3.json': '46d42fad',
 'data/la81_frozen_c1_1950.json': '9aa4be95', 'data/la81_frozen_c1_1950_lexall.json': '92388edf',
 'data/la81_frozen_c2_1950.json': '926a6d06', 'data/la81_c3_plan.json': '4bf1afec', 'data/la15/c3_predictions.json': None}
HK = ['sha256', 'sha256_of_rest', 'sha256_of_body_without_this_field', 'hash']
def cands(path):
    raw = open(path, 'rb').read(); yield 'raw', hashlib.sha256(raw).hexdigest()
    txt = raw.decode()
    if '=====JSON=====\n' in txt: yield 'json-block', hashlib.sha256(txt.split('=====JSON=====\n')[1].encode()).hexdigest()
    try: obj = json.loads(txt)
    except Exception: return
    objs = [('whole', obj)]
    if isinstance(obj, dict):
        objs.append(('minus-hash-keys', {k: v for k, v in obj.items() if k not in HK}))
        for k, v in obj.items():
            if isinstance(v, dict): objs.append((f'field:{k}', v)); objs.append((f'field:{k}-minus', {a: b for a, b in v.items() if a not in HK}))
            if isinstance(v, (list, dict)): objs.append((f'value:{k}', v))
    for (nm, o), ind, ea in itertools.product(objs, [None, 1], [True, False]):
        try: s = json.dumps(o, sort_keys=True, indent=ind, ensure_ascii=ea, default=str)
        except Exception: continue
        yield f'{nm} indent={ind} ascii={ea}', hashlib.sha256(s.encode()).hexdigest()
out = {}
for f, c in CLAIMS.items():
    p = os.path.join(D, f)
    if not os.path.exists(p): out[f] = 'MISSING'; continue
    hit = None
    for m, h in cands(p):
        if c and h.startswith(c): hit = (m, h); break
    out[f] = dict(claimed=c, verified=bool(hit), method=hit[0] if hit else None, sha256=hit[1] if hit else hashlib.sha256(open(p,'rb').read()).hexdigest())
print(json.dumps(out, indent=1))
