"""pe83 cycle 3 calibration: the same outline pipeline on Uruk III proto-cuneiform CDLI photos.
'header' analogue: obverse line 1 has signs and no numerals. usage: python3 pe83_protocun.py <photo_dir> <cdli.atf>"""
import sys, os, json, re
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe83_common as P

D = sys.argv[1] if len(sys.argv) > 1 else None


def parse(atf, ids):
    out = {}; cur = None; surf = None
    for line in open(atf, encoding='utf-8', errors='replace'):
        if line.startswith('&P'):
            pid = line[1:8]; cur = pid if pid in ids else None; surf = None
            if cur:
                out[cur] = dict(lines=[], obv=[])
            continue
        if not cur:
            continue
        s = line.strip()
        if s.startswith('@'):
            w = s[1:].split()[0] if len(s) > 1 else ''
            if w in ('obverse', 'reverse', 'top', 'bottom', 'left', 'right', 'edge', 'surface', 'seal'):
                surf = w
            continue
        m = re.match(r"^([0-9]+[a-z.']*)\.?\s+(.*)$", s)
        if m:
            out[cur]['lines'].append((surf, s))
            if surf == 'obverse':
                out[cur]['obv'].append(m.group(2))
    return out


def toks(txt):
    nums = re.findall(r'\d+\((N\d+)', txt)
    rest = re.sub(r'\d+\(N[^)]*\)\S*', ' ', txt)
    signs = [t for t in re.split(r'[\s,]+', rest) if t and not re.match(r'^[\[\]x.#?!<>()$-]+$', t) and 'X' != t.strip('#?[]')]
    return nums, signs


def measure(pid):
    try:
        A = P.load_img(pid, D)
    except Exception:
        return pid, None
    g = A.mean(2); H, W = g.shape
    B = P.blobs(g); ob = P.pick_obverse(B, H)
    if ob is None:
        return pid, None
    m = ob[1]; c = P.corner_metrics(m)
    f = dict(cf=float(np.mean(c)), cf_top=float((c[0] + c[1]) / 2), cf_bot=float((c[2] + c[3]) / 2), asp=float(np.log(m.shape[0] / m.shape[1])),
             la=float(np.log(m.sum())), bg=float(np.median(np.concatenate([g[:3].ravel(), g[-3:].ravel()]))))
    rv = P.pick_reverse(B, H, ob)
    if rv is not None:
        rc = P.corner_metrics(rv[1]); f['rev_lower'] = float((rc[2] + rc[3]) / 2); f['rev_upper'] = float((rc[0] + rc[1]) / 2)
    return pid, f


if __name__ == '__main__':
    import glob
    ids = {os.path.basename(p)[:7] for p in glob.glob(os.path.join(D, 'P*.jpg')) if os.path.getsize(p) > 2000}
    A = parse(sys.argv[2], ids)
    with Pool(int(os.environ.get('NW', 2))) as pool:
        S = dict(pool.map(measure, sorted(ids), chunksize=10))
    R = []
    for pid, f in S.items():
        a = A.get(pid)
        if not f or not a or not a['obv']:
            continue
        nums, sg = toks(a['obv'][0])
        dam = any(ch in ''.join(x for _, x in a['lines']) for ch in '[#') or ' x' in ' '.join(x for _, x in a['lines'])
        l1dam = any(ch in a['obv'][0] for ch in '[#') or 'X' in a['obv'][0].split() or 'x' in a['obv'][0].split()
        rev = any(s != 'obverse' for s, _ in a['lines'])
        R.append(dict(f, id=pid, hd=float(bool(sg) and not nums), nl=np.log(len(a['lines'])), dx=float(dam), l1dam=float(l1dam), rev=float(rev)))
    hd = np.array([r['hd'] for r in R]); rng = np.random.default_rng(833)
    col = lambda k: np.array([r.get(k, np.nan) for r in R], float)
    Z = np.column_stack([np.ones(len(R)), col('la'), col('asp'), col('nl'), col('dx'), col('rev'), col('l1dam'), col('la') ** 2, col('nl') ** 2])
    out = dict(n=len(R), header_rate=round(float(hd.mean()), 3))
    for k in ('cf', 'cf_top', 'cf_bot', 'rev_lower', 'rev_upper'):
        out[k] = P.partial(col(k), hd, Z, 2000, rng)
    intact = col('l1dam') == 0
    out['cf_top_line1_intact'] = P.partial(col('cf_top')[intact], hd[intact], Z[intact], 2000, rng)
    json.dump(dict(summary=out, rows=[{k: v for k, v in r.items()} for r in R]), open(os.path.join(P.CK, 'protocun.json'), 'w'))
    print(json.dumps(out, indent=1))
