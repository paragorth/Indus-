"""pe80 cycle 2 calibration: do KNOWN Ur III document kinds differ in breakage / wear beyond size?
Umma + Girsu, CDLI ATF: BREAK = share of text lines with '[' ; WEAR = share of words with '#'.
Kinds (known from words, read openly here because this is the answer key): balanced account
(nig2-ka9-ak), summary (szunigin), sealed receipt (kiszib3), letter / legal (CDLI genre).
Residualised on provenience, log area, thickness, log lines; null = permutation within
provenience x line tercile. Then the same test thinned to PE size (287 tablets, 200 draws)."""
import re, os, json, math, random, collections, sys
import numpy as np
import pe80_common as pc, pe75_common as p75
cat = p75.catalogue()
ids = pc.ur3_ids()
tabs = {}; cur = None
for raw in open(os.path.join(pc.SCRATCH, 'ur.atf'), encoding='utf-8', errors='replace'):
    if raw.startswith('&P'):
        pid = raw[1:8]; cur = pid if pid in ids else None
        if cur: tabs[cur] = []
        continue
    if cur and re.match(r"^\d+'?\.\s", raw):
        tabs[cur].append(raw.split('.', 1)[1].strip())
rows = []
for pid, L in tabs.items():
    if len(L) < 4: continue
    k = cat.get(pid) or {}
    words = [w for l in L for w in l.split()]
    brk = np.mean(['[' in l for l in L]); wear = np.mean(['#' in w for w in words])
    txt = ' '.join(L)
    rows.append(dict(id=pid, prov=ids[pid], nl=len(L), h=k.get('h'), w=k.get('w'), th=k.get('t'), genre=(k.get('genre') or ''),
                     brk=brk, wear=wear,
                     f={'nigkaak': 'nig2-ka9' in txt, 'szunigin': 'szu-nigin2' in txt or 'szunigin' in txt,
                        'kiszib': 'kiszib3' in txt, 'letter': (k.get('genre') or '').startswith('Letter'),
                        'legal': (k.get('genre') or '').startswith('Legal')}))
print('tablets', len(rows))
def resid(rows, y):
    pv = sorted({r['prov'] for r in rows}); X = []
    for r in rows:
        a = (r['h'] or 0) * (r['w'] or 0)
        X.append([1.0 * (r['prov'] == p) for p in pv] + [math.log(a) if a > 0 else 0, a <= 0, r['th'] or 0, not r['th'], math.log(r['nl'])])
    X = np.array(X, float); b, *_ = np.linalg.lstsq(X, y, rcond=None); return y - X @ b
rng = np.random.default_rng(5)
def test(rows, R=200):
    out = {}
    nl = np.array([r['nl'] for r in rows]); terc = np.digitize(nl, np.quantile(nl, [1/3, 2/3]))
    st = np.array([hash(r['prov']) % 1000 * 10 for r in rows]) + terc
    for tgt in ['brk', 'wear']:
        y = resid(rows, np.array([r[tgt] for r in rows], float))
        for f in rows[0]['f']:
            x = np.array([r['f'][f] for r in rows], float)
            if x.sum() < 5 or x.sum() > len(x) - 5: continue
            d = y[x == 1].mean() - y[x == 0].mean()
            nd = []
            for _ in range(R):
                yy = y.copy()
                for s in np.unique(st):
                    m = np.flatnonzero(st == s); yy[m] = y[rng.permutation(m)]
                nd.append(yy[x == 1].mean() - yy[x == 0].mean())
            nd = np.array(nd); out[(tgt, f)] = (round(float(d / y.std()), 3), float((1 + (np.abs(nd) >= abs(d)).sum()) / (R + 1)), int(x.sum()))
    return out
full = test(rows, 100)
for k, v in full.items(): print('FULL', k, v)
hits = collections.Counter(); nd = 0
for rep in range(40):
    sub = random.Random(rep).sample(rows, 287)
    r = test(sub, 100)
    for k, v in r.items():
        if v[1] < 0.05: hits[k] += 1
    nd += 1
print('PE-size (287) share of draws with p<0.05:', {str(k): v / nd for k, v in hits.items()})
json.dump(dict(full={str(k): v for k, v in full.items()}, pe_size={str(k): v / nd for k, v in hits.items()}), open(os.path.join(pc.CK, 'c2_ur3.json'), 'w'), indent=1)
