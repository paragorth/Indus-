"""v67 cycle 2b: properties of the selected line-token streams (rules chosen on odd folios in c2):
lag profile of the off-diagonal MI z (lags 1,2,3,5 lines within page), asymmetry, recurring ordered
token trigrams across pages vs within-page shuffle, Zipf, REP, section information real vs page-word twin."""
import sys, os, json, pickle, random
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v67_lib as X

C = pickle.load(open(os.path.join(X.CK, 'c2_corpora.pkl'), 'rb'))
R = [json.loads(l) for l in open(os.path.join(X.CK, 'c2.jsonl'))]
ALPH = X.alphabet(C['ZL'][0])

def lagged(toks, pages, lag, nshuf=40, seed=0):
    T, V = X.encode(toks); pd = {}; P = np.array([pd.setdefault(p, len(pd)) for p in pages])
    def st(T, P):
        same = P[lag:] == P[:-lag]
        a, b = T[:-lag][same], T[lag:][same]; off = a != b; a, b = a[off], b[off]; n = len(a)
        u, c = np.unique(a * V + b, return_counts=True); ca = np.bincount(a, minlength=V); cb = np.bincount(b, minlength=V)
        return float(np.sum(c / n * np.log2(c * n / (ca[u // V] * cb[u % V])))), float(np.mean(T[:-lag][same] == T[lag:][same]))
    o = st(T, P); rng = np.random.default_rng(seed); S = []
    for _ in range(nshuf):
        k = np.lexsort((rng.random(len(T)), P)); S.append(st(T[k], P[k]))
    S = np.array(S)
    return round((o[0] - S[:, 0].mean()) / S[:, 0].std(), 2), round(o[1] / max(1e-9, S[:, 1].mean()), 2)

def trigrams(toks, pages, nshuf=40, seed=0):
    def rec(tk, pg):
        g = Counter(); seen = {}
        for i in range(len(tk) - 2):
            if pg[i] == pg[i + 2] and len({tk[i], tk[i + 1], tk[i + 2]}) == 3:
                seen.setdefault((tk[i], tk[i + 1], tk[i + 2]), set()).add(pg[i])
        return sum(1 for v in seen.values() if len(v) >= 2), seen
    o, seen = rec(toks, pages); rng = random.Random(seed); S = []
    byp = {}
    for i, p in enumerate(pages): byp.setdefault(p, []).append(i)
    for _ in range(nshuf):
        tk = list(toks)
        for p, idx in byp.items():
            v = [toks[i] for i in idx]; rng.shuffle(v)
            for i, x in zip(idx, v): tk[i] = x
        S.append(rec(tk, pages)[0])
    top = sorted(((len(v), k) for k, v in seen.items() if len(v) >= 2), reverse=True)[:5]
    return o, float(np.mean(S)), float(np.std(S) + 1e-9), top

def pick(n, key):
    o = np.array([r['res'][n]['o'][key] for r in R]); return R[int(np.argmax(o))]

out = {}
for n in C:
    c, tw, par = C[n]
    for key in ('agx', 'MI_z', 'LANG'):
        r = pick(n, key) if n not in ('IT',) else pick('ZL', key)  # IT uses the rule chosen on ZL (replication)
        tk, ag = X.line_tokens(c, r['r'], ALPH); tk2, _ = X.line_tokens(tw, r['r'], ALPH)
        pages = [L['folio'] for L in c]
        lags = {L: lagged(tk, pages, L) for L in (1, 2, 3, 5)}
        s = X.score(tk, pages, nshuf=100)
        tri = trigrams(tk, pages)
        secr = X.nmi(tk, [L['sec'] for L in c]); sect = X.nmi(tk2, [L['sec'] for L in c])
        d = {'rule': r['rule'], 'lags(MIz,REPratio)': lags, 'score': {k: round(v, 3) for k, v in s.items()},
             'tri_rec': [tri[0], round(tri[1], 1), round((tri[0] - tri[1]) / tri[2], 2)], 'tri_top': [[k, list(t)] for k, t in tri[3]],
             'sec_nmi_real': round(secr, 4), 'sec_nmi_twin': round(sect, 4), 'mean_agree': round(sum(ag) / len(ag), 3)}
        if 'truth' in c[0]: d['nmix'] = round(X.nmi_ex(tk, [L['truth'] for L in c]), 3)
        out[f'{n}/{key}'] = d
        print(n, key, json.dumps(d), flush=True)
json.dump(out, open(os.path.join(X.CK, 'c2b.json'), 'w'), indent=1)
