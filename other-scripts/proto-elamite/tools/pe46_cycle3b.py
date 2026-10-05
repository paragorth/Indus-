"""pe46 cycle 3b: unordered SET codes. Perhaps the fields of a PE code are not positions but sign
alphabets: a code takes at most one sign from each field, in any order. Random guessing: 4,000 random
partitions of the 40 commonest signs into k = 2..6 fields (+ hill-climbing from the best 5), fitted on
half the tablets: exclusivity = share of strings (with >= 2 top signs) that use at most one sign per
field. Top 20 re-scored on held-out tablets. Null for each corpus: the same pipeline after signs are
shuffled ACROSS strings (string lengths and sign frequencies kept). Lift = real held-out / null held-out.
Controls: planted code with each string's signs shuffled (a set code: must pass), HTS groups shuffled
within each code (real code, unordered), Ur III herd attributes, Ur III and Linear B names.
"""
import sys, os, json, random
from collections import Counter
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe46_lib as L

NP = 4000; TOP = 40


def build():
    pe = L.pe_tokens(); n = len(pe)
    return dict(PE=pe, PEv=L.pe_tokens(base_signs=False), PLANTset=L.null_signshuf(L.planted(pe), 41),
                HTSset=L.null_signshuf(L.hts(n), 42), HERD=L.ur3_herd(n), UR3N=L.ur3_names(n),
                LBF=L.linb(True), LBA=L.linb(False))


def across(toks, seed):
    rng = random.Random(seed); pool = [u for t in toks for u in t['w']]; rng.shuffle(pool)
    out = []; i = 0
    for t in toks:
        k = len(t['w']); out.append(dict(t, w=tuple(pool[i:i + k]))); i += k
    return out


def matrix(words, top):
    idx = {s: i for i, s in enumerate(top)}
    rows = [[idx[u] for u in w if u in idx] for w in words]
    rows = [r for r in rows if len(r) >= 2]
    m = max(len(r) for r in rows)
    S = np.full((len(rows), m), -1, dtype=np.int64)
    for i, r in enumerate(rows):
        S[i, :len(r)] = r
    return S


def excl(S, f):
    F = np.where(S >= 0, f[np.clip(S, 0, None)], -1 - np.arange(S.shape[1])[None, :])
    F = np.sort(F, axis=1)
    return float(1 - (F[:, 1:] == F[:, :-1]).any(1).mean())


def search(toks, seed, split):
    C = L.make(toks, 'x'); A = L.tab_split(C, split)
    wa = [w for w, a in zip(C['w'], A) if a]; wb = [w for w, a in zip(C['w'], A) if not a]
    top = [u for u, _ in Counter(u for w in wa for u in set(w)).most_common(TOP)]
    SA, SB = matrix(wa, top), matrix(wb, top)
    rng = np.random.default_rng(seed); res = []
    for _ in range(NP):
        k = int(rng.integers(2, 7)); f = rng.integers(0, k, len(top))
        res.append((excl(SA, f), k, f))
    res.sort(key=lambda x: -x[0])
    best = res[:20]
    for j in range(5):                       # hill-climb the best five
        e, k, f = best[j]; f = f.copy()
        for _ in range(300):
            g = f.copy(); g[rng.integers(len(top))] = rng.integers(k)
            eg = excl(SA, g)
            if eg >= e:
                e, f = eg, g
        best[j] = (e, k, f)
    out = [dict(k=k, fit=e, held=excl(SB, f), fields=[[top[i] for i in range(len(top)) if f[i] == q] for q in range(k)])
           for e, k, f in best]
    return out, len(SA), len(SB)


def run(arg):
    k, toks = arg
    fn = os.path.join(L.CK, f'c3b_{k}.json')
    if os.path.exists(fn):
        return k, json.load(open(fn))
    d = dict(corpus=k, res=[])
    for split in (0, 1):
        r, na, nb = search(toks, 500 + split, split)
        rn, _, _ = search(across(toks, 600 + split), 700 + split, split)
        hr = float(np.median([x['held'] for x in r])); hn = float(np.median([x['held'] for x in rn]))
        d['res'].append(dict(split=split, nA=na, nB=nb, held=hr, null=hn, lift=hr / max(1e-9, hn),
                             best=max(r, key=lambda x: x['held'])))
    json.dump(d, open(fn, 'w'))
    return k, d


if __name__ == '__main__':
    B = build()
    with Pool(2) as p:
        for k, d in p.imap_unordered(run, list(B.items())):
            print(k, [(x['nB'], round(x['held'], 3), round(x['null'], 3), round(x['lift'], 3)) for x in d['res']], flush=True)
