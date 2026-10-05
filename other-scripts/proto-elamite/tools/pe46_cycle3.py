"""pe46 cycle 3: a code system predicts unseen codes.
For each corpus: random schema search on fit tablets (A); for the 5 best fit schemas, the slot-value
marginals on A give an independence (code) model; the M most probable slot-value combinations NOT
seen on A are the predicted codes. Score on held-out tablets B: share of B's NEW strings that are
predicted codes (hit rate), and lift = hit rate / hit rate of the same M combos drawn by a sign
bigram model fitted on A (a name-like generator) -- both predictors get the same budget M.
Corpora: PE (base signs), its nulls (sig, tab, Markov), planted code, HTS, Ur III herd, Ur III names,
Linear B first word / full designation.
Also freezes the PE top-40 predicted codes (full corpus fit) with a SHA-256 hash.
usage: python3 pe46_cycle3.py [N_schemas] [M]
"""
import sys, os, json, time, hashlib, math, random
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe46_lib as L

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
M = int(sys.argv[2]) if len(sys.argv) > 2 else 200


def build():
    pe = L.pe_tokens(); n = len(pe)
    return dict(PE=pe, **{'PE~sig': L.null_signshuf(pe, 31), 'PE~tab': L.null_tabshuf(pe, 32),
                          'PE~mkv': L.null_markov(pe, 33)},
                PLANT=L.planted(pe), HTS=L.hts(n), HERD=L.ur3_herd(n), UR3N=L.ur3_names(n),
                LBF=L.linb(True), LBA=L.linb(False))


def code_predictions(words, sch, M):
    """Top-M unseen combinations under slot independence (with slot-emptiness as a value)."""
    K = len(sch['rules']) + 1
    cnt = [Counter() for _ in range(K)]
    seen = set(words)
    for w in words:
        for s, v in enumerate(L.cut(w, sch['rules'])):
            cnt[s][v] += 1
    n = len(words)
    tops = [[(v, c / n) for v, c in c_.most_common(25)] for c_ in cnt]
    # beam over slots
    beam = [((), 0.0)]
    for s in range(K):
        nb = []
        for pre, lp in beam:
            for v, p in tops[s]:
                nb.append((pre + v, lp + math.log(p)))
        nb.sort(key=lambda x: -x[1]); beam = nb[:20 * M]
    out = []; used = set()
    for w, lp in beam:
        if len(w) >= 2 and w not in seen and w not in used:
            out.append(w); used.add(w)
        if len(out) >= M:
            break
    return out


def bigram_predictions(words, M, rng):
    """Same budget: the M most frequent unseen strings among 50*M draws of a sign bigram chain."""
    cnt = defaultdict(Counter)
    for w in words:
        s = ('^',) + w + ('$',)
        for a, b in zip(s, s[1:]):
            cnt[a][b] += 1
    tab = {a: (list(c), np.cumsum(list(c.values()))) for a, c in cnt.items()}
    seen = set(words); draws = Counter()
    for _ in range(50 * M):
        w = []; h = '^'
        while len(w) < 10:
            ks, cs = tab[h]; u = ks[int(np.searchsorted(cs, rng.random() * cs[-1], side='right'))]
            if u == '$':
                break
            w.append(u); h = u
        w = tuple(w)
        if len(w) >= 2 and w not in seen:
            draws[w] += 1
    return [w for w, _ in draws.most_common(M)]


def run(arg):
    k, toks = arg
    f = os.path.join(L.CK, f'c3_{k}.json')
    if os.path.exists(f):
        return k, json.load(open(f))
    C = L.make(toks, k); res = []
    for split in (0, 1):
        A = L.tab_split(C, split); CA = L.subset(C, A); CB = L.subset(C, ~A)
        out = L.search(C, NS, seed=300 + split, split_seed=split, top=5)
        seenA = set(CA['w'])
        newB = [w for w in CB['w'] if w not in seenA]
        bg = set(bigram_predictions(CA['w'], M, random.Random(split)))
        hb = sum(w in bg for w in newB) / max(1, len(newB))
        for o in out:
            pr = set(code_predictions(CA['w'], o['sch'], M))
            hc = sum(w in pr for w in newB) / max(1, len(newB))
            res.append(dict(split=split, s=L.sch_str(o['sch']), held=o['held']['CODE'], new=len(newB),
                            hit_code=hc, hit_bigram=hb, lift=(hc + 1e-3) / (hb + 1e-3)))
    d = dict(corpus=k, res=res,
             hit_code=float(np.median([r['hit_code'] for r in res])),
             hit_bigram=float(np.median([r['hit_bigram'] for r in res])),
             lift=float(np.median([r['lift'] for r in res])))
    json.dump(d, open(f, 'w'))
    return k, d


def freeze_pe():
    pe = L.pe_tokens(); C = L.make(pe, 'PE')
    out = L.search(C, NS, seed=399, split_seed=0, top=1)
    sch = out[0]['sch']
    pr = code_predictions(C['w'], sch, 40)
    txt = '\n'.join(' '.join(w) for w in pr)
    h = hashlib.sha256(txt.encode()).hexdigest()[:16]
    json.dump(dict(schema=L.sch_str(sch), predictions=[list(w) for w in pr], sha=h),
              open(os.path.join(L.DATA, 'pe46_frozen_predicted_codes.json'), 'w'), indent=1)
    return L.sch_str(sch), h, pr[:10]


if __name__ == '__main__':
    B = build()
    with Pool(2) as p:
        for k, d in p.imap_unordered(run, list(B.items())):
            print(k, 'hit_code', round(d['hit_code'], 4), 'hit_bigram', round(d['hit_bigram'], 4),
                  'lift', round(d['lift'], 2), 'new', d['res'][0]['new'], flush=True)
    print('FROZEN', freeze_pe(), flush=True)
