"""v39 cycle 3: MASSIVE RANDOM MERGE GUESSING.  If some hidden set of glyph identities (not necessarily the look-alike
twins) is what makes the Voynich look generated, a blind search over thousands of merges should find merges that
make it language-like, and those merges should hold on held-out text.
Discovery: half of the ZL pages.  NSET random merge sets (1-8 pairs among units with >= 100 tokens, union-find).
Cheap battery: S = P(language or conlang) - P(generator) (v31 classifier), arrow, gap ratio, ttr.
Held-out: the top 12 sets and 40 random sets are rescored on the other ZL half and on IT2a (other transcription,
same held-out pages); a set survives if its held-out S beats the 95th percentile of the random sets' held-out S in
both.  Calibration (positive): the same search on Latin herbal with the planted free twin split (5 pairs): top sets
must be enriched for planted pairs and survive.  Enrichment test: share of the v35 twin pairs among pairs of the
top sets vs all sets.  Usage: python3 v39_cycle3.py [nworkers]"""
import sys, time, random
from multiprocessing import Pool
import numpy as np
import v39_lib as L
from v39_cycle1 import base as base1, plant_spec

NSET = {'V': 260, 'Pla': 160}
_B = {}


def halves(name):
    if name in _B: return _B[name]
    if name == 'V':
        P = L.voynich('ZL3b'); PI = L.voynich('IT2a')
    else:
        P0 = base1('la'); letters, ratios = plant_spec(P0)
        P, undo = L.plant_split(P0, letters, ratios, seed=0, mode='free'); PI = None
        _B['planted'] = {frozenset((k, v)) for k, v in undo.items()}
    rng = random.Random(39)
    ids = sorted(p['id'] for p in P); rng.shuffle(ids)
    h = set(ids[:len(ids) // 2])
    d = [p for p in P if p['id'] in h]; t = [p for p in P if p['id'] not in h]
    ti = [p for p in PI if p['id'] not in h] if PI else None
    _B[name] = (d, t, ti)
    return _B[name]


def rand_set(name, seed):
    d, _, _ = halves(name)
    c = L.unit_freq(d)
    U = sorted(u for u, n in c.items() if n >= 100)
    rng = random.Random(seed * 7 + 1)
    k = rng.randint(1, 8)
    return [tuple(rng.sample(U, 2)) for _ in range(k)]


def score(P, excl):
    r = L.v31_probs(P, excl, maxs=30)
    return dict(S=r['pL'] + r['pI'] - r['pG'], pLI=r['pL'] + r['pI'], pG=r['pG'], arrow=L.freq_arrow(P, nprobe=200),
                gap=L.gap_ratio(P), ttr=L.lexstats(P)['ttr'])


EXCL = {'V': (), 'Pla': ('L_Isidore',)}


def job(spec):
    name, part, seed = spec
    fn = f'c3_{name}|{part}|{seed}.json'
    if L.load(fn) is not None: return spec, 'cached'
    d, t, ti = halves(name)
    P = {'d': d, 't': t, 'ti': ti}[part]
    pairs = rand_set(name, seed) if seed >= 0 else []
    r = score(L.apply_map(P, L.merge_map(pairs)), EXCL[name])
    r['pairs'] = pairs
    L.save(fn, r)
    return spec, round(r['S'], 3)


def run(S, nw):
    t0 = time.time()
    with Pool(nw) as P:
        for k, s in P.imap_unordered(job, S):
            print(k, s, round(time.time() - t0), flush=True)


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for name in ('Pla', 'V'):
        run([(name, 'd', s) for s in range(-1, NSET[name])], nw)
        R = {s: L.load(f'c3_{name}|d|{s}.json') for s in range(NSET[name])}
        top = sorted(R, key=lambda s: -R[s]['S'])[:12]
        rnd = random.Random(5).sample(sorted(R), 40)
        parts = ('t', 'ti') if name == 'V' else ('t',)
        run([(name, p, s) for p in parts for s in sorted(set(top) | set(rnd) | {-1})], nw)
        L.save(f'c3_{name}_sel.json', dict(top=top, rnd=rnd))
    print('C3DONE', flush=True)
