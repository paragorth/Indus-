"""v39 cycle 1: merge the v35 twin pairs (each, subsets, all) and rerun the battery; random merges of equally frequent
pairs as the null; planted homophone splits of Latin / German / Italian (free variants, rank- and ratio-matched to
the Voynich twins) as the sensitivity control; random merges in the languages (must not make them more
language-like); Copiale cipher vs Copiale merged by its published key (real homophone cipher, positive control) vs
random merges of the cipher with the same class sizes.
Usage: python3 v39_cycle1.py [nworkers]"""
import sys, time, random
from multiprocessing import Pool
import v39_lib as L

EXCL = {'la': ('L_Isidore',), 'de': ('L_msG_Alem',), 'it': (), 'cop': ()}
NRAND_V, NRAND_L, NRAND_C = 60, 15, 15
CONFIGS = {'none': [], 'kt': ['kt'], 'pf': ['pf'], 'KT': ['KT'], 'PF': ['PF'], 'CS': ['CS'],
           'gal4': ['kt', 'pf', 'KT', 'PF'], 'all5': ['kt', 'pf', 'KT', 'PF', 'CS'],
           'fam': 'fam'}   # fam: k=t=p=f, K=T=P=F, C=S (whole gallows family collapsed)
_B = {}


def base(name):
    if name not in _B:
        if name == 'V': _B[name] = L.voynich('ZL3b')
        elif name in ('la', 'de', 'it'): _B[name] = L.lang(name)
        elif name in ('cop', 'copK'):
            Pc, Pp, _ = L.copiale(); _B['cop'] = Pc; _B['copK'] = Pp
    return _B[name]


def pairs_of(cfg):
    if cfg == 'fam': return [('k', 't'), ('k', 'p'), ('k', 'f'), ('K', 'T'), ('K', 'P'), ('K', 'F'), ('C', 'S')]
    return [L.TWINS[c] for c in cfg]


def plant_spec(lang_P):
    """letters at the same frequency ranks as the merged Voynich twin units, split at the Voynich minor share."""
    V = base('V'); cv = L.unit_freq(V)
    merged = L.apply_map(V, L.merge_map(pairs_of(CONFIGS['all5'])))
    cm = [u for u, _ in L.unit_freq(merged).most_common()]
    cl = [u for u, _ in L.unit_freq(lang_P).most_common()]
    letters, ratios = [], []
    for a, b in pairs_of(CONFIGS['all5']):
        r = cm.index(a)
        letters.append(cl[min(r, len(cl) - 1)])
        ratios.append(min(cv[a], cv[b]) / (cv[a] + cv[b]))
    return letters, ratios


def job(spec):
    key = '|'.join(map(str, spec))
    fn = f'c1_{key}.json'.replace('/', '_')
    r = L.load(fn)
    if r is not None: return key, 'cached'
    corp, kind, arg = spec
    t0 = time.time()
    P = base(corp)
    info = {}
    if kind == 'cfg':
        mp = L.merge_map(pairs_of(CONFIGS[arg])); P = L.apply_map(P, mp); info['map'] = mp
    elif kind == 'rand':       # random freq-matched merges, same number of pairs as cfg all5 / or single
        n, seed = arg
        rng = random.Random(seed)
        real = pairs_of(CONFIGS['all5'])[:n] if corp == 'V' else None
        if corp == 'V':
            if n == 1: real = [pairs_of([['kt', 'pf', 'KT', 'PF', 'CS'][seed % 5]])[0]]
            pr = L.random_pairs(P, real, rng)
        else:   # language: random merges matched to the planted letters (as if they were the twins)
            letters, _ = plant_spec(P)
            fr = [u for u, _ in L.unit_freq(P).most_common()]
            real = [(c, fr[min(fr.index(c) + 1, len(fr) - 1)]) for c in letters]
            pr = L.random_pairs(P, real, rng)
        mp = L.merge_map(pr); P = L.apply_map(P, mp); info['pairs'] = pr
    elif kind == 'plant':
        mode, seed = arg
        letters, ratios = plant_spec(P)
        P, undo = L.plant_split(P, letters, ratios, seed=seed, mode=mode)
        info['letters'] = letters; info['ratios'] = ratios
        if seed == 0:
            back = L.apply_map(P, undo)
            info['undo_exact'] = L.tokens(back) == L.tokens(base(corp))
    elif kind == 'coprand':    # random merges of the cipher with the key's class-size structure
        seed = arg
        rng = random.Random(seed)
        Pc, Pp, enc = L.copiale()
        inv = {v: k for k, v in enc.items()}
        # class sizes of the key over cipher units present
        cls = {}
        for k, ch in enc.items():
            if k.startswith('c:'):
                kb = __import__('v35_lib').COP_NAME.get(k[2:])
                pl = L._COP_PLAIN.get(kb) if kb else None
                cls.setdefault(pl or k, []).append(ch)
        units = [u for u in L.unit_freq(Pc)]
        rng.shuffle(units)
        mp, i = {}, 0
        for grp in sorted(cls.values(), key=len, reverse=True):
            g = [u for u in grp if u in set(units)]
            sz = len(g)
            if sz == 0: continue
            take = units[i:i + sz]; i += sz
            for u in take[1:]: mp[u] = take[0]
        P = L.apply_map(Pc, mp); info['nmerge'] = len(mp)
    excl = EXCL.get(corp.rstrip('K'), ())
    r = L.battery(P, exclude=excl)
    r['info'] = info; r['sec'] = round(time.time() - t0, 1)
    L.save(fn, r)
    return key, r['sec']


def specs():
    S = [('V', 'cfg', c) for c in CONFIGS]
    S += [('V', 'rand', (5, s)) for s in range(NRAND_V)]
    S += [('V', 'rand', (1, s)) for s in range(25)]
    for lg in ('la', 'de', 'it'):
        S += [(lg, 'cfg', 'none'), (lg, 'plant', ('free', 0)), (lg, 'plant', ('pos', 0))]
        S += [(lg, 'rand', (5, s)) for s in range(NRAND_L)]
    S += [('cop', 'cfg', 'none'), ('copK', 'cfg', 'none')] + [('cop', 'coprand', s) for s in range(NRAND_C)]
    return S


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    t0 = time.time()
    S = specs()
    # 'none' for languages / copiale must not try twin configs
    with Pool(nw) as P:
        for k, s in P.imap_unordered(job, S):
            print(k, s, round(time.time() - t0), flush=True)
