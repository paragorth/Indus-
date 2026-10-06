"""pe58 cycle 1: controls. (a) Ur III distributive ration/fodder lines made opaque: do the pairwise exchange
rates give back the ladder (gurusz:geme2 2, geme2:dumu 2-3, gurusz:dumu 4, gu4:udu fodder 4-6), and does the
lattice test fire above quantity-shuffled-within-tablet pools? (b) planted ladders / arbitrary factors /
physical-constant factors on the PE skeleton: recovery error, lattice test, physics test."""
import sys, os, json, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe58_lib import *
from multiprocessing import Pool

FN = 'pe58_cycle1.txt'


def frequent(C, mino=15, mintab=5):
    G = groups(C)
    occ = Counter(s for r in C for s in set(r['w']))
    out = []
    for s in occ:
        if occ[s] >= mino:
            sg = sign_groups(C, G, s)
            if len(sg) >= mintab:
                out.append(s)
    return sorted(out)


def pair_table(C, signs, minpair=5, floor=0.12, maxw=0.8):
    E = Est(C)
    out = {}
    for i, a in enumerate(signs):
        for b in signs[i + 1:]:
            p = E.pair(a, b)
            if p and p['ntab'] >= minpair:
                out[(a, b)] = p
    use = [p['mode'] for p in out.values() if abs(p['mode']) >= floor and p['mhi'] - p['mlo'] <= maxw]
    return out, use


def shuf_pool(C, signs, seeds):
    pool = []
    for sd in seeds:
        _, u = pair_table(qshuf_tab(C, 1000 + sd), signs)
        pool += u
    return pool


def ur3_job(total):
    C, voc = ur3_ta(total=total)
    S = frequent(C)
    tab, use = pair_table(C, S)
    pool = shuf_pool(C, S, range(6))
    lt = lattice_test(use, pool)
    inv = {v: k for k, v in voc.items()}
    truth = {('gurusz', 'geme2'): (0.69, 0.69), ('geme2', 'dumu'): (0.69, 1.10), ('gurusz', 'dumu'): (1.10, 1.79),
             ('gu4', 'udu'): (1.39, 1.79)}
    rec = {}
    for (a, b), (lo, hi) in truth.items():
        A, B = voc[a], voc[b]
        p = tab.get((A, B)) or tab.get((B, A))
        if p is None:
            p0 = Est(C).pair(A, B)
            p = p0
            sg = 1
        else:
            sg = 1 if (A, B) in tab else -1
        if p is None:
            rec[f'{a}:{b}'] = None; continue
        m = sg * p['mode']; ml, mh = sorted([sg * p['mlo'], sg * p['mhi']])
        rec[f'{a}:{b}'] = dict(mode=round(m, 3), x=round(math.exp(m), 2), ci=(round(ml, 2), round(mh, 2)),
                                truth=(round(math.exp(lo), 2), round(math.exp(hi), 2)), hit=not (mh < lo - 0.05 or ml > hi + 0.05))
    return dict(corpus='UR3ta_total' if total else 'UR3ta_rate', nsign=len(S), npairs=len(tab), nuse=len(use),
                lattice=lt, ladder=rec)


def plant_job(args):
    kind, seed = args
    pe = pe_corpus()
    W = json.load(open(os.path.join(DATA, 'pe52_frozen_weights.json')))['signs']
    C, f = planted(pe, seed, kind, k=10, exclude=set(W))
    E = Est(C)
    err = []; F = {}
    for s, v in f.items():
        r = E.factor(s, n=120, seed=seed)
        p_s = None
        if r:
            err.append(abs(r['med'] - math.log(v)))
            F[s] = (r['med'] - r['iqr'], r['med'] + r['iqr'])
    S = sorted(f)
    tab, use = pair_table(C, S, minpair=3)
    pool = shuf_pool(C, S, range(4))
    # empirical pool: same pipeline on 10 random non-planted signs
    rng = random.Random(seed)
    others = [s for s in frequent(C, 20, 8) if s not in f and s not in W]
    oth = rng.sample(others, min(10, len(others)))
    _, opool = pair_table(C, oth, minpair=3)
    lt = lattice_test(use, pool + opool)
    single = [math.log(v) for v in f.values()]
    est = [E.factor(s, n=120, seed=seed + 1) for s in S]
    estl = [e['med'] for e in est if e and abs(e['med']) >= 0.12]
    lt1 = lattice_test(estl) if estl else None
    ph = phys_test(F, n=3000, seed=seed)
    ph.pop('hits')
    return dict(corpus=f'PLANT_{kind}_{seed}', mae=float(np.mean(err)) if err else None, nrec=len(err),
                corr=float(np.corrcoef([math.log(f[s]) for s, e in zip(S, est) if e], [e['med'] for e in est if e])[0, 1]),
                npairs=len(tab), nuse=len(use), lattice_pairs=lt, lattice_single=lt1, phys=ph)


if __name__ == '__main__':
    jobs = [('ladder', s) for s in range(3)] + [('arb', s) for s in range(3)] + [('phys', s) for s in range(3)]
    with Pool(2) as P:
        r_u = P.map(ur3_job, [False, True])
        r_p = P.map(plant_job, jobs)
    res = r_u + r_p
    json.dump(res, open(os.path.join(CK, 'c1.json'), 'w'), indent=1, default=str)
    for r in res:
        print(json.dumps(r, default=str)[:900])
