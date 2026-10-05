"""pe50 cycle 1: count the invisible (capture-recapture on tablets as capture events).

Parts
 U  Ur III control: Drehem / Umma, entity types OFF (named officials) and ENT2 (entry strings),
    thinned to PE size (n tablets carrying the entity type = PE's 607); estimators must cover
    the full archive's observed count S_full.  Also the thinning-mixture q estimator.
 P  planted archives: known persons N, offices with local pools, T written tablets, thinned
    to PE size; estimators must cover N_seen_in_full (persons on >= 1 written tablet).
 S  shuffle null: entity labels shuffled across tablets (set sizes kept); natural-group
    (excavation volume) three-list models on real vs shuffled.
 E  PE ensemble: random (entity definition, tablet subset, estimator, grouping) choices.
Output: data/pe50_ckpt/cycle1.json
"""
import json, os, sys, random, math, collections, time
from multiprocessing import Pool
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe50_lib import chao2, ichao2, jack, chapman, ll3, thin, thinfit  # noqa

CK = os.path.join(HERE, '..', 'data', 'pe50_ckpt')
D = json.load(open(os.path.join(CK, 'caps.json')))
CAPS, META = D['caps'], D['meta']
NPE = 607
EST = ('chao2', 'ichao2', 'jack1', 'jack2', 'chapman', 'll3')


def run_est(inc, rng, which=EST, groups=None):
    out = {}
    for w in which:
        if w == 'chao2':
            out[w] = chao2(inc)
        elif w == 'ichao2':
            out[w] = ichao2(inc)
        elif w == 'jack1':
            out[w] = jack(inc, 1)
        elif w == 'jack2':
            out[w] = jack(inc, 2)
        elif w == 'chapman':
            out[w] = chapman(inc, rng)
        elif w == 'll3':
            r = ll3(inc, rng, groups=groups)
            if r:
                out[w] = r[:3]
    return out


def job_ur3(args):
    arch, d, rep, n = args
    rng = random.Random(hash((arch, d, rep, n)) & 0xffffffff)
    inc = [set(s) for s in CAPS[arch][d] if s]
    Sfull = len(set().union(*inc))
    s = rng.sample(inc, n)
    r = run_est(s, rng)
    r['S'] = len(set().union(*s))
    if rep < 6:
        tf = thinfit(s)
        r['thin_q'] = tf['q']
        r['thin_qtrue'] = n / len(inc)
    return arch, d, n, Sfull, r


def planted(rng):
    N = rng.choice([1000, 2000, 4000, 8000, 16000])
    s = rng.uniform(0.3, 2.0)
    O = rng.choice([5, 10, 20, 40])
    Tw = rng.choice([6000, 12000, 30000, 60000])
    lam = rng.uniform(0.8, 2.5)
    act = np.exp(np.random.default_rng(rng.randrange(1 << 30)).normal(0, s, N))
    off = [rng.randrange(O) for _ in range(N)]
    pools = collections.defaultdict(list)
    for i, o in enumerate(off):
        pools[o].append(i)
    osz = np.array([len(pools[o]) for o in range(O)], float)
    nr = np.random.default_rng(rng.randrange(1 << 30))
    cw = {o: act[pools[o]] / act[pools[o]].sum() for o in range(O) if pools[o]}
    tabs = []
    for _ in range(Tw):
        o = int(nr.choice(O, p=osz / osz.sum()))
        if not pools[o]:
            continue
        k = 1 + nr.poisson(lam)
        k = min(k, len(pools[o]))
        tabs.append(set(nr.choice(pools[o], size=k, replace=False, p=cw[o]).tolist()))
    return {'N': N, 's': s, 'O': O, 'Tw': Tw, 'lam': lam}, tabs


def job_plant(rep):
    rng = random.Random(1000 + rep)
    par, tabs = planted(rng)
    Nseen = len(set().union(*tabs))
    s = rng.sample(tabs, NPE)
    r = run_est(s, rng)
    r['S'] = len(set().union(*s))
    if rep < 20:
        tf = thinfit(s)
        r['thin_q'] = tf['q']
        r['thin_qtrue'] = NPE / len(tabs)
    return par, Nseen, r


def pe_subsets():
    m = META['PE']
    ids = list(range(len(m)))
    herd = {'P008283', 'P008294', 'P008295', 'P008389', 'P008905', 'P008939', 'P008259', 'P009151'}
    return {
        'all': ids,
        'susa': [i for i in ids if m[i]['prov'].startswith('Susa')],
        'susa_noherd': [i for i in ids if m[i]['prov'].startswith('Susa') and m[i]['id'] not in herd],
        'small_tabs': [i for i in ids if m[i]['nent'] <= 8],
        'big_tabs': [i for i in ids if m[i]['nent'] > 8],
    }


def job_pe(seed):
    rng = random.Random(seed)
    d = rng.choice(['MID2', 'MID3', 'FULL2', 'VAR2', 'DIRTY2', 'NSIGN', 'HDR', 'DENT'])
    subs = pe_subsets()
    sub = rng.choice(list(subs))
    est = rng.choice(['chao2', 'ichao2', 'jack1', 'jack2', 'chapman', 'll3', 'll3vol'])
    idx = [i for i in subs[sub] if CAPS['PE'][d][i]]
    inc = [set(CAPS['PE'][d][i]) for i in idx]
    if len(inc) < 20:
        return None
    groups = None
    if est == 'll3vol':
        vols = sorted({META['PE'][i]['vol'] for i in idx})
        g = {v: rng.randrange(3) for v in vols}
        groups = [g[META['PE'][i]['vol']] for i in idx]
        if len(set(groups)) < 3:
            return None
    r = run_est(inc, rng, which=('ll3' if est.startswith('ll3') else est,), groups=groups)
    if not r:
        return None
    v = list(r.values())[0]
    return {'def': d, 'sub': sub, 'est': est, 'n': len(inc), 'S': len(set().union(*inc)), 'N': v[0], 'lo': v[1], 'hi': v[2]}


def job_shuffle(seed):
    """volume-grouped LL3 on real vs label-shuffled PE MID2 (Susa)."""
    rng = random.Random(seed)
    subs = pe_subsets()
    idx = [i for i in subs['susa'] if CAPS['PE']['MID2'][i]]
    inc = [set(CAPS['PE']['MID2'][i]) for i in idx]
    vols = sorted({META['PE'][i]['vol'] for i in idx})
    g = {v: rng.randrange(3) for v in vols}
    groups = [g[META['PE'][i]['vol']] for i in idx]
    out = {}
    if len(set(groups)) < 3:
        return None
    real = ll3(inc, rng, groups=groups, boot=5)
    pool = [e for s in inc for e in s]
    rng.shuffle(pool)
    sh, k = [], 0
    for s in inc:
        sh.append(set(pool[k:k + len(s)]))
        k += len(s)
    shf = ll3(sh, rng, groups=groups, boot=5)
    # within-volume recapture: fraction of repeated entities whose tablets share a volume
    def same_vol(I):
        where = collections.defaultdict(list)
        for s, i in zip(I, idx):
            for e in s:
                where[e].append(META['PE'][i]['vol'])
        rep = [v for v in where.values() if len(v) >= 2]
        return sum(len(set(v)) == 1 for v in rep) / max(len(rep), 1), len(rep)
    out['real'] = (real[0], real[3]) if real else None
    out['shuf'] = (shf[0], shf[3]) if shf else None
    out['samevol_real'] = same_vol(inc)
    out['samevol_shuf'] = same_vol(sh)
    return out


if __name__ == '__main__':
    t0 = time.time()
    res = {}
    with Pool(2) as P:
        jobs = [(a, d, r, NPE) for a in ('DREHEM', 'UMMA') for d in ('OFF', 'ENT2') for r in range(60)]
        res['ur3'] = P.map(job_ur3, jobs, chunksize=4)
        print('ur3 done', time.time() - t0, flush=True)
        res['plant'] = P.map(job_plant, range(60), chunksize=2)
        print('plant done', time.time() - t0, flush=True)
        res['shuffle'] = [x for x in P.map(job_shuffle, range(200), chunksize=10) if x]
        print('shuffle done', time.time() - t0, flush=True)
        res['pe'] = [x for x in P.map(job_pe, range(4000), chunksize=50) if x]
        print('pe done', time.time() - t0, flush=True)
    json.dump(res, open(os.path.join(CK, 'cycle1.json'), 'w'), default=float)
