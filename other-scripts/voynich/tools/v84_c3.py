"""v84 cycle 3: POSITIVE CONTROLS AND A DISCRIMINATING STATISTIC.
(1) fit: the cycle-2 generator family (v84_lib.gen_page) fitted to each real control (herbals Konrad, Culpeper;
    recipe books Apicius, Forme of Cury, all through the merge code + v72 surface) so that the generator matches the
    control's own PB, PB_far / PB and surface statistics (bands: +-25% or the v82 floors). 120 random settings + 8
    climbing rounds x 20 per control.
(2) stats: v84_c3lib statistics on each control, its page-shuffled twin, and its best fitted generators (best 3
    settings x 2 fresh seeds); then on ZL3b, IT2a and the Voynich-fitted generators of cycle 2.
PRE-REGISTERED SELECTION RULE (before any Voynich value of these statistics is computed): a statistic discriminates
if in >= 3 of the 4 controls the real value lies outside the range of all 6 fitted-generator runs, on the same side in
every such control. The Voynich then counts as content-like on that statistic if ZL3b and IT2a both lie beyond the
range of every Voynich-fitted generator run, on the real side.
"""
import os, sys, json, random, copy, math, time, zlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v84_lib as K, v82_lib as K82, v84_c3lib as T
import v84_c2 as C2
from multiprocessing import Pool

CTRLS = ['HERB_KONRAD', 'HERB_CULP', 'RECI_APIC', 'RECI_CURY']
SRC = None; BND = None; TGT = None; MC = {}


def init(name):
    global SRC, BND, TGT
    SRC = K.controls()[name]
    s = K82.surf_stats(SRC); BND = K82.bands(s, s)
    r = K.PB(SRC, nnull=10, variants=('PB', 'PB_far'))
    TGT = (r['PB'][0], r['PB_far'][0] / r['PB'][0])


def score(args):
    P, seed = args
    try:
        X = K.gen_page(SRC, P, seed, MC)
    except Exception as e:
        return dict(P=P, seed=seed, err=repr(e))
    r = K.PB(X, nnull=4, variants=('PB', 'PB_far'))
    s = K82.surf_stats(X); ok, bad = K82.match(s, BND)
    return dict(P=P, seed=seed, pb=r['PB'][0], far=r['PB_far'][0], stats=s, ok=ok, bad=bad)


def fit(r):
    if 'err' in r: return -1e9
    d = sum(max(0.0, abs(r['stats'][k] - BND[k][0]) / BND[k][1] - 1) for k in K82.STAT_KEYS)
    pb = max(r['pb'], 1e-3)
    return -(2 * abs(math.log(pb / TGT[0])) + 3 * abs(r['far'] / pb - TGT[1]) + 0.5 * len(r['bad']) + 0.2 * d)


def jl(name, recs):
    with open(os.path.join(K.CK, name), 'a') as f:
        for r in recs: f.write(json.dumps(r, default=float) + '\n')


def load(name):
    p = os.path.join(K.CK, name)
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def stat_job(args):
    label, pages = args
    s = T.stats(pages); s['T_conf'] = T.confined(pages)
    r = K.PB(pages, nnull=6, variants=('PB', 'PB_far'))
    s['PB'] = r['PB'][0]; s['T_far'] = r['PB_far'][0] / max(r['PB'][0], 1e-3)
    return label, s


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'fit':
        name = sys.argv[2]; init(name)
        rng = random.Random(8430 + zlib.crc32(name.encode()) % 1000)
        fn = 'c3_fit_%s.jsonl' % name
        pop = [r for r in load(fn) if 'err' not in r]
        with Pool(2, initializer=init, initargs=(name,)) as Pl:
            if len(pop) < 80:
                jobs = [(K.sample_params(rng), 200 + i) for i in range(80 - len(pop))]
                new = list(Pl.imap_unordered(score, jobs, chunksize=2)); jl(fn, new); pop += [r for r in new if 'err' not in r]
            for rd in range(6):
                pop.sort(key=fit, reverse=True); top = pop[:10]
                jobs = [(C2.perturb(t['P'], rng), 7000 + rd * 100 + j * 10 + k) for j in range(2) for k, t in enumerate(top)]
                new = [r for r in Pl.imap_unordered(score, jobs) if 'err' not in r]; jl(fn, new); pop += new
                b = max(pop, key=fit)
                print(name, 'round', rd, 'target pb %.3f far %.2f | best pb %.3f far %.2f bad %s' % (TGT[0], TGT[1], b['pb'], b['far'] / max(b['pb'], 1e-3), b['bad']), flush=True)
    elif stage == 'stats':
        C = K.controls(); jobs = []
        for name in CTRLS:
            init(name)
            pop = sorted([r for r in load('c3_fit_%s.jsonl' % name) if 'err' not in r], key=fit, reverse=True)
            jobs.append((name + '|real', C[name])); jobs.append((name + '|shuf', C[name + '_SHUF']))
            seen = []
            for r in pop:
                if len(seen) == 3: break
                if any(json.dumps(r['P'], sort_keys=True) == x for x in seen): continue
                seen.append(json.dumps(r['P'], sort_keys=True))
                for sd in (9301, 9302):
                    jobs.append(('%s|gen%d_%d|%s' % (name, len(seen), sd, C2.mechs(r['P'])), K.gen_page(SRC, r['P'], sd, {})))
        with Pool(2) as Pl:
            res = dict(Pl.imap(stat_job, jobs))
        json.dump(res, open(os.path.join(K.CK, 'c3_ctrl_stats.json'), 'w'), default=float, indent=0)
        for k, v in res.items(): print(k, {a: round(b, 3) for a, b in v.items()})
    elif stage == 'voynich':
        # applied only after the selection rule has been run on the controls (c3_ctrl_stats.json)
        Z = K.voynich('ZL3b'); I = K.voynich('IT2a')
        fin = [r for r in C2.load('c2_final.jsonl') if 'err' not in r]
        C2.init(); fin.sort(key=C2.fit, reverse=True)
        jobs = [('VOY|ZL', Z), ('VOY|IT', I)]
        seen = []
        for r in fin:
            key = json.dumps(r['P'], sort_keys=True)
            if key in seen or len(seen) >= 6: continue
            seen.append(key)
            for sd in (9401, 9402):
                jobs.append(('VOY|gen%d_%d|%s' % (len(seen), sd, C2.mechs(r['P'])), K.gen_page(Z, r['P'], sd, {})))
        with Pool(2) as Pl:
            res = dict(Pl.imap(stat_job, jobs))
        json.dump(res, open(os.path.join(K.CK, 'c3_voy_stats.json'), 'w'), default=float, indent=0)
        for k, v in res.items(): print(k, {a: round(b, 3) for a, b in v.items()})
