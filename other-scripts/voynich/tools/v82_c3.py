"""v82 cycle 3: SHAPE KILL. Cycle 1 killed the number (+0.03 bits) but the killing generator has the wrong shape:
its excess pair table avoids repeating an onset (same-onset excess -0.12/-0.16) while the Voynich's is dominated by
onset repeats (+0.09/+0.13) and a symmetric ok<->chd exchange, and the Voynich link does not cross line breaks.
Here the same generator family is searched for the Voynich's whole SHAPE: 10-dim vector
  g0 g1 g2 (frozen gains), diag1 same1 asym1 (excess table, d1), cross1 cross0 (line-break gains), lag1 (lag2/lag1, d1)
distance = mean over features of ((x - ZL)/scale)^2 with scale = max(|ZL - IT2a|, floor); plus 1 per statistic
outside its band. Yardsticks: IT2a (the same text, other transcription), Swahili and Hebrew through the v72 surface,
GEN_KILL. A setting closer to ZL than the prose controls and within ~2x the IT2a distance would say the whole shape is
mechanical. 400 random settings + 12 climbing rounds x 24.
"""
import os, sys, json, random, copy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v82_lib as K, v82_c1 as C1, v82_c2 as C2, v81_lib as L81, v72_lib as V
from multiprocessing import Pool

FEATS = ['g0', 'g1', 'g2', 'diag1', 'same1', 'asym1', 'cross1', 'cross0', 'lag1']
FLOOR = dict(g0=0.01, g1=0.01, g2=0.015, diag1=0.05, same1=0.03, asym1=0.03, cross1=0.015, cross0=0.015, lag1=0.1)


def shape(c, nrep=10, seed=83):
    rng = np.random.default_rng(seed); f = {}
    fz = K.frozen(c, nrep=nrep)
    f['g0'], f['g1'], f['g2'] = [x[0] for x in fz]
    o1 = L81.onset(c, K.DEFS[1])
    D, Nm, _ = C2.ext_table(c, o1, o1, rng, nrep=nrep)
    pos = D.clip(min=0); f['diag1'] = float(np.trace(pos) / (pos.sum() + 1e-9))
    f['same1'] = float(np.trace(D) / (np.trace(Nm) + 1e-9)); f['asym1'] = C2.asym(D)
    f['cross1'] = C2.cross(c, o1, rng, nrep=nrep)['cross'][0]
    o0 = L81.onset(c, K.DEFS[0]); f['cross0'] = C2.cross(c, o0, rng, nrep=nrep)['cross'][0]
    lg = K.lag_ratio(c, nrep=5)[1]; f['lag1'] = float(np.clip(lg[1] / max(lg[0], 1e-3), -1, 2))
    return f


REF = None; SC = None


def dist(f, nbad=0):
    return float(np.mean([((f[k] - REF[k]) / SC[k]) ** 2 for k in FEATS])) + nbad


def init():
    global REF, SC
    C1.init()
    R = json.load(open(os.path.join(K.CK, 'c3_ref.json')))
    REF = R['VOY_ZL']; SC = {k: max(abs(R['VOY_ZL'][k] - R['VOY_IT'][k]), FLOOR[k]) for k in FEATS}


def score(args):
    P, seed = args
    X = K.gen_moodagr(C1.Z, P, seed, C1.MC.setdefault('ZL', {}))
    c = K.Mini(X); f = shape(c)
    s = K.surf_stats(X); ok, bad = K.match(s, C1.B)
    return dict(P=P, seed=seed, f=f, bad=bad, ok=ok, d=dist(f, len(bad)))


def ref_one(name):
    C = K.pload('c2_corpora.pkl')
    return name, shape(K.Mini(C[name]))


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'ref':
        names = ['VOY_ZL', 'VOY_IT', 'SW', 'HE', 'LA', 'EO', 'CS', 'SW_REV', 'GEN_KILL', 'GEN_NEAR', 'GEN_PMI', 'GEN_STACK']
        with Pool(2) as Pl:
            R = dict(Pl.map(ref_one, names))
        json.dump(R, open(os.path.join(K.CK, 'c3_ref.json'), 'w'), default=float)
        for n, f in R.items(): print(n, ' '.join('%s %+.3f' % (k, f[k]) for k in FEATS))
    elif stage == 'search':
        rng = random.Random(8301)
        prev = C1.load('c1b.jsonl')
        seeds = [r['P'] for r in prev[::10]]
        with Pool(2, initializer=init) as Pl:
            jobs = [(K.sample_params(rng), 300 + i) for i in range(400)] + [(P, 290) for P in seeds]
            pop = list(Pl.imap_unordered(score, jobs, chunksize=4)); C1.jl('c3_search.jsonl', pop)
            print('random: best d %.2f' % min(r['d'] for r in pop), flush=True)
            for rd in range(12):
                pop.sort(key=lambda r: r['d']); top = pop[:8]
                jobs = [(C1.perturb(t['P'], rng), 1000 + rd * 50 + j) for j in range(3) for t in top]
                new = list(Pl.imap_unordered(score, jobs)); C1.jl('c3_search.jsonl', new); pop += new
                b = min(pop, key=lambda r: r['d'])
                print('round', rd, 'best d %.2f' % b['d'], b['bad'], {k: round(b['f'][k], 3) for k in FEATS}, flush=True)
    elif stage == 'confirm':
        init()
        pop = C1.load('c3_search.jsonl'); pop.sort(key=lambda r: r['d'])
        cands = []; seen = set()
        for r in pop:
            key = json.dumps(r['P'], sort_keys=True)
            if key not in seen: seen.add(key); cands.append(r['P'])
            if len(cands) == 3: break
        with Pool(2, initializer=init) as Pl:
            out = list(Pl.imap(score, [(P, 9300 + s) for P in cands for s in range(5)]))
        for i, r in enumerate(out): r['cand'] = i // 5
        C1.jl('c3_confirm.jsonl', out)
