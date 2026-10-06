"""pe58 cycle 3: does any sign convert at ONE fixed rate?
a  spike test calibrated: Ur III rate lines (gurusz, geme2, dumu, gu4, udu) and a planted ladder, vs PE.
b  minimal pairs: entries whose sign sets differ by exactly one added sign (w vs w+s) or one swapped sign
   (w+a vs w+b), within tablet and corpus-wide. Per sign: n, mode, share within +-0.05 of the mode, share at
   exactly 1:1. Nulls: quantities shuffled within tablet (within-tablet pairs) and across the corpus (corpus pairs).
c  1:1 pairing: share of exactly equal quantities between entries of two different weighted-sign classes on
   the same tablet, vs quantities shuffled within tablet.
d  split-half replication of the factors (20 random tablet halves): weighted vs unweighted signs."""
import sys, os, json, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe58_lib import *
from pe58_lib import _agg
from multiprocessing import Pool

W = json.load(open(os.path.join(DATA, 'pe52_frozen_weights.json')))['signs']


def spike_z(C, signs, nsh=10):
    E = Est(C)
    obs = {s: E.spike(s) for s in signs}
    nul = defaultdict(list)
    for k in range(nsh):
        Es = Est(qshuf_tab(C, 300 + k))
        for s in signs:
            r = Es.spike(s)
            if r:
                nul[s].append(r[0])
    out = {}
    for s in signs:
        if obs[s] and nul[s]:
            out[s] = dict(conc=obs[s][0], mode=obs[s][1], null=float(np.mean(nul[s])),
                          z=float((obs[s][0] - np.mean(nul[s])) / (np.std(nul[s]) + 1e-9)))
    return out


def minimal_pairs(C, scope):
    """returns {('add', s) or ('swap', a, b): [log ratios]}"""
    by = defaultdict(list)
    for i, r in enumerate(C):
        key = (r['tab'], r['sys']) if scope == 'tab' else r['sys']
        by[key].append(i)
    lq = [math.log(r['q']) for r in C]
    out = defaultdict(list)
    for key, idx in by.items():
        sets = {}
        for i in idx:
            sets.setdefault(frozenset(C[i]['w']), []).append(i)
        keys = list(sets)
        if scope != 'tab':
            # index by every one-sign deletion for speed
            dele = defaultdict(list)
            for k in keys:
                for s in k:
                    dele[k - {s}].append((k, s))
            for k in keys:
                for (k2, s) in dele.get(k, []):
                    out[('add', s)].append(float(np.mean([lq[i] for i in sets[k2]]) - np.mean([lq[i] for i in sets[k]])))
            for base, lst in dele.items():
                if len(lst) > 1:
                    for x in range(len(lst)):
                        for y in range(x + 1, len(lst)):
                            (ka, a), (kb, b) = lst[x], lst[y]
                            if a > b:
                                (ka, a), (kb, b) = (kb, b), (ka, a)
                            out[('swap', a, b)].append(float(np.mean([lq[i] for i in sets[ka]]) - np.mean([lq[i] for i in sets[kb]])))
        else:
            for x in range(len(keys)):
                for y in range(len(keys)):
                    if x == y:
                        continue
                    A, B = keys[x], keys[y]
                    d = A ^ B
                    if len(d) == 1 and B < A:
                        (s,) = tuple(d)
                        out[('add', s)].append(float(np.mean([lq[i] for i in sets[A]]) - np.mean([lq[i] for i in sets[B]])))
                    elif len(d) == 2 and len(A - B) == 1 and x < y:
                        (a,), (b,) = tuple(A - B), tuple(B - A)
                        v = float(np.mean([lq[i] for i in sets[A]]) - np.mean([lq[i] for i in sets[B]]))
                        if a > b:
                            a, b, v = b, a, -v
                        out[('swap', a, b)].append(v)
    return out


def mp_stats(v):
    v = np.array(v)
    if len(v) < 4:
        return None
    m = _agg(v, np.ones(len(v)), 'mode', 0.05)
    return dict(n=len(v), mode=float(m), at_mode=float(np.mean(np.abs(v - m) <= 0.05)),
                at_one=float(np.mean(np.abs(v) <= 0.01)), med=float(np.median(v)))


def mp_job(args):
    name, scope = args
    C = load(name)
    obs = minimal_pairs(C, scope)
    keys = [k for k, v in obs.items() if len(v) >= 6]
    nul = defaultdict(list)
    for k in range(8):
        Cs = qshuf_tab(C, 700 + k) if scope == 'tab' else qshuf_sys(C, 700 + k)
        ns = minimal_pairs(Cs, scope)
        for key in keys:
            st = mp_stats(ns.get(key, []))
            if st:
                nul[key].append((st['at_mode'], st['at_one']))
    res = {}
    for key in keys:
        st = mp_stats(obs[key])
        if not st or not nul[key]:
            continue
        a = np.array(nul[key])
        st.update(null_at_mode=float(a[:, 0].mean()), null_at_one=float(a[:, 1].mean()),
                  z_mode=float((st['at_mode'] - a[:, 0].mean()) / (a[:, 0].std() + 0.02)),
                  z_one=float((st['at_one'] - a[:, 1].mean()) / (a[:, 1].std() + 0.02)))
        res['|'.join(key)] = st
    return dict(job=f'MP_{name}_{scope}', res=res)


_cache = {}


def load(name):
    if name in _cache:
        return _cache[name]
    if name == 'PE':
        C = pe_corpus()
    elif name == 'UR3':
        C, voc = ur3_ta(opaque=False)
    elif name.startswith('PLANT'):
        C, f = planted(pe_corpus(), int(name[-1]), 'ladder', k=17, exclude=set(W))
        json.dump(f, open(os.path.join(CK, f'c3_{name}_truth.json'), 'w'))
    _cache[name] = C
    return C


def spike_job(name):
    C = load(name)
    if name == 'UR3':
        S = ['gurusz', 'geme2', 'dumu', 'gu4', 'udu', 'niga', 'erin2']
    elif name == 'PE':
        S = sorted(W)
    else:
        S = sorted(json.load(open(os.path.join(CK, f'c3_{name}_truth.json'))))
    return dict(job=f'SPIKE_{name}', res=spike_z(C, S))


def one_job(name):
    """1:1 pairing between different weighted classes on the same tablet"""
    C = load(name)
    if name == 'PE':
        S = set(W)
    elif name == 'UR3':
        S = {'gurusz', 'geme2', 'dumu', 'gu4', 'udu', 'erin2', 'niga'}
    else:
        S = set(json.load(open(os.path.join(CK, f'c3_{name}_truth.json'))))
    def stat(C):
        G = groups(C); eq = 0; tot = 0; eq_any = 0; tot_any = 0
        for g, idx in G.items():
            cl = [(i, frozenset(set(C[i]['w']) & S)) for i in idx]
            for x in range(len(cl)):
                for y in range(x + 1, len(cl)):
                    i, a = cl[x]; j, b = cl[y]
                    same = C[i]['q'] == C[j]['q']
                    if a and b and a != b:
                        tot += 1; eq += same
                    else:
                        tot_any += 1; eq_any += same
        return eq / max(1, tot), tot, eq_any / max(1, tot_any)
    o = stat(C)
    ns = [stat(qshuf_tab(C, 800 + k)) for k in range(30)]
    a = np.array([x[0] for x in ns])
    return dict(job=f'ONE_{name}', obs=o, null=float(a.mean()), sd=float(a.std()),
                p=float((np.sum(a >= o[0]) + 1) / (len(a) + 1)))


def split_job(seed):
    C = load('PE')
    tabs = sorted({r['tab'] for r in C}); rng = random.Random(seed); rng.shuffle(tabs)
    h = set(tabs[:len(tabs) // 2])
    A = [r for r in C if r['tab'] in h]; B = [r for r in C if r['tab'] not in h]
    oth = [s for s in frequent(C, 25, 8) if s not in W]
    S = sorted(W) + oth
    EA, EB = Est(A, W=set(W)), Est(B, W=set(W))
    out = {}
    for s in S:
        a = EA.factor(s, n=60, seed=seed, spec2=True); b = EB.factor(s, n=60, seed=seed, spec2=True)
        out[s] = (a['med'] if a else None, b['med'] if b else None)
    return dict(job=f'SPLIT_{seed}', res=out, nW=len(W))


def run(a):
    kind, x = a
    return {'mp': mp_job, 'spike': spike_job, 'one': one_job, 'split': split_job}[kind](x)


if __name__ == '__main__':
    jobs = [('spike', 'UR3'), ('spike', 'PLANT0'), ('one', 'PE'), ('one', 'UR3'), ('one', 'PLANT0')]
    jobs += [('mp', (n, s)) for n in ('PE', 'UR3', 'PLANT0') for s in ('tab', 'corpus')]
    jobs += [('split', s) for s in range(20)]
    with Pool(2) as P:
        res = P.map(run, jobs, chunksize=1)
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), default=str)
    print('done', len(res))
