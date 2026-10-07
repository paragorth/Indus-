#!/usr/bin/env python3
"""LA-84 cycle 1: THE READER'S PULL.  ABC over random reader models.

Premise (physics): SigLA stipple is surface damage; it does not know the text. So a worn sign-group, before it was
read, is a draw from the unworn groups of the same stratum (site HT/other x tablet/other x length 2/3/4+).
A reader model (m share of signs worn, e misread rate per worn sign, g share of misreads pulled to a sign that makes
an attested group, a frequency exponent of that pull, b frequency exponent of a random misread) is applied to matched
unworn groups; the summary (exact-familiar rate, near-only rate, mean log count) is compared with the worn groups.
Usage: la84_c1.py bank N SEED | la84_c1.py abc
"""
import sys, os, json, random, math
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la84_common as C

TOKS, LEX = C.load()
LX = C.Lex(LEX)
BY = {}
for t in TOKS:
    BY.setdefault((t['cls'], t['stratum']), []).append(t)
WORN = [t for t in TOKS if t['cls'] == 'WORN']
STRATA = {}
for t in WORN:
    STRATA[t['stratum']] = STRATA.get(t['stratum'], 0) + 1


def read_word(w, doc, P, rng):
    m, e, g, a, b = P
    w = list(w)
    pos = [i for i in range(len(w)) if rng.random() < m] or [rng.randrange(len(w))]
    for i in pos:
        if rng.random() >= e:
            continue
        cur = w[i]
        done = False
        if rng.random() < g:
            key = (len(w), i, tuple(w[:i]) + ('_',) + tuple(w[i + 1:]))
            cand = [(x, LX.count(v, doc)) for x, v in LX.nb.get(key, ()) if x != cur]
            cand = [(x, c) for x, c in cand if c > 0]
            if cand:
                ws = [c ** a for _, c in cand]
                w[i] = rng.choices([x for x, _ in cand], ws)[0]
                done = True
        if not done:
            while True:
                x = rng.choices(LX.signs, LX.wb[b])[0]
                if x != cur:
                    break
            w[i] = x
    return tuple(w)


BGRID = [round(0.1 * k, 1) for k in range(11)]
LX.wb = {bb: [f ** bb for f in LX.sfreq] for bb in BGRID}


def sample_set(rng, P=None, cls='READ', strata=None):
    out = []
    for s, n in (strata or STRATA).items():
        pool = BY.get((cls, s), [])
        for _ in range(n):
            t = rng.choice(pool)
            w = read_word(t['s'], t['doc'], P, rng) if P else t['s']
            out.append((w, t['doc']))
    return out


def rand_params(rng):
    return (rng.uniform(0.15, 1.0), rng.uniform(0, 0.8), rng.uniform(0, 1), rng.uniform(0, 2), rng.choice(BGRID))


def job(args):
    seed, n = args
    rng = random.Random(seed)
    res = []
    for _ in range(n):
        P = rand_params(rng)
        res.append(list(P) + list(LX.stats(sample_set(rng, P))))
    return res


def bank(N, seed):
    chunks = [(seed * 1000 + k, N // 20) for k in range(20)]
    with Pool(2) as pool:
        out = sum(pool.map(job, chunks), [])
    np.save(os.path.join(C.CK, 'c1_bank_%d.npy' % seed), np.array(out))


def abc(B, obs, frac=0.01):
    S = B[:, 5:8]
    sd = S.std(0)
    d = np.sqrt((((S - obs) / sd) ** 2).sum(1))
    k = max(50, int(frac * len(B)))
    idx = np.argsort(d)[:k]
    return B[idx, :5], d[idx]


if __name__ == '__main__':
    if sys.argv[1] == 'bank':
        bank(int(sys.argv[2]), int(sys.argv[3]))
        sys.exit()
    files = [f for f in os.listdir(C.CK) if f.startswith('c1_bank_')]
    B = np.concatenate([np.load(os.path.join(C.CK, f)) for f in files])
    rng = random.Random(84)
    out = dict(n_bank=len(B), strata={'/'.join(map(str, k)): v for k, v in STRATA.items()}, n_worn=len(WORN))
    obs = np.array(LX.stats([(t['s'], t['doc']) for t in WORN]))
    ref = [LX.stats(sample_set(rng)) for _ in range(200)]
    out['obs_worn'] = obs.tolist()
    out['matched_read_mean'] = np.mean(ref, 0).tolist()
    out['matched_read_sd'] = np.std(ref, 0).tolist()
    post, _ = abc(B, obs)
    out['post_real'] = dict(zip('m e g a b'.split(), [np.percentile(post[:, j], [5, 50, 95]).tolist() for j in range(5)]))
    # controls
    ctl = {}
    for name, P in [('null_noreader', None), ('planted_random', (0.5, 0.4, 0.0, 1.0, 0.5)),
                    ('planted_gravity', (0.5, 0.4, 0.85, 1.0, 0.5))]:
        rows = []
        for r in range(20):
            o = np.array(LX.stats(sample_set(rng, P)))
            p, _ = abc(B, o)
            rows.append([float(np.median(p[:, 1])), float(np.median(p[:, 2])),
                         float(np.mean(p[:, 1] * p[:, 2])), float(np.mean(p[:, 1] * (1 - p[:, 2])))])
        rows = np.array(rows)
        ctl[name] = dict(e_med=rows[:, 0].tolist(), g_med=rows[:, 1].tolist(),
                         mean_e=float(rows[:, 0].mean()), mean_g=float(rows[:, 1].mean()),
                         pull=float(rows[:, 2].mean()), noise=float(rows[:, 3].mean()))
    out['controls'] = ctl
    out['real_pull'] = float(np.mean(post[:, 1] * post[:, 2]))
    out['real_noise'] = float(np.mean(post[:, 1] * (1 - post[:, 2])))
    json.dump(out, open(os.path.join(C.CK, 'c1_abc.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
