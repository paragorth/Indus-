#!/usr/bin/env python3
"""LA-40 cycle 3: held-out stability and transfer.

(a) The pre-registered system of cycle 1 and the top learned systems of cycle 2 are run on
    random halves of the Linear A tablets (disjoint), on whole Linear A, and on word-shuffled
    Linear A. Held-out agreement = share of word types seen in both halves that get the same
    modal role, against the agreement expected from the two halves' role marginals.
(b) The same on Pylos halves (Linear A size per half) as the control: if roles are structure,
    halves agree beyond the marginals.
"""
import collections, json, os, random, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la40_common as C
import la40_c2 as C2

NSPLIT = int(os.environ.get('NSPLIT', 6))
STEPS = int(os.environ.get('STEPS', 300000))
NCH = int(os.environ.get('NCH', 4))
PYSCALE = float(os.environ.get('PYSCALE', 2))
TAG = os.environ.get('TAG', '')
ONLY = os.environ.get('ONLY', '')


def systems():
    c2 = json.load(open(os.path.join(C.CK, 'c2.json')))
    top = sorted(zip(c2['test_top'], c2['top']), reverse=True)[:3]
    out = [('prereg', None)]
    for sc, h in top:
        out.append(('learned%d' % h, h))
    return out


def run(docs, sysname, h, seed):
    B = C.build(docs)
    if h is None:
        S = C.run_chains(B, nchains=NCH, steps=STEPS, thin=STEPS // 100, seed0=seed)
    else:
        s = C2.draw_sys(np.random.default_rng(C.seed('la40c2-sys-%d' % h)))
        S = C2.run_sys(B, C2.occ_onehot(B), s, seed=seed, steps=STEPS, nch=NCH)
    mode, freq, agree, _ = C.summarize(S)
    return {w: (C.ROLES[mode[t]], float(freq[t]), int(B['nocc'][t])) for t, w in enumerate(B['types'])}


def agreement(A, Bm):
    shared = [w for w in A if w in Bm]
    if not shared:
        return 0, float('nan'), float('nan'), []
    obs = np.mean([A[w][0] == Bm[w][0] for w in shared])
    pa = collections.Counter(A[w][0] for w in shared); pb = collections.Counter(Bm[w][0] for w in shared)
    n = len(shared)
    exp = sum(pa[r] * pb[r] for r in C.ROLES) / n / n
    return n, float(obs), float(exp), [(w, A[w][0], Bm[w][0]) for w in shared]


def job(args):
    corpus, sysname, h, k = args
    part = os.path.join(C.CK, 'c3_parts', '%s%s_%s_%d.json' % (corpus, TAG, sysname, k))
    if os.path.exists(part):
        return json.load(open(part))
    r = _job(args)
    json.dump(r, open(part + '.tmp', 'w')); os.replace(part + '.tmp', part)
    return r


def _job(args):
    corpus, sysname, h, k = args
    rng = random.Random(C.seed('la40c3-%s-%d' % (corpus, k)))
    if corpus == 'LA':
        docs = C.la_docs()
    else:
        D = [d for d in C.lb_docs_all() if d['site'] == 'PY']
        LA = C.la_docs(); ntok = sum(1 for d in LA for ln in d['lines'] for t in ln if C.is_word(t))
        docs = C.lb_sample(D, int(PYSCALE * ntok), rng)
    idx = list(range(len(docs))); rng.shuffle(idx)
    A = [docs[i] for i in idx[:len(idx) // 2]]; Bd = [docs[i] for i in idx[len(idx) // 2:]]
    ra = run(A, sysname, h, 100 + k); rb = run(Bd, sysname, h, 200 + k)
    n, obs, exp, pairs = agreement(ra, rb)
    # shuffle control: same halves with words shuffled inside each half
    rsa = run(C.shuffle_words(A, rng), sysname, h, 300 + k); rsb = run(C.shuffle_words(Bd, rng), sysname, h, 400 + k)
    ns, obss, exps, _ = agreement(rsa, rsb)
    return dict(corpus=corpus, sys=sysname, k=k, n=n, obs=obs, exp=exp, n_s=ns, obs_s=obss, exp_s=exps, pairs=pairs)


def main():
    t = time.time()
    os.makedirs(os.path.join(C.CK, 'c3_parts'), exist_ok=True)
    jobs = [(c, s, h, k) for (s, h) in systems() for c in ((ONLY,) if ONLY else ('LA', 'PY')) for k in range(NSPLIT)]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(C.CK, 'c3%s.json' % TAG), 'w'))
    by = collections.defaultdict(list)
    for r in res: by[(r['corpus'], r['sys'])].append(r)
    for key, rs in sorted(by.items()):
        ex = np.mean([r['obs'] - r['exp'] for r in rs]); exs = np.mean([r['obs_s'] - r['exp_s'] for r in rs])
        print('%s %-12s shared %4.0f  agree %.3f vs marg %.3f (excess %+.3f) | shuffled excess %+.3f  wins %d/%d' % (
            key[0], key[1], np.mean([r['n'] for r in rs]), np.mean([r['obs'] for r in rs]), np.mean([r['exp'] for r in rs]),
            ex, exs, sum((r['obs'] - r['exp']) > (r['obs_s'] - r['exp_s']) for r in rs), len(rs)))
    print('done %.0fs' % (time.time() - t))


if __name__ == '__main__':
    main()
