"""X-4 cycle 1: arrow of time for administrative lists (Voynich v23 battery adapted).

Unit of exchange = the document (tablet / page / 6-line prose chunk); the time reversal reads a document
backwards (lines, words and signs). All corpora are cut to the same word budget (whole documents).
Measures, per corpus x budget x seed x condition:
  sign level  ep1   lag-1 sign-pair entropy production KL(P(a,b)||P(b,a)) in bits over the doc stream
                    (word and line separators kept as symbols), minus its mean over 4 random-direction copies
              kn    KN-3 sign model fwd vs bwd, 5-fold by document, per-doc bits difference, sign-flip z
                    (+ = written direction is easier)
              sprob random sign-partition probes (300, lags 1-6): held-out survivors
  word level  random probes on the document word stream (1,000: 200 each of freq / random / len / first /
              last class maps, lags 1-4), survivors = |z_A| >= 3 and z_B >= 2 same sign on a random half split
Conditions: real; rand0/rand1 (each doc read in a random direction: exact null, survivors must be ~0; word
level reverses word order only, sign level reverses everything);
body (first and last line of every doc dropped: header / total layout control); generators fitted to the
same subsample (WBG word bigram, TRI sign trigram, SLOT independent words, CPV copy-and-vary).
Out: data/x4_ckpt/c1/<job>.json, loops/x4_cycle1.txt (rows written by x4_cycle1_sum.py).
"""
import os, sys, json, math, random, time, zlib
from collections import Counter
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x4_lib as X

OUT = os.path.join(X.CK, 'c1'); os.makedirs(OUT, exist_ok=True)
KINDS = ['freq', 'random', 'len', 'first', 'last']
NPK = 200
SPROBE = 300


def rev_doc(d):
    return [[w[::-1] for w in l[::-1]] for l in d[::-1]]


def pairs(seqs_by_doc, lag, ix):
    I, J, D = [], [], []
    for di, seqs in enumerate(seqs_by_doc):
        for s in seqs:
            ids = [ix[x] for x in s]
            for i in range(len(ids) - lag):
                I.append(ids[i]); J.append(ids[i + lag]); D.append(di)
    return np.array(I, int), np.array(J, int), np.array(D, int)


def probe_run(seqs_by_doc, alpha, classes_fn, lags, nprobe, A, rng):
    ix = {x: i for i, x in enumerate(alpha)}
    P = {k: pairs(seqs_by_doc, k, ix) for k in lags}
    nd = len(seqs_by_doc)
    inA = np.zeros(nd, bool); inA[list(A)] = True
    surv, nA = Counter(), Counter()
    for _ in range(nprobe):
        c, m, kind = classes_fn(rng)
        k = rng.choice(list(lags)); a, b = rng.sample(range(m), 2)
        I, J, Dd = P[k]
        if len(I) == 0: continue
        f = (c[I] == a) & (c[J] == b); g = (c[I] == b) & (c[J] == a)
        D = np.bincount(Dd[f], minlength=nd) - np.bincount(Dd[g], minlength=nd)
        DA, DB = D[inA].astype(float), D[~inA].astype(float)
        zA = DA.sum() / math.sqrt((DA ** 2).sum() + 1e-9); zB = DB.sum() / math.sqrt((DB ** 2).sum() + 1e-9)
        if abs(zA) >= 3:
            nA[kind] += 1
            if np.sign(zA) == np.sign(zB) and abs(zB) >= 2: surv[kind] += 1
    return dict(surv), dict(nA)


def word_classes(alpha, gf, rng):
    V = len(alpha)
    def f(r):
        m = r.choice([2, 3, 4]); kind = KINDS[f.i % 5]; f.i += 1
        if kind == 'random':
            c = np.array([r.randrange(m) for _ in alpha])
        elif kind in ('first', 'last'):
            key = (lambda a: a[0]) if kind == 'first' else (lambda a: a[-1])
            g = {}
            c = np.array([g.setdefault(key(a), r.randrange(m)) for a in alpha])
        elif kind == 'len':
            th = sorted(r.sample(range(1, 7), m - 1))
            c = np.array([sum(len(a) > t for t in th) for a in alpha])
        else:
            th = sorted(r.sample(range(1, 8), m - 1))
            c = np.array([sum(math.log2(gf.get(a, 0) + 1) > t for t in th) for a in alpha])
        return c, m, kind
    f.i = 0
    return f


def sign_classes(alpha):
    def f(r):
        m = r.choice([2, 3, 4])
        c = np.array([m if a in (' ', '|') else r.randrange(m) for a in alpha])
        return c, m + 1, 'sign'
    return f


def sign_seq(d):
    return list('|'.join(' '.join(l) for l in d))


def kl_asym(cnt, eps=0.5):
    keys = set(cnt) | {(b, a) for a, b in cnt}
    tot = sum(cnt.values()) + eps * len(keys); s = 0.0
    for (a, b) in keys:
        p = (cnt.get((a, b), 0) + eps) / tot; q = (cnt.get((b, a), 0) + eps) / tot
        s += p * math.log2(p / q)
    return s


def ep1(docs):
    c = Counter()
    for d in docs:
        s = sign_seq(d); c.update(zip(s, s[1:]))
    return kl_asym(c)


def kn_dir(docs, seed=0):
    import v23_lib as V
    seqs = [[sign_seq(d)] for d in docs]
    out = V.kn_arrow(seqs, 3, nfold=5, seed=seed)
    a = [bb - bf for bf, bb, ns in out if ns]
    z, p = V.signflip(a, nperm=4000, seed=seed)
    return z, float(np.sum(a) / max(1, sum(ns for _, _, ns in out)))


def run(job):
    name, budget, seed, cond = job
    fn = os.path.join(OUT, f'{name}_{budget}_{seed}_{cond}.json')
    if os.path.exists(fn): return job
    t0 = time.time()
    C = X.corpora()[name]
    base = X.subsample(C, budget, seed)
    rng = random.Random(zlib.crc32(f'{name}{budget}{seed}{cond}'.encode()))
    if cond == 'real': docs = base
    elif cond.startswith('rand'):
        flips = [rng.random() < 0.5 for d in base]
        docs = [rev_doc(d) if f else d for d, f in zip(base, flips)]
        # word level: reverse word ORDER only (reversed spellings would be new word types and break the null)
        docs_w = [[l[::-1] for l in d[::-1]] if f else d for d, f in zip(base, flips)]
    elif cond == 'body': docs = X.body_only(base)
    else: docs = X.GENS[cond](base, rng)
    if not cond.startswith('rand'): docs_w = docs
    keep = [i for i, d in enumerate(docs) if X.ntok([d])]
    docs = [docs[i] for i in keep]; docs_w = [docs_w[i] for i in keep]
    res = {'name': name, 'budget': budget, 'seed': seed, 'cond': cond, 'ndoc': len(docs), 'ntok': X.ntok(docs)}
    split = random.Random(5 + seed)
    A = set(split.sample(range(len(docs)), len(docs) // 2))
    # word probes
    gf = Counter(w for d in docs_w for w in X.stream(d))
    alpha = sorted(gf)
    wseq = [[X.stream(d)] for d in docs_w]
    res['wsurv'], res['wdisc'] = probe_run(wseq, alpha, word_classes(alpha, gf, None), range(1, 5),
                                           NPK * 5, A, random.Random(17))
    # sign probes
    sseq = [[sign_seq(d)] for d in docs]
    salpha = sorted({x for s in sseq for x in s[0]})
    res['ssurv'], res['sdisc'] = probe_run(sseq, salpha, sign_classes(salpha), range(1, 7), SPROBE, A,
                                           random.Random(19))
    # entropy production, excess over random-direction copies
    e = ep1(docs); r2 = random.Random(23)
    nul = [ep1([rev_doc(d) if r2.random() < 0.5 else d for d in docs]) for _ in range(4)]
    res['ep1'] = e; res['ep1_null'] = float(np.mean(nul)); res['ep1_x'] = e - float(np.mean(nul))
    res['kn_z'], res['kn_bits'] = kn_dir(docs, seed)
    res['sec'] = time.time() - t0
    json.dump(res, open(fn, 'w'))
    print(name, budget, seed, cond, res['wsurv'], res['ssurv'].get('sign', 0), round(res['ep1_x'], 3),
          round(res['kn_z'], 2), round(res['sec']), flush=True)
    return job


def jobs():
    C = X.corpora()
    J = []
    for budget in (3000, 8000):
        for name in X.LIST + ['VOY'] + X.PROSE:
            if X.ntok(C[name]) < budget * 0.95: continue
            for seed in (0, 1):
                for cond in ['real', 'rand0', 'rand1', 'body', 'WBG', 'TRI', 'SLOT', 'CPV']:
                    J.append((name, budget, seed, cond))
    return J


if __name__ == '__main__':
    J = jobs()
    print(len(J), 'jobs', flush=True)
    with Pool(int(os.environ.get('W', '2'))) as pool:
        for _ in pool.imap_unordered(run, J): pass
