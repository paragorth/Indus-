"""v96 cycle 2 (N5): TWO DIALS. Simulate writers from real texts with a TOPIC dial and a KEY dial, and ask which dial
settings make books whose page-level fingerprint (each statistic minus the same statistic on the book's own line-dealt
twin) looks like the Voynich's.
Book = a real text through the v72 merge code + surface in its real page order (P_X), with a fraction 1-tau of its lines
dealt among the line slots of their page group (tau = 1 keeps every page's topic, tau = 0 destroys it), then a random
page key (none with probability 1/4; else 6-16 rules, strength 0.3-0.9, as in cycle 1).
Twin = the same book with every line dealt (topic and key both destroyed). The Voynich gets the same twin treatment, so
every summary is relative to the corpus's own page-free version.
Summaries (book minus twin): dC, dR = cycle-1 common / rare stratum page spelling (8 frozen cycle-1 views, both halves);
dW, dF = v92 recurrence profile of rare words / frames; conf = share of page-recurring rare words confined to their page;
loc = mean gap between successive lines holding a page-recurring rare word, observed / expected under line-order
permutation (book only; < 1 = clustered, the v92 topic signature).
ABC: random books, regression (kNN + ridge) of tau and of key-present on the summaries, cross-validated by text
(leave-one-text-out); applied to fresh W1 plants (tau = 1 truth), to their dealt twins (tau = 0), to generators and to the
Voynich (frozen first)."""
import os, sys, json, time, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v96_lib as L
import v92_lib as L92
import v96_c1 as C1

NSIM = int(os.environ.get('V96_NSIM', '100'))     # books per text
FZ1 = json.load(open(os.path.join(L.DATA, 'v96_frozen_c1.json')))
VIEWS = FZ1['views'][:8]


def locality(pages, rng, nperm=20):
    cnt = Counter(w for p in pages for l in p['lines'] for w in l['w'])
    obs = exp = 0.0
    for p in pages:
        occ = defaultdict(list)
        for li, l in enumerate(p['lines']):
            for w in set(l['w']):
                if 2 <= cnt[w] <= 20: occ[w].append(li)
        sets = [v for v in occ.values() if len(v) >= 2]
        if not sets: continue
        n = len(p['lines'])
        o = np.mean([np.mean(np.diff(sorted(s))) for s in sets]); e = 0.0
        for _ in range(nperm):
            perm = rng.permutation(n)
            e += np.mean([np.mean(np.diff(sorted(perm[s]))) for s in [np.array(x) for x in sets]])
        obs += o * len(sets); exp += e / nperm * len(sets)
    return obs / exp if exp else 1.0


def confinement(pages):
    cnt = Counter(w for p in pages for l in p['lines'] for w in l['w'])
    tot = conf = 0
    for p in pages:
        pc = Counter(w for l in p['lines'] for w in set(l['w']))
        lc = Counter(w for l in p['lines'] for w in l['w'])
        for w, nl in pc.items():
            if nl >= 2 and 2 <= cnt[w] <= 20:
                tot += 1; conf += lc[w] == cnt[w]
    return conf / tot if tot else 0.0


def raw(pages, seed):
    dC = []; dR = []
    for v in VIEWS:
        for h in (0, 1):
            r = C1.stats(pages, v, h)
            if r: dC.append(r['aC'] - 0.5); dR.append(r['aR'] - 0.5)
    rw, _ = L92.recurrence_profile(pages, unit=lambda w: w); rf, _ = L92.recurrence_profile(pages, unit=L92.frame)
    return dict(C=float(np.mean(dC)) if dC else 0.0, R=float(np.mean(dR)) if dR else 0.0, W=rw, F=rf, conf=confinement(pages),
                loc=locality(pages, np.random.default_rng(seed)))


def summary(pages, seed):
    a = raw(pages, seed); b = raw(L.deal_frac(pages, seed + 7, 1.0), seed + 1)
    return dict(dC=a['C'] - b['C'], dR=a['R'] - b['R'], dW=a['W'] - b['W'], dF=a['F'] - b['F'], conf=a['conf'] - b['conf'],
                loc=a['loc'], C=a['C'], R=a['R'])


def sim(args):
    X, k = args
    out = os.path.join(L.CK, 'c2_sim_%s_%03d.json' % (X, k))
    if os.path.exists(out): return out, 0.0
    t0 = time.time()
    rng = random.Random(96200 + 1000 * L.TEXTS.index(X) + k)
    tau = rng.random(); keyon = rng.random() >= 0.25
    pages = L.deal_frac(L92.corpus('P_' + X), rng.randrange(1 << 30), 1.0 - tau)
    spec = None
    if keyon:
        spec = L.key_spec(pages, rng.randrange(1 << 30)); pages = L.apply_key(pages, spec)
    s = summary(pages, rng.randrange(1 << 30))
    json.dump(dict(X=X, k=k, tau=tau, key=keyon, R=spec['R'] if spec else 0, s_=spec['s'] if spec else 0.0, sm=s),
              open(out, 'w'))
    return out, time.time() - t0


def target(name):
    out = os.path.join(L.CK, 'c2_tgt_%s.json' % name)
    if os.path.exists(out): return out, 0.0
    t0 = time.time()
    if name.endswith('_DEALT'):
        pages = L.deal_frac(L.corpus(name[:-6]), 96999, 1.0)
    else:
        pages = L.corpus(name)
    json.dump(dict(name=name, sm=summary(pages, 96777)), open(out, 'w'))
    return out, time.time() - t0


def job(a): return sim(a) if isinstance(a, tuple) else target(a)


if __name__ == '__main__':
    from multiprocessing import Pool
    if sys.argv[1:2] == ['targets']:
        jobs = sys.argv[2:]
    else:
        jobs = [(X, k) for k in range(NSIM) for X in L.TEXTS]
    with Pool(2) as P:
        for i, (o, s) in enumerate(P.imap_unordered(job, jobs)):
            if i % 25 == 0 or isinstance(jobs[0], str): print(i, os.path.basename(o), '%.0fs' % s, flush=True)
