"""LA-79 cycle 2: the survival law.

Generate N random sign-group role hypotheses (random partitions of signs by first / last /
second sign, first x last, last x length, any-sign membership, whole-word classes; K 2-5) that
predict what follows a sign-group (number / logogram / sign-group / end; la78 'prior' scoring).
For each, compute properties knowable in 1950 (from documents published by 1950):
  ins    in-sample gain on the past
  loso   leave-one-site-out on the past
  lofo   leave-one-findspot (deposit)-out on the past
  ent    site entanglement: mutual information between the hypothesis's classes and site (label-free)
  bal    class balance (normalised entropy), K, family
and its first future gain W1 (fit <= 1950, score 1951-1976).
Outcome = survival into the LATER futures: gain on W2 (fit <= 1976, score after) and W3 (<= 1988, after).
Question: which 1950-knowable property predicts survival? Controls: shuffled publication years;
planted stable / drifting grammars with the real site confound.
usage: la79_c2.py real N seed | shuf N seed | plant N seed
"""
import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la78_engine as E
from la78_common import load
from la79_common import CK, DATA

E.MODE['mode'] = 'prior'
docs = load()
T = E.tokens(docs)
C = {d['id']: d for d in json.load(open(os.path.join(DATA, 'corpus_ra.json')))}
signs = sorted({s for t in T for s in t['syl']})
S = {s: i for i, s in enumerate(signs)}
first = np.array([S[t['syl'][0]] for t in T]); last = np.array([S[t['syl'][-1]] for t in T])
second = np.array([S[t['syl'][1]] for t in T])
ln = np.array([min(len(t['syl']), 4) - 2 for t in T])
words = sorted({t['word'] for t in T}); W = {w: i for i, w in enumerate(words)}
wid = np.array([W[t['word']] for t in T])
member = np.zeros((len(T), len(signs)), bool)
for i, t in enumerate(T):
    for s in t['syl']:
        member[i, S[s]] = True
y1 = np.array([t['y1'] for t in T])
docid = np.array([t['doc'] for t in T])
year0 = np.array([docs[t['doc']]['year'] for t in T])
site = np.array([docs[t['doc']]['site'] for t in T])
fspot = np.array([docs[t['doc']]['site'] + '|' + (C[docs[t['doc']]['id']].get('findspot') or '?') for t in T])
FAMS = ['FIN', 'INI', 'SEC', 'PAIR', 'LENFIN', 'ANY', 'WORD']


def gen(rng, fam):
    K = int(rng.integers(2, 6)) if fam not in ('PAIR', 'LENFIN', 'ANY') else 2
    part = rng.integers(0, K, len(signs))
    if fam == 'FIN': f = part[last]
    elif fam == 'INI': f = part[first]
    elif fam == 'SEC': f = part[second]
    elif fam == 'PAIR': f = part[first] * 2 + part[last]
    elif fam == 'LENFIN': f = part[last] * 3 + ln
    elif fam == 'ANY':
        sel = rng.random(len(signs)) < rng.uniform(0.05, 0.3)
        f = (member[:, sel].any(1)).astype(np.int64)
    else:
        f = rng.integers(0, K, len(words))[wid]
    _, f = np.unique(f, return_inverse=True)
    return f.astype(np.int64), K


def mi(a, b):
    ua, a = np.unique(a, return_inverse=True); ub, b = np.unique(b, return_inverse=True)
    c = np.zeros((len(ua), len(ub))); np.add.at(c, (a, b), 1); p = c / c.sum()
    pa = p.sum(1, keepdims=True); pb = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pa @ pb)[nz])).sum())


def lofo(f, Y, tr, grp):
    tot = n = 0.0
    for g in np.unique(grp[tr]):
        te = tr & (grp == g); tr2 = tr & (grp != g)
        if te.sum() < 8 or tr2.sum() < 50:
            continue
        v = E.score(f, Y, tr2, te)
        if not np.isnan(v):
            tot += v * te.sum(); n += te.sum()
    return tot / max(n, 1)


def run(N, seed, year, Y, extra=None):
    rng = np.random.default_rng(seed)
    p50 = year <= 1950
    W1 = (p50, (year > 1950) & (year <= 1976))
    W2 = (year <= 1976, year > 1976)
    W3 = (year <= 1988, year > 1988)
    rows = []
    hyps = []
    if extra is not None:
        hyps.append(('TRUE', extra[0], extra[1]))
    for i in range(N):
        fam = FAMS[i % len(FAMS)]
        f, K = gen(rng, fam)
        hyps.append((fam, f, K))
    for fam, f, K in hyps:
        fp = f[p50]
        cnt = np.bincount(fp, minlength=f.max() + 1) / max(len(fp), 1)
        nz = cnt[cnt > 0]
        bal = float(-(nz * np.log2(nz)).sum() / max(math.log2(len(cnt)), 1e-9)) if len(cnt) > 1 else 0.0
        rows.append([FAMS.index(fam) if fam in FAMS else -1, K,
                     E.score(f, Y, p50, p50), E.loso_score(f, Y, p50, site), lofo(f, Y, p50, fspot),
                     mi(fp, site[p50]), bal,
                     E.score(f, Y, *W1), E.score(f, Y, *W2), E.score(f, Y, *W3)])
    return np.array(rows, float)


COLS = ['fam', 'K', 'ins', 'loso', 'lofo', 'ent', 'bal', 'W1', 'W2', 'W3']

if __name__ == '__main__':
    mode, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    Y = [(y1, 4)]
    if mode == 'real':
        R = run(N, seed, year0, Y)
        np.save(os.path.join(CK, 'c2_real_%d.npy' % seed), R)
    elif mode == 'shuf':
        rng = np.random.default_rng(seed + 999)
        dy = np.array([d['year'] for d in docs]); perm = rng.permutation(len(docs))
        year = dy[perm][docid]
        R = run(N, seed, year, Y)
        np.save(os.path.join(CK, 'c2_shuf_%d.npy' % seed), R)
    else:
        # planted world: labels from a true FIN/INI K3 partition + real per-site label mix; stable or drift
        rng = np.random.default_rng(seed + 555)
        sites = sorted(set(site)); LP = np.zeros((len(T), 4))
        for s in sites:
            c = np.bincount(y1[site == s], minlength=4) + 1.0; LP[site == s] = np.log(c / c.sum())
        variant = 'stable' if seed % 2 == 0 else 'drift'
        key = last if (seed // 2) % 2 == 0 else first
        part = rng.integers(0, 3, len(signs)); ft = part[key]; Em = rng.normal(0, 1, (3, 4))
        logit = LP + 1.0 * Em[ft]
        if variant == 'drift':
            p2 = rng.integers(0, 3, len(signs)); E2 = rng.normal(0, 1, (3, 4)); late = year0 > 1960
            logit[late] = LP[late] + 1.0 * E2[p2[key][late]]
        p = np.exp(logit - logit.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        yy = (p.cumsum(1) > rng.random((len(T), 1))).argmax(1)
        _, ftu = np.unique(ft, return_inverse=True)
        R = run(N, seed, year0, [(yy, 4)], extra=(ftu.astype(np.int64), 3))
        np.save(os.path.join(CK, 'c2_plant_%s_%d.npy' % (variant, seed)), R)
    print('done', mode, N, seed, flush=True)
