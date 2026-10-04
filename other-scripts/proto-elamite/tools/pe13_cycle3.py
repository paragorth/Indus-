"""pe13 cycle 3: whose memory?  Massive random guessing over tablet groups.

Per tablet (entries only): local-memory score S = sum over targets of
  log2 P_local - log2 P_flat,   P = (1-lam) Pbase + lam * cache(K)
with K_local = exp(-(d-1)/2) (no floor) and K_flat = 1, lam fitted on half A.
A tablet written from short-term memory (priming, or a list sorted into runs) has S > 0;
a topic tablet S ~ 0 or < 0.
Hypotheses: 3,000 random tablet groups defined by 1-2 conjoined features (header sign,
class signs present, number system, size, reverse present, columns, publication volume,
site).  Score on half A: Welch t of mean S (group vs rest).  Family-wise null: the same
3,000-hypothesis search on 20 within-tablet-shuffled copies of half A (max t).  Survivors
(top 20 with t above the FWER 95% line, or the top 20 regardless if none) re-tested on half B
(one-sided, Bonferroni x20) against 200 within-tablet shuffles of half B.
Planted control: PE entries with a PRIME kernel planted only in tablets whose header is M157,
must be found by the same pipeline.
usage: python3 pe13_cycle3.py [workers]
"""
import json, math, os, random, re, sys
import numpy as np
from collections import Counter
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe13_common as C  # noqa
from pe13_cycle2 import variant  # noqa
from common import load, system_of  # noqa

OUT = os.path.join(C.CK, 'c3')
os.makedirs(OUT, exist_ok=True)
KL = C.kernel({'shape': 'exp', 'tau': 2.0})
KF = C.kernel({'shape': 'flat'})


def tablet_scores(tabs, lam=None, base_tabs=None):
    P = C.prep(tabs, 'ent')
    pb = C.base_probs(P, np.arange(P['ntab']))[P['w']]
    cl = C.cache_prob(P, None, KL)
    cf = C.cache_prob(P, None, KF)
    if lam is None:
        lam = (C.LAMS[int(np.argmax(C.ll_grid(pb, cl)))], C.LAMS[int(np.argmax(C.ll_grid(pb, cf)))])
    s = (np.log2((1 - lam[0]) * pb + lam[0] * cl) - np.log2((1 - lam[1]) * pb + lam[1] * cf))
    S = np.zeros(P['ntab'])
    np.add.at(S, P['tab'], s)
    n = np.bincount(P['tab'], minlength=P['ntab'])
    return S / np.maximum(n, 1), lam   # per-token mean


def features(tabs, meta):
    F = []
    for t in tabs:
        m = meta[t['id']]
        f = set()
        f.add('hdr=' + (m['hdr'] or 'none'))
        for l in t['lines']:
            if l['cls']:
                f.add('cls=' + l['cls'])
        f.add('sys=' + m['sys'])
        n = len(t['lines'])
        f.add('size=' + ('3-5' if n <= 5 else '6-10' if n <= 10 else '11-20' if n <= 20 else '21+'))
        f.add('rev=%d' % any(l['surf'] for l in t['lines']))
        f.add('cols=%d' % (len(set(l['col'] for l in t['lines'])) > 1))
        f.add('vol=' + m['vol'])
        f.add('site=' + m['site'])
        F.append(f)
    return F


def meta_pe():
    M = {}
    for t in load():
        hdr = None
        if t['lines'] and not t['lines'][0]['numerals'] and t['lines'][0]['signs']:
            hdr = t['lines'][0]['signs'][0].split('~')[0]
        sy = Counter(system_of(l['numerals']) for l in t['lines'] if l['numerals'])
        sys_ = sy.most_common(1)[0][0] if sy else 'none'
        vol = re.sub(r',.*', '', t.get('designation', '')).strip()
        M[t['id']] = {'hdr': hdr, 'sys': str(sys_), 'vol': vol, 'site': 'Susa' if 'Susa' in t['provenience'] else 'other'}
    return M


def hypotheses(F, rng, n=3000, minsize=15):
    cnt = Counter(x for f in F for x in f)
    feats = [x for x, c in cnt.items() if c >= minsize and c <= len(F) - minsize]
    H = set()
    for x in feats:
        H.add((x,))
    tries = 0
    while len(H) < n and tries < 200000:
        tries += 1
        a, b = rng.sample(feats, 2)
        g = sum(1 for f in F if a in f and b in f)
        if minsize <= g <= len(F) - minsize:
            H.add(tuple(sorted((a, b))))
    return sorted(H)


def tstats(S, F, H):
    out = np.zeros(len(H))
    for i, h in enumerate(H):
        g = np.array([all(x in f for x in h) for f in F])
        a, b = S[g], S[~g]
        if len(a) < 5 or len(b) < 5:
            continue
        out[i] = (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b) + 1e-12)
    return out


def null_job(args):
    seed, tabsA, F, H, lam = args
    fn = os.path.join(OUT, 'null_%s_%02d.npy' % (args[5], seed))
    if os.path.exists(fn):
        return np.load(fn)
    sh = C.shuffle_lines(tabsA, random.Random(seed))
    S, _ = tablet_scores(sh, lam)
    t = tstats(S, F, H)
    np.save(fn, t)
    return t


def pipeline(name, ent, meta, pool, nnull=20):
    ids = sorted(t['id'] for t in ent)
    rr = random.Random(313)
    A = set(rr.sample(ids, len(ids) // 2))
    TA = [t for t in ent if t['id'] in A]
    TB = [t for t in ent if t['id'] not in A]
    FA, FB = features(TA, meta), features(TB, meta)
    H = hypotheses(FA, random.Random(9))
    SA, lam = tablet_scores(TA)
    tA = tstats(SA, FA, H)
    nulls = pool.map(null_job, [(s, TA, FA, H, lam, name) for s in range(nnull)])
    mx = np.array([n.max() for n in nulls])
    thr = float(np.quantile(mx, 0.95))
    order = np.argsort(-tA)[:20]
    surv = [int(i) for i in order if tA[i] > thr]
    test = surv if surv else [int(i) for i in order]
    SB, _ = tablet_scores(TB, lam)
    HB = [H[i] for i in test]
    tB = tstats(SB, FB, HB)
    rng = random.Random(4)
    nullB = []
    for s in range(200):
        Ss, _ = tablet_scores(C.shuffle_lines(TB, rng), lam)
        nullB.append(tstats(Ss, FB, HB))
    nullB = np.array(nullB)
    pB = [(1 + (nullB[:, j] >= tB[j]).sum()) / 201 for j in range(len(HB))]
    res = {'name': name, 'nH': len(H), 'lam': [float(x) for x in lam],
           'meanS_A': float(SA.mean()), 'meanS_B': float(SB.mean()),
           'fwer_thr': thr, 'null_max': mx.tolist(), 'n_surv': len(surv),
           'top': [{'h': list(H[i]), 'tA': float(tA[i]), 'tB': float(tB[j]), 'pB': float(pB[j]),
                    'nA': int(sum(all(x in f for x in H[i]) for f in FA))} for j, i in enumerate(test)]}
    json.dump(res, open(os.path.join(OUT, name + '.json'), 'w'))
    print(name, 'meanS A/B %.4f %.4f' % (SA.mean(), SB.mean()), 'thr %.2f' % thr, 'surv', len(surv), flush=True)
    for x in res['top'][:10]:
        print('   ', x, flush=True)
    return res


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    CO = C.corpora()
    ent = variant(CO['PE'], 'ENT')
    meta = meta_pe()
    uni = Counter(x for t in ent for l in t['lines'] for x in l['toks'])
    # planted: PRIME only on M157-header tablets, TOPIC elsewhere
    m157 = [t for t in ent if meta[t['id']]['hdr'] == 'M157']
    rest = [t for t in ent if meta[t['id']]['hdr'] != 'M157']
    plant = C.gen_planted(m157, 'PRIME', 5, uni) + C.gen_planted(rest, 'TOPIC', 6, uni)
    with Pool(nw) as pool:
        pipeline('PLANT_M157', plant, meta, pool)
        pipeline('PE', ent, meta, pool)
