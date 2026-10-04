"""pe13 cycle 3: the rhythm.  Cycles 1-2 found a PE kernel that dips at gap 1 and peaks at
gap 2.  Is the scribe's memory periodic (entries written in pairs / two interleaved lists)?

(A) Exact-gap profile R(d), d = 1..10: pooled over signs, observed pairs of lines sharing a
    sign at gap d / exact expectation under random line order (k lines with the sign among L:
    (L-d) k(k-1) / (L(L-1))).  Shuffle bands from 100 within-tablet line shuffles.
    Parity index PI = mean R(2,4,6) - mean R(3,5,7).
    Corpora: PE entries, PE middles, PE class signs, Ur III, Linear B, proto-cuneiform admin and
    lexical; PLANTED on PE's skeleton: TOPIC, PRIME, REF (flat + gap-1 dip only), ALT (two
    interleaved topic streams, odd and even lines).
(B) Per-sign parity z (even-gap minus odd-gap excess, 200 shuffles) on PE; false positives from
    10 shuffled corpora; split-half replication.
(C) Massive random guessing: 3,000 random tablet groups (1-2 conjoined features: header sign,
    class signs present, number system, size, reverse, columns, volume, site); per-tablet parity
    score; Welch t on half A; family-wise null = whole search on 20 shuffled half-A corpora;
    top 20 re-tested on half B against 200 shuffles.  PLANTED: ALT only on M157-header tablets.
usage: python3 pe13_cycle3.py [workers]
"""
import json, math, os, random, re, sys
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe13_common as C  # noqa
from pe13_cycle2 import variant  # noqa
from common import load, system_of  # noqa

OUT = os.path.join(C.CK, 'c3')
os.makedirs(OUT, exist_ok=True)
DM = 10
EVEN, ODD = [2, 4, 6], [3, 5, 7]


def tab_gaps(t):
    """obs[d], exp[d] for d = 1..DM for one tablet (pooled over signs)."""
    L = len(t['lines'])
    pos = defaultdict(list)
    for i, l in enumerate(t['lines']):
        for s in set(l['toks']):
            pos[s].append(i)
    o = np.zeros(DM + 1)
    e = np.zeros(DM + 1)
    if L < 2:
        return o, e
    for s, P in pos.items():
        k = len(P)
        if k < 2:
            continue
        for a in range(k):
            for b in range(a + 1, k):
                d = P[b] - P[a]
                if d <= DM:
                    o[d] += 1
        f = k * (k - 1) / (L * (L - 1))
        for d in range(1, min(DM, L - 1) + 1):
            e[d] += (L - d) * f
    return o, e


def profile(tabs):
    O = np.zeros(DM + 1)
    E = np.zeros(DM + 1)
    for t in tabs:
        o, e = tab_gaps(t)
        O += o
        E += e
    R = O[1:] / np.maximum(E[1:], 1e-9)
    return R


def parity(R):
    return float(np.mean([R[d - 1] for d in EVEN]) - np.mean([R[d - 1] for d in ODD]))


def gen_alt(skel, seed, uni):
    """two interleaved topic streams: even lines one cache, odd lines another."""
    ev = [{'id': t['id'], 'lines': t['lines'][0::2]} for t in skel]
    od = [{'id': t['id'], 'lines': t['lines'][1::2]} for t in skel]
    E = C.gen_planted(ev, 'TOPIC', seed, uni, {'lam': 0.6})
    O = C.gen_planted(od, 'TOPIC', seed + 1, uni, {'lam': 0.6})
    out = []
    for t, a, b in zip(skel, E, O):
        L = []
        for i in range(len(t['lines'])):
            L.append(a['lines'][i // 2] if i % 2 == 0 else b['lines'][i // 2])
        out.append({'id': t['id'], 'lines': L})
    return out


def job_profile(args):
    name, tabs, nshuf = args
    fn = os.path.join(OUT, 'prof_' + name + '.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    R = profile(tabs)
    rng = random.Random(7)
    S = np.array([profile(C.shuffle_lines(tabs, rng)) for _ in range(nshuf)])
    PS = np.array([parity(r) for r in S])
    res = {'name': name, 'R': R.tolist(), 'lo': np.quantile(S, 0.025, 0).tolist(), 'hi': np.quantile(S, 0.975, 0).tolist(),
           'PI': parity(R), 'PI_null_mean': float(PS.mean()), 'PI_null_sd': float(PS.std()),
           'PI_z': float((parity(R) - PS.mean()) / (PS.std() + 1e-9)), 'ntab': len(tabs)}
    json.dump(res, open(fn, 'w'))
    print('PROFILE %-10s PI %.3f z %.1f  R %s' % (name, res['PI'], res['PI_z'], ' '.join('%.2f' % x for x in R)), flush=True)
    return res


# ---------- per sign parity ----------
def sign_counts(tabs, signs):
    ev = {s: [0.0, 0.0] for s in signs}  # obs even, exp even
    od = {s: [0.0, 0.0] for s in signs}
    for t in tabs:
        L = len(t['lines'])
        pos = defaultdict(list)
        for i, l in enumerate(t['lines']):
            for s in set(l['toks']):
                if s in ev:
                    pos[s].append(i)
        for s, P in pos.items():
            k = len(P)
            if k < 2:
                continue
            for a in range(k):
                for b in range(a + 1, k):
                    d = P[b] - P[a]
                    if d in EVEN:
                        ev[s][0] += 1
                    elif d in ODD:
                        od[s][0] += 1
            f = k * (k - 1) / (L * (L - 1))
            for d in EVEN:
                ev[s][1] += max(L - d, 0) * f
            for d in ODD:
                od[s][1] += max(L - d, 0) * f
    return {s: (ev[s][0] - ev[s][1]) - (od[s][0] - od[s][1]) for s in signs}, ev, od


def job_signs(args):
    name, tabs, signs, nshuf, seed = args
    fn = os.path.join(OUT, 'sign_' + name + '.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    obs, ev, od = sign_counts(tabs, signs)
    rng = random.Random(seed)
    S = defaultdict(list)
    for _ in range(nshuf):
        o, _, _ = sign_counts(C.shuffle_lines(tabs, rng), signs)
        for s in signs:
            S[s].append(o[s])
    res = {s: {'stat': obs[s], 'z': float((obs[s] - np.mean(S[s])) / (np.std(S[s]) + 1e-9)),
               'ev': ev[s], 'od': od[s]} for s in signs}
    json.dump(res, open(fn, 'w'))
    print('SIGNS', name, 'z>=3:', sum(1 for r in res.values() if r['z'] >= 3), 'z<=-3:', sum(1 for r in res.values() if r['z'] <= -3), flush=True)
    return res


# ---------- random tablet-group search ----------
def tab_scores(tabs):
    S = np.zeros(len(tabs))
    for i, t in enumerate(tabs):
        o, e = tab_gaps(t)
        S[i] = (sum(o[d] - e[d] for d in EVEN) - sum(o[d] - e[d] for d in ODD)) / math.sqrt(1 + sum(e[d] for d in EVEN + ODD))
    return S


def features(tabs, meta):
    F = []
    for t in tabs:
        m = meta[t['id']]
        f = {'hdr=' + (m['hdr'] or 'none'), 'sys=' + m['sys'], 'vol=' + m['vol'], 'site=' + m['site']}
        for l in t['lines']:
            if l['cls']:
                f.add('cls=' + l['cls'])
        n = len(t['lines'])
        f.add('size=' + ('3-5' if n <= 5 else '6-10' if n <= 10 else '11-20' if n <= 20 else '21+'))
        f.add('rev=%d' % any(l['surf'] for l in t['lines']))
        f.add('cols=%d' % (len(set(l['col'] for l in t['lines'])) > 1))
        F.append(f)
    return F


def meta_pe():
    M = {}
    for t in load():
        hdr = None
        if t['lines'] and not t['lines'][0]['numerals'] and t['lines'][0]['signs']:
            hdr = t['lines'][0]['signs'][0].split('~')[0]
        sy = Counter(system_of(l['numerals']) for l in t['lines'] if l['numerals'])
        vol = re.sub(r',.*', '', t.get('designation', '')).strip()
        M[t['id']] = {'hdr': hdr, 'sys': str(sy.most_common(1)[0][0]) if sy else 'none', 'vol': vol,
                      'site': 'Susa' if 'Susa' in t['provenience'] else 'other'}
    return M


def hypotheses(F, rng, n=3000, minsize=15):
    cnt = Counter(x for f in F for x in f)
    feats = sorted(x for x, c in cnt.items() if minsize <= c <= len(F) - minsize)
    H = set((x,) for x in feats)
    tries = 0
    while len(H) < n and tries < 300000:
        tries += 1
        a, b = rng.sample(feats, 2)
        g = sum(1 for f in F if a in f and b in f)
        if minsize <= g <= len(F) - minsize:
            H.add(tuple(sorted((a, b))))
    return sorted(H)


def masks(F, H):
    return np.array([[all(x in f for x in h) for f in F] for h in H])


def tstats(S, MK):
    out = np.zeros(len(MK))
    for i, g in enumerate(MK):
        a, b = S[g], S[~g]
        if len(a) < 5 or len(b) < 5:
            continue
        out[i] = (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b) + 1e-12)
    return out


def search(name, ent, meta):
    ids = sorted(t['id'] for t in ent)
    A = set(random.Random(313).sample(ids, len(ids) // 2))
    TA = [t for t in ent if t['id'] in A]
    TB = [t for t in ent if t['id'] not in A]
    FA, FB = features(TA, meta), features(TB, meta)
    H = hypotheses(FA, random.Random(9))
    MA = masks(FA, H)
    tA = tstats(tab_scores(TA), MA)
    rng = random.Random(21)
    mx = np.array([tstats(tab_scores(C.shuffle_lines(TA, rng)), MA).max() for _ in range(20)])
    thr = float(np.quantile(mx, 0.95))
    order = [int(i) for i in np.argsort(-tA)[:20]]
    surv = [i for i in order if tA[i] > thr]
    HB = [H[i] for i in order]
    MB = masks(FB, HB)
    tB = tstats(tab_scores(TB), MB)
    nB = np.array([tstats(tab_scores(C.shuffle_lines(TB, rng)), MB) for _ in range(200)])
    pB = [(1 + (nB[:, j] >= tB[j]).sum()) / 201 for j in range(len(HB))]
    res = {'name': name, 'nH': len(H), 'thr': thr, 'null_max': mx.tolist(), 'n_surv': len(surv),
           'top': [{'h': list(H[i]), 'tA': float(tA[i]), 'nA': int(MA[i].sum()), 'tB': float(tB[j]),
                    'nB': int(MB[j].sum()), 'pB': float(pB[j]), 'surv': i in surv} for j, i in enumerate(order)]}
    json.dump(res, open(os.path.join(OUT, 'search_' + name + '.json'), 'w'))
    print('SEARCH', name, 'thr %.2f' % thr, 'survivors', len(surv), flush=True)
    for x in res['top'][:8]:
        print('   ', x, flush=True)
    return res


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    CO = C.corpora()
    ent = variant(CO['PE'], 'ENT')
    uni = Counter(x for t in ent for l in t['lines'] for x in l['toks'])
    J = [('PE_ENT', ent, 100), ('PE_MID', variant(CO['PE'], 'MID'), 100), ('PE_CLS', variant(CO['PE'], 'CLS'), 100),
         ('PE_ALL', CO['PE'], 100),
         ('PL_TOPIC', C.gen_planted(ent, 'TOPIC', 41, uni), 100), ('PL_PRIME', C.gen_planted(ent, 'PRIME', 42, uni), 100),
         ('PL_REF', C.gen_planted(ent, 'TOPIC', 43, uni, {'rho': 0.1}), 100), ('PL_ALT', gen_alt(ent, 44, uni), 100)]
    for k in ('UR3', 'LINB', 'ARCH_ADM', 'ARCH_LEX'):
        J.append((k, random.Random(5).sample(CO[k], min(1000, len(CO[k]))), 50))
    tc = Counter(s for t in ent for s in set(x for l in t['lines'] for x in l['toks']))
    signs = sorted(s for s, n in tc.items() if n >= 8)
    ids = sorted(t['id'] for t in ent)
    HA = set(random.Random(77).sample(ids, len(ids) // 2))
    SJ = [('PE', ent, signs, 200, 1), ('HALFA', [t for t in ent if t['id'] in HA], signs, 200, 2),
          ('HALFB', [t for t in ent if t['id'] not in HA], signs, 200, 3)]
    for s in range(10):
        SJ.append(('SHUF%02d' % s, C.shuffle_lines(ent, random.Random(500 + s)), signs, 100, 10 + s))
    with Pool(nw) as p:
        p.map(job_profile, J, chunksize=1)
        p.map(job_signs, SJ, chunksize=1)
    meta = meta_pe()
    m157 = [t for t in ent if meta[t['id']]['hdr'] == 'M157']
    rest = [t for t in ent if meta[t['id']]['hdr'] != 'M157']
    plant = gen_alt(m157, 61, uni) + C.gen_planted(rest, 'TOPIC', 62, uni)
    search('PLANT_ALT_M157', plant, meta)
    search('PE', ent, meta)
