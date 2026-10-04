"""pe10 cycle 2: mine long<->short 'abbreviation' pairs from squeezed vs open positions.

Positions (per entry, intact obverse/reverse runs):
  Q (squeezed) = last 2 entries of an intact obverse on a crowded tablet (top tercile of
      obverse lines per cm of height) + last entry of the reverse on spill tablets whose
      reverse end is intact and crowded.
  everything else = not squeezed.
Candidates:
  short S = any whole entry string (base forms, no x/breaks) occurring >= 2 times;
  long  L = any contiguous n-gram (n=2..4) contained in entries, occurring >= 2 times,
      longer than S.  All S x L pairs scored (~millions).  'sub' = S is a subsequence of L.
Score(S,L) = (z_S - z_L)/sqrt2 where z = binomial z of being in Q against the base Q rate
  (S counted as whole entries, L as containing entries); pairs gated on context
  similarity cos >= 0.5 (header opener, number system, final sign of tablet neighbours),
  computed from ALL occurrences, so fixed under the null.
Search-size correction: identical search with Q labels shuffled among entries within the
  same tablet (200 reps): FWER threshold = 95th pct of the null max score; FDR count.
Planted control: overwrite m random Q entries with S* and m random non-Q entries
  (same tablets' openers) with L* for 5 random pairs; m = 3, 5, 8.  Recovery = pair
  above the FWER threshold.
"""
import sys, json, itertools
import numpy as np
from pe10_common import *

rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def system(nums):
    codes = {c.split('@')[0] for _, c in nums}
    C = {'N39B', 'N30C', 'N24', 'N30D', 'N39C', 'N39A', 'N29B', 'N28'}
    if codes & C:
        return 'C'
    if codes & {'N51', 'N54', 'N46'}:
        return 'B'
    if codes & {'N02', 'N08', 'N08A'}:
        return 'F'
    return 'S'


def collect(T):
    obv_cr = []
    for t in T:
        n = sum(1 for u in t['units'] if u['face'] == 'obverse')
        if t['h']:
            obv_cr.append(n / (t['h'] / 10))
    q2 = np.quantile(obv_cr, 2 / 3)
    E = []   # entries
    for ti, t in enumerate(T):
        U = t['units']
        hdr = base(U[0]['signs'][0]) if U and U[0]['header'] and U[0]['signs'] else '-'
        broken_faces = {m[0] for m in t['markers'] if 'broken' in m[3].lower() or 'missing' in m[3].lower()}
        for face in ('obverse', 'reverse'):
            F = [u for u in U if u['face'] == face]
            if not F:
                continue
            ent = [u for u in F if u['entry'] and not u['total']]
            intact_end = face not in broken_faces and not any(u['prime'] for u in F) and F and F[-1].get('entry') and not F[-1]['total']
            nF = len(F)
            crowd = nF / (t['h'] / 10) if t['h'] else None
            if face == 'reverse':
                crowd = (nF + sum(1 for u in U if u['face'] == 'obverse')) / (2 * t['h'] / 10) if t['h'] else None
            crowded = crowd is not None and crowd >= q2
            for k, u in enumerate(ent):
                if u['broken'] or 'x' in u['signs']:
                    continue
                q = False
                if intact_end and crowded:
                    if face == 'obverse' and k >= len(ent) - 2:
                        q = True
                    if face == 'reverse' and t['spill'] and k == len(ent) - 1:
                        q = True
                E.append({'tab': ti, 's': tuple(base(x) for x in u['signs']), 'q': q,
                          'sys': system(u['nums']), 'hdr': hdr})
    return E, q2


def build_index(E):
    # short strings
    cS = Counter(e['s'] for e in E)
    S_types = [s for s, c in cS.items() if c >= 2]
    # long n-grams
    cL = Counter()
    occL = defaultdict(set)
    for i, e in enumerate(E):
        s = e['s']
        seen = set()
        for n in range(2, 5):
            for j in range(len(s) - n + 1):
                g = s[j:j + n]
                if g not in seen:
                    seen.add(g); occL[g].add(i)
    L_types = [g for g, o in occL.items() if len(o) >= 2]
    occS = defaultdict(list)
    for i, e in enumerate(E):
        occS[e['s']].append(i)
    return S_types, L_types, occS, occL


def ctx_vec(E, idxs, feats):
    v = np.zeros(len(feats))
    for i in idxs:
        e = E[i]
        for f in ('h:' + e['hdr'], 'y:' + e['sys']):
            v[feats[f]] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


def is_subseq(a, b):
    it = iter(b)
    return all(x in it for x in a)


def zmat(occ_lists, qvec, p):
    """binomial z for each type: occ_lists = list of index arrays."""
    out = np.zeros(len(occ_lists))
    for k, idx in enumerate(occ_lists):
        n = len(idx); x = qvec[idx].sum()
        out[k] = (x - n * p) / np.sqrt(n * p * (1 - p))
    return out


def search(E, S_types, L_types, occS, occL, qvec, gate, keepn=0):
    p = qvec.mean()
    oS = [np.array(occS[s]) for s in S_types]
    oL = [np.array(sorted(occL[g])) for g in L_types]
    zS = zmat(oS, qvec, p)
    zL = zmat(oL, qvec, p)
    sc = (zS[:, None] - zL[None, :]) / np.sqrt(2)
    sc = np.where(gate, sc, -np.inf)
    mx = float(sc.max())
    top = []
    if keepn:
        flat = np.argsort(sc, axis=None)[::-1][:keepn]
        for f in flat:
            i, j = np.unravel_index(f, sc.shape)
            top.append((S_types[i], L_types[j], float(sc[i, j]), float(zS[i]), float(zL[j])))
    return mx, sc, top


def shuffle_q(E, qvec):
    q = qvec.copy()
    bytab = defaultdict(list)
    for i, e in enumerate(E):
        bytab[e['tab']].append(i)
    for idx in bytab.values():
        idx = np.array(idx)
        q[idx] = q[rng.permutation(idx)]
    return q


def prepare(E):
    S_types, L_types, occS, occL = build_index(E)
    feats = {}
    for e in E:
        for f in ('h:' + e['hdr'], 'y:' + e['sys']):
            feats.setdefault(f, len(feats))
    VS = np.array([ctx_vec(E, occS[s], feats) for s in S_types])
    VL = np.array([ctx_vec(E, occL[g], feats) for g in L_types])
    cos = VS @ VL.T
    lenok = np.array([[len(s) < len(g) for g in L_types] for s in S_types])
    gate = (cos >= 0.5) & lenok
    sub = np.array([[is_subseq(s, g) for g in L_types] for s in S_types]) & gate
    return S_types, L_types, occS, occL, gate, sub


def run(E, tag, nrep=NREP, keepn=30):
    qvec = np.array([e['q'] for e in E], float)
    S_types, L_types, occS, occL, gate, sub = prepare(E)
    mx, sc, top = search(E, S_types, L_types, occS, occL, qvec, gate, keepn)
    mxs = float(np.where(sub, sc, -np.inf).max())
    nul, nuls, nul_cnt4 = [], [], []
    for r in range(nrep):
        q = shuffle_q(E, qvec)
        m, s2, _ = search(E, S_types, L_types, occS, occL, q, gate)
        nul.append(m); nuls.append(float(np.where(sub, s2, -np.inf).max()))
        nul_cnt4.append(int((s2 > 4).sum()))
    nul = np.array(nul); nuls = np.array(nuls)
    thr = float(np.quantile(nul, 0.95)); thrs = float(np.quantile(nuls, 0.95))
    surv = [t for t in top if t[2] > thr]
    subtop = []
    flat = np.argsort(np.where(sub, sc, -np.inf), axis=None)[::-1][:keepn]
    for f in flat:
        i, j = np.unravel_index(f, sc.shape)
        subtop.append((S_types[i], L_types[j], float(sc[i, j])))
    return {'tag': tag, 'n_entries': len(E), 'n_Q': int(qvec.sum()), 'n_S': len(S_types), 'n_L': len(L_types),
            'n_pairs_gated': int(gate.sum()), 'n_sub_pairs': int(sub.sum()), 'n_pairs_total': len(S_types) * len(L_types),
            'obs_max': mx, 'null_max_mean': float(nul.mean()), 'fwer_thr': thr, 'p_max': float((np.sum(nul >= mx) + 1) / (nrep + 1)),
            'obs_max_sub': mxs, 'fwer_thr_sub': thrs, 'p_max_sub': float((np.sum(nuls >= mxs) + 1) / (nrep + 1)),
            'obs_count_gt4': int((sc > 4).sum()), 'null_count_gt4_mean': float(np.mean(nul_cnt4)),
            'top': [(' '.join(a), ' '.join(b), round(c, 2), round(d, 2), round(e_, 2)) for a, b, c, d, e_ in top[:15]],
            'top_sub': [(' '.join(a), ' '.join(b), round(c, 2)) for a, b, c in subtop[:15]],
            'survivors': [(' '.join(a), ' '.join(b), round(c, 2)) for a, b, c, *_ in surv]}


def plant(E, m, npairs=5):
    E2 = [dict(e) for e in E]
    S_types, L_types, occS, occL = build_index(E)
    Lc = [g for g in L_types if len(occL[g]) >= 3 and len(g) >= 2]
    planted = []
    Qi = [i for i, e in enumerate(E2) if e['q']]
    Oi = [i for i, e in enumerate(E2) if not e['q']]
    used = set()
    for _ in range(npairs):
        g = Lc[rng.integers(len(Lc))]
        k = rng.integers(1, len(g))
        pos = sorted(rng.choice(len(g), k, replace=False))
        s = tuple(g[i] for i in pos)
        qs = [i for i in rng.permutation(Qi) if i not in used][:m]
        hdrs = Counter(E2[i]['hdr'] for i in qs)
        os_ = [i for i in rng.permutation(Oi) if i not in used and E2[i]['hdr'] in hdrs][:m]
        for i in qs:
            E2[i]['s'] = s; used.add(i)
        for i in os_:
            E2[i]['s'] = g; used.add(i)
        planted.append((' '.join(s), ' '.join(g)))
    return E2, planted


if __name__ == '__main__':
    T = load()
    E, q2 = collect(T)
    out = {'q2_lines_per_cm': float(q2)}
    ck = os.path.join(CKPT, 'c2.json')
    if os.path.exists(ck):
        out = json.load(open(ck))
    if 'real' not in out:
        out['real'] = run(E, 'real'); json.dump(out, open(ck, 'w'), indent=1)
        print('real', json.dumps(out['real'], indent=1), flush=True)
    for m in (3, 5, 8):
        key = 'plant_m%d' % m
        if key in out:
            continue
        E2, planted = plant(E, m)
        r = run(E2, key, nrep=100)
        surv = {(a, b) for a, b, _ in r['survivors']}
        # check recovery directly from scores for planted pairs
        r['planted'] = planted
        r['recovered'] = sum(1 for p in planted if p in surv)
        out[key] = r; json.dump(out, open(ck, 'w'), indent=1)
        print(key, 'recovered', r['recovered'], 'of', len(planted), 'thr', r['fwer_thr'], flush=True)
    if 'negative' not in out:
        En = [dict(e) for e in E]
        qs = shuffle_q(E, np.array([e['q'] for e in E], float))
        for e, q in zip(En, qs):
            e['q'] = bool(q)
        out['negative'] = run(En, 'negative', nrep=100); json.dump(out, open(ck, 'w'), indent=1)
        print('neg', out['negative']['obs_max'], out['negative']['fwer_thr'], out['negative']['survivors'][:5], flush=True)
