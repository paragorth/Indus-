"""pe44 cycle 3: plan-versus-actual TABLETS from numbers alone.

Tablet fingerprints (numbers only, own system, sign-blind):
  TWIN   pairs (A, B) on the tablet, same system, A one-denomination (round), B < A, (A-B)/A <= 0.25,
         B not round  ->  'target and its shortfall'
  DIFF   triples A - B = C among distinct entries (deficit = obligation - delivery)
  MIX    min(share round, share ragged) among values >= 2nd denomination
Controls:
  Ur III: balanced accounts (sag-nig2-gur11 + la2-ia3, n=197) vs other Ur III tablets matched on
          number of quantities; AUC of each fingerprint (the wording is never used by the score).
  PE planted: 30 random PE tablets get one planted twin (an entry's value v -> round anchor A >= v
          and a new entry A - Geom deficit).
  PE null:  values shuffled across tablets within system (200 reps): the excess of real tablets
          over the null is the plan-vs-actual signal; per-tablet p-values with BH.
"""
import json, math, os, random, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe44_common import *  # noqa
from pe44_cycle2 import shuffle_within_system


def is_round(v, den):
    return value_feats(v, den)[0] == 1


def tab_fp(recs, dmax=0.25):
    by = defaultdict(list)
    for r in recs:
        by[r['sys']].append(r['val'])
    twin = diff = 0
    nr = nrag = nbig = 0
    for s, vals in by.items():
        den = DEN[s]
        rd = [is_round(v, den) for v in vals]
        for v, r_ in zip(vals, rd):
            if v >= den[1]:
                nbig += 1
                nr += r_
                nrag += (not r_)
        n = len(vals)
        for a in range(n):
            if not rd[a]:
                continue
            A = vals[a]
            for b in range(n):
                if b != a and not rd[b] and 0 < A - vals[b] <= dmax * A:
                    twin += 1
        if n <= 40:
            st = Counter(vals)
            for a in range(n):
                for b in range(n):
                    if a != b and vals[a] > vals[b]:
                        c = vals[a] - vals[b]
                        if st[c] - (c == vals[a]) - (c == vals[b]) > 0:
                            diff += 1
    mix = min(nr, nrag) / nbig if nbig else 0.0
    return {'TWIN': min(twin, 5), 'TWIN1': int(twin > 0), 'DIFF': min(diff, 5), 'DIFF1': int(diff > 0), 'MIX': mix}


FPS = ['TWIN1', 'TWIN', 'DIFF1', 'DIFF', 'MIX']


def ur3_control(rng):
    u = ur3_tabs()
    pos = [t for t, d in u.items() if d['bal']]
    bylen = defaultdict(list)
    for t, d in u.items():
        if not d['bal']:
            bylen[min(60, len(d['recs']))].append(t)
    neg = []
    for t in pos:
        L = min(60, len(u[t]['recs']))
        for dl in (0, 1, -1, 2, -2, 3, -3, 5, -5, 10, -10):
            if bylen.get(L + dl):
                neg += list(rng.choice(bylen[L + dl], min(5, len(bylen[L + dl])), replace=False))
                break
    fp = {t: tab_fp(u[t]['recs']) for t in set(pos) | set(neg)}
    res = {}
    for f in FPS:
        res[f] = {'auc': auc([fp[t][f] for t in pos], [fp[t][f] for t in neg]),
                  'pos_mean': float(np.mean([fp[t][f] for t in pos])), 'neg_mean': float(np.mean([fp[t][f] for t in neg]))}
    # label-shuffle control
    allt = pos + neg
    lab = np.array([1] * len(pos) + [0] * len(neg))
    sh = []
    for _ in range(200):
        p = rng.permutation(lab)
        sh.append(auc([fp[t]['TWIN'] for t, l in zip(allt, p) if l], [fp[t]['TWIN'] for t, l in zip(allt, p) if not l]))
    res['TWIN_shuffled_auc_95'] = float(np.percentile(sh, 95))
    res['n_pos'], res['n_neg'] = len(pos), len(neg)
    return res


def pe_excess(tabs, rng, reps=200):
    real = {t: tab_fp(r) for t, r in tabs.items()}
    tot = {f: sum(v[f] for v in real.values()) for f in FPS}
    nulls = defaultdict(list)
    per = defaultdict(list)
    for rep in range(reps):
        tb = shuffle_within_system(tabs, random.Random(int(rng.integers(1e9))))
        fp = {t: tab_fp(r) for t, r in tb.items()}
        for f in FPS:
            nulls[f].append(sum(v[f] for v in fp.values()))
        for t in tabs:
            per[t].append(fp[t]['TWIN'])
    res = {}
    for f in FPS:
        a = np.array(nulls[f])
        res[f] = {'real': tot[f], 'null': float(a.mean()), 'sd': float(a.std()),
                  'p_hi': float((1 + (a >= tot[f]).sum()) / (1 + len(a))), 'z': float((tot[f] - a.mean()) / (a.std() + 1e-9))}
    ptab = {t: float((1 + sum(x >= real[t]['TWIN'] for x in per[t])) / (1 + reps)) for t in tabs if real[t]['TWIN'] > 0}
    return res, real, ptab


def plant_twins(tabs, rng, n=30):
    tb = {t: [dict(r) for r in recs] for t, recs in tabs.items()}
    cand = [t for t, recs in tb.items() if len(recs) >= 3]
    pick = rng.sample(cand, n)
    for t in pick:
        recs = tb[t]
        i = rng.randrange(len(recs))
        r = recs[i]; den = DEN[r['sys']]
        v = max(r['val'], den[1])
        hi = max(k for k, d in enumerate(den) if d <= v)
        A = int(math.ceil(v / den[hi])) * den[hi]
        A = A if is_round(A, den) else den[min(hi + 1, len(den) - 1)]
        r['val'] = A
        unit = den[hi - 1] if hi > 0 else 1
        d = int(np.random.default_rng(rng.randrange(1 << 30)).geometric(0.3)) * unit
        B = max(1, A - max(1, min(d, int(0.2 * A))))
        recs.insert(i + 1, dict(r, val=B, item='PLANT'))
    return tb, set(pick)


if __name__ == '__main__':
    rng = np.random.default_rng(21)
    out = {}
    out['ur3'] = ur3_control(rng)
    print('UR3', json.dumps(out['ur3']), flush=True)
    tabs = pe_tabs()
    tbP, pick = plant_twins(tabs, random.Random(4))
    rp, realp, ptp = pe_excess(tbP, rng, reps=100)
    qs = P23.bh(list(ptp.values()))
    hitP = [t for t, q in zip(ptp, qs) if t in pick and ptp[t] < 0.05]
    out['planted'] = {'excess': rp, 'planted_tabs_p<0.05': len(hitP), 'n_planted': len(pick),
                      'other_tabs_p<0.05': sum(1 for t in ptp if t not in pick and ptp[t] < 0.05)}
    print('PLANT', json.dumps(out['planted']), flush=True)
    r, real, pt = pe_excess(tabs, rng, reps=200)
    out['pe'] = r
    tops = sorted(pt.items(), key=lambda kv: kv[1])[:20]
    q = dict(zip(pt, P23.bh(list(pt.values()))))
    out['pe_top'] = []
    for t, p in tops:
        recs = tabs[t]
        out['pe_top'].append({'tab': t, 'p': p, 'q': float(q[t]), 'fp': real[t],
                              'lines': [(x['raw'], x['sys'], x['val']) for x in recs]})
    print('PE', json.dumps(r), flush=True)
    for x in out['pe_top'][:10]:
        print(x['tab'], x['p'], round(x['q'], 3), x['fp'], flush=True)
    json.dump(out, open(os.path.join(CK, 'c3.json'), 'w'), indent=1)
