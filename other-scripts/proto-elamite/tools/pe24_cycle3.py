"""pe24 cycle 3: which entries would have to be red ink? (deficit-subset enrichment)

For every tablet whose total does not close with all entries +1, enumerate every
set S of 1-3 member entries (any hypothesis, any value set) such that
  negating S closes it  (sum E - 2 sum_S E == T)   'neg'
  dropping S closes it  (sum E -   sum_S E == T)   'drop'
Statistic per mode: over rescued tablets, share of the minimal rescuing entries
that carry a variant marker (any '~x' / '@x' on a sign), minus the share of all
member entries that carry one (tablet-weighted). Null: marker flags permuted
among the members of each tablet (2,000x). Also: how many non-closing tablets a
1-3-entry flip can rescue at all, vs totals replaced by random numbers of the
same size (100x).
Planted control: on the tablets that close at baseline, one random marked entry
(where present) is flipped to -1 (T recomputed); does the statistic find it?
Also per marker: enrichment of each marker among rescuing entries.
Usage: python3 pe24_cycle3.py
"""
import time
from pe24_common import *
from pe24_cycle1 import load_pe, rand_totals

MAXK = 3


def rescues(c, mode):
    """Return list of minimal rescuing sets (as tuples of (hyp, pair, member idx))
    for a tablet with a single-pair-per-hypothesis structure; multi-pair hyps:
    each pair must be rescued by its own set, we take pairs that fail."""
    sols = []
    for hi, h in enumerate(c['hyps']):
        nv = len(h[0]['T'])
        for v in range(nv):
            per_pair = []
            ok = True
            for pi, p in enumerate(h):
                E = [float(e[v]) for e in p['E']]; T = float(p['T'][v])
                d = sum(E) - T
                if abs(d) < 1e-9:
                    per_pair.append([()]); continue
                found = []
                for k in range(1, MAXK + 1):
                    for S in itertools.combinations(range(len(E)), k):
                        s = sum(E[i] for i in S)
                        if abs((2 * s if mode == 'neg' else s) - d) < 1e-9:
                            found.append(S)
                    if found:
                        break
                if not found:
                    ok = False; break
                per_pair.append([(pi, S) for S in found])
            if ok:
                for combo in itertools.product(*per_pair):
                    sols.append(tuple((hi, x[0], x[1]) for x in combo if x))
    if not sols:
        return []
    m = min(sum(len(x[2]) for x in s) for s in sols)
    return [s for s in sols if sum(len(x[2]) for x in s) == m]


def marked(c, hi, pi, i):
    return len(c['hyps'][hi][pi]['mk'][i]) > 0


def stat(C, idx_nonclose, mode, flags=None):
    """flags: optional dict (tablet k) -> function(hi,pi,i)->bool."""
    diffs, rescued = [], 0
    per_marker = collections.Counter(); per_marker_base = collections.Counter()
    for k in idx_nonclose:
        c = C[k]
        sols = SOLS[mode].get(k) if flags is None or 'sols' not in flags else None
        if sols is None:
            sols = rescues(c, mode)
        if not sols:
            continue
        rescued += 1
        f = (flags or {}).get(k, lambda hi, pi, i: marked(c, hi, pi, i))
        ent = [(hi, pi, i) for s in sols for (hi, pi, S) in s for i in S]
        share = np.mean([f(*e) for e in ent])
        h0 = c['hyps'][0]
        allm = [f(0, pi, i) for pi, p in enumerate(h0) for i in range(len(p['E']))]
        diffs.append(share - np.mean(allm))
        if flags is None:
            for e in ent:
                for m in c['hyps'][e[0]][e[1]]['mk'][e[2]]:
                    per_marker[m] += 1.0 / len(ent)
            for pi, p in enumerate(h0):
                for i in range(len(p['E'])):
                    for m in p['mk'][i]:
                        per_marker_base[m] += 1.0 / len(allm)
    return (float(np.mean(diffs)) if diffs else 0.0), rescued, per_marker, per_marker_base


SOLS = {'neg': {}, 'drop': {}}

if __name__ == '__main__':
    t0 = time.time()
    C = load_pe()
    P = Problem(C, 'mk'); base = P.baseline()
    closed = [k for k in range(len(C)) if P.closes(k, base)]
    nonc = [k for k in range(len(C)) if k not in closed]
    rng = np.random.default_rng(5)
    out = {'n': len(C), 'closed': len(closed), 'nonclosed': len(nonc)}
    for mode in ('neg', 'drop'):
        for k in nonc:
            SOLS[mode][k] = rescues(C[k], mode)
        st, resc, pm, pb = stat(C, nonc, mode)
        # null: permute marker flags among members within each tablet (first hyp's members;
        # members shared across hyps keep their identity by sign tuple)
        null = []
        for r in range(2000):
            fl = {}
            for k in nonc:
                c = C[k]
                keys = {}
                for h in c['hyps']:
                    for p in h:
                        for i, sg in enumerate(p['sg']):
                            keys.setdefault(tuple(sg), len(p['mk'][i]) > 0)
                kl = list(keys); vals = [keys[x] for x in kl]; rng.shuffle(vals)
                mp = dict(zip(kl, vals))
                fl[k] = (lambda c, mp: (lambda hi, pi, i: mp[tuple(c['hyps'][hi][pi]['sg'][i])]))(c, mp)
            null.append(stat(C, nonc, mode, fl)[0])
        null = np.array(null)
        # random totals: how many non-closing tablets are rescued by a 1-3 flip
        rres = []
        for r in range(100):
            Cr = rand_totals(C, rng)
            Pr = Problem(Cr, 'mk')
            nk = [k for k in range(len(Cr)) if not Pr.closes(k, Pr.baseline())]
            rres.append(sum(1 for k in nk if rescues(Cr[k], mode)) / max(1, len(nk)))
        pmk = {m: {'rescue_share': round(pm[m] / max(1, resc), 3), 'base_share': round(pb[m] / max(1, resc), 3)}
               for m in sorted(set(pm) | set(pb), key=lambda m: -pm[m])[:10]}
        out[mode] = {'enrichment': st, 'null_mean': float(null.mean()), 'null_sd': float(null.std()),
                     'p_enriched': float((1 + (null >= st).sum()) / (1 + len(null))),
                     'p_depleted': float((1 + (null <= st).sum()) / (1 + len(null))),
                     'rescued': resc, 'rescued_share': resc / len(nonc),
                     'randtot_rescued_share_mean': float(np.mean(rres)),
                     'randtot_p': float((1 + sum(x >= resc / len(nonc) for x in rres)) / 101),
                     'per_marker': pmk}
        print(mode, json.dumps(out[mode], default=str), flush=True)
    # planted: flip one marked member to -1 on baseline-closing tablets
    plant_stats = []
    for rep in range(20):
        Cp = [dict(c) for c in C]
        flipped = []
        for k in closed:
            c = C[k]
            h0 = c['hyps'][0]
            cand = [(pi, i) for pi, p in enumerate(h0) for i in range(len(p['E'])) if p['mk'][i]]
            if not cand:
                continue
            pi, i = cand[rng.integers(len(cand))]
            key = tuple(h0[pi]['sg'][i])
            hs = []
            for h in c['hyps']:
                nh = []
                for p in h:
                    sg = [(-1 if tuple(s) == key else 1) for s in p['sg']]
                    T = [sum(s * e[v] for s, e in zip(sg, p['E'])) for v in range(len(p['T']))]
                    nh.append(dict(p, T=T))
                hs.append(nh)
            Cp[k] = dict(c, hyps=hs); flipped.append(k)
        SOLS['neg'] = {k: rescues(Cp[k], 'neg') for k in flipped}
        st = stat(Cp, flipped, 'neg')[0]
        plant_stats.append((st, len(flipped)))
    out['plant'] = {'enrichment_mean': float(np.mean([x[0] for x in plant_stats])),
                    'n_flipped': plant_stats[0][1]}
    print('plant', out['plant'], flush=True)
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(DATA, 'pe24_cycle3.json'), 'w'), indent=1, default=str)
