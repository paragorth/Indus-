"""la63 summary: loss profiles -> loss classes, calibrated against pseudo-words.

usage: python3 la63_sum.py CORPUS [-v] [-models a:b]
For type w, loss key k (task + L = same entry as an erased token / R = elsewhere) and model m:
  excess_m = mean change when w is erased - mean change over R random same-count erasures in the
  same documents; pseudo_m,j = null draw j - mean of the other draws (a pseudo-word with exactly
  w's documents and token count). Both are averaged over models. Per key, z = excess / sd of the
  pseudo values of all types in the same frequency bin. Group score = max z over its keys; p = share
  of pseudo group scores in the bin >= the type's score. Class = group with the smallest p if
  p < ALPHA, else SILENT. False-class rate = pseudo-words classified the same way (leave-one-out).
"""
import glob, json, math, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la63_lib as L

GROUPS = {'NUM-L': ['MAG-L', 'FRAC-L'], 'NUM-R': ['MAG-R', 'FRAC-R'], 'LOGO': ['LOGO-L', 'LOGO-R'],
          'TOT': ['TOT-L', 'TOT-R'], 'ENT': ['ENT-L', 'ENT-R'], 'HEAD': ['HEAD-L', 'HEAD-R']}
KEYS = sorted({k for v in GROUPS.values() for k in v})
ALPHA = None      # calibrated: FALSE_RATE of pseudo-words are classed
FALSE_RATE = 0.02


def fbin(c):
    return 0 if c < 5 else 1 if c < 10 else 2 if c < 25 else 3


def load(name, models=None):
    F = [f for f in glob.glob(os.path.join(L.CK, 'del_%s_*.json' % name))
         if re.match(r'^del_%s_\d+\.json$' % re.escape(name), os.path.basename(f))]
    D = [json.load(open(f)) for f in F]
    if models is not None:
        D = [d for d in D if d['m'] in models]
    return sorted(D, key=lambda d: d['m'])


def tables(D, minn=2):
    """-> ex[w][k] = list over models of excess; ps[w][k] = list over models of [pseudo_j]."""
    ex = collections.defaultdict(lambda: collections.defaultdict(list))
    ps = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in D:
        for w, r in d['res'].items():
            for k, a in r['acc'].items():
                s, n, ns, nn = a
                ok = [j for j in range(len(nn)) if nn[j] >= 1]
                if n < minn or len(ok) < max(3, len(nn) // 2):
                    continue
                nul = {j: ns[j] / nn[j] for j in ok}
                mu = float(np.mean(list(nul.values())))
                ex[w][k].append(s / n - mu)
                R = len(nn)
                tot = sum(nul.values()); K = len(nul)
                ps[w][k].append([(nul[j] - (tot - nul[j]) / (K - 1)) if j in nul else np.nan for j in range(R)])
    return ex, ps


def analyse(name, models=None, alpha=ALPHA, minM=3, D=None):
    if D is None:
        D = load(name, models)
    if len(D) < 3:
        return None
    count = {}
    for d in D:
        count.update(d['count'])
    ex, ps = tables(D)
    W = [w for w in ex if any(len(v) >= minM for v in ex[w].values())]
    E, P = {}, {}
    for w in W:
        E[w], P[w] = {}, {}
        for k in KEYS:
            if len(ex[w].get(k, [])) >= minM:
                E[w][k] = float(np.mean(ex[w][k]))
                P[w][k] = np.nanmean(np.array(ps[w][k], dtype=float), axis=0)   # R pseudo values (nan if never drawn)
    # per bin and key: sd of pseudo values
    SD = {}
    for b in range(4):
        for k in KEYS:
            v = [x for w in W if fbin(count[w]) == b and k in P[w] for x in P[w][k] if x == x]
            SD[(b, k)] = float(np.std(v)) if len(v) >= 20 else None
    # fall back to pooled sd when a bin is thin
    for k in KEYS:
        v = [x for w in W if k in P[w] for x in P[w][k] if x == x]
        pooled = float(np.std(v)) if v else 1.0
        for b in range(4):
            if SD[(b, k)] is None:
                SD[(b, k)] = pooled
    Z, ZP = {}, {}
    for w in W:
        b = fbin(count[w])
        Z[w] = {k: E[w][k] / (SD[(b, k)] + 1e-9) for k in E[w]}
        R = len(next(iter(P[w].values()))) if P[w] else 0
        ZP[w] = [{k: P[w][k][j] / (SD[(b, k)] + 1e-9) for k in P[w] if P[w][k][j] == P[w][k][j]} for j in range(R)]

    def gscore(z):
        return {g: max([z[k] for k in ks if k in z], default=-99.0) for g, ks in GROUPS.items()}

    G = {w: gscore(Z[w]) for w in W}
    GP = {w: [gscore(z) for z in ZP[w]] for w in W}
    # z is already scaled within frequency bins, so the pseudo pool is shared by all bins
    pool = {g: np.sort([gp[g] for w in W for gp in GP[w] if gp[g] > -98]) for g in GROUPS}

    def pval(b, g, s, drop=None):
        arr = pool[g]
        k = len(arr) - np.searchsorted(arr, s, side='left')
        if drop is not None:
            k -= 1
        return (1 + k) / (1 + len(arr))

    # calibrate the class threshold on pseudo-words: the min-p over groups of a pseudo-word
    pmin = []
    for w in W:
        b = fbin(count[w])
        for gp in GP[w]:
            pmin.append(min((pval(b, g, s, drop=True) if s > -98 else 1.0) for g, s in gp.items()))
    pmin = np.sort(pmin)
    if alpha is None:   # threshold that lets FALSE_RATE of pseudo-words through
        alpha = float(pmin[max(0, int(FALSE_RATE * len(pmin)) - 1)]) if len(pmin) else 0.01
    PV, C = {}, {}
    for w in W:
        b = fbin(count[w])
        PV[w] = {g: (pval(b, g, s) if s > -98 else 1.0) for g, s in G[w].items()}
        g = min(PV[w], key=PV[w].get)
        C[w] = g if PV[w][g] <= alpha else 'SILENT'
    fc, nf = int(np.sum(pmin <= alpha)), len(pmin)
    base = {}
    for d in D:
        for k, v in d['base'].items():
            base.setdefault(k, []).append(v)
    return dict(name=name, models=[d['m'] for d in D], count=count, E=E, Z=Z, G=G, PV=PV, C=C,
                fp=fc / max(1, nf), alpha=alpha, base={k: float(np.mean(v)) for k, v in base.items()})


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    s = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return s / len(pos) / len(neg)


def key_for(name):
    if name == 'dose':
        return dict(L.DOSE_KEY)
    if name.startswith('plant'):
        return {w: r for w, r in L.PLANT_KEY.items()}
    if name == 'lb':
        return L.lb_key(L.build_lb())
    if name == 'ur3':
        u = L.build_ur3()
        return L.ur3_key(u['truth'], u['docs'])
    return None


def report(R, key=None, top=30):
    out = []
    C = R['C']
    out.append('%s: models %d, types %d, false-class rate %.3f (calibrated alpha %.4f); base %s' % (
        R['name'], len(R['models']), len(C), R['fp'], R['alpha'], {k: round(v, 2) for k, v in R['base'].items()}))
    out.append('  classes %s' % dict(collections.Counter(C.values())))
    if key:
        roles = collections.defaultdict(list)
        for w in C:
            if w in key:
                roles[key[w]].append(w)
        for r, ws in sorted(roles.items()):
            out.append('  key %-7s n %3d -> %s' % (r, len(ws), dict(collections.Counter(C[w] for w in ws))))
        for g in GROUPS:
            sc = {w: -math.log10(R['PV'][w][g]) for w in C}
            for pos_r, neg_r in ((('COM', 'QUAL', 'UNI'), ('PER', 'PLA')), (('UNI',), ('PER', 'PLA')),
                                 (('QUAL',), ('PER', 'PLA')), (('LOGO',), ('SILENT',)), (('NUM',), ('SILENT',))):
                pos = [sc[w] for w in C if key.get(w) in pos_r]
                neg = [sc[w] for w in C if key.get(w) in neg_r]
                if len(pos) >= 2 and len(neg) >= 2:
                    out.append('  AUC %-6s %s vs %s: %.3f (n %d/%d)' % (g, '+'.join(pos_r), '+'.join(neg_r), auc(pos, neg), len(pos), len(neg)))
    for g in list(GROUPS) + ['SILENT']:
        ws = sorted([w for w in C if C[w] == g], key=lambda w: -R['count'][w])
        out.append('  %-6s (%d): %s' % (g, len(ws), ' '.join('%s[%d,p%.3f]' % (w, R['count'][w], min(R['PV'][w].values())) for w in ws[:top])))
    return out


def profile(R, w):
    if w not in R['C']:
        return '%-14s not measured' % w
    return '%-14s n%-3d %-6s %s' % (w, R['count'][w], R['C'][w], ' '.join(
        '%s z%+.1f p%.3f' % (g, R['G'][w][g], R['PV'][w][g]) for g in GROUPS if R['G'][w][g] > -98))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    models = None
    for a in sys.argv[1:]:
        if a.startswith('-models'):
            lo, hi = a.split('=')[1].split(':')
            models = set(range(int(lo), int(hi)))
    for name in args:
        R = analyse(name, models)
        if R is None:
            print(name, 'no data'); continue
        print('\n'.join(report(R, key_for(name))))
        if '-v' in sys.argv:
            for w in sorted(R['C'], key=lambda w: -R['count'][w])[:60]:
                print('   ' + profile(R, w))
