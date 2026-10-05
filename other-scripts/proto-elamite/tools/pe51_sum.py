"""pe51 summary: loss profiles -> loss classes, stability, nulls, answer-key scoring.

usage: python3 pe51_sum.py CORPUS [CORPUS ...]
Per sign s, task k and model m: excess = mean loss change when s is erased minus the mean change
when as many random other sign tokens are erased in the same tablets (R draws).
t_k(s) = mean_m excess / (sd_m / sqrt(M)). A pseudo-sign (null draw 0 scored against draws 1..R-1)
gives the calibrated null of t. Class = largest task group with t above the threshold that lets
<= 5% of pseudo-signs through; signs with no group above it are SILENT (the 'name' class).
Groups: NUM = SYS or MAG, TOT, HEAD, ENT.
"""
import glob, json, math, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe51_lib as L

TASKS = ['SYS', 'MAG', 'TOT', 'HEAD', 'ENT']
GROUPS = {'NUM': ['SYS', 'MAG'], 'TOT': ['TOT'], 'HEAD': ['HEAD'], 'ENT': ['ENT']}


def load(name, models=None):
    F = sorted(glob.glob(os.path.join(L.CK, 'del_%s_*.json' % name)))
    D = [json.load(open(f)) for f in F]
    if models is not None:
        D = [d for d in D if d['m'] in models]
    return D


def excess_table(D, halves=(0, 1), pseudo=False):
    """-> sign -> task -> list of per-model excess (mean per target)."""
    X = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in D:
        for w, acc in d['res'].items():
            for k in TASKS:
                s, n, nul = 0.0, 0, None
                for h in halves:
                    a = acc.get(str(h), {}).get(k)
                    if not a:
                        continue
                    s += a[0]; n += a[1]
                    nul = a[2] if nul is None else [x + y for x, y in zip(nul, a[2])]
                if n < 3:
                    continue
                if pseudo:
                    real = nul[0] / n; rest = nul[1:]
                else:
                    real = s / n; rest = nul
                X[w][k].append(real - sum(rest) / len(rest) / n)
    return X


def tstats(X, minM=3):
    T = {}
    for w, kk in X.items():
        T[w] = {}
        for k, v in kk.items():
            if len(v) < minM:
                continue
            v = np.array(v)
            sd = v.std(ddof=1) if len(v) > 1 else 1.0
            T[w][k] = float(v.mean() / (sd / math.sqrt(len(v)) + 1e-4))
    return T


def group_scores(T):
    G = {}
    for w, t in T.items():
        G[w] = {g: max([t.get(k, -99) for k in ks]) for g, ks in GROUPS.items()}
    return G


def threshold(Gp, q=0.95):
    """per-group threshold from pseudo-signs (any group above -> false class)."""
    mx = [max(g.values()) for g in Gp.values() if g]
    return max(3.0, float(np.quantile(mx, q))) if mx else 3.0


def classify(G, thr):
    C = {}
    for w, g in G.items():
        best = max(g, key=g.get)
        C[w] = best if g[best] > thr else 'SILENT'
    return C


def analyse(name, quiet=False):
    D = load(name)
    if not D:
        return None
    ms = sorted(d['m'] for d in D)
    Gp = group_scores(tstats(excess_table(D, pseudo=True)))
    thr = threshold(Gp)
    G = group_scores(tstats(excess_table(D)))
    C = classify(G, thr)
    # stability: tablet halves and model halves (seeds)
    Ch = [classify(group_scores(tstats(excess_table(D, halves=(h,)))), thr) for h in (0, 1)]
    A = [d for d in D if d['m'] < 5]; B = [d for d in D if d['m'] >= 5]
    Cs = [classify(group_scores(tstats(excess_table(x), minM=2)), thr) if x else {} for x in (A, B)]
    stable = {}
    for w, c in C.items():
        ok_h = all(Ch[i].get(w) == c for i in (0, 1))
        ok_s = all(Cs[i].get(w) == c for i in (0, 1)) if Cs[0] and Cs[1] else True
        stable[w] = ok_h and ok_s
    # false-class rate of pseudo signs
    fp = np.mean([max(g.values()) > thr for g in Gp.values() if g])
    count = {}
    for d in D:
        count.update(d['count'])
    base = {k: float(np.mean([d['base'].get(k, np.nan) for d in D])) for k in TASKS}
    return dict(name=name, models=ms, thr=thr, fp=float(fp), G=G, C=C, stable=stable, count=count,
                base=base, Ch=Ch, Cs=Cs)


def agree(a, b):
    ks = [w for w in a if w in b]
    return sum(a[w] == b[w] for w in ks) / max(1, len(ks)), len(ks)


def report(R, key=None):
    lines = []
    C, S = R['C'], R['stable']
    cnt = collections.Counter(C.values())
    stc = collections.Counter(c for w, c in C.items() if S[w])
    lines.append('%s: models %d, signs %d, threshold t>%.2f (pseudo-sign false-class %.3f); base %s' % (
        R['name'], len(R['models']), len(C), R['thr'], R['fp'], {k: round(v, 2) for k, v in R['base'].items()}))
    lines.append('  classes %s; stable in tablet halves and seed halves %s' % (dict(cnt), dict(stc)))
    ah = agree(R['Ch'][0], R['Ch'][1]); as_ = agree(R['Cs'][0], R['Cs'][1]) if R['Cs'][0] else (float('nan'), 0)
    lines.append('  agreement half0/half1 %.2f (n %d); seeds A/B %.2f (n %d)' % (ah[0], ah[1], as_[0], as_[1]))
    if key:
        for role, ws in key.items():
            ws = [w for w in ws if w in C]
            if not ws:
                continue
            lines.append('  key %-10s n %3d -> %s' % (role, len(ws), dict(collections.Counter(C[w] for w in ws))))
    return lines


def auc(pos, neg):
    if not pos or not neg:
        return float('nan')
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else 0.5 if p == n else 0.0
    return s / len(pos) / len(neg)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        R = analyse(name)
        if R is None:
            print(name, 'no data'); continue
        key = {'plant': L.PLANT_KEY, 'pc': L.PC_KEY}.get(name)
        if name == 'ur3':
            key = dict(L.UR3_KEY); key['name'] = sorted(L.ur3_names(L.build_ur3()))
        print('\n'.join(report(R, key)))
        if key:
            for g, roles in [('NUM', ['commodity', 'unit']), ('TOT', ['total']), ('HEAD', ['doctype'])]:
                pos = [R['G'][w][g] for r in roles for w in key.get(r, []) if w in R['G']]
                neg = [R['G'][w][g] for w in R['G'] if not any(w in key.get(r, []) for r in roles)]
                print('  AUC %s score for %s vs rest: %.3f (n %d)' % (g, roles, auc(pos, neg), len(pos)))
        for c in ['NUM', 'TOT', 'HEAD', 'ENT', 'SILENT']:
            ws = sorted([w for w in R['C'] if R['C'][w] == c and R['stable'][w]], key=lambda w: -R['count'][w])
            print('  %s stable: %s' % (c, ' '.join(ws[:40])))
