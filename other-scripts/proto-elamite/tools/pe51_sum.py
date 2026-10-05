"""pe51 summary: loss profiles -> loss classes, stability, nulls, answer-key scoring.

usage: python3 pe51_sum.py CORPUS [CORPUS ...]
Per sign s, task k and model m: excess = mean loss change when s is erased minus the mean change
when as many random other sign tokens are erased in the same tablets (R draws).
t_k(s) = mean_m excess / (sd_m / sqrt(M)). A pseudo-sign (null draw 0 scored against draws 1..R-1)
gives the calibrated null of t. Class = largest task group with t above the threshold that lets
<= 5% of pseudo-signs through; signs with no group above it are SILENT (the 'name' class).
Groups: NUM = SYS or MAG, TOT, HEAD, ENT.
"""
import glob, json, math, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe51_lib as L

TASKS = ['SYS', 'MAG', 'TOT', 'HEAD', 'ENT']
GROUPS = {'NUM': ['SYS', 'MAG'], 'TOT': ['TOT'], 'HEAD': ['HEAD'], 'ENT': ['ENT']}


KEYS = [k + l for k in TASKS for l in 'LR']
GROUPS = {'NUM-L': ['SYSL', 'MAGL'], 'NUM-R': ['SYSR', 'MAGR'], 'TOT': ['TOTL', 'TOTR'],
          'HEAD': ['HEADL', 'HEADR'], 'ENT': ['ENTR']}
FLOOR = 0.02  # nats per target


def load(name, models=None):
    F = sorted(f for f in glob.glob(os.path.join(L.CK, 'del_%s_*.json' % name))
               if re.match(r'^del_%s_\d+\.json$' % re.escape(name), os.path.basename(f)))
    D = [json.load(open(f)) for f in F]
    if models is not None:
        D = [d for d in D if d['m'] in models]
    return D


def excess_table(D, halves=(0, 1), pseudo=False):
    """-> sign -> key -> list of per-model excess: mean loss change per target when the sign is
    erased minus the same for as many random other sign tokens erased in the same tablets.
    pseudo=True: null draw 0 plays the sign, draws 1.. are the null."""
    X = collections.defaultdict(lambda: collections.defaultdict(list))
    for d in D:
        for w, acc in d['res'].items():
            for k in KEYS:
                s, n, ns, nn = 0.0, 0, None, None
                for h in halves:
                    a = acc.get(str(h), {}).get(k)
                    if not a:
                        continue
                    s += a[0]; n += a[1]
                    ns = a[2] if ns is None else [x + y for x, y in zip(ns, a[2])]
                    nn = a[3] if nn is None else [x + y for x, y in zip(nn, a[3])]
                if ns is None:
                    continue
                if pseudo:
                    if nn[0] < 3 or sum(nn[1:]) < 3:
                        continue
                    X[w][k].append(ns[0] / nn[0] - sum(ns[1:]) / sum(nn[1:]))
                else:
                    if n < 3 or sum(nn) < 3:
                        continue
                    X[w][k].append(s / n - sum(ns) / sum(nn))
    return X


def tstats(X, minM=3):
    T, M = {}, {}
    for w, kk in X.items():
        T[w], M[w] = {}, {}
        for k, v in kk.items():
            if len(v) < minM:
                continue
            v = np.array(v)
            sd = v.std(ddof=1)
            M[w][k] = float(v.mean())
            T[w][k] = float(v.mean() / (sd / math.sqrt(len(v)) + 1e-6))
    return T, M


def group_scores(TM):
    T, M = TM
    G = {}
    for w in T:
        g = {}
        for gn, ks in GROUPS.items():
            vals = [(T[w][k], M[w][k]) for k in ks if k in T[w]]
            ok = [t for t, m in vals if m >= FLOOR]
            g[gn] = max(ok) if ok else (max([t for t, m in vals]) if vals else -99) * 0 - 1
        G[w] = g
    return G


def classify(G, thr):
    C = {}
    for w, g in G.items():
        best = max(g, key=g.get)
        C[w] = best if g[best] > thr else 'SILENT'
    return C


def analyse(name, thr=3.0, models=None):
    D = load(name, models)
    if len(D) < 3:
        return None
    ms = sorted(d['m'] for d in D)
    Gp = group_scores(tstats(excess_table(D, pseudo=True)))
    G = group_scores(tstats(excess_table(D)))
    C = classify(G, thr)
    Cp = classify(Gp, thr)
    Ch = [classify(group_scores(tstats(excess_table(D, halves=(h,)))), thr) for h in (0, 1)]
    A = [d for d in D if d['m'] % 2 == 0]; B = [d for d in D if d['m'] % 2 == 1]
    Cs = [classify(group_scores(tstats(excess_table(x), minM=2)), thr) for x in (A, B)]
    stable = {w: all(Ch[i].get(w) == c for i in (0, 1)) and all(Cs[i].get(w) == c for i in (0, 1)) for w, c in C.items()}
    fp = float(np.mean([c != 'SILENT' for c in Cp.values()])) if Cp else float('nan')
    count = {}
    for d in D:
        count.update(d['count'])
    base = {k: float(np.mean([d['base'].get(k, np.nan) for d in D])) for k in TASKS}
    TM = tstats(excess_table(D))
    return dict(name=name, models=ms, thr=thr, fp=fp, G=G, C=C, stable=stable, count=count,
                base=base, Ch=Ch, Cs=Cs, T=TM[0], M=TM[1], Cp=Cp)


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
    for name in [a for a in sys.argv[1:] if not a.startswith('-')]:
        R = analyse(name)
        if R is None:
            print(name, 'no data'); continue
        key = {'plant': L.PLANT_KEY, 'pc': L.PC_KEY}.get(name)
        if name == 'ur3':
            key = dict(L.UR3_KEY); key['name'] = sorted(L.ur3_names(L.build_ur3()))
        print('\n'.join(report(R, key)))
        if '-v' in sys.argv or name == 'plant':
            for w in sorted(R['C'], key=lambda w: -R['count'][w])[:40]:
                print('   %-12s %-6s %s' % (w, R['C'][w], ' '.join('%s %+.3f(%.1f)' % (k, R['M'][w][k], R['T'][w][k]) for k in KEYS if k in R['M'][w] and abs(R['T'][w][k]) > 2)))
        if key:
            for g, roles in [('NUM-L', ['commodity', 'unit']), ('TOT', ['total']), ('HEAD', ['doctype']),
                             ('ENT', ['doctype']), ('NUM-R', ['doctype'])]:
                pos = [R['G'][w][g] for r in roles for w in key.get(r, []) if w in R['G']]
                neg = [R['G'][w][g] for w in R['G'] if not any(w in key.get(r, []) for r in roles)]
                print('  AUC %s score for %s vs rest: %.3f (n %d)' % (g, roles, auc(pos, neg), len(pos)))
        for c in list(GROUPS) + ['SILENT']:
            ws = sorted([w for w in R['C'] if R['C'][w] == c and R['stable'][w]], key=lambda w: -R['count'][w])
            print('  %s stable: %s' % (c, ' '.join(ws[:40])))


EARLIER = {
    'pe45 name-frame': ['M056', 'M010', 'M001', 'M157', 'M210', 'M054', 'M009'],
    'GRAIN office': ['M010', 'M106', 'M002', 'M243', 'M075', 'M081', 'M265', 'M266', 'M296', 'M112'],
    'CLASS office': ['M387', 'M388', 'M218', 'M124', 'M009', 'M066', 'M057'],
    'BARE-COUNT office': ['M054', 'M367', 'M001', 'M370', 'M032', '|M036+1(N30D)|', 'M206', 'M269', 'M059', 'M102'],
    'header twig': ['M288', 'M157', 'M153', 'M175', 'M106', 'M010'],
    'capacity twig': ['M387', 'M297', 'M036', 'M260', 'M111', 'M264', 'M265', 'M002'],
    'name twig 1': ['M388', 'M218', 'M263', 'M057', 'M066'],
    'name twig 2': ['M371', 'M377', 'M320', 'M347', 'M386'],
    'never-capacity': ['M346', 'M263', 'M376', 'M003', 'M032', 'M373', 'M264', 'M102', 'M362', 'M317', 'M149', 'M046'],
    'pe39 SZE/U4/B': ['M136', 'M111', 'M147', 'M354'],
}


def mean_excess(R, w, g):
    vals = [R['M'][w].get(k) for k in GROUPS[g] if R['M'][w].get(k) is not None]
    return max(vals) if vals else float('nan')


def compare_earlier(R):
    out = []
    ws = list(R['G'])
    for name, mem in EARLIER.items():
        mem = [w for w in mem if w in R['G']]
        if len(mem) < 2:
            continue
        row = []
        for g in GROUPS:
            pos = [mean_excess(R, w, g) for w in mem]
            neg = [mean_excess(R, w, g) for w in ws if w not in mem]
            pos = [x for x in pos if x == x]; neg = [x for x in neg if x == x]
            row.append('%s %.2f' % (g, auc(pos, neg)))
        out.append('  %-18s n %2d  AUC(excess): %s  classes %s' % (name, len(mem), ' '.join(row),
                   dict(collections.Counter(R['C'][w] for w in mem))))
    return out


def top_by_group(R, n=12):
    out = []
    for g in GROUPS:
        ws = sorted(R['G'], key=lambda w: -(mean_excess(R, w, g) if mean_excess(R, w, g) == mean_excess(R, w, g) else -9))
        out.append('  top %-6s ' % g + ' '.join('%s(%+.3f,t%.1f)' % (w, mean_excess(R, w, g), R['G'][w][g]) for w in ws[:n]))
    return out
