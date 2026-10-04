"""v6 cycle 5: is the left-edge link a column (glyph above -> glyph below), and does it hold out of sample?
(a) Which property of line r best predicts the first glyph of line r+1? first glyph of r, its first word (class),
    last glyph of r, length bin of r, q-mode bin of r. MI excess over within-paragraph line permutation (200x).
(b) Held-out prediction: fit a smoothed transition table first-glyph(r) -> first-glyph(r+1) on even folios, score
    bits per line on odd folios against a unigram line-initial model fitted on the same half (and vice versa);
    null = same with lines permuted within paragraph in the TEST half (keeps the test half's positional marginals).
(c) Does the column follow any known Voynich transition rule? Spearman correlation of the left-edge log(obs/exp)
    table with: within-word glyph bigrams, word junction (last -> next first) and first-first along rows.
(d) Currier A / B and hands separately (ZL)."""
import sys, os, json, random, math
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from v6_gridlib import mi, units_voy as U
from v6_cycle4 import voy_meta, perm_within, zstat

REPS = int(os.environ.get('REPS', 200))

def qscore(l):
    q = sum(w.startswith('q') for w in l); a = sum(('ai' in w) or ('ar' in w) for w in l)
    return (q - a) / len(l)

def bin3(x, cuts): return 0 if x < cuts[0] else (1 if x < cuts[1] else 2)

def props(paras):
    allq = sorted(qscore(l) for p in paras for l in p); alll = sorted(len(l) for p in paras for l in p)
    qc = (allq[len(allq) // 3], allq[2 * len(allq) // 3]); lc = (alll[len(alll) // 3], alll[2 * len(alll) // 3])
    cnt = Counter(l[0] for p in paras for l in p[1:])
    return {'first glyph': lambda l: U(l[0])[0], 'first word': lambda l: l[0] if cnt[l[0]] >= 8 else '*' + U(l[0])[0],
            'last glyph': lambda l: U(l[-1])[-1], 'length bin': lambda l: bin3(len(l), lc),
            'q-mode bin': lambda l: bin3(qscore(l), qc), 'second word first glyph': lambda l: U(l[1])[0] if len(l) > 1 else '#'}

def ppairs(paras, f):
    return [(f(p[i]), U(p[i + 1][0])[0]) for p in paras for i in range(1, len(p) - 1)]

def heldout(train, test, alpha=0.5):
    T = defaultdict(Counter); uni = Counter()
    for p in train:
        for i in range(1, len(p)):
            uni[U(p[i][0])[0]] += 1
            if i < len(p) - 1: T[U(p[i][0])[0]][U(p[i + 1][0])[0]] += 1
    V = sorted(set(uni) | {U(l[0])[0] for p in test for l in p}); N = sum(uni.values())
    pu = {v: (uni[v] + alpha) / (N + alpha * len(V)) for v in V}
    def score(paras):
        gain, n = 0.0, 0
        for p in paras:
            for i in range(1, len(p) - 1):
                a, b = U(p[i][0])[0], U(p[i + 1][0])[0]; row = T[a]; s = sum(row.values())
                pc = (row[b] + 5 * pu[b]) / (s + 5)          # back-off to unigram, weight 5
                gain += math.log2(pc / pu[b]); n += 1
        return gain / n
    return score

def table(P):
    a = Counter(x for x, _ in P); b = Counter(y for _, y in P); ab = Counter(P); n = len(P)
    return {k: math.log((v + .5) / (a[k[0]] * b[k[1]] / n + .5)) for k, v in ab.items() if a[k[0]] >= 30 and b[k[1]] >= 30}

def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for j, i in enumerate(o): r[i] = j
        return r
    rx, ry = rk(x), rk(y); n = len(x); mx = sum(rx) / n; my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5

if __name__ == '__main__':
    res = {}
    for name in ('ZL3b', 'IT2a'):
        vm = voy_meta(name); paras = [p['lines'] for p in vm]; R = {}
        sims = [perm_within(paras, random.Random(s)) for s in range(REPS)]
        for k, f in props(paras).items():
            R['(a) ' + k] = zstat(mi(ppairs(paras, f)), [mi(ppairs(q, f)) for q in sims])
        # (b) held-out
        even = [p['lines'] for p in vm if int(''.join(c for c in p['folio'] if c.isdigit()) or 0) % 2 == 0]
        odd = [p['lines'] for p in vm if int(''.join(c for c in p['folio'] if c.isdigit()) or 0) % 2 == 1]
        for tr, te, lab in ((even, odd, 'train even/test odd'), (odd, even, 'train odd/test even')):
            sc = heldout(tr, te)
            R['(b) ' + lab + ' bits/line'] = zstat(sc(te), [sc(perm_within(te, random.Random(s))) for s in range(REPS)])
        # (c) correlation with other transition tables
        edge = table([(U(p[i][0])[0], U(p[i + 1][0])[0]) for p in paras for i in range(1, len(p) - 1)])
        inword, junc, ff = [], [], []
        for p in paras:
            for l in p:
                for w in l:
                    u = U(w); inword += list(zip(u, u[1:]))
                for a, b in zip(l, l[1:]):
                    junc.append((U(a)[-1], U(b)[0])); ff.append((U(a)[0], U(b)[0]))
        for lab, P in (('within-word bigram', inword), ('junction last->first', junc), ('row first->first', ff)):
            t = table(P); ks = [k for k in edge if k in t]
            R['(c) rho vs ' + lab] = {'rho': spearman([edge[k] for k in ks], [t[k] for k in ks]), 'n': len(ks)}
        # (d) A/B and hands (first glyph only)
        if name == 'ZL3b':
            lines = vlib.load_voynich(name, drop_uncertain=True)
            meta = {}
            for L in lines: meta.setdefault(L['folio'], (L.get('lang'), L.get('hand')))
            for key, idx in (('lang', 0), ('hand', 1)):
                groups = defaultdict(list)
                for p in vm: groups[meta[p['folio']][idx]].append(p['lines'])
                for g, ps in sorted(groups.items(), key=lambda x: str(x[0])):
                    if sum(len(p) - 2 for p in ps) < 150: continue
                    f = props(paras)['first glyph']
                    R['(d) %s %s (n=%d pairs)' % (key, g, sum(len(p) - 2 for p in ps))] = zstat(
                        mi(ppairs(ps, f)), [mi(ppairs(perm_within(ps, random.Random(s)), f)) for s in range(REPS)])
        res[name] = R
        print('\n==', name)
        for k, v in R.items():
            if 'rho' in v: print('  %-44s rho %+.3f (n=%d)' % (k, v['rho'], v['n']))
            else: print('  %-44s obs %.4f null %.4f excess %+.4f z %6.2f' % (k, v['obs'], v['null'], v['ex'], v['z']))
    json.dump(res, open(os.path.join(vlib.RES, 'v6_cycle5.json'), 'w'), indent=0)
