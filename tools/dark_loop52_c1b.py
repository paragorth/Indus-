"""Loop 52, cycle 1b: rescoring of the cycle-1 graph after its first lesson.

Cycle 1a (tools/dark_loop52_c1.py) defined E1 as 'MI(form, neighbour) above a permutation null' and E3 as 'fine
similarity >= the 25th percentile of the S268 strong merges (0.667)'. E1 was then positive for 57% of all pairs
(any two signs with different collocations pass) and E3 for 31%, so E1+E3 chained 212 signs into one class
(jar, marked jars, numerals): the S268 lesson again. Here:
E1 = contextual allography in the S-DARK-46 sense: (i) same slot (position bins initial/medial/final/alone,
     chi-square P > 0.05), (ii) MI permutation P <= 0.01 (from cycle 1a), (iii) a near-deterministic rule: some
     neighbour with >= 10 tokens of the pair selects one form at >= 90%.
E3 = loop28 strokes/doubling edge, or fine similarity >= 0.85 (top 0.2% of all sign pairs; S268's cut was 0.90
     on the coarse matrix). E2 and E4 unchanged. Same vetoes, same ladder, same control.
Rewrites loop52_merges.json and writes loop52_cycle1.txt (1a kept as loop52_cycle1a.txt).
"""
import json, collections, itertools, random, os
import numpy as np
from scipy.stats import chi2_contingency

ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/'
random.seed(521)
corpus = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
levels = json.load(open(ROOT + 'data/derived/sign_allographs_levels.json'))['merges']
M = json.load(open(OUT + 'loop52_merges.json'))
if not os.path.exists(OUT + 'loop52_cycle1a.txt'):
    os.rename(OUT + 'loop52_cycle1.txt', OUT + 'loop52_cycle1a.txt')
rows = M['pairs']
FINE_CUT = 0.85
tok = collections.Counter(s for r in corpus for s in r['seq_raw'])
signs = sorted(tok)
BOUND = -1

# position bins and neighbour tables
pos = collections.defaultdict(lambda: np.zeros(4))
prevc = collections.defaultdict(collections.Counter); nextc = collections.defaultdict(collections.Counter)
for r in corpus:
    q = r['seq_raw']
    for i, s in enumerate(q):
        if len(q) == 1: pos[s][3] += 1
        elif i == 0: pos[s][0] += 1
        elif i == len(q) - 1: pos[s][2] += 1
        else: pos[s][1] += 1
        prevc[s][q[i - 1] if i > 0 else BOUND] += 1
        nextc[s][q[i + 1] if i + 1 < len(q) else BOUND] += 1


def same_slot(a, b):
    tab = np.array([pos[a], pos[b]]); tab = tab[:, tab.sum(0) > 0]
    if tab.shape[1] < 2: return True, 1.0
    _, p, _, _ = chi2_contingency(tab)
    return p > 0.05, float(p)


def rule(a, b):
    best = (0.0, None, 0)
    for side, tabs in (('prev', prevc), ('next', nextc)):
        for k in set(tabs[a]) | set(tabs[b]):
            n = tabs[a][k] + tabs[b][k]
            if n >= 10:
                share = max(tabs[a][k], tabs[b][k]) / n
                if share > best[0]: best = (share, '%s %s' % (side, k), n)
    return best


for r in rows:
    a, b = r['a'], r['b']
    f = r['fine']
    g28 = set(r['loop28'])
    r['ev']['E3'] = bool((g28 & {'strokes', 'doubling'}) or (f is not None and f >= FINE_CUT))
    if r.get('rare'):
        r['ev']['E1'] = False
    else:
        ok, p = same_slot(a, b); sh, ctx, n = rule(a, b)
        r['slot_P'] = round(p, 3); r['rule'] = [round(sh, 2), ctx, n]
        r['ev']['E1'] = bool(ok and min(r['MI_prev_P'], r['MI_next_P']) <= 0.01 and sh >= 0.9)
    r['n_ev'] = sum(r['ev'].values())


def canonical_map(level):
    allowed = {'raw': set(), 'strong': {'strong'}, 'all': {'strong', 'probable'}}[level]
    return {m['form']: m['into'] for m in levels if m['level'] in allowed}


class UF:
    def __init__(self): self.p = {}
    def f(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def u(self, a, b): self.p[self.f(a)] = self.f(b)


def principled(ev, exclude_types=()):
    """one non-distributional witness (E3 shape or E4 reader) AND one distributional witness (E1 or E2)"""
    e = {t: (v and t not in exclude_types) for t, v in ev.items()}
    return (e['E3'] or e['E4']) and (e['E1'] or e['E2'])


def partition(level, k, use_rows=None, exclude_types=()):
    uf = UF()
    for s in signs: uf.f(s)
    for f_, h in canonical_map(level).items(): uf.u(f_, h)
    for r in (use_rows if use_rows is not None else rows):
        if r['veto']: continue
        if k == 'p':
            if principled(r['ev'], exclude_types): uf.u(r['a'], r['b'])
        elif sum(v for t, v in r['ev'].items() if t not in exclude_types) >= k: uf.u(r['a'], r['b'])
    cls = collections.defaultdict(set)
    for s in signs: cls[uf.f(s)].add(s)
    return cls


def chao1(counts):
    c = np.array(list(counts)); S = len(c); f1 = int((c == 1).sum()); f2 = int((c == 2).sum())
    return S + (f1 * (f1 - 1) / (2 * (f2 + 1)) if f2 == 0 else f1 * f1 / (2 * f2))


def merged_counts(cls):
    return [sum(tok[s] for s in m) for m in cls.values()]


out = ['# Loop 52 cycle 1 (rescored, see 1a for the first pass): four-evidence merge graph (seq_raw, %d types, %d tokens)' % (len(signs), sum(tok.values()))]
out.append('E1 = same slot + MI P <= 0.01 + a neighbour rule >= 0.90 (n >= 10); E3 = loop28 strokes/doubling or fine >= %.2f; E2, E4 as in 1a' % FINE_CUT)
usable = [r for r in rows if not r.get('rare')]
out.append('testable pairs (both >= 8 tokens) %d, rare-sign pairs %d' % (len(usable), len(rows) - len(usable)))
for t in ('E1', 'E2', 'E3', 'E4'):
    out.append('  %s positive: %d of %d pairs (%d after vetoes)' % (t, sum(r['ev'][t] for r in rows), len(rows), sum(r['ev'][t] for r in rows if not r['veto'])))
pairs_k = {k: [r for r in rows if not r['veto'] and r['n_ev'] >= k] for k in (1, 2, 3, 4)}
out.append('pairs by evidence count (after vetoes): ' + ', '.join('>=%d: %d' % (k, len(v)) for k, v in pairs_k.items()))
out.append('vetoed pairs with >= 2 evidence types: %d' % sum(1 for r in rows if r['veto'] and r['n_ev'] >= 2))
out.append('\n## agreement between evidence types (testable pairs, no veto)')
ok = [r for r in usable if not r['veto']]
for t1, t2 in itertools.combinations(('E1', 'E2', 'E3', 'E4'), 2):
    n11 = sum(r['ev'][t1] and r['ev'][t2] for r in ok); n1 = sum(r['ev'][t1] for r in ok); n2 = sum(r['ev'][t2] for r in ok)
    exp = n1 * n2 / max(1, len(ok))
    out.append('  %s & %s: %d observed vs %.1f if independent (ratio %.1f)' % (t1, t2, n11, exp, n11 / exp if exp else float('nan')))
# S268 recovery: how many of the 24 canonical merges does the graph find at k>=2 / k>=1 (start raw)?
canon = {tuple(sorted((m['form'], m['into']))) for m in levels}
found = {k: sum(1 for r in pairs_k[k] if (r['a'], r['b']) in canon) for k in (1, 2, 3)}
out.append('S268 canonical merges (27 pairs) recovered by the graph: k>=1 %d, k>=2 %d, k>=3 %d' % (found[1], found[2], found[3]))

out.append('\n## inventory ladder (types with >= 1 token; Chao1 on merged counts)')
out.append('| start | evidence k | classes | merged signs | largest class | Chao1 | f1 | f2 |')
ladder = {}
for level in ('raw', 'strong', 'all'):
    for k in (None, 3, 'p', 2, 1):
        cls = partition(level, 99 if k is None else k)
        label = 'canonical %s' % level + ('' if k is None else (' + graph principled (E3|E4)&(E1|E2)' if k == 'p' else ' + graph >= %d' % k))
        mc = merged_counts(cls); big = max(len(v) for v in cls.values())
        ladder[(level, k)] = {'classes': len(cls), 'chao1': chao1(mc), 'largest': big, 'f1': sum(1 for c in mc if c == 1), 'f2': sum(1 for c in mc if c == 2)}
        out.append('| %s | %s | %d | %d | %d | %.0f | %d | %d |' % (label, k if k else '-', len(cls), len(signs) - len(cls), big, chao1(mc), ladder[(level, k)]['f1'], ladder[(level, k)]['f2']))
out.append('\n## sensitivity (start all): drop one evidence type, k >= 2 and principled')
for t in ('E1', 'E2', 'E3', 'E4'):
    cls = partition('all', 2, exclude_types=(t,)); clp = partition('all', 'p', exclude_types=(t,))
    out.append('  without %s: k>=2 %d classes (Chao1 %.0f, largest %d); principled %d classes (Chao1 %.0f, largest %d)' % (t, len(cls), chao1(merged_counts(cls)), max(len(v) for v in cls.values()), len(clp), chao1(merged_counts(clp)), max(len(v) for v in clp.values())))
pk = [r for r in rows if not r['veto'] and principled(r['ev'])]
out.append('principled pairs: %d; of them in S268 canonical: %d; new: %s' % (len(pk), sum(1 for r in pk if (r['a'], r['b']) in canon), ', '.join('%d-%d' % (r['a'], r['b']) for r in pk if (r['a'], r['b']) not in canon)))
for t, name in (('E1', 'E1 alone (context-only)'), ('E2', 'E2 alone (free interchange only)'), ('E3', 'E3 alone (graphic only)'), ('E4', 'E4 alone (transcriber only)')):
    cls = partition('all', 1, exclude_types=tuple(x for x in ('E1', 'E2', 'E3', 'E4') if x != t))
    out.append('  %s: %d classes, largest %d' % (name, len(cls), max(len(v) for v in cls.values())))
out.append('\n## control: the k>=2 edge count drawn at random from the candidate universe (200x), start all')
n_edges = len(pairs_k[2]); pool = [r for r in rows if not r['veto']]; ctrl = []
for _ in range(200):
    fake = [dict(r, ev={'E1': True, 'E2': True, 'E3': False, 'E4': False}) for r in random.sample(pool, n_edges)]
    cls = partition('all', 2, use_rows=fake); ctrl.append((len(cls), max(len(v) for v in cls.values())))
ctrl = np.array(ctrl)
out.append('  real: %d classes, largest %d; random edge sets: %.1f +/- %.1f classes, largest %.1f' % (ladder[('all', 2)]['classes'], ladder[('all', 2)]['largest'], ctrl[:, 0].mean(), ctrl[:, 0].std(), ctrl[:, 1].mean()))
for k in ('p', 2, 1):
    cls = partition('all', k)
    multi = sorted([sorted(v, key=lambda s: -tok[s]) for v in cls.values() if len(v) > 1], key=lambda v: -sum(tok[s] for s in v))
    out.append('\n## merged classes at %s (start all): %d classes with >= 2 members, %d signs' % ('principled' if k == 'p' else 'k >= %d' % k, len(multi), sum(len(v) for v in multi)))
    for v in multi[:80 if k == 1 else 400]:
        out.append('  ' + '/'.join('%d(%d)' % (s, tok[s]) for s in v))
    if k == 1 and len(multi) > 80: out.append('  ... %d more' % (len(multi) - 80))
out.append('\n## k >= 2 edges with their evidence')
for r in sorted(pairs_k[2], key=lambda r: -(r['na'] + r['nb'])):
    out.append('  %d-%d n=%d/%d ev=%s cos=%s P_E2=%s minpairs=%d MI_P=%s/%s slot_P=%s rule=%s fine=%s loop28=%s E4=%s' % (
        r['a'], r['b'], r['na'], r['nb'], ''.join(t for t in ('E1', 'E2', 'E3', 'E4') if r['ev'][t]), r['cos'], r['P_E2'], r['minpairs'], r['MI_prev_P'], r['MI_next_P'], r.get('slot_P'), r.get('rule'), r['fine'], r['loop28'], r['E4']))
out.append('\n## vetoed pairs with >= 2 evidence types')
for r in sorted([r for r in rows if r['veto'] and r['n_ev'] >= 2], key=lambda r: -(r['na'] + r['nb'])):
    out.append('  %d-%d n=%d/%d ev=%s veto=%s' % (r['a'], r['b'], r['na'], r['nb'], ''.join(t for t in ('E1', 'E2', 'E3', 'E4') if r['ev'][t]), r['veto']))
open(OUT + 'loop52_cycle1.txt', 'w').write('\n'.join(out) + '\n')
M['note'] += '; rescored by dark_loop52_c1b.py: E1 = same slot + MI P<=0.01 + neighbour rule >=0.9; E3 = strokes/doubling or fine >= 0.85'
M['fine_cut'] = FINE_CUT; M['pairs'] = rows
M['classes'] = {'%s_k%s' % (lv, k): [sorted(v) for v in partition(lv, k).values() if len(v) > 1] for lv in ('raw', 'strong', 'all') for k in (1, 2, 3, 'p')}
M['ladder'] = {'%s_k%s' % (lv, k): v for (lv, k), v in ladder.items()}
json.dump(M, open(OUT + 'loop52_merges.json', 'w'), indent=0)
print('\n'.join(out[:90]))
