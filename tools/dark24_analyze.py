"""S-DARK-24 cycle 1: classify every aligned position of the Wells/IM77 object pairs, validate
against Mahadevan's own doubtful marks, build the graphic-confusion matrix with a permutation null,
and compare confused pairs' glyph similarity with the null. Usage: python3 dark24_analyze.py [maxcost]"""
import json, collections, random, sys, math
import numpy as np
R = '/home/user/Indus-/'; OUT = R + 'data/derived/dark/'
MAXCOST = float(sys.argv[1]) if len(sys.argv) > 1 else 9
corr = {int(k): set(v) for k, v in json.load(open(OUT + 'loop24_corr.json')).items()}
bridge = {int(k): set(v) for k, v in json.load(open(R + 'data/derived/bridge_extended.json')).items()}
inv = collections.defaultdict(set)
for w, ms in corr.items():
    for m in ms: inv[m].add(w)
gs = json.load(open(R + 'data/derived/glyph_sim_signs.json')); gidx = {s: i for i, s in enumerate(gs)}
G = np.load(R + 'data/derived/glyph_sim.npy')
levels = json.load(open(R + 'data/derived/sign_allographs_levels.json'))
merged_pairs = {frozenset((m['form'], m['into'])) for m in levels['merges']}
def gsim(a, b):
    if a in gidx and b in gidx: return float(G[gidx[a], gidx[b]])
    return None

def classify(rec):
    """Return list of position records for one object pair."""
    out = []
    ops = [o for o in rec['ops'] if o[0] != 'L']
    lineflip = any(o[0] == 'L' for o in rec['ops'])
    ws, ms, doubt = rec['wells']['seq'], rec['im']['seq'], rec['im']['doubt']
    n = len(ops)
    for k, o in enumerate(ops):
        kind, w, m = o[0], o[1], o[2]
        d = bool(doubt[o[4]]) if o[4] is not None else None
        if kind == 'S':
            if w == 0 and m == 0: c = 'both_lost'
            elif w == 0: c = 'lostW'          # Wells marks lost, Mahadevan reads a sign
            elif m == 0: c = 'lostM'          # Mahadevan marks lost, Wells reads a sign
            elif w not in corr: c = 'unbridged'
            elif m in corr[w]: c = 'agree'
            else: c = 'sub'
        elif kind == 'M':
            # Mahadevan-only sign: doubling convention if neighbour S op carries the same M sign
            nb = [ops[j] for j in (k - 1, k + 1) if 0 <= j < n and ops[j][0] == 'S']
            c = 'segM' if any(x[2] == m for x in nb) else ('lostM_extra' if m == 0 else 'insM')
        else:
            nb = [ops[j] for j in (k - 1, k + 1) if 0 <= j < n and ops[j][0] == 'S']
            c = 'segW' if any(x[1] == w for x in nb) else ('lostW_extra' if w == 0 else 'insW')
        out.append(dict(cls=c, w=w, m=m, doubt=d, pos=k, n=n, lineflip=lineflip))
    return out

def load(fn, maxcost=MAXCOST):
    P = json.load(open(OUT + fn))
    return [r for r in P if r['accepted'] and r['cost'] <= maxcost]

def summarize(pairs, label, lines, do_null=True, seed=0):
    rng = random.Random(seed)
    pos = []
    for r in pairs:
        for p in classify(r): p['cisi'] = r['cisi']; p['text_no'] = r['text_no']; p['site'] = r['site_code']; pos.append(p)
    cc = collections.Counter(p['cls'] for p in pos)
    lines.append(f'[{label}] object pairs {len(pairs)}, aligned positions {len(pos)}: ' + ', '.join(f'{k} {v}' for k, v in cc.most_common()))
    comparable = [p for p in pos if p['cls'] in ('agree', 'sub')]
    subs = [p for p in pos if p['cls'] == 'sub']
    rate = len(subs) / max(1, len(comparable))
    lines.append(f'  substitution rate among bridged sign-to-sign positions: {len(subs)}/{len(comparable)} = {100*rate:.2f}%; '
                 f'insertions (one reads a sign the other omits) insM {cc["insM"]} insW {cc["insW"]}; segmentation (doubling) segM {cc["segM"]} segW {cc["segW"]}; '
                 f'lost-vs-sign lostW {cc["lostW"]} lostM {cc["lostM"]}')
    # validation against Mahadevan's doubtful marks
    dc = collections.Counter((p['cls'], p['doubt']) for p in pos if p['doubt'] is not None)
    for c in ('agree', 'sub', 'lostW', 'insM', 'unbridged'):
        t = dc[(c, True)] + dc[(c, False)]
        if t: lines.append(f'  Mahadevan doubtful mark rate at {c}: {dc[(c, True)]}/{t} = {100*dc[(c,True)]/t:.1f}%')
    # position dependence: substitutions by relative position
    rel = collections.Counter(); relall = collections.Counter()
    for p in comparable:
        b = 'first' if p['pos'] == 0 else ('last' if p['pos'] == p['n'] - 1 else 'middle')
        relall[b] += 1
        if p['cls'] == 'sub': rel[b] += 1
    lines.append('  substitution rate by position: ' + ', '.join(f'{b} {rel[b]}/{relall[b]} ({100*rel[b]/max(1,relall[b]):.1f}%)' for b in ('first', 'middle', 'last')))
    if not do_null: return pos, subs
    # ---------- confusion matrix ----------
    pairs_c = collections.Counter()
    weights = []
    for p in subs:
        ys = inv.get(p['m'], set())
        ys = {y for y in ys if y != p['w']}
        if not ys: pairs_c[(p['w'], f'M{p["m"]}')] += 1; continue
        for y in ys: pairs_c[(p['w'], y)] += 1 / len(ys)
    # symmetrize (a,b) == (b,a)
    sym = collections.Counter()
    for (a, b), v in pairs_c.items(): sym[frozenset((a, b)) if not isinstance(b, str) else (a, b)] += v
    top = sorted(sym.items(), key=lambda z: -z[1])[:40]
    def fmt(k):
        if isinstance(k, frozenset): return '<->'.join(f'W{x}' for x in sorted(k))
        return f'W{k[0]}->{k[1]}'
    lines.append('  most frequent substitution pairs (weighted; W<->W via the correspondence table; W->M where M has no Wells partner): ' +
                 '; '.join(f'{fmt(k)} {v:.1f}' for k, v in top))
    # null: permute the M identities among substitution positions (keeps every sign's error load, randomizes partners)
    def stats(sub_m_list):
        c = collections.Counter()
        for p, m in zip(subs, sub_m_list):
            ys = {y for y in inv.get(m, set()) if y != p['w']}
            if not ys: c[(p['w'], f'M{m}')] += 1; continue
            for y in ys: c[frozenset((p['w'], y))] += 1 / len(ys)
        vals = sorted(c.values(), reverse=True)
        n2 = sum(1 for v in c.values() if v >= 2); n3 = sum(1 for v in c.values() if v >= 3)
        # glyph similarity of W<->W pairs (weighted mean)
        gs_, wt = 0.0, 0.0
        for k, v in c.items():
            if isinstance(k, frozenset) and len(k) == 2:
                a, b = sorted(k); g = gsim(a, b)
                if g is not None: gs_ += g * v; wt += v
        # share of pairs that are already in a Wells merge class
        mg = sum(v for k, v in c.items() if isinstance(k, frozenset) and k in merged_pairs)
        return dict(max=vals[0] if vals else 0, n2=n2, n3=n3, gsim=gs_ / wt if wt else float('nan'), merged=mg, distinct=len(c))
    obs = stats([p['m'] for p in subs])
    ms = [p['m'] for p in subs]
    nulls = []
    for _ in range(1000):
        sh = ms[:]; rng.shuffle(sh); nulls.append(stats(sh))
    def P(key, hi=True):
        o = obs[key]; vs = [x[key] for x in nulls]
        return (sum(1 for v in vs if v >= o) + 1) / (len(vs) + 1) if hi else (sum(1 for v in vs if v <= o) + 1) / (len(vs) + 1)
    def med(key): vs = sorted(x[key] for x in nulls); return vs[len(vs)//2], vs[int(0.95*len(vs))]
    lines.append(f'  confusion matrix vs permuted-partner null (1000x): distinct pairs {obs["distinct"]} (null median {med("distinct")[0]}); '
                 f'pairs recurring >=2: {obs["n2"]} vs null median {med("n2")[0]} 95% {med("n2")[1]} (P={P("n2"):.3f}); '
                 f'>=3: {obs["n3"]} vs {med("n3")[0]} / {med("n3")[1]} (P={P("n3"):.3f}); max {obs["max"]:.1f} vs {med("max")[0]:.1f} / {med("max")[1]:.1f} (P={P("max"):.3f}); '
                 f'mean glyph similarity of confused W pairs {obs["gsim"]:.3f} vs null {med("gsim")[0]:.3f} / 95% {med("gsim")[1]:.3f} (P={P("gsim"):.3f}); '
                 f'weight on pairs already in a Wells merge class {obs["merged"]:.1f} vs null {med("merged")[0]:.1f} / {med("merged")[1]:.1f} (P={P("merged"):.3f})')
    return pos, subs

if __name__ == '__main__':
    lines = [f'S-DARK-24 cycle 1 analysis (object pairs with alignment cost <= {MAXCOST})']
    pairs = load('loop24_pairs.json')
    nullp = load('loop24_pairs_null.json')
    pos, subs = summarize(pairs, 'HOME', lines)
    # by site
    for sc in ('MD', 'HP', 'LL', 'KB', 'CD', 'OS', 'WA'):
        sub = [r for r in pairs if r['site_code'] == sc]
        if len(sub) >= 20: summarize(sub, f'site {sc}', lines, do_null=(sc in ('MD', 'HP')), seed=1)
    # cost bands
    for lo, hi in ((0, 0), (0.25, 0.75), (1.0, 1.5), (1.75, 9)):
        sub = [r for r in pairs if lo <= r['cost'] <= hi]
        if sub: summarize(sub, f'cost {lo}-{hi}', lines, do_null=False)
    summarize(nullp, 'WRONG-SITE NULL PAIRS (chance object matches)', lines, do_null=False)
    # all substitution positions to file for later cycles
    json.dump([dict(p) for p in pos], open(OUT + 'loop24_positions.json', 'w'))
    open(OUT + 'loop24_cycle1_log.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
