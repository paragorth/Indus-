"""Compare the stable co-category graphs from the loop-10 cycles (data/derived/dark/loop10_*.json): Jaccard of stable
pair sets across seeds, sites (Mohenjo-daro / Harappa / other) and sequence levels; consensus classes; a STRATEGIES row.
Usage: python3 tools/strat_dark_loop10_final.py
"""
import json, os, glob, collections, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'data/derived/dark')
import sys; sys.path.insert(0, os.path.join(ROOT, 'tools'))
from strat_dark_loop10 import KNOWN

def load(tag):
    p = os.path.join(D, 'loop10_%s.json' % tag)
    return json.load(open(p)) if os.path.exists(p) else None

def pairset(J, thr=0.70, margin=0.20):
    top = J['top']; G = np.array(J['G']); GP = np.array(J['GP']); GS = np.array(J['GS'])
    s = set()
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            if G[i, j] >= thr and G[i, j] - max(GP[i, j], GS[i, j]) >= margin: s.add((min(top[i], top[j]), max(top[i], top[j])))
    return s

def rate(J, x, y):
    top = J['top']
    if x not in top or y not in top: return None
    return np.array(J['G'])[top.index(x), top.index(y)]

def jacc(a, b): return len(a & b) / max(1, len(a | b))

tags = ['cycle1', 'cycle2', 'cycle3_md', 'cycle3_ha', 'cycle3_other', 'cycle4_raw', 'cycle4_strong']
J = {t: load(t) for t in tags}; J = {t: v for t, v in J.items() if v}
P = {t: pairset(v) for t, v in J.items()}
out = []
out.append('LOOP 10 FINAL: stable co-category pairs per run and their overlap')
for t in J: out.append('  %-14s stable pairs %3d  components %s' % (t, len(P[t]), [['W%d' % x for x in c] for c in J[t]['comps']][:8]))
out.append('')
out.append('Jaccard overlap of stable pair sets:')
for a, b in itertools.combinations(J, 2):
    out.append('  %-14s %-14s J=%.2f  shared=%d' % (a, b, jacc(P[a], P[b]), len(P[a] & P[b])))
# consensus: pairs stable in both full-corpus seed runs and in at least one of each other split
full = P.get('cycle1', set()) & P.get('cycle2', set())
site_rep = {p for p in full if sum(p in P.get(t, set()) for t in ('cycle3_md', 'cycle3_ha', 'cycle3_other')) >= 1}
lvl_rep = {p for p in full if all(p in P.get(t, set()) for t in ('cycle4_raw', 'cycle4_strong') if t in P)}
# also: random-expectation of overlap between two runs = (|A||B|)/1770
def label(x): return ','.join(k for k, v in KNOWN.items() if x in v) or 'new'
out.append('')
out.append('CONSENSUS (stable in both full-corpus seeds, cycle1 & cycle2): %d pairs; chance overlap %.1f' %
           (len(full), len(P.get('cycle1', set())) * len(P.get('cycle2', set())) / 1770))
for x, y in sorted(full, key=lambda p: -(rate(J['cycle1'], *p) or 0)):
    reps = ''.join('M' if (x, y) in P.get('cycle3_md', set()) else '-', ) + ('H' if (x, y) in P.get('cycle3_ha', set()) else '-') + \
           ('O' if (x, y) in P.get('cycle3_other', set()) else '-') + ('r' if (x, y) in P.get('cycle4_raw', set()) else '-') + \
           ('s' if (x, y) in P.get('cycle4_strong', set()) else '-')
    r = [rate(J[t], x, y) for t in J]
    out.append('  W%-4d W%-4d  rates %s  rep[MD,Ha,other,raw,strong]=%s  [%s | %s]' %
               (x, y, ' '.join('%.2f' % v if v is not None else ' -- ' for v in r), reps, label(x), label(y)))
out.append('  replicated at >=1 held-out site split: %d; at all sequence levels: %d' % (len(site_rep), len(lvl_rep)))
# consensus components
adj = collections.defaultdict(set)
for x, y in full: adj[x].add(y); adj[y].add(x)
seen = set(); comps = []
for v in adj:
    if v in seen: continue
    st = [v]; c = set()
    while st:
        u = st.pop()
        if u in c: continue
        c.add(u); st.extend(adj[u] - c)
    seen |= c; comps.append(sorted(c))
out.append('')
out.append('CONSENSUS CLASSES:')
for c in sorted(comps, key=len, reverse=True):
    out.append('  {%s}  known: %s' % (', '.join('W%d' % x for x in c), '; '.join('W%d=%s' % (x, label(x)) for x in c)))
# held-out beat summary across runs
out.append('')
out.append('HELD-OUT: ontologies beating the max of both controls, per criterion, summed over full-corpus runs (cycle1, cycle2, cycle4_*):')
agg = collections.defaultdict(lambda: [0, 0, [], [], []])
for t in ('cycle1', 'cycle2', 'cycle4_raw', 'cycle4_strong'):
    if t not in J: continue
    for r, p, s in zip(J[t]['real'], J[t]['ctlP'], J[t]['ctlS']):
        k = r['crit'][0] + ('/' + r['crit'][1] if r['crit'][1] else '')
        a = agg[k]; a[1] += 1; a[2].append(r['held']); a[3].append(p['held']); a[4].append(s['held'])
for k, a in sorted(agg.items()):
    mx = max(a[3] + a[4]); beats = sum(1 for v in a[2] if v > mx)
    out.append('  %-16s n=%3d  real mean %.3f max %.3f | P mean %.3f max %.3f | S mean %.3f max %.3f | real > max(all controls): %d'
               % (k, a[1], np.mean(a[2]), np.max(a[2]), np.mean(a[3]), np.max(a[3]), np.mean(a[4]), np.max(a[4]), beats))
txt = '\n'.join(out)
open(os.path.join(D, 'loop10_final_tables.txt'), 'w').write(txt + '\n')
print(txt)

# ---- per-criterion stable graphs pooled over all full-corpus runs, each against its own controls
def pooled(kind, runs=('cycle1', 'cycle2', 'cycle4_raw', 'cycle4_strong')):
    sel = [(r, p, s, J[t]['top']) for t in runs if t in J for r, p, s in zip(J[t]['real'], J[t]['ctlP'], J[t]['ctlS']) if r['crit'][0] == kind]
    if not sel: return None, 0
    acc = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r, p, s, top in sel:
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                k = (min(top[i], top[j]), max(top[i], top[j])); a = acc[k]
                a[0] += r['A'][i] == r['A'][j]; a[1] += p['A'][i] == p['A'][j]; a[2] += s['A'][i] == s['A'][j]; a[3] += 1
    return acc, len(sel)

out2 = ['', 'PER-CRITERION STABLE PAIRS pooled over full-corpus runs (rate >= 0.70, >= 0.20 above both controls):']
percrit = {}
for kind in ('mdl', 'reuse', 'fact', 'quantity'):
    acc, n = pooled(kind)
    if not acc: continue
    st = sorted(((a[0] / a[3], a[1] / a[3], a[2] / a[3], k) for k, a in acc.items() if a[3] >= 8 and a[0] / a[3] >= 0.7 and a[0] / a[3] - (a[2] if kind == 'mdl' else max(a[1], a[2])) / a[3] >= 0.2), reverse=True)  # P control = real for mdl (no facts used)
    percrit[kind] = {k for _, _, _, k in st}
    out2.append('  %-9s n=%2d ontologies, %d stable pairs: %s' % (kind, n, len(st), '  '.join('%d-%d(%.2f;P%.2f,S%.2f)[%s|%s]' % (k[0], k[1], g, gp, gs, label(k[0]), label(k[1])) for g, gp, gs, k in st[:40])))
if len(percrit) > 1:
    for a, b in itertools.combinations(percrit, 2):
        out2.append('  overlap %s/%s: %d shared of %d/%d' % (a, b, len(percrit[a] & percrit[b]), len(percrit[a]), len(percrit[b])))
txt2 = '\n'.join(out2)
open(os.path.join(D, 'loop10_final_tables.txt'), 'a').write(txt2 + '\n')
print(txt2)
