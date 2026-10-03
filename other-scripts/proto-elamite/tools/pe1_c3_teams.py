#!/usr/bin/env python3
"""PE-1.3: teams. Bipartite graph recurring designations (on >=2 tablets) x
tablets. (i) Repeat co-occurrence: designation pairs sharing >=2 tablets,
against a degree-preserving bipartite shuffle (curveball). (ii) Teams =
connected components of the designation co-occurrence graph; tablet team =
team holding most of its designations. MI(team; header opener) and MI(team;
dominant system), MI(team; dominant class sign) against the same shuffle
(teams rebuilt each run). (iii) Tablet pairs linked by a shared designation vs
pairs linked by a shared ONE-sign entry type of matched frequency: same opener,
same dominant system, same publication volume."""
import json, random, math
from pe1_lib import *
T = load(); meta = tablet_meta(T)
E_all = entries(T)
dom_sys, dom_fin = {}, {}
tmp = collections.defaultdict(list); tmpf = collections.defaultdict(list)
for e in E_all:
    if e['system']: tmp[e['tablet']].append('C' if e['system'] in ('C', 'C*') else e['system'])
    if e['signs'][-1] in CLASS: tmpf[e['tablet']].append(e['signs'][-1])
for t, v in tmp.items(): dom_sys[t] = collections.Counter(v).most_common(1)[0][0]
for t, v in tmpf.items(): dom_fin[t] = collections.Counter(v).most_common(1)[0][0]

def mi(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if not n: return 0.0, 0
    ca = collections.Counter(a for a, _ in pairs); cb = collections.Counter(b for _, b in pairs); cab = collections.Counter(pairs)
    return sum(v / n * math.log2(v * n / (ca[a] * cb[b])) for (a, b), v in cab.items()), n

def components(edges, nodes):
    par = {x: x for x in nodes}
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges: par[f(a)] = f(b)
    return {x: f(x) for x in nodes}

def analyse(B):
    """B: dict tablet -> set of designations (recurring ones only)."""
    co = collections.Counter()
    for t, ds in B.items():
        ds = sorted(ds)
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)): co[(ds[i], ds[j])] += 1
    rep2 = sum(1 for v in co.values() if v >= 2)
    nodes = {d for ds in B.values() for d in ds}
    comp = components([k for k, v in co.items() if v >= 1], nodes)
    team = {}
    for t, ds in B.items():
        if ds: team[t] = collections.Counter(comp[d] for d in ds).most_common(1)[0][0]
    tsize = collections.Counter(comp.values())
    multi = {t: c for t, c in team.items() if tsize[c] >= 2}   # tablets in teams of >=2 designations
    s = {'rep_cooc_pairs': rep2, 'cooc_pairs': len(co), 'n_teams_ge2': sum(1 for v in tsize.values() if v >= 2),
         'largest_team': max(tsize.values()), 'tablets_in_teams': len(multi)}
    s['MI_header'], s['n_h'] = mi([(c, meta[t]['header']) for t, c in multi.items()])
    s['MI_sys'], s['n_s'] = mi([(c, dom_sys.get(t)) for t, c in multi.items()])
    s['MI_fin'], s['n_f'] = mi([(c, dom_fin.get(t)) for t, c in multi.items()])
    s['MI_vol'], _ = mi([(c, meta[t]['vol']) for t, c in multi.items()])
    return s, comp

def curveball(B, rng, swaps=None):
    L = {t: set(v) for t, v in B.items() if v}
    ks = list(L)
    for _ in range(swaps or 5 * len(ks)):
        a, b = rng.sample(ks, 2)
        A, Bb = L[a], L[b]
        oa, ob = list(A - Bb), list(Bb - A)
        if not oa and not ob: continue
        pool = oa + ob; rng.shuffle(pool)
        na = set(pool[:len(oa)]); nb = set(pool[len(oa):])
        L[a] = (A & Bb) | na; L[b] = (A & Bb) | nb
    return L

res = {}
for lab, kw in [('len>=2', dict(minlen=2)), ('len>=2 prefix-stripped', dict(minlen=2, strip_prefix=True)), ('len>=3', dict(minlen=3))]:
    DE = designation_entries(T, **kw)
    tabs = collections.defaultdict(set)
    for e in DE: tabs[e['des']].add(e['tablet'])
    rec = {d for d, ts in tabs.items() if len(ts) >= 2}
    B = collections.defaultdict(set)
    for e in DE:
        if e['des'] in rec: B[e['tablet']].add(e['des'])
    obs, comp = analyse(B)
    rng = random.Random(9); nl = [analyse(curveball(B, rng))[0] for _ in range(500)]
    out = {'n_recurring': len(rec), 'n_tablets': len(B), 'obs': obs}
    for k in ['rep_cooc_pairs', 'cooc_pairs', 'largest_team', 'tablets_in_teams', 'MI_header', 'MI_sys', 'MI_fin', 'MI_vol']:
        out[k] = zp(obs[k], [x[k] for x in nl])
    # teams listing
    teams = collections.defaultdict(list)
    for d, c in comp.items(): teams[c].append(' '.join(d))
    out['teams'] = sorted([v for v in teams.values() if len(v) >= 2], key=len, reverse=True)[:15]
    res[lab] = out
    print('==', lab, 'recurring', len(rec), 'tablets', len(B), obs)
    for k, v in out.items():
        if isinstance(v, dict) and 'z' in v: print('  %-18s obs %.3f null %.3f sd %.3f z %6.2f p %.4f' % (k, v['obs'], v['null_mean'], v['null_sd'], v['z'], v['p']))
    print('  teams:', out['teams'][:6])

# (iii) tablet-pair coherence: shared designation vs shared one-sign entry
DE = designation_entries(T, minlen=2)
def linked_pairs(item_tabs):
    P = set()
    for ts in item_tabs.values():
        ts = sorted(ts)
        if len(ts) > 12: continue
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)): P.add((ts[i], ts[j]))
    return P
dt = collections.defaultdict(set)
for e in DE: dt[e['des']].add(e['tablet'])
dt = {d: v for d, v in dt.items() if len(v) >= 2}
st = collections.defaultdict(set)
for e in E_all:
    if len(e['signs']) == 1 and e['signs'][0] != 'x': st[e['signs'][0]].add(e['tablet'])
st = {d: v for d, v in st.items() if 2 <= len(v) <= 12}   # match the <=12 tablet spread used for designations
# also a strict baseline: one-sign entries that are NOT class signs
st_nc = {d: v for d, v in st.items() if d not in CLASS}
def coh(P):
    o = {}
    for nm, f in [('same_header', lambda t: meta[t]['header']), ('same_sys', dom_sys.get), ('same_fin', dom_fin.get), ('same_vol', lambda t: meta[t]['vol'])]:
        q = [(a, b) for a, b in P if f(a) and f(b)]
        o[nm] = round(sum(f(a) == f(b) for a, b in q) / max(1, len(q)), 3); o['n_' + nm] = len(q)
    return o
alltabs = sorted({t for t in meta})
rng = random.Random(1); rp = set()
while len(rp) < 20000:
    a, b = rng.sample(alltabs, 2); rp.add((min(a, b), max(a, b)))
res['pair_coherence'] = {'designation_linked': coh(linked_pairs(dt)), 'one_sign_linked': coh(linked_pairs(st)),
                         'one_sign_nonclass_linked': coh(linked_pairs(st_nc)), 'random_pairs': coh(rp)}
for k, v in res['pair_coherence'].items(): print(k, v)
json.dump(res, open(os.path.join(DATA, 'pe1_c3_teams.json'), 'w'), indent=1)
