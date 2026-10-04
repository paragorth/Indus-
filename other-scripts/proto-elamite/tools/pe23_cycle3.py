"""pe23 cycle 3.
(a) Scan again with repeats conditioned on tablet size; 5 value-shuffled calibration corpora.
(b) Headers: tablet-level permutation (header labels permuted among tablets of the same
    size bin and dominant system), statistic = mean COMP / ROUND of the header's entries.
(c) Held-out: 5 random tablet halves; flags (q < 0.1, not REP) found on half A re-tested on B.
(d) TOTALS FORENSICS: on tablets with one reverse total and clean obverse entries of one system,
    is the written total rounder than the sum of its entries (a rounded / estimated total)?
    Null: total' = sum + d, d drawn from the observed |total - sum| with random sign (2,000x).
    Planted: totals replaced by the sum rounded to its leading denomination on 30% of tablets.
(e) Typing: effect vectors (ROUND, FIVE, LEAD1, LOW/levels) of PE groups vs Ur III types;
    control = 200 random frequency-matched final-sign groups per system.
"""
import sys, json, math
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pe23_common import *
from pe23_scan import scan, indicators
import pe23_cycle1_lib as L

rng = np.random.default_rng(2323)
R = {}
pe = pe_entries('B')
real = [e for e in pe if e['sys'] in ('CNT', 'CAP', 'AMBCAP')]
KEYS = ('final', 'first', 'anysign', 'hdr', 'office', 'site', 'surface', 'size', 'nsign', 'sys')
FE = ('ROUND', 'LOW', 'FIVE', 'LEAD1', 'REP', 'COMP')

# (a)
res = scan(real, keys=KEYS, feats=FE, strata='mag')
sig = sorted([r for r in res if r['q'] < 0.05], key=lambda r: -abs(r['z']))
cal = [sum(r['q'] < 0.05 for r in scan(L.shuffle_values(real, rng), keys=KEYS, feats=FE, strata='mag'))
       for _ in range(5)]
R['a'] = {'ntests': len(res), 'sig': sig, 'calib': cal}
print(f'(a) REAL {len(sig)} of {len(res)} q<0.05; calibration flags {cal}')
for r in sig:
    print('  ', r['key'], r['group'], r['sys'], r['feat'], 'n', r['n'], 'tabs', r['tabs'],
          f"obs {r['obs']:.3f} exp {r['exp']:.3f} z {r['z']:+.1f} q {r['q']:.3g}")

# (b) headers, tablet-level permutation
ents = indicators([dict(e) for e in real])
bytab = defaultdict(list)
for e in ents:
    bytab[e['tab']].append(e)
tinfo = {}
for t, es in bytab.items():
    dom = Counter(e['sys'] for e in es).most_common(1)[0][0]
    tinfo[t] = (es[0]['hdr'] or 'NONE', (dom, int(math.log2(max(1, es[0]['nent'])))))
R['b'] = {}
for f in ('COMP', 'ROUND'):
    # per-tablet sums
    tsum = {t: sum(e[f] for e in es if e[f] is not None) for t, es in bytab.items()}
    tcnt = {t: sum(1 for e in es if e[f] is not None) for t, es in bytab.items()}
    tabs = [t for t in bytab if tcnt[t] > 0]
    hdrs = np.array([tinfo[t][0] for t in tabs])
    strata = defaultdict(list)
    for i, t in enumerate(tabs):
        strata[tinfo[t][1]].append(i)
    S = np.array([tsum[t] for t in tabs])
    N = np.array([tcnt[t] for t in tabs])
    cands = [h for h, n in Counter(hdrs.tolist()).items() if n >= 8 and h != 'NONE']
    obs = {h: S[hdrs == h].sum() / N[hdrs == h].sum() for h in cands}
    ge = {h: 0 for h in cands}
    le = {h: 0 for h in cands}
    NP = 2000
    for _ in range(NP):
        perm = hdrs.copy()
        for idx in strata.values():
            perm[idx] = rng.permutation(hdrs[idx])
        for h in cands:
            m = perm == h
            v = S[m].sum() / max(1, N[m].sum())
            ge[h] += v >= obs[h]
            le[h] += v <= obs[h]
    pv = {h: min(1, 2 * min(ge[h] + 1, le[h] + 1) / (NP + 1)) for h in cands}
    q = dict(zip(cands, bh([pv[h] for h in cands])))
    R['b'][f] = {h: {'obs': float(obs[h]), 'tabs': int((hdrs == h).sum()), 'p': pv[h], 'q': float(q[h]),
                     'dir': 'up' if ge[h] < le[h] else 'down'} for h in cands}
    print(f'(b) header {f}:', {h: (R["b"][f][h]['tabs'], round(R["b"][f][h]['obs'], 3), R['b'][f][h]['dir'], round(pv[h], 4), round(q[h], 3)) for h in cands})

# (c) held-out
R['c'] = []
for rep in range(5):
    tabs = sorted({e['tab'] for e in real})
    rng.shuffle(tabs)
    A = set(tabs[::2])
    ra = scan([e for e in real if e['tab'] in A], keys=KEYS, feats=FE, strata='mag', min_n=10, min_tabs=4)
    rb = scan([e for e in real if e['tab'] not in A], keys=KEYS, feats=FE, strata='mag', min_n=10, min_tabs=4)
    bi = {(r['key'], r['group'], r['sys'], r['feat']): r for r in rb}
    fa = [r for r in ra if r['q'] < 0.1 and r['feat'] != 'REP']
    rep_ok = []
    for r in fa:
        k = (r['key'], r['group'], r['sys'], r['feat'])
        b = bi.get(k)
        ok = b is not None and b['p'] < 0.05 and np.sign(b['z']) == np.sign(r['z'])
        rep_ok.append((k, round(r['z'], 1), round(b['z'], 1) if b else None, ok))
    R['c'].append(rep_ok)
    print(f'(c) split {rep}: {len(fa)} flags on A, replicated on B: {sum(x[3] for x in rep_ok)}',
          [x for x in rep_ok if x[3]][:12])

# (d) totals forensics
corp = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
pairs = []
for t in corp:
    obv, rev = [], []
    bad = False
    for l in t['lines']:
        if not l['numerals']:
            continue
        if l['lacuna'] or '...' in l['raw'] or l['damaged']:
            bad = True
            break
        codes = {c.split('@')[0] for _, c in l['numerals']}
        if any(n is None or c.startswith('n') for n, c in l['numerals']) or codes & FRACSET:
            bad = True
            break
        if l['surface'] == 'obverse':
            obv.append(l)
        elif l['surface'] == 'reverse':
            rev.append(l)
    if bad or len(rev) != 1 or len(obv) < 2:
        continue
    allc = [{c.split('@')[0] for _, c in l['numerals']} for l in obv + rev]
    if all(c <= CNTSET for c in allc):
        s, V = 'CNT', CNT_VALS['B']
    elif all(c & CAPSET or c <= {'N01', 'N14'} for c in allc) and any(c & CAPSET for c in allc):
        s, V = 'CAP', CAP_VALS['B']
    else:
        continue
    if not all(c <= set(V) for c in allc):
        continue
    val = lambda l: sum(n * V[c.split('@')[0]] for n, c in l['numerals'])
    pairs.append({'tab': t['id'], 'sys': s, 'sum': sum(val(l) for l in obv), 'tot': val(rev[0])})
print('(d) total/sum pairs', len(pairs), Counter(p['sys'] for p in pairs),
      'equal', sum(p['sum'] == p['tot'] for p in pairs))


def roundness(v, s):
    den = pe_den(s)
    nlev, low, lead = canon_arrays([v], den)
    return int(nlev[0]), int(low[0])


def tot_stat(pp):
    """Among mismatched pairs (tot >= 10): share where total uses FEWER denominations than the sum."""
    m = [p for p in pp if p['tot'] != p['sum'] and p['tot'] >= 10]
    if not m:
        return np.nan, 0
    sc = [roundness(p['tot'], p['sys'])[0] < roundness(p['sum'], p['sys'])[0] for p in m]
    sc2 = [roundness(p['tot'], p['sys'])[0] == 1 for p in m]
    return float(np.mean(sc)), float(np.mean(sc2)), len(m)


def tot_null(pp, rng, n=2000):
    mism = [p for p in pp if p['tot'] != p['sum']]
    ds = [abs(p['tot'] - p['sum']) for p in mism]
    out = []
    for _ in range(n):
        q = []
        for p in pp:
            if p['tot'] == p['sum']:
                q.append(p)
                continue
            d = ds[rng.integers(len(ds))] * (1 if rng.random() < 0.5 else -1)
            q.append(dict(p, tot=max(1, p['sum'] + d)))
        out.append(tot_stat(q))
    return out


obs = tot_stat(pairs)
nul = tot_null(pairs, rng)
fewer = np.array([x[0] for x in nul])
single = np.array([x[1] for x in nul])
R['d'] = {'n_pairs': len(pairs), 'obs_fewer': obs[0], 'obs_single': obs[1], 'n_mismatch': obs[2],
          'null_fewer': [float(fewer.mean()), float(np.percentile(fewer, 95))],
          'null_single': [float(single.mean()), float(np.percentile(single, 95))],
          'p_fewer': float((fewer >= obs[0]).mean()), 'p_single': float((single >= obs[1]).mean())}
print('(d) REAL', R['d'])
# planted rounded totals
pl = []
for rep in range(20):
    pp = []
    for p in pairs:
        p = dict(p)
        if rng.random() < 0.3 and p['sum'] >= 10:
            den = pe_den(p['sys'])
            hi = max(d for d in den if d <= p['sum'])
            r = int(max(1, round(p['sum'] / hi)) * hi)
            if r != p['sum']:
                p['tot'] = r
        pp.append(p)
    o = tot_stat(pp)
    nn = np.array([x[0] for x in tot_null(pp, rng, 300)])
    pl.append(float((nn >= o[0]).mean()))
R['d']['planted_p'] = pl
print('(d) planted rounded totals: p-values', [round(x, 3) for x in pl], 'detected', sum(x < 0.05 for x in pl), '/20')

# (e) typing groups by nearest Ur III type
ur = ur3_entries()
TYPE = {'GRAIN': 'measured', 'RATION': 'allocated', 'ESTIMATE': 'estimated', 'TARGET': 'estimated',
        'LIVESTOCK': 'counted', 'PEOPLE': 'counted'}
F4 = ('ROUND', 'FIVE', 'LEAD1', 'LOWN')


def evec(vals, tabs, den, reps=60):
    f, eff, z = effects(vals, tabs, den, rng, reps=reps)
    v = [eff['ROUND'], eff['FIVE'], eff['LEAD1'], eff['LOW'] / (len(den) - 1)]
    return [0.0 if np.isnan(x) else float(x) for x in v]


cent = defaultdict(list)
for c, ty in TYPE.items():
    xs = [e for e in ur if e['cls'] == c]
    for g in L.groups_for(xs, {e['tab'] for e in xs}, 15, rng, 60):
        cent[ty].append(evec([e['val'] for e in g], [e['tab'] for e in g], ur_den(c)))
allv = np.array([v for vs in cent.values() for v in vs])
mu, sd = allv.mean(0), allv.std(0) + 1e-9
C = {ty: ((np.array(vs) - mu) / sd).mean(0) for ty, vs in cent.items()}


def nearest(v):
    x = (np.array(v) - mu) / sd
    d = {ty: float(np.sqrt(((x - c) ** 2).sum())) for ty, c in C.items()}
    return min(d, key=d.get), d


groups = {}
for s in ('CNT', 'CAP'):
    xs = [e for e in real if e['sys'] == s]
    for o, S in OFFICES.items():
        groups[f'office {o} {s}'] = [e for e in xs if set(e['signs']) & S]
    for h in ('M157', '|M327+M342|', 'M327', 'M136', 'M388', 'M005'):
        groups[f'hdr {h} {s}'] = [e for e in xs if e['hdr'] == h]
    for f_ in ('M376', 'M288', 'M297', 'M263', 'M346', 'M002', 'M036', 'M243', 'M264', 'M003', 'M072'):
        groups[f'final {f_} {s}'] = [e for e in xs if e['final'] == f_]
    groups[f'no-sign {s}'] = [e for e in xs if e['nsign'] == 0]
    groups[f'reverse {s}'] = [e for e in xs if e['surface'] == 'reverse']
    groups[f'top-edge {s}'] = [e for e in xs if e['surface'] == 'top']
    groups[f'ALL {s}'] = xs
# random controls: frequency-matched final-sign groups
ctrl = defaultdict(Counter)
for s in ('CNT', 'CAP'):
    xs = [e for e in real if e['sys'] == s]
    fs = sorted({e['final'] for e in xs if e['final']})
    for _ in range(200):
        pick = set(rng.choice(fs, max(1, len(fs) // 10), replace=False).tolist())
        g = [e for e in xs if e['final'] in pick]
        if sum(e['val'] >= 10 for e in g) < 20:
            continue
        ctrl[s][nearest(evec([e['val'] for e in g], [e['tab'] for e in g], pe_den(s), reps=20))[0]] += 1
R['e'] = {'ctrl': {s: dict(c) for s, c in ctrl.items()}, 'groups': {}}
print('(e) random final-sign groups, nearest Ur III type:', R['e']['ctrl'])
for name, g in groups.items():
    if sum(e['val'] >= 10 for e in g) < 20:
        continue
    s = name.split()[-1]
    v = evec([e['val'] for e in g], [e['tab'] for e in g], pe_den(s))
    ty, d = nearest(v)
    share = ctrl[s][ty] / max(1, sum(ctrl[s].values()))
    R['e']['groups'][name] = {'n': len(g), 'vec': v, 'nearest': ty, 'dist': d, 'ctrl_share_of_type': share}
    print(f'  {name:28s} n {len(g):4d} vec {[round(x, 2) for x in v]} -> {ty} (random groups -> {ty}: {share:.2f})',
          {k: round(x, 2) for k, x in d.items()})
json.dump(R, open(os.path.join(CK, 'c3.json'), 'w'), indent=1, default=float)
