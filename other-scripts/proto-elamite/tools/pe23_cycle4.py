"""pe23 cycle 4: is the M157 / M136 'rounder counts' result real?
(a) Held-out: 20 random tablet halves; tablet-level stratified header permutation on each half.
(b) Calibration: a FAKE header given to random tablets (same count as M157) -> false positive rate;
    planted: 30% / 15% of entries >= 10 on the fake-header tablets rounded -> power.
(c) Confounds: Susa only; drop sign-less lines and edges; drop the 3 largest tablets per header.
(d) Which values carry it: top values on M157 tablets vs other tablets, entries >= 10, same stratum mix.
(e) Totals with a relaxed filter (damage allowed off the numerals).
"""
import sys, json, math
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pe23_common import *
from pe23_scan import indicators
import pe23_cycle1_lib as L

rng = np.random.default_rng(2324)
R = {}
pe = pe_entries('B')
base_ents = [e for e in pe if e['sys'] in ('CNT', 'CAP', 'AMBCAP')]


def header_test(ents, heads, f='COMP', NP=1000, fake=None):
    ents = indicators([dict(e) for e in ents])
    bytab = defaultdict(list)
    for e in ents:
        bytab[e['tab']].append(e)
    tabs = [t for t, es in bytab.items() if any(e[f] is not None for e in es)]
    lab = np.array([(fake.get(t, 'OTHER') if fake is not None else (bytab[t][0]['hdr'] or 'NONE')) for t in tabs])
    st = defaultdict(list)
    for i, t in enumerate(tabs):
        es = bytab[t]
        st[(Counter(e['sys'] for e in es).most_common(1)[0][0], int(math.log2(max(1, es[0]['nent']))))].append(i)
    S = np.array([sum(e[f] for e in bytab[t] if e[f] is not None) for t in tabs])
    N = np.array([sum(1 for e in bytab[t] if e[f] is not None) for t in tabs])
    out = {}
    for h in heads:
        m = lab == h
        if m.sum() < 4:
            continue
        obs = S[m].sum() / N[m].sum()
        null = []
        for _ in range(NP):
            perm = lab.copy()
            for idx in st.values():
                perm[idx] = rng.permutation(lab[idx])
            mm = perm == h
            null.append(S[mm].sum() / max(1, N[mm].sum()))
        null = np.array(null)
        out[h] = {'tabs': int(m.sum()), 'obs': float(obs), 'null': float(null.mean()),
                  'p_up': float(((null >= obs).sum() + 1) / (NP + 1))}
    return out


# (a) held-out halves
R['a'] = []
tabs_all = sorted({e['tab'] for e in base_ents})
for rep in range(20):
    rng.shuffle(tabs_all)
    A = set(tabs_all[::2])
    r = header_test([e for e in base_ents if e['tab'] in A], ['M157', 'M136'], NP=500)
    R['a'].append(r)
for h in ('M157', 'M136'):
    ps = [r[h]['p_up'] for r in R['a'] if h in r]
    print(f'(a) {h}: halves with p_up < 0.05: {sum(p < 0.05 for p in ps)}/{len(ps)}; median p {np.median(ps):.3f}')
    R[f'a_{h}'] = {'n_sig': int(sum(p < 0.05 for p in ps)), 'n': len(ps), 'median_p': float(np.median(ps))}

# (b) calibration + planted
tabs_cnt = sorted({e['tab'] for e in base_ents})
fp, pw = [], defaultdict(list)
for rep in range(20):
    fake_tabs = set(rng.choice(tabs_cnt, 194, replace=False).tolist())
    fake = {t: 'FAKE' for t in fake_tabs}
    r = header_test(base_ents, ['FAKE'], NP=300, fake=fake)
    fp.append(r['FAKE']['p_up'])
    for p in (0.3, 0.15):
        pl = []
        for e in base_ents:
            e = dict(e)
            if e['tab'] in fake_tabs and e['val'] >= 10 and rng.random() < p:
                den = pe_den(e['sys'])
                hi = max(d for d in den if d <= e['val'])
                e['val'] = int(max(1, round(e['val'] / hi)) * hi)
            pl.append(e)
        pw[p].append(header_test(pl, ['FAKE'], NP=300, fake=fake)['FAKE']['p_up'])
R['b'] = {'fake_p': fp, 'fp_rate': float(np.mean(np.array(fp) < 0.05)),
          'planted': {str(p): float(np.mean(np.array(v) < 0.05)) for p, v in pw.items()}}
print('(b) fake header false-positive rate', R['b']['fp_rate'], '; planted power', R['b']['planted'])

# (c) confounds
R['c'] = {}
variants = {
    'susa_only': [e for e in base_ents if e['site'].startswith('Susa')],
    'no_signless_no_edge': [e for e in base_ents if e['nsign'] > 0 and e['surface'] in ('obverse', 'reverse')],
    'obverse_only': [e for e in base_ents if e['surface'] == 'obverse' and e['nsign'] > 0],
    'CNT_only': [e for e in base_ents if e['sys'] == 'CNT'],
    'ROUND_feature': None,
}
for k, ents in variants.items():
    if ents is None:
        r = header_test(base_ents, ['M157', 'M136'], f='ROUND', NP=1000)
    else:
        r = header_test(ents, ['M157', 'M136'], NP=1000)
    R['c'][k] = r
    print('(c)', k, {h: (v['tabs'], round(v['obs'], 3), round(v['null'], 3), round(v['p_up'], 4)) for h, v in r.items()})
# drop the largest M157 tablets
big = Counter(e['tab'] for e in base_ents if e['hdr'] == 'M157')
drop = {t for t, _ in big.most_common(10)}
r = header_test([e for e in base_ents if e['tab'] not in drop], ['M157'], NP=1000)
R['c']['drop10largest'] = r
print('(c) drop 10 largest M157 tablets', r)

# (d) which values
m157 = [e for e in base_ents if e['hdr'] == 'M157' and e['sys'] == 'CNT' and e['val'] >= 10]
oth = [e for e in base_ents if e['hdr'] != 'M157' and e['sys'] == 'CNT' and e['val'] >= 10]
cm, co = Counter(e['val'] for e in m157), Counter(e['val'] for e in oth)
R['d'] = {'n157': len(m157), 'noth': len(oth),
          'top157': [(v, n, round(n / len(m157), 3), round(co[v] / len(oth), 3)) for v, n in cm.most_common(15)]}
print('(d) M157 CNT >= 10 values (value, n, share, share elsewhere):', R['d']['top157'])
fs157 = Counter(e['final'] for e in m157 if (canon_arrays([e['val']], pe_den('CNT'))[0][0] == 1))
R['d']['round_finals_157'] = fs157.most_common(12)
print('(d) final signs of ROUND values on M157 tablets:', fs157.most_common(12))
json.dump(R, open(os.path.join(CK, 'c4.json'), 'w'), indent=1, default=float)
