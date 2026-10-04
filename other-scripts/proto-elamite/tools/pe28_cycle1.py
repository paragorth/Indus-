"""pe28 cycle 1: crossword propagation from the pe27 anchors over the whole corpus.
Controls: (a) Ur III ration / fodder tablets, one anchor (gurusz -> sze-bi = 60 sila), recovered rates
checked against the '-ta' rates written on the same tablets (never used by the solver); (b) a planted
rate network (6 counted signs -> target PLB, one of them given as anchor) on PE tablets; (c) null: target
values re-dealt among targets with the same (system, final sign) across tablets, NP times, same solver,
same anchors.  Statistics: number of new fixed keys; per key, tablets voting for its value, compared with
the null distribution of the MAX votes of any new key in the same system (search-corrected)."""
import os, sys, json, time
import numpy as np
from collections import Counter, defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe28_common import *  # noqa

NP = int(os.environ.get('NP', 100))
t0 = time.time()
res = {}


def summarise(info, anchors):
    return [{'key': keystr(k), 'v': str(v['v']), 'n': v['n'], 'n2': v['n2'], 'round': v['round'],
             'prop': v['prop'], 'tabs': v['tabs'][:10]} for k, v in sorted(info.items(), key=lambda a: -a[1]['n'])]


def null_run(E, anchors, nperm, seed, **kw):
    rng = np.random.default_rng(seed)
    nk, mx = [], defaultdict(list)
    for _ in range(nperm):
        Ep = redeal(E, rng)
        _, inf = propagate(Ep, anchors, **kw)
        nk.append(len(inf))
        m = defaultdict(int)
        for k, v in inf.items():
            m[k[2]] = max(m[k[2]], v['n'])
        for s in ('CAP', 'CNT', 'UR'):
            mx[s].append(m.get(s, 0))
    return np.array(nk), {s: np.array(v) for s, v in mx.items()}


# ------------------------------------------------------------ (a) Ur III
US = ur_seqs()
rng = np.random.default_rng(0)
idx = rng.choice(len(US), 800, replace=False)
USs = [US[i] for i in idx]
EU = build(USs, ur=True)
anc_u = {('gurusz', 'sze-bi', 'UR'): Fr(60)}
fx, inf = propagate(EU, anc_u, gsys='UR')
truth = []
for k, v in inf.items():
    ok = sum(1 for t in v['tabs'] if v['v'] in [Fr(r) for r in USs[t][2]])
    wr = sum(1 for t in v['tabs'] if USs[t][2])
    truth.append({'key': keystr(k), 'v': str(v['v']), 'n': v['n'], 'prop': v['prop'],
                  'written_rate_matches': ok, 'tabs_with_written_rate': wr})
nk, mx = null_run(EU, anc_u, NP, 1, gsys='UR')
res['ur3'] = {'n_tabs': len(USs), 'n_targets': len(EU), 'new_keys': len(inf), 'null_new_keys_mean': float(nk.mean()),
              'null_new_keys_q95': float(np.quantile(nk, .95)),
              'p_new_keys': float((1 + (nk >= len(inf)).sum()) / (1 + NP)),
              'null_maxvotes_q95': float(np.quantile(mx['UR'], .95)), 'keys': truth}
print('UR3', len(EU), len(inf), nk.mean(), np.quantile(mx['UR'], .95), round(time.time() - t0), flush=True)
for r in sorted(truth, key=lambda a: -a['n'])[:15]:
    print('   ', r, flush=True)

# ------------------------------------------------------------ (b) planted network on PE
S = pe_seqs()
adj = Counter()
for tid, L in S:
    for j in range(1, len(L)):
        if L[j][1] == 'CAP' and L[j - 1][1] == 'CNT':
            adj[L[j - 1][3]] += 1
PS = [s for s, _ in adj.most_common(6)]
rp = np.random.default_rng(5)
PR = {s: G_CAP[int(rp.integers(4, len(G_CAP) - 8))] for s in PS}
Sp, npl = [], 0
for tid, L in S:
    L = list(L)
    for j in range(1, len(L)):
        if L[j][1] != 'CAP':
            continue
        win = [L[j - 1]] if L[j - 1][1] == 'CNT' else []
        if j >= 2 and win and L[j - 2][1] == 'CNT' and rp.random() < 0.5:
            win.append(L[j - 2])
        if win and all(w[3] in PS for w in win) and rp.random() < 0.35:
            y = sum(PR[w[3]] * w[2] for w in win)
            L[j] = (L[j][0], 'CAP', y, 'PLB', 'PLB')
            npl += 1
    Sp.append((tid, L))
Ep = build(Sp)
anc_p = dict(anchors_pe())
anc_p[(PS[0], 'PLB', 'CAP')] = PR[PS[0]]
fxp, infp = propagate(Ep, anc_p)
rec = {s: (str(PR[s]), str(fxp.get((s, 'PLB', 'CAP')))) for s in PS}
wrong = [keystr(k) for k, v in infp.items() if k[1] == 'PLB' and PR.get(k[0]) != v['v']]
res['plant'] = {'signs': PS, 'rates': {s: str(r) for s, r in PR.items()}, 'n_planted_lines': npl,
                'recovered': rec, 'n_correct': sum(1 for s in PS[1:] if fxp.get((s, 'PLB', 'CAP')) == PR[s]),
                'wrong_PLB_keys': wrong}
print('PLANT', npl, rec, wrong, round(time.time() - t0), flush=True)

# ------------------------------------------------------------ (c) real PE
E = build(S)
A = anchors_pe()
fxr, infr = propagate(E, A)
nk, mx = null_run(E, A, NP, 2)
rows = summarise(infr, A)
for r in rows:
    s = r['key'].split('[')[1][:-1]
    r['p_corr'] = float((1 + (mx[s] >= r['n']).sum()) / (1 + NP))
res['pe'] = {'n_tabs': len(S), 'n_targets': len(E), 'new_keys': len(infr), 'null_new_keys_mean': float(nk.mean()),
             'null_new_keys_q95': float(np.quantile(nk, .95)), 'p_new_keys': float((1 + (nk >= len(infr)).sum()) / (1 + NP)),
             'null_max_CAP_q95': float(np.quantile(mx['CAP'], .95)), 'null_max_CNT_q95': float(np.quantile(mx['CNT'], .95)),
             'keys': rows}
print('PE', len(E), len(infr), nk.mean(), np.quantile(nk, .95), np.quantile(mx['CAP'], .95), np.quantile(mx['CNT'], .95),
      round(time.time() - t0), flush=True)
for r in rows[:20]:
    print('   ', r, flush=True)
json.dump(res, open(os.path.join(CK, 'cycle1.json'), 'w'), indent=1, default=str)
print('done', round(time.time() - t0))
