"""v87 cycle 1: calibration (leave-source-out), Voynich posterior, prior-only, shuffled and generator controls,
held-out statistics as posterior predictive checks; freeze the posterior."""
import os, sys, json, random, math
import numpy as np
import v87_lib as L
import v87_common as C

R, F, P = C.load_bank('base')
N = len(R); print('bank', N)
sid = np.array([r['sid'] for r in R]); unit = np.array([r['unit'] for r in R])
scale = L.scale_of(F[:, C.TI])
rng = np.random.default_rng(87)
out = {'bank': N}


def auc(score, y):
    o = np.argsort(score); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---- 1. calibration: planted texts with hidden random schemes, their source held out of the bank
test = rng.choice(N, 600, replace=False)
cal = []
for i in test:
    po = L.abc(F[:, C.TI], P, F[i, C.TI], scale, q=0.02, mask=(sid != sid[i]))
    cal.append(po)
truth = P[test]
pl = np.array([c['is_list'] for c in cal])
res = {'auc_list': auc(pl, truth[:, 0].astype(int))}
for j, p in enumerate(L.PROPS[1:], 1):
    est = np.array([c[p] for c in cal])
    res['r_' + p] = float(np.corrcoef(est, truth[:, j])[0, 1])
    lo = np.array([c[p + '_lo'] for c in cal]); hi = np.array([c[p + '_hi'] for c in cal])
    res['cov80_' + p] = float(((truth[:, j] >= lo) & (truth[:, j] <= hi)).mean())
    res['mae_' + p] = float(np.abs(est - truth[:, j]).mean()); res['prior_mae_' + p] = float(np.abs(P[:, j].mean() - truth[:, j]).mean())
# reliability of P(list)
bins = [0, .2, .4, .6, .8, 1.01]
res['reliab'] = [[b, float(truth[(pl >= b) & (pl < bins[k + 1]), 0].mean()) if ((pl >= b) & (pl < bins[k + 1])).any() else None,
                  int(((pl >= b) & (pl < bins[k + 1])).sum())] for k, b in enumerate(bins[:-1])]
# per encoder family
res['by_unit'] = {}
for u in ['letter', 'chunk', 'word', 'nomen']:
    m = unit[test] == u
    if m.sum() > 20:
        res['by_unit'][u] = dict(n=int(m.sum()), auc=auc(pl[m], truth[m, 0].astype(int)),
                                 r_ptt=float(np.corrcoef([c['ptt'] for c, mm in zip(cal, m) if mm], truth[m, 1])[0, 1]))
out['calibration'] = res; print('calib', json.dumps(res))

# ---- 2. prior
out['prior'] = dict(is_list=float(P[:, 0].mean()), ptt=float(P[:, 1].mean()), pwl=float(P[:, 2].mean()),
                    ptt_sd=float(P[:, 1].std()), pwl_sd=float(P[:, 2].std()),
                    types_per_1000=float(1000 * math.exp(P[:, 1].mean())))
print('prior', out['prior'])

# ---- 3. Voynich posteriors and controls
V = {}
for name in ['ZL3b', 'IT2a']:
    vl = L.voy_lines(name); ch = L.voy_chunks(vl)['all']
    vecs = [C.fvec(c) for c in ch]
    posts = C.run_abc(F, P, vecs, scale)
    V[name] = dict(n_chunks=len(ch), post=C.summarize(posts), ppc=C.ppc(F, posts, vecs),
                   per_chunk_list=[p['is_list'] for p in posts], per_chunk_ptt=[p['ptt'] for p in posts])
    accepted = np.concatenate([p['idx'] for p in posts])
    from collections import Counter
    V[name]['accepted_sources'] = Counter(sid[accepted].tolist()).most_common(8)
    V[name]['accepted_units'] = Counter(unit[accepted].tolist()).most_common()
    for cname, fn in [('glyph_shuffle', C.glyph_shuffle), ('word_shuffle', C.word_shuffle), ('trigram_resynth', C.trigram_resynth)]:
        cv = [C.fvec(fn(c, seed=k)) for k, c in enumerate(ch)]
        cp = C.run_abc(F, P, cv, scale)
        V[name][cname] = dict(post=C.summarize(cp),
                              d_list=[a['is_list'] - b['is_list'] for a, b in zip(cp, posts)],
                              d_ptt=[a['ptt'] - b['ptt'] for a, b in zip(cp, posts)])
    print(name, json.dumps({k: v for k, v in V[name].items() if k in ('post', 'accepted_sources', 'accepted_units')}))
out['voynich'] = V

# ---- 4. does the generator control have power? resynthesise 80 planted texts and measure the posterior shift
sh = []
for i in test[:80]:
    _, _, _, lines = C.regenerate(int(R[i]['seed']))
    a = L.abc(F[:, C.TI], P, C.fvec(lines)[C.TI], scale, mask=(sid != sid[i]))
    b = L.abc(F[:, C.TI], P, C.fvec(C.trigram_resynth(lines))[C.TI], scale, mask=(sid != sid[i]))
    sh.append([a['is_list'], b['is_list'], a['ptt'], b['ptt'], P[i, 0], P[i, 1]])
sh = np.array(sh)
out['resynth_plants'] = dict(mean_abs_dlist=float(np.abs(sh[:, 1] - sh[:, 0]).mean()), mean_abs_dptt=float(np.abs(sh[:, 3] - sh[:, 2]).mean()),
                             auc_real=auc(sh[:, 0], sh[:, 4].astype(int)), auc_resynth=auc(sh[:, 1], sh[:, 4].astype(int)),
                             r_ptt_real=float(np.corrcoef(sh[:, 2], sh[:, 5])[0, 1]), r_ptt_resynth=float(np.corrcoef(sh[:, 3], sh[:, 5])[0, 1]))
print('resynth plants', out['resynth_plants'])

json.dump(out, open(os.path.join(L.CK, 'c1.json'), 'w'), indent=1, default=float)
# ---- freeze
frozen = dict(loop='v87', bank=N, train_stats=L.TRAIN, held_stats=L.HELD, prior=out['prior'],
              ZL3b=V['ZL3b']['post'], IT2a=V['IT2a']['post'])
h = L.sha(frozen)
D = os.path.join(L.VD, 'data')
json.dump(dict(frozen=frozen, sha256=h), open(os.path.join(D, 'v87_posterior.json'), 'w'), indent=1, default=float)
open(os.path.join(D, 'v87_posterior.sha256'), 'w').write(h + '  v87_posterior.json[frozen]\n')
print('sha256', h)
