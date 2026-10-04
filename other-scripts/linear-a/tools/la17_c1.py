"""LA-17 cycle 1: SI(R) outbreak with unknown adoption times, all sites observed at the end.
Targets: source site and adoption order. Controls: held-out planted outbreaks, a planted
PH-source outbreak, shuffled (doc-label) incidence, Linear B (KN must come first)."""
import sys, json
from la17_abc import *

mode = sys.argv[1] if len(sys.argv) > 1 else 'si'
tgt = 't' if mode == 'si' else 'tau'
rows, res = [], {}
docs = load_la(); d = la_data(docs); codes = d['codes']; K = len(codes)
dates, ndated = la_dates()
T, S = load_sims('la', mode)
abc = ABC(T, S, K, target=tgt)
acc, cov, rho = abc.heldout()
s_obs = summaries(d['inc'])
ps_rf, th_rf = abc.rf(s_obs)
ps_rj, rk = abc.reject(s_obs)
lo, hi = np.percentile(rk, [2.5, 97.5], 0); mr = rk.mean(0)
r_rf, p_rf = date_match(th_rf, codes, dates)
r_rj, p_rj = date_match(mr, codes, dates)
# baselines that need no epidemic: big sites first; richness first; distance from the RF source
r_E, p_E = date_match(-d['E'], codes, dates)
res['la'] = dict(n=len(S), heldout=[acc, cov, rho], ps_rf=ps_rf.tolist(), th_rf=th_rf.tolist(),
                 ps_rj=ps_rj.tolist(), rank_mean=mr.tolist(), rank_lo=lo.tolist(), rank_hi=hi.tolist(),
                 r_rf=r_rf, p_rf=p_rf, r_rj=r_rj, p_rj=p_rj, r_E=r_E, p_E=p_E)
print('LA', mode, len(S), 'heldout acc/cov90/rho', acc, cov, rho)
print(' RF source', fmt_p(ps_rf, codes), '| rej', fmt_p(ps_rj, codes))
print(' RF order', order_str(th_rf, codes))
print(' rej ranks', ', '.join(f'{c} {m:.1f} [{a:.0f}-{b:.0f}]' for c, m, a, b in sorted(zip(codes, mr, lo, hi), key=lambda x: x[1])))
print(' date rho RF', r_rf, p_rf, 'rej', r_rj, p_rj, 'effort baseline', r_E, p_E)

# planted outbreak: source PH, adoption by distance from PH, mid parameters
rng = np.random.default_rng(42)
pl = []
for rep in range(20):
    p = draw_prior(K, rng, deposit=(mode == 'dep'))
    ph = codes.index('PH'); p['src'] = ph
    p['t'] = d['D'][ph] / d['D'][ph].max() * 0.9; p['t'][ph] = 0
    if mode == 'dep':
        p['tau'] = 0.1 + d['D'][ph] / d['D'][ph].max() * 0.85
    p['logbeta'] = math.log(3); p['phi'] = 0.5; p['gamma'] = 1.0 if mode == 'dep' else 0.3
    o = simulate(p, d['E'], d['D'], 2000, rng, d['inc'].sum())
    pp, th = abc.rf(summaries(o))
    truth = p['t'] if mode == 'si' else p['tau']
    pl.append((pp[ph], int(pp.argmax() == ph), spearmanr(th, truth)[0]))
pl = np.array(pl)
res['planted_PH'] = pl.mean(0).tolist()
print(' planted PH: mean P(PH) %.2f, top-1 %.2f, order rho %.2f' % tuple(pl.mean(0)))

# shuffled incidence: document site labels permuted
sh = []
for rep in range(30):
    rr = np.random.default_rng(100 + rep)
    import random
    rs = random.Random(100 + rep)
    dd = la_data(doc_shuffle(docs, LA_SITES, rs))
    pp, th = abc.rf(summaries(dd['inc']))
    r, _ = date_match(th, codes, dates, nperm=1)
    sh.append((pp.max(), -(pp * np.log(pp + 1e-12)).sum(), r, codes[pp.argmax()]))
mx = np.array([x[0] for x in sh]); en = np.array([x[1] for x in sh]); rs_ = np.array([x[2] for x in sh])
import collections
top = collections.Counter(x[3] for x in sh)
ent_obs = -(ps_rf * np.log(ps_rf + 1e-12)).sum()
res['shuffle'] = dict(maxp=mx.tolist(), ent=en.tolist(), rho=rs_.tolist(), top=dict(top), ent_obs=ent_obs)
print(' shuffle: max P %.2f (obs %.2f), entropy %.2f (obs %.2f, flat %.2f), date rho %.2f +- %.2f (obs %.2f), P(sh>=obs) %.2f, top %s' % (
    mx.mean(), ps_rf.max(), en.mean(), ent_obs, math.log(K), rs_.mean(), rs_.std(), r_rf, np.mean(rs_ >= r_rf), dict(top)))
json.dump(res, open(os.path.join(OUT, f'c1_{mode}_la.json'), 'w'), indent=1, default=float)
