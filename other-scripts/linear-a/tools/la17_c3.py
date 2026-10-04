"""LA-17 cycle 3: (a) misspecification control - outbreaks planted by a different generator
(scribal lexicon copying, la17_alt) scored by the ABC-RF trained on the epidemic simulator;
(b) sign types as the pathogen (LA and LB), same controls."""
import sys, json, random, collections
from la17_abc import *
from la17_alt import alt_world

part = sys.argv[1]
res = {}
if part == 'alt':
    docs = load_la(); d = la_data(docs); codes = d['codes']; K = len(codes)
    T, S = load_sims('la', 'si')
    abc = ABC(T, S, K, target='t')
    rng = np.random.default_rng(7)
    out = []
    for rep in range(80):
        src = int(rng.integers(K)); t = rng.uniform(0, 1, K); t[src] = 0
        inc = alt_world(d['E'], d['D'], src, t, rng, inherit=rng.uniform(0.3, 0.9))
        ps, th = abc.rf(summaries(inc[inc.any(1)]))
        out.append((int(ps.argmax() == src), ps[src], spearmanr(th, t)[0],
                    spearmanr(-d['E'], t)[0]))
    o = np.array(out)
    print('alt-generator planted (80): top-1 %.3f (chance %.3f), P(true src) %.3f, order rho %.3f [%.2f..%.2f], effort-baseline rho %.3f' % (
        o[:, 0].mean(), 1 / K, o[:, 1].mean(), o[:, 2].mean(), *np.percentile(o[:, 2], [5, 95]), o[:, 3].mean()))
    res['alt'] = o.tolist()
elif part in ('las', 'lbs'):
    isla = part == 'las'
    docs = load_la() if isla else load_lb()
    sites = LA_SITES if isla else LB_SITES
    d = build_signs(docs, sites); codes = d['codes']; K = len(codes)
    T, S = load_sims(part, 'si')
    abc = ABC(T, S, K, target='t')
    acc, cov, rho = abc.heldout()
    ps, th = abc.rf(summaries(d['inc'])); pr, rk = abc.reject(summaries(d['inc']))
    print(part, len(S), 'heldout acc/cov/rho', acc, cov, rho)
    print(' RF source', fmt_p(ps, codes, 5), '| rej', fmt_p(pr, codes, 5))
    print(' RF order', order_str(th, codes))
    lo, hi = np.percentile(rk, [2.5, 97.5], 0)
    print(' rej ranks', ', '.join(f'{c} {m:.1f} [{a:.0f}-{b:.0f}]' for c, m, a, b in sorted(zip(codes, rk.mean(0), lo, hi), key=lambda x: x[1])))
    dates = la_dates()[0] if isla else LB_DATES
    r, p = date_match(th, codes, dates); r2, p2 = date_match(rk.mean(0), codes, dates)
    rE, pE = date_match(-d['E'], codes, dates)
    print(' date rho RF %.3f (P %.3f) rej %.3f (P %.3f) effort %.3f (P %.3f)' % (r, p, r2, p2, rE, pE))
    sh = []
    for rep in range(30):
        dd = build_signs(doc_shuffle(docs, sites, random.Random(rep)), sites)
        pp, t2 = abc.rf(summaries(dd['inc']))
        sh.append((pp.max(), codes[pp.argmax()], date_match(t2, codes, dates, nperm=1)[0]))
    print(' shuffle: max P %.2f (obs %.2f), tops %s, date rho %.2f +- %.2f, P(sh>=obs) %.2f' % (
        np.mean([x[0] for x in sh]), ps.max(), dict(collections.Counter(x[1] for x in sh)),
        np.mean([x[2] for x in sh]), np.std([x[2] for x in sh]), np.mean([x[2] >= r for x in sh])))
    res = dict(ps=ps.tolist(), th=th.tolist(), heldout=[acc, cov, rho], r=r, p=p, shuffle=sh)
json.dump(res, open(os.path.join(OUT, f'c3_{part}.json'), 'w'), default=float)
