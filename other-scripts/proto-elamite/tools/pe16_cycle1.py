"""pe16 cycle 1: POSITIVE CONTROL on Ur III distributive ration/fodder lines ('n NOUN amount-ta').
Truth: sila ~ 1 litre; human rations monthly (60/30/10-20 sila), hired-man wages and animal
fodder daily.  The hunger ruler must give back u ~ 1 l per sila blind, and the right period per
tablet.  Also: a single-period subset (expected degeneracy), shuffled-amount null, PE-sized
subsamples, and a 1,000-ruler random search (is real biology special?)."""
import json, sys, time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from pe16_common import *  # noqa

OUT = os.path.join(CK, 'cycle1.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
U = ur3_for_model(ur3_entries())
rng = np.random.default_rng(16)
COARSE = np.linspace(np.log(1e-4), np.log(1e4), 61)


def save():
    json.dump(res, open(OUT, 'w'), indent=1)


def run(E, J, seed, grid=None, **kw):
    la, gi, gt, T = prep(E)
    r = np.random.default_rng(seed)
    M, sig = kw.pop('ruler', None) or draw_shape(J, r)
    LL = loglik_t(la, gi, gt, T, M, sig, r, lu_grid=grid, **kw)
    g = LU if grid is None else grid
    J_ = LL.shape[0]
    lm = np.logaddexp.reduce(LL, axis=0) - np.log(J_)
    ev = float(np.logaddexp.reduce(lm) - np.log(len(g)))
    p = np.exp(lm - lm.max()); p /= p.sum()
    c = np.cumsum(p)
    q = lambda a: float(np.exp(g[min(np.searchsorted(c, a), len(g) - 1)]))
    L = to_litres(p, g, np.random.default_rng(seed + 1))
    return {'map': float(np.exp(g[np.argmax(p)])), 'q05': q(.05), 'q50': q(.5), 'q95': q(.95),
            'L05': float(np.percentile(L, 5)), 'L50': float(np.percentile(L, 50)),
            'L95': float(np.percentile(L, 95)),
            'ev': ev, 'n': len(E), 'tabs': T}, p, (la, gi, gt, T, M, sig)


if 'A' not in res:
    t = time.time()
    s, p, (la, gi, gt, T, M, sig) = run(U, 160, 1)
    s['post'] = p.tolist()
    # per-tablet period at the posterior median
    E = sorted(U, key=lambda e: (e['tab'], str(e['label'])))
    tabs = list(dict.fromkeys(e['tab'] for e in E))
    TP = tab_period_post(la, gi, gt, T, M[:40], sig[:40], np.random.default_rng(5), np.log(s['map']))
    best = {tabs[i]: PERIODS[int(np.argmax(TP[i]))] for i in range(T)}
    chk = {}
    for name, cond, want in (('hired man wage (hun-ga2)', lambda e: 'hun-ga2' in e['raw'], 1.0),
                             ('gurusz 60 sila', lambda e: e['cls'] == 'man' and e['value'] == 60 and 'hun-ga2' not in e['raw'], 30.0),
                             ('geme2 30 sila', lambda e: e['cls'] == 'woman' and e['value'] == 30, 30.0),
                             ('dumu 10-20 sila', lambda e: e['cls'] == 'child' and 10 <= e['value'] <= 20, 30.0),
                             ('sheep 0.5-2 sila', lambda e: e['cls'] in ('sheep', 'lamb') and e['value'] <= 2, 1.0),
                             ('cattle 2-12 sila', lambda e: e['cls'] in ('cattle',) and 2 <= e['value'] <= 12, 1.0)):
        tb = {e['tab'] for e in U if cond(e)}
        hit = [best[t_] == want for t_ in tb]
        chk[name] = {'tablets': len(tb), 'right_period': int(sum(hit)), 'want': want}
    s['period_check'] = chk
    s['sec'] = time.time() - t
    res['A'] = s; save()
    print('A', {k: v for k, v in s.items() if k != 'post'}, flush=True)

if 'B' not in res:   # single-period subset: monthly human rations only
    E = [e for e in U if e['cls'] in ('man', 'woman', 'child') and 'hun-ga2' not in e['raw'] and e['value'] >= 10]
    s, p, _ = run(E, 120, 2)
    # mass near 1 l vs near 1/30 l vs near 12 l (factor-3 windows)
    w = lambda c: float(p[(LU > np.log(c / 3)) & (LU < np.log(c * 3))].sum())
    s['mass_near_manday'] = {'0.5 (monthly, true)': w(0.5), '15 (daily)': w(15), '0.042 (yearly)': w(0.5 / 12)}
    s['post'] = p.tolist()
    res['B'] = s; save()
    print('B', {k: v for k, v in s.items() if k != 'post'}, flush=True)

if 'C' not in res:   # shuffled amounts across all entries (labels, tablets kept)
    real, _, _ = run(U, 40, 3, grid=COARSE)
    sh = []
    for k in range(5):
        vals = [e['value'] for e in U]
        r = np.random.default_rng(100 + k); r.shuffle(vals)
        E = [dict(e, value=v) for e, v in zip(U, vals)]
        s, _, _ = run(E, 40, 3, grid=COARSE)
        sh.append(s)
    res['C'] = {'real': real, 'shuffled': sh,
                'dlogZ': [real['ev'] - s['ev'] for s in sh]}
    save(); print('C', res['C']['dlogZ'], [s['q50'] for s in sh], flush=True)

if 'D' not in res:   # PE-sized subsamples (record-sized ~130 entries, capacity-sized ~1,300)
    tabs = sorted({e['tab'] for e in U})
    out = []
    for size in (130, 1300):
        for k in range(12):
            r = np.random.default_rng(200 + k + size)
            order = r.permutation(len(tabs))
            pick, n = set(), 0
            cnt = Counter(e['tab'] for e in U)
            for i in order:
                if n >= size:
                    break
                pick.add(tabs[i]); n += cnt[tabs[i]]
            E = [e for e in U if e['tab'] in pick]
            s, _, _ = run(E, 40, 300 + k, grid=COARSE)
            s['size'] = size
            s['covers1'] = s['L05'] <= 1.0 <= s['L95']
            s['uprime_ok'] = 0.25 <= s['q50'] <= 1.0   # truth 0.5 man-day per sila (60 sila/month)
            out.append(s)
            print('D', size, k, 'u_manday', round(s['q05'], 3), round(s['q50'], 3), round(s['q95'], 3), 'L', round(s['L05'], 2), round(s['L50'], 2), round(s['L95'], 2), flush=True)
    res['D'] = out; save()

if 'E' not in res:   # random-ruler search: is real biology + calendar special?
    tabs = sorted({e['tab'] for e in U})
    r = np.random.default_rng(7)
    half = set(r.choice(tabs, len(tabs) // 2, replace=False))
    A = [e for e in U if e['tab'] in half]
    Bh = [e for e in U if e['tab'] not in half]
    J = 4
    true_A = []
    for k in range(4):
        s, _, _ = run(A, J, 900 + k, grid=COARSE); true_A.append(s['ev'])
    rul = []
    t = time.time()
    for k in range(1000):
        rr = np.random.default_rng(10_000 + k)
        med, per = random_ruler(rr)
        M, sig = draw_ruler(J, rr, med)
        s, _, _ = run(A, J, 10_000 + k, grid=COARSE, ruler=(M, sig), periods=per)
        rul.append({'k': k, 'ev': s['ev'], 'med': med.tolist(), 'per': per, 'u': s['q50']})
        if k % 100 == 0:
            print('E', k, time.time() - t, flush=True)
    evs = np.array([x['ev'] for x in rul])
    top = sorted(rul, key=lambda x: -x['ev'])[:10]
    # held-out re-test of top 10 random rulers and the true ruler
    for x in top:
        rr = np.random.default_rng(50_000 + x['k'])
        M, sig = draw_ruler(J, rr, np.array(x['med']))
        s, _, _ = run(Bh, J, 50_000 + x['k'], grid=COARSE, ruler=(M, sig), periods=tuple(x['per']))
        x['ev_B'] = s['ev']
    true_B = [run(Bh, J, 950 + k, grid=COARSE)[0]['ev'] for k in range(4)]
    res['E'] = {'true_ev_A': true_A, 'true_ev_B': true_B,
                'rank_A': int((evs > np.mean(true_A)).sum()), 'n_rulers': len(rul),
                'rand_ev_q': np.percentile(evs, [50, 95, 99, 100]).tolist(),
                'top10': top}
    save()
    print('E rank', res['E']['rank_A'], 'trueA', np.mean(true_A), 'trueB', np.mean(true_B),
          [(round(x['ev'], 1), round(x['ev_B'], 1)) for x in top], flush=True)
