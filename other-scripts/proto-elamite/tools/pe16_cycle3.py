"""pe16 cycle 3: who eats, the alias lattice, a planted PE-shaped control, a 2,000-ruler random
search on the PE records with held-out re-test, and the vessel check.
(a) PLANTED: the PE record tablets/labels with values replaced by draws from the model at a
    known u' (=3 man-days per N39C) and known periods (mixed 30/360, or all 30), rounded to whole
    N39C.  Does the inference return u' and does the 90% interval cover it?
(b) WHO EATS: per record tablet, period posterior at the R mode(s); implied man-day fraction per
    day for main and companion lines -> class (adult man >= 0.85; woman 0.65-0.85; child/sheep
    0.3-0.65; equid/cattle > 1.6).
(c) RULERS: 2,000 random rulers (random class needs, random 3-period calendar) vs real biology
    on half A of the record tablets; top 20 re-tested on half B.
(d) VESSEL: litres per PE unit on the alias lattice {u'/30, u', 12u'}; which unit falls in the
    measured bevelled-rim-bowl range 0.5-1.2 l (Chogha Mish 0.7 +- 0.2 l; most 0.78-1.21 l)."""
import json, sys, time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from pe16_common import *  # noqa

OUT = os.path.join(CK, 'cycle3.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
R = [r for r in pe_record_entries() if r['capacity'] and r['value']]
VS = VSETS['apriori']


def save():
    json.dump(res, open(OUT, 'w'), indent=1)


def run(E, J, seed, ruler=None, periods=PERIODS):
    la, gi, gt, T = prep(E)
    r = np.random.default_rng(seed)
    M, sig = ruler if ruler is not None else draw_shape(J, r)
    LL = loglik_t(la, gi, gt, T, M, sig, r, periods=periods)
    lm = np.logaddexp.reduce(LL, axis=0) - np.log(LL.shape[0])
    ev = float(np.logaddexp.reduce(lm) - np.log(len(LU)))
    p = np.exp(lm - lm.max()); p /= p.sum()
    c = np.cumsum(p)
    q = lambda a: float(np.exp(LU[min(np.searchsorted(c, a), len(LU) - 1)]))
    return {'map': float(np.exp(LU[np.argmax(p)])), 'q05': q(.05), 'q50': q(.5), 'q95': q(.95),
            'ev': ev}, p, (la, gi, gt, T, M, sig)


# (a) planted control ---------------------------------------------------------------
if 'planted' not in res:
    out = []
    tabs = sorted({e['tab'] for e in R})
    for mode in ('mixed', 'monthly'):
        for k in range(10):
            rng = np.random.default_rng(700 + k + (0 if mode == 'mixed' else 50))
            M1, _ = draw_shape(1, rng)
            sig = rng.uniform(0.15, 0.4)
            Dt = {t: (rng.choice([30.0, 360.0]) if mode == 'mixed' else 30.0) for t in tabs}
            cls = {}
            E = []
            for e in R:
                key = (e['tab'], e['label'])
                if key not in cls:
                    cls[key] = rng.choice([0, 1, 2, 3], p=[.4, .25, .2, .15])
                a = M1[0, cls[key]] * Dt[e['tab']] / 3.0 * np.exp(rng.normal(0, sig))
                E.append(dict(e, value=max(1, int(round(a)))))
            s, p, _ = run(E, 60, 800 + k)
            s.update(mode=mode, cover=bool(s['q05'] <= 3.0 <= s['q95']),
                     alias=float(s['q50'] / 3.0))
            out.append(s)
            print('planted', mode, k, round(s['q05'], 2), round(s['q50'], 2), round(s['q95'], 2), s['cover'], flush=True)
    res['planted'] = out; save()

# (b) who eats --------------------------------------------------------------------
if 'who' not in res:
    c2 = json.load(open(os.path.join(CK, 'cycle2.json')))
    p = np.array(c2['R']['real']['post'])
    modes = [m for m, _ in c2['R']['real']['modes']]
    la, gi, gt, T = prep(R)
    E = sorted(R, key=lambda e: (e['tab'], str(e['label'])))
    tabs = list(dict.fromkeys(e['tab'] for e in E))
    rng = np.random.default_rng(9)
    M, sig = draw_shape(80, rng)
    who = {}
    for m in modes[:3]:
        TP = tab_period_post(la, gi, gt, T, M, sig, rng, np.log(m))
        rows = []
        for i, t in enumerate(tabs):
            D = PERIODS[int(np.argmax(TP[i]))]
            for e in [x for x in E if x['tab'] == t]:
                f = e['value'] * m / D
                cl = ('large animal' if f > 1.6 else 'adult man' if f >= 0.85 else 'woman' if f >= 0.65
                      else 'child or sheep' if f >= 0.3 else 'below any recipient')
                rows.append({'tab': t, 'D': D, 'pD': float(TP[i].max()), 'role': e['role'],
                             'label': e['label'], 'value': e['value'], 'manday_per_day': f, 'class': cl,
                             'raw': e['raw']})
        who[str(m)] = rows
        cnt = Counter((r_['role'], r_['class']) for r_ in rows)
        print('who @u', round(m, 2), dict(cnt), Counter(r_['D'] for r_ in rows), flush=True)
    res['who'] = who; save()

# (c) random rulers ---------------------------------------------------------------
if 'rulers' not in res:
    tabs = sorted({e['tab'] for e in R})
    rng = np.random.default_rng(31)
    # halves balanced by size
    order = sorted(tabs, key=lambda t: -sum(e['tab'] == t for e in R))
    A_t = set(order[0::2]); A = [e for e in R if e['tab'] in A_t]; B = [e for e in R if e['tab'] not in A_t]
    J = 8
    true_A = [run(A, J, 1000 + k)[0]['ev'] for k in range(8)]
    true_B = [run(B, J, 1100 + k)[0]['ev'] for k in range(8)]
    rul = []
    t0 = time.time()
    for k in range(2000):
        rr = np.random.default_rng(20_000 + k)
        med, per = random_ruler(rr)
        M, sig = draw_ruler(J, rr, med)
        s, _, _ = run(A, J, 20_000 + k, ruler=(M, sig), periods=per)
        rul.append({'k': k, 'ev': s['ev'], 'med': med.tolist(), 'per': list(per)})
        if k % 250 == 0:
            print('rulers', k, round(time.time() - t0), flush=True)
    evs = np.array([x['ev'] for x in rul])
    top = sorted(rul, key=lambda x: -x['ev'])[:20]
    for x in top:
        rr = np.random.default_rng(60_000 + x['k'])
        M, sig = draw_ruler(J, rr, np.array(x['med']))
        x['ev_B'] = run(B, J, 60_000 + x['k'], ruler=(M, sig), periods=tuple(x['per']))[0]['ev']
    # held-out: are top-A rulers better than real biology on B?
    res['rulers'] = {'true_A': true_A, 'true_B': true_B, 'nA': len(A), 'nB': len(B),
                     'rank_A': int((evs > np.mean(true_A)).sum()), 'n': len(rul),
                     'q': np.percentile(evs, [50, 90, 99, 100]).tolist(), 'top': top,
                     'top_beat_true_B': int(sum(x['ev_B'] > np.mean(true_B) for x in top))}
    save()
    print('rulers rank', res['rulers']['rank_A'], 'of', len(rul), 'trueA', round(np.mean(true_A), 1),
          'q', [round(v, 1) for v in res['rulers']['q']], 'top beat true on B', res['rulers']['top_beat_true_B'], flush=True)

# (d) vessel check ----------------------------------------------------------------
if 'vessel' not in res:
    c2 = json.load(open(os.path.join(CK, 'cycle2.json')))
    p = np.array(c2['R']['real']['post'])
    rng = np.random.default_rng(5)
    out = {}
    for alias, fac in (('daily (u/30)', 1 / 30), ('as fitted', 1.0), ('yearly (12u)', 12.0)):
        L = to_litres(p, LU, rng) * fac
        row = {}
        for unit in ('N39C', 'N30D', 'N30C', 'N24', 'N39B', 'N01'):
            x = L * VS[unit]
            row[unit] = {'L50': float(np.median(x)), 'L05': float(np.percentile(x, 5)),
                         'L95': float(np.percentile(x, 95)),
                         'P_in_BRB': float(((x >= 0.5) & (x <= 1.2)).mean())}
        out[alias] = row
        print('vessel', alias, {u: (round(v['L50'], 2), round(v['P_in_BRB'], 2)) for u, v in row.items()}, flush=True)
    res['vessel'] = out; save()
