"""pe16 cycle 2: the hunger ruler on Proto-Elamite.
Data: (R) capacity lines inside fixed two-line records (per-person by pe12/pe14 grade B/C);
(S) all capacity entries on capacity tablets, with an unknown number of recipients k per entry
(prior p(k) ~ 1/k on 1..300).  Calibration of the k-model: Ur III per-person amounts multiplied
by the line's head count (known totals) must still give back ~1 l per sila.
Nulls: amounts shuffled across entries; random numbers with the same numeral-code structure;
5 alternative PE value sets.  Unit = litres per N39C (apriori set N39C 1 ... N01 120)."""
import json, sys, time, re
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from pe16_common import *  # noqa

OUT = os.path.join(CK, 'cycle2.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
COARSE = np.linspace(np.log(1e-4), np.log(1e4), 121)
KS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300]
KW = np.log(1.0 / np.array(KS, float)); KW -= np.logaddexp.reduce(KW)
KP = (KS, KW.tolist())


def save():
    json.dump(res, open(OUT, 'w'), indent=1)


def run(E, J, seed, grid=COARSE, M=None, **kw):
    la, gi, gt, T = prep(E)
    r = np.random.default_rng(seed)
    if M is None:
        M, sig = draw_shape(J, r)
    else:
        M, sig = M
    LL = loglik_t(la, gi, gt, T, M, sig, r, lu_grid=grid, **kw)
    lm = np.logaddexp.reduce(LL, axis=0) - np.log(LL.shape[0])
    ev = float(np.logaddexp.reduce(lm) - np.log(len(grid)))
    p = np.exp(lm - lm.max()); p /= p.sum()
    c = np.cumsum(p)
    q = lambda a: float(np.exp(grid[min(np.searchsorted(c, a), len(grid) - 1)]))
    # mass in windows around each period-mode is not separable here; report quantiles + modes
    loc = [i for i in range(1, len(p) - 1) if p[i] >= p[i - 1] and p[i] >= p[i + 1] and p[i] > 0.02]
    modes = sorted([(float(np.exp(grid[i])), float(p[max(0, i - 3):i + 4].sum())) for i in loc],
                   key=lambda x: -x[1])[:4]
    L = to_litres(p, grid, np.random.default_rng(seed + 1))
    return {'map': float(np.exp(grid[np.argmax(p)])), 'q05': q(.05), 'q50': q(.5), 'q95': q(.95),
            'L05': float(np.percentile(L, 5)), 'L50': float(np.percentile(L, 50)),
            'L95': float(np.percentile(L, 95)), 'ev': ev, 'n': len(E), 'tabs': T, 'modes': modes, 'post': p.tolist()}


def count_of(raw):
    """Head count before the noun in a Ur III '-ta' line; None if the first numeral run is the
    amount itself or uses la2."""
    toks = re.sub(r'[\[\]#?!<>]', '', raw).split()
    tot = 0; seen = False; end = None
    for i, tk in enumerate(toks):
        m = re.match(r"^(\d+)\((disz|u|gesz2|gesz'u)\)$", tk)
        if not m:
            if seen:
                end = i
                break
            continue
        mult = {'disz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600}[m.group(2)]
        tot += int(m.group(1)) * mult; seen = True
    if not seen or end is None:
        return None
    nxt = toks[end]
    if nxt.startswith(('sila3', 'gin2', 'la2')) or nxt == 'gur' or re.match(r'^[\d/]+\((barig|ban2|asz)\)', nxt):
        return None
    return tot if tot > 0 else None


def digit_null(E, rng):
    """Same numeral codes per entry, counts redrawn uniformly from the counts seen for each code."""
    seen = defaultdict(list)
    for e in E:
        for n, c in e['nums']:
            if isinstance(n, int):
                seen[c.split('@')[0]].append(n)
    vs = VSETS['apriori']
    out = []
    for e in E:
        v = 0
        for n, c in e['nums']:
            c0 = c.split('@')[0]
            v += int(rng.choice(seen[c0])) * vs[c0]
        out.append(dict(e, value=v))
    return out


if __name__ != '__main__':
    raise SystemExit
R = [r for r in pe_record_entries() if r['capacity'] and r['value']]
S = [e for e in pe_capacity_entries() if not e['damaged'] or True]
print('records', len(R), len({r['tab'] for r in R}), 'capacity entries', len(S), len({e['tab'] for e in S}), flush=True)

if 'UR3tot' not in res:   # calibration of the k-model
    U = ur3_for_model(ur3_entries())
    E = []
    for u in U:
        k = count_of(u['raw'])
        if k:
            E.append(dict(u, value=u['value'] * k, k=k))
    s1 = run(E, 60, 11, kprior=KP)
    s0 = run(E, 60, 11)
    res['UR3tot'] = {'kmodel': s1, 'no_k': s0, 'n': len(E)}
    save()
    print('UR3tot k-model u_manday', round(s1['q05'], 3), round(s1['q50'], 3), round(s1['q95'], 3), 'L', round(s1['L05'], 2), round(s1['L50'], 2), round(s1['L95'], 2), s1['modes'][:3],
          '| without k', round(s0['q50'], 3), flush=True)

for key, E, kw, J in (('R', R, {}, 200), ('S', S, {'kprior': KP}, 60)):
    if key in res:
        continue
    t = time.time()
    s = run(E, J, 21, **kw)
    nulls = []
    for k in range(6):
        r = np.random.default_rng(400 + k)
        vals = [e['value'] for e in E]; r.shuffle(vals)
        Esh = [dict(e, value=v) for e, v in zip(E, vals)]
        ns = run(Esh, J // 2, 22 + k, **kw); ns['kind'] = 'shuffle'
        Edg = digit_null(E, r)
        nd = run(Edg, J // 2, 32 + k, **kw); nd['kind'] = 'digits'
        for x in (ns, nd):
            x.pop('post')
        nulls += [ns, nd]
    alts = {}
    for vn in VSETS:
        if vn == 'apriori':
            continue
        Ev = [r_ for r_ in (pe_record_entries(vn) if key == 'R' else pe_capacity_entries(vn))
              if r_['value'] and (key == 'S' or r_['capacity'])]
        a = run(Ev, J // 2, 41, **kw); a.pop('post')
        alts[vn] = a
    res[key] = {'real': s, 'nulls': nulls, 'alts': alts, 'sec': time.time() - t}
    save()
    print(key, 'u(man-days/N39C)', round(s['q05'], 3), round(s['q50'], 3), round(s['q95'], 3), 'litres', round(s['L05'], 3), round(s['L50'], 3), round(s['L95'], 3), 'modes', s['modes'],
          'ev', round(s['ev'], 1), flush=True)
    for x in nulls:
        print('   null', x['kind'], round(x['q50'], 3), round(x['L50'], 3), round(x['ev'], 1), flush=True)
    for vn, a in alts.items():
        print('   alt', vn, round(a['q05'], 3), round(a['q50'], 3), round(a['q95'], 3), 'L', round(a['L50'], 3), flush=True)
