"""v34 cycle 3: (a) power: plant crowding-driven abbreviation into the real Voynich lines
(keeping their measured slack) and ask how often the cycle-1 statistic finds it;
(b) Latin manuscripts and the greedy-wrap signature (calibrates V-34.1.4);
(c) does Latin abbreviation rise toward the right margin inside the line (non-final
words, x-position tercile), i.e. is there ANY room-driven abbreviation in the control?
"""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v34_lib as V

NP = 400
REPS = 20


def plant(lines, p, mode, rng):
    """tight = slack below the page's 30th percentile; with prob p the last word is
    abbreviated: mode 'm' = drop last 1-2 glyphs and write final m; 'trunc' = drop 2."""
    q = {}
    for l in lines:
        q.setdefault(l['page'], []).append(l['slack'])
    out = []
    for l in lines:
        g = V.glyphs(l['words'][-1])
        if l['slack'] <= np.percentile(q[l['page']], 30) and rng.random() < p and len(g) >= 3:
            k = rng.integers(1, 3)
            g = g[:-k] + (['m'] if mode == 'm' else [])
        out.append(g)
    return out


def power(vo):
    t = [-l['slack'] for l in vo]; grp = [l['page'] for l in vo]
    res = {}
    for mode in ['m', 'trunc']:
        for p in [0.0, 0.1, 0.2, 0.35, 0.5]:
            zs = []
            for r in range(REPS):
                rng = np.random.default_rng(100 + r)
                G = plant(vo, p, mode, rng)
                F = {'final_m': [g[-1] == 'm' for g in G], 'nglyph_last': [len(g) for g in G]}
                o = V.multi_perm(t, F, grp, NP, seed=r)
                zs.append((o['final_m']['z'], o['nglyph_last']['z']))
            zs = np.array(zs)
            key = f'{mode}_p{p}'
            res[key] = {'z_final_m': float(zs[:, 0].mean()), 'z_len': float(zs[:, 1].mean()),
                        'detect_m': float(np.mean(zs[:, 0] > 2)), 'detect_len': float(np.mean(zs[:, 1] < -2))}
            print(key, res[key], flush=True)
    return res


def latin_next(la_all):
    nxt = {}
    for l in la_all:
        nxt.setdefault(l['page'], []).append(l)
    rows = []
    for pg, L in nxt.items():
        for a, b in zip(L[:-1], L[1:]):
            if a['ok']:
                rows.append((a, V.base_letters(b['words'][0]) + V.abbr_count(b['words'][0])))
    t = [-a['slack'] for a, _ in rows]
    o = V.multi_perm(t, {'next_first_len': [n for _, n in rows]}, [a['page'] for a, _ in rows], 2000)
    print('latin next-line first token', len(rows), o, flush=True)
    return o


def latin_xpos(li):
    """abbreviation marks per non-final, non-initial word by x-tercile of the line span."""
    out = {0: [], 1: [], 2: []}
    perpage = {}
    for l in li:
        n = len(l['words'])
        if n < 5 or len(l['rawtoks']) != n:
            continue
        span = l['x1'][-1] - l['x0'][0]
        for k in range(1, n - 1):
            x = ((l['x0'][k] + l['x1'][k]) / 2 - l['x0'][0]) / span
            tr = min(2, int(3 * x))
            a = V.abbr_count(l['rawtoks'][k]) / max(1, V.base_letters(l['rawtoks'][k]))
            out[tr].append(a)
            perpage.setdefault(l['page'], {0: [], 1: [], 2: []})[tr].append(a)
    m = {k: float(np.mean(v)) for k, v in out.items()}
    d = [np.mean(v[2]) - np.mean(v[1]) for v in perpage.values() if v[2] and v[1]]
    print('latin abbreviation per letter by x-tercile (non-final words)', m, 'right-minus-middle per page: mean',
          round(float(np.mean(d)), 4), 'pages up', int(np.sum(np.array(d) > 0)), '/', len(d), flush=True)
    return {'tercile_means': m, 'page_diffs': [float(x) for x in d]}


def voy_xpos(vo):
    """Voynich analogue: per-glyph rate of each common glyph in non-final, non-initial words
    by x-tercile; right-minus-middle difference with a sign test over pages."""
    from collections import Counter
    pp = {}
    for l in vo:
        n = len(l['words'])
        if n < 5:
            continue
        span = l['x1'][-1] - l['x0'][0]
        for k in range(1, n - 1):
            x = ((l['x0'][k] + l['x1'][k]) / 2 - l['x0'][0]) / span
            tr = min(2, int(3 * x))
            pp.setdefault(l['page'], {0: Counter(), 1: Counter(), 2: Counter()})[tr].update(V.glyphs(l['words'][k]))
    res = {}
    for gl in ['m', 'g', 'n', 'y', 'l', 'r', 's', 'd', 'k', 't', 'ch', 'sh', 'e', 'o', 'a', 'i', 'q', 'p', 'f']:
        d = []
        for c in pp.values():
            n1, n2 = sum(c[1].values()), sum(c[2].values())
            if n1 and n2:
                d.append(c[2][gl] / n2 - c[1][gl] / n1)
        d = np.array(d)
        res[gl] = {'mean_diff': float(d.mean()), 'up': int((d > 0).sum()), 'n': len(d)}
    from scipy.stats import binomtest
    for gl, v in res.items():
        v['p_sign'] = float(binomtest(v['up'], v['n']).pvalue)
    print('voynich glyph rate right-minus-middle tercile:',
          {k: (round(v['mean_diff'], 4), f"{v['up']}/{v['n']}", round(v['p_sign'], 3)) for k, v in res.items()}, flush=True)
    return res


if __name__ == '__main__':
    out = {}
    vo = [l for l in V.voynich_lines() if l['ok']]
    out['latin_next'] = latin_next(V.latin_lines())
    li = [l for l in V.latin_img_lines() if l['ok']]
    out['latin_xpos'] = latin_xpos(li)
    out['voy_xpos'] = voy_xpos(vo)
    out['power'] = power(vo)
    json.dump(out, open(os.path.join(V.CKPT, 'cycle3.json'), 'w'), indent=1)
