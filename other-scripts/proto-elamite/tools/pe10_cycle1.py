"""pe10 cycle 1: is there a squeeze effect at all?

1A  end-of-face shortening: sign count of the last intact obverse entry minus the tablet's
    obverse mean, vs a within-tablet shuffle of entry order (10,000 perms).  Split by
    crowding (obverse lines per cm of height) and by spill (entries continue on reverse).
    Squeeze predicts: last entry shorter, more so on crowded tablets.
1B  within-line trade-off: an entry with many numeral impressions has less room for signs.
    Statistic: within-tablet correlation of (sign count, numeral impressions); null =
    shuffle numerals among entries of the same tablet.  Squeeze-specific test: the
    trade-off is steeper on NARROW tablets (width per mm) -- null permutes width
    among tablets.
1C  between tablets: mean sign count per entry vs mm of height per obverse line,
    null = permute mm/line among tablets of the same entry-count bin.
Planted control: on the real corpus, drop one sign (if >=2) from the last obverse entry on
    crowded tablets with prob 0.5, and from entries with >= 6 numeral impressions on narrow
    tablets with prob 0.5. Must be detected.  Negative control: same pipeline on a corpus
    with entry order shuffled within tablet (no layout) must be null.
"""
import sys, json, random, math
import numpy as np
from pe10_common import *

rng = np.random.default_rng(10)
random.seed(10)


def obv_profile(t):
    """intact obverse entries in order (list of units), or None."""
    obv = [u for u in t['units'] if u['face'] == 'obverse']
    if not obv:
        return None
    # obverse end must be intact: no break marker on obverse after last unit, last unit not broken
    for (face, col, pos, txt) in t['markers']:
        if face == 'obverse' and ('broken' in txt.lower() or 'missing' in txt.lower()):
            return None
    if any(u['prime'] for u in obv):
        return None
    if len({u['col'] for u in obv}) > 1:
        return None
    ent = [u for u in obv if u['entry']]
    if len(ent) < 5:
        return None
    if obv[-1] is not ent[-1]:
        return None  # last obverse line is not an entry (subscript, total, blank)
    return ent


def ns(u):
    return len(u['signs'])


def nimp(u):
    return sum(n for n, _ in u['nums'])


def build(T):
    rows = []
    for t in T:
        ent = obv_profile(t)
        if ent is None:
            continue
        L = np.array([ns(u) for u in ent], float)
        ok = np.array([not u['broken'] for u in ent])
        if not ok[-1] or ok.sum() < 4:
            continue
        crowd = (len([u for u in t['units'] if u['face'] == 'obverse']) / (t['h'] / 10.0)) if t['h'] else None
        rows.append({'id': t['id'], 'L': L, 'ok': ok, 'crowd': crowd, 'spill': t['spill'],
                     'w': t['w'], 'h': t['h'],
                     'imp': np.array([nimp(u) for u in ent], float)})
    return rows


def stat_last(rows, perms=10000, sel=None):
    """mean over tablets of L[last] - mean(L[ok]); null by within-tablet shuffle."""
    R = [r for r in rows if (sel is None or sel(r))]
    if not R:
        return None
    obs = np.mean([r['L'][-1] - r['L'][r['ok']].mean() for r in R])
    # null: the 'last' slot gets a random ok entry
    nul = np.zeros(perms)
    for r in R:
        v = r['L'][r['ok']]
        nul += rng.choice(v, perms) - v.mean()
    nul /= len(R)
    p = (np.sum(nul <= obs) + 1) / (perms + 1)
    return {'n': len(R), 'obs': float(obs), 'null_sd': float(nul.std()), 'z': float(obs / nul.std()),
            'p_low': float(p)}


def stat_pos_profile(rows):
    """residual sign count by relative position bin (5 bins)."""
    acc = defaultdict(list)
    for r in rows:
        n = len(r['L'])
        m = r['L'][r['ok']].mean()
        for i in range(n):
            if not r['ok'][i]:
                continue
            b = 'last' if i == n - 1 else ('first' if i == 0 else 'q%d' % min(3, int(4 * i / n)))
            acc[b].append(r['L'][i] - m)
    return {k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(acc.items())}


def tradeoff(rows, perms=2000):
    """within-tablet corr(sign count, impressions); null shuffles imp within tablet.
    Also slope difference narrow vs wide, null permutes width labels."""
    R = [r for r in rows if r['w']]
    def per_tab(r, imp):
        k = r['ok']
        L, I = r['L'][k], imp[k]
        if L.std() == 0 or I.std() == 0:
            return None
        return float(np.corrcoef(L, I)[0, 1])
    cors = []
    W = []
    keep = []
    for r in R:
        c = per_tab(r, r['imp'])
        if c is None:
            continue
        cors.append(c); W.append(r['w']); keep.append(r)
    cors = np.array(cors); W = np.array(W)
    obs = cors.mean()
    nul = []
    for _ in range(perms):
        cs = []
        for r in keep:
            c = per_tab(r, rng.permutation(r['imp']))
            cs.append(0.0 if c is None else c)
        nul.append(np.mean(cs))
    nul = np.array(nul)
    med = np.median(W)
    narrow = W < med
    dif = cors[narrow].mean() - cors[~narrow].mean()
    dnul = []
    for _ in range(10000):
        s = rng.permutation(narrow)
        dnul.append(cors[s].mean() - cors[~s].mean())
    dnul = np.array(dnul)
    return {'n_tab': int(len(keep)), 'mean_r': float(obs), 'null_mean': float(nul.mean()), 'null_sd': float(nul.std()),
            'z': float((obs - nul.mean()) / nul.std()),
            'narrow_minus_wide': float(dif), 'p_narrow_steeper': float((np.sum(dnul <= dif) + 1) / 10001),
            'r_narrow': float(cors[narrow].mean()), 'r_wide': float(cors[~narrow].mean())}


def between(rows, perms=10000):
    R = [r for r in rows if r['crowd']]
    mml = np.array([10.0 / r['crowd'] for r in R])           # mm per obverse line
    ml = np.array([r['L'][r['ok']].mean() for r in R])
    n = np.array([len(r['L']) for r in R])
    bins = np.digitize(n, [7, 10, 14, 20])
    def cor(x):
        return float(np.corrcoef(x, ml)[0, 1])
    obs = cor(mml)
    nul = []
    for _ in range(perms):
        x = mml.copy()
        for b in np.unique(bins):
            idx = np.where(bins == b)[0]
            x[idx] = x[rng.permutation(idx)]
        nul.append(cor(x))
    nul = np.array(nul)
    return {'n': len(R), 'r_mm_per_line_vs_meanlen': obs, 'null_mean': float(nul.mean()), 'null_sd': float(nul.std()),
            'p_two': float((np.sum(np.abs(nul - nul.mean()) >= abs(obs - nul.mean())) + 1) / (perms + 1))}


def run_all(rows, tag):
    cr = np.array([r['crowd'] for r in rows if r['crowd']])
    q1, q2 = np.quantile(cr, [1 / 3, 2 / 3])
    out = {'tag': tag, 'n_tablets': len(rows)}
    out['last_all'] = stat_last(rows)
    out['last_crowded'] = stat_last(rows, sel=lambda r: r['crowd'] and r['crowd'] >= q2)
    out['last_loose'] = stat_last(rows, sel=lambda r: r['crowd'] and r['crowd'] < q1)
    out['last_spill'] = stat_last(rows, sel=lambda r: r['spill'])
    out['last_nospill'] = stat_last(rows, sel=lambda r: not r['spill'])
    out['profile'] = stat_pos_profile(rows)
    out['tradeoff'] = tradeoff(rows)
    out['between'] = between(rows)
    return out, (q1, q2)


def plant(rows, q2, wmed):
    P = []
    for r in rows:
        r2 = dict(r); L = r['L'].copy()
        if r['crowd'] and r['crowd'] >= q2 and L[-1] >= 2 and rng.random() < 0.5:
            L[-1] -= 1
        if r['w'] and r['w'] < wmed:
            for i in range(len(L)):
                if r['imp'][i] >= 6 and L[i] >= 2 and rng.random() < 0.5:
                    L[i] -= 1
        r2['L'] = L
        P.append(r2)
    return P


def shuffled(rows):
    S = []
    for r in rows:
        r2 = dict(r)
        perm = rng.permutation(len(r['L']))
        r2['L'] = r['L'][perm]; r2['ok'] = r['ok'][perm]; r2['imp'] = r['imp'][perm]
        # keep the 'last is ok' requirement
        if not r2['ok'][-1]:
            j = np.where(r2['ok'])[0][0]
            for k in ('L', 'ok', 'imp'):
                a = r2[k].copy(); a[[j, -1]] = a[[-1, j]]; r2[k] = a
        S.append(r2)
    return S


if __name__ == '__main__':
    T = load()
    rows = build(T)
    real, (q1, q2) = run_all(rows, 'real')
    wmed = float(np.median([r['w'] for r in rows if r['w']]))
    pl, _ = run_all(plant(rows, q2, wmed), 'planted')
    neg, _ = run_all(shuffled(rows), 'shuffled-order')
    res = {'real': real, 'planted': pl, 'negative': neg, 'crowd_terciles_lines_per_cm': [q1, q2], 'w_median_mm': wmed}
    json.dump(res, open(os.path.join(CKPT, 'c1_pe.json'), 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
