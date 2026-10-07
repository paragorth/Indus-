"""pe80 cycle 3: HOW BIG IS A THING THAT IS COUNTED IN HUNDREDS? (abundance-size law in ledgers)

Ecology's abundance-size rule (Damuth): small things are counted in large numbers. If a ledger
obeys it, a counted word's typical count ranks its physical size. Calibration on Ur III with a
mass table frozen before looking (data/pe80_mass_table_frozen.json): estimator blind to word
identity; Spearman with log mass; mass-permutation null; kill = counts shuffled across lines
within tablet; power at PE size by thinning.
usage: python3 pe80_c3.py ur3 | pe
"""
import os, re, sys, json, math, random, collections
import numpy as np
from scipy.stats import spearmanr
import pe80_common as pc
import common

UNITS = {'disz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'szar2': 3600}
NUMRE = re.compile(r"(\d+)\(([^)]+)\)")
MEAS = {'sila3', 'gur', 'ban2', 'gin2', 'ma-na', 'sar', 'iku', 'bur3', 'esze3', 'barig', 'u4', 'iti', 'mu', 'kusz3', 'ninda', 'gu2'}


def ur3_lines():
    fn = os.path.join(pc.CK, 'c3_ur3_lines.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    ids = pc.ur3_ids()
    out = []; cur = None
    for raw in open(os.path.join(pc.SCRATCH, 'ur.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            pid = raw[1:8]; cur = pid if pid in ids else None
            continue
        if not cur:
            continue
        m = re.match(r"^\d+'?\.\s+(.*)$", raw.strip())
        if not m:
            continue
        txt = m.group(1)
        if '[' in txt or ' x ' in ' ' + txt + ' ':
            continue
        txt = re.sub(r"[#?!*]", '', txt)
        lead = re.match(r"^((?:\d+\([^)]+\)\s+)+)(\S+)", txt)
        if not lead:
            continue
        nums = NUMRE.findall(lead.group(1))
        if any(u not in UNITS for _, u in nums):
            continue
        v = sum(int(n) * UNITS[u] for n, u in nums)
        w = lead.group(2).strip('_')
        if v <= 0 or w in MEAS or w.startswith('sila3'):
            continue
        out.append((cur, w, v))
    json.dump(out, open(fn, 'w'))
    return out


def pe_lines():
    out = []
    for t in common.load():
        for i, l in enumerate(t['lines']):
            if i == 0 or not l['numerals'] or l.get('lacuna') or not l['signs']:
                continue
            sy = common.system_of(l['numerals'])
            if sy != 'SDB':
                continue
            if any(n is None for n, _ in l['numerals']):
                continue
            val = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600}
            if any(c not in val for _, c in l['numerals']):
                continue
            v = sum(n * val[c] for n, c in l['numerals'])
            w = common.base(l['signs'][-1])
            if w == 'x' or v <= 0:
                continue
            out.append((t['id'], w, v))
    return out


def scales(lines, minn=20, iters=25):
    """E1 = median log count per word; E2 = word fixed effect with tablet fixed effects."""
    c = collections.Counter(w for _, w, _ in lines)
    keep = [(t, w, math.log10(v)) for t, w, v in lines if c[w] >= minn]
    words = sorted({w for _, w, _ in keep})
    E1 = {w: float(np.median([y for _, ww, y in keep if ww == w])) for w in words}
    tabs = sorted({t for t, _, _ in keep})
    ti = {t: i for i, t in enumerate(tabs)}; wi = {w: i for i, w in enumerate(words)}
    T = np.array([ti[t] for t, _, _ in keep]); W = np.array([wi[w] for _, w, _ in keep]); Y = np.array([y for _, _, y in keep])
    a = np.zeros(len(tabs)); s = np.array([E1[w] for w in words])
    nt = np.bincount(T, minlength=len(tabs)); nw = np.bincount(W, minlength=len(words))
    for _ in range(iters):
        a = np.bincount(T, Y - s[W], minlength=len(tabs)) / np.maximum(nt, 1)
        a -= a.mean()
        s = np.bincount(W, Y - a[T], minlength=len(words)) / np.maximum(nw, 1)
    E2 = {w: float(s[wi[w]]) for w in words}
    return E1, E2, dict(c)


def corr_with_mass(E, mass, nperm=2000, rng=None):
    ks = [w for w in E if w in mass]
    if len(ks) < 5:
        return dict(n=len(ks), rho=None, p=None)
    x = np.array([E[w] for w in ks]); y = np.log10([mass[w] for w in ks])
    rho = spearmanr(x, y).correlation
    rng = rng or np.random.default_rng(0)
    null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(nperm)])
    return dict(n=len(ks), rho=float(rho), p=float((1 + (null <= rho).sum()) / (nperm + 1)), words=ks)


def shuffle_within_tablet(lines, rng):
    by = collections.defaultdict(list)
    for i, (t, w, v) in enumerate(lines):
        by[t].append(i)
    out = list(lines)
    for t, idx in by.items():
        vs = [lines[i][2] for i in idx]
        rng.shuffle(vs)
        for i, v in zip(idx, vs):
            out[i] = (lines[i][0], lines[i][1], v)
    return out


def main():
    run = sys.argv[1]
    rng = np.random.default_rng(3)
    mass = json.load(open(os.path.join(pc.DATA, 'pe80_mass_table_frozen.json')))['mass_kg']
    res = {}
    if run == 'ur3':
        L = ur3_lines()
        npe = len(pe_lines())
        E1, E2, c = scales(L)
        res['full'] = dict(n_lines=len(L), E1=corr_with_mass(E1, mass, rng=rng), E2=corr_with_mass(E2, mass, rng=rng),
                           scale_E2={w: round(E2[w], 2) for w in E2 if w in mass}, counts={w: c[w] for w in E2 if w in mass})
        prng = random.Random(1)
        Ls = shuffle_within_tablet(L, prng)
        a, b, _ = scales(Ls)
        res['kill_within_tablet_shuffle'] = dict(E1=corr_with_mass(a, mass, rng=rng), E2=corr_with_mass(b, mass, rng=rng))
        # PE size: thin by tablets to ~npe lines
        by = collections.defaultdict(list)
        for x in L:
            by[x[0]].append(x)
        tabs = list(by)
        thin = []
        for d in range(100):
            prng.shuffle(tabs)
            sub = []
            for t in tabs:
                sub += by[t]
                if len(sub) >= npe:
                    break
            e1, e2, _ = scales(sub, minn=10)
            r1 = corr_with_mass(e1, mass, 500, rng); r2 = corr_with_mass(e2, mass, 500, rng)
            thin.append((r1['n'], r1['rho'], r1['p'], r2['rho'], r2['p']))
        th = np.array([[x if x is not None else np.nan for x in r] for r in thin], float)
        res['pe_size'] = dict(n_pe_lines=npe, mean_n_words=float(np.nanmean(th[:, 0])), E1_mean_rho=float(np.nanmean(th[:, 1])),
                              E1_share_p05=float(np.nanmean(th[:, 2] < 0.05)), E2_mean_rho=float(np.nanmean(th[:, 3])),
                              E2_share_p05=float(np.nanmean(th[:, 4] < 0.05)))
    json.dump(res, open(os.path.join(pc.CK, 'c3_%s.json' % run), 'w'), indent=1)
    print(json.dumps(res, indent=1)[:4000])


if __name__ == '__main__':
    main()
