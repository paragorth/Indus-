"""pe71 cycle 2: does the partner sign in |M153+X| behave like a NAME or like an OFFICE, and where does it sit?

(a) Slot profile. A family = the occurrences (tablet, value) of one slot; PE: compounds containing sign S, value = the
    partner sign(s); Ur III control: personal names vs titles taken from the seal legends of sealed tablets (words opaque
    in the test; the name/title split is used only to know which is which).
    Metrics at matched size (50 tablet-occurrences): V/n (distinct values per occurrence), top share, and LOCALITY =
    share of same-value tablet pairs that fall in the same find-lot, divided by the same share with values permuted
    within the family (1,000 perms).  PE find-lot: same volume, publication numbers within 10.  Ur III find-lot: same
    site and same reign year.  Ur III drawn at PE size (1,581 tablets of one site).
(b) One party one seal: for the M153 partner values on tablets with a seal id, same value + same seal pairs vs values
    permuted among those tablets within volume.  Ur III control: legend-name words in the text of same-seal tablets.
(c) Line role of M153 compounds (header line, bare line, in an M288 line, line just before an M288 line, numbered entry)
    vs every other compound type, decoys = compound families of matched size.
usage: pe71_c2.py -> data/pe71_ckpt/c2.json
"""
import json, os, re, collections
import numpy as np
from pe71_lib import load, comps, CK
from pe70_common import get

TITLES = {'dumu', 'dub-sar', 'ARAD2-zu', 'ARAD2', 'ensi2', 'lugal', 'szabra', 'nu-banda3', 'kuruszda', 'sanga', 'ugula',
          'muhaldim', 'sukkal', 'nita', 'kal-ga', 'an-ub-da', 'limmu2-ba', 'sza13-dub-ba', 'sza13-dub-ba-ka', 'ummaki',
          'uri5ki-ma', 'szusz3', 'gudu4', 'ugula-e2', 'aga3-us2', 'kisal-luh', 'sipa', 'nar', 'ka-guru7', 'u3', 'lu2'}


def profile(occ, lot, rng, nperm=1000):
    """occ: list of (tablet_index, value); lot(i, j) -> bool."""
    n = len(occ)
    vals = [v for _, v in occ]
    c = collections.Counter(vals)
    tabs = [t for t, _ in occ]
    L = np.array([[lot(a, b) for b in tabs] for a in tabs], bool)
    np.fill_diagonal(L, False)
    va = np.array([hash(v) for v in vals])

    def same_lot_share(v):
        S = (v[:, None] == v[None, :]); np.fill_diagonal(S, False)
        k = S.sum()
        return (S & L).sum() / k if k else np.nan
    real = same_lot_share(va)
    nul = np.array([same_lot_share(rng.permutation(va)) for _ in range(nperm)])
    base = np.nanmean(nul)
    return dict(n=n, V=len(c), v_per_n=len(c) / n, top=c.most_common(1)[0][1] / n, top_val=c.most_common(1)[0][0],
                loc=float(real), loc_null=float(base), loc_ratio=float(real / base) if base > 0 else np.nan,
                p_loc=float(np.mean(nul >= real)) if not np.isnan(real) else np.nan)


def pe_part(rng):
    R = load()
    pub = [r['pub'] for r in R]; vol = [r['vol'] for r in R]

    def lot(a, b):
        return vol[a] == vol[b] and not np.isnan(pub[a]) and not np.isnan(pub[b]) and abs(pub[a] - pub[b]) <= 10
    fam = collections.defaultdict(list)
    for i, r in enumerate(R):
        for t in r['toks']:
            if t.startswith('|'):
                cs = comps(t)
                for s in set(cs):
                    partner = '+'.join(x for x in cs if x != s) or s
                    fam[s].append((i, partner))
    out = {}
    for s, occ in fam.items():
        if len(occ) < 15:
            continue
        occ = sorted(set(occ))
        prof = []
        for k in range(20 if len(occ) > 50 else 1):
            sub = [occ[j] for j in sorted(rng.choice(len(occ), min(50, len(occ)), replace=False))]
            prof.append(profile(sub, lot, rng, 300))
        out[s] = {m: float(np.nanmean([p[m] for p in prof])) for m in ('n', 'V', 'v_per_n', 'top', 'loc_ratio', 'p_loc')}
        out[s]['top_val'] = prof[0]['top_val']
    # (b) same value same seal
    occ = [(i, '+'.join(x for x in comps(t) if x != 'M153')) for i, r in enumerate(R) for t in r['toks']
           if t.startswith('|') and 'M153' in comps(t) and r['seals']]
    unit = [r['unit'] for r in R]
    tabs = np.array([i for i, _ in occ]); va = np.array([v for _, v in occ])

    uc = np.array([hash(unit[i]) for i in tabs])
    M = (uc[:, None] == uc[None, :]) & (tabs[:, None] != tabs[None, :])

    def ss(v):
        return int(((v[:, None] == v[None, :]) & M).sum() // 2)
    real = ss(va)
    vv = np.array([vol[i] for i in tabs])
    nul = []
    for _ in range(2000):
        v2 = va.copy()
        for x in np.unique(vv):
            ii = np.where(vv == x)[0]; v2[ii] = va[rng.permutation(ii)]
        nul.append(ss(v2))
    nul = np.array(nul)
    b = dict(n_occ=len(occ), units=len({unit[i] for i in tabs}), same_pairs=real, null=float(nul.mean()), p=float((nul >= real).mean()),
             by_unit=collections.Counter((unit[i], v) for i, v in occ).most_common(10))
    # (c) roles
    ROLE = ['header', 'bare', 'in_M288_line', 'before_M288_line', 'entry']
    roles = collections.defaultdict(collections.Counter)
    for r in R:
        L = r['lines']
        for k, l in enumerate(L):
            nxt = L[k + 1] if k + 1 < len(L) else None
            if 'M288' in l['signs']:
                role = 'in_M288_line'
            elif nxt is not None and 'M288' in nxt['signs']:
                role = 'before_M288_line'
            elif not l['nums']:
                role = 'header' if k == 0 else 'bare'
            else:
                role = 'entry'
            for t in set(l['signs']):
                if t.startswith('|'):
                    roles[t][role] += 1
    m153 = collections.Counter()
    for t, c in roles.items():
        if 'M153' in comps(t):
            m153 += c
    tot = sum(m153.values())
    others = [t for t in roles if 'M153' not in comps(t)]
    dec = []
    for _ in range(2000):
        fam = collections.Counter();
        for t in rng.permutation(others):
            if sum(fam.values()) >= tot:
                break
            fam += roles[t]
        s = sum(fam.values()); dec.append([fam[x] / s for x in ROLE])
    dec = np.array(dec)
    obs = np.array([m153[x] / tot for x in ROLE])
    allc = sum((roles[t] for t in others), collections.Counter()); sa = sum(allc.values())
    c = dict(n=tot, obs=dict(zip(ROLE, obs.round(3))), all_other=dict(zip(ROLE, [round(allc[x] / sa, 3) for x in ROLE])),
             p_high=dict(zip(ROLE, [float((dec[:, j] >= obs[j]).mean()) for j in range(5)])),
             p_low=dict(zip(ROLE, [float((dec[:, j] <= obs[j]).mean()) for j in range(5)])),
             per_type={t: dict(roles[t]) for t in roles if 'M153' in comps(t)})
    return out, b, c


def ur3_part(rng, ndraw=10):
    U = [u for u in get('ur3') if u['vol'] == 'Umma']
    res = {'name': [], 'title': [], 'sameseal_name': []}
    for d in range(ndraw):
        idx = rng.choice(len(U), 1581, replace=False)
        S = [U[i] for i in idx]
        yr = [re.sub(r'\.\d\d\.\d\d.*$', '', s['date']) for s in S]

        def lot(a, b):
            return yr[a] == yr[b] and not yr[a].startswith(('--', '00'))
        legend_names = {t for s in S for t in s['legend'] if t not in TITLES}
        legend_titles = {t for s in S for t in s['legend'] if t in TITLES}
        for nm, vocab in (('name', legend_names), ('title', legend_titles)):
            occ = sorted({(i, t) for i, s in enumerate(S) for t in s['toks'] if t in vocab})
            if len(occ) < 50:
                continue
            sub = [occ[j] for j in sorted(rng.choice(len(occ), 50, replace=False))]
            res[nm].append(profile(sub, lot, rng, 300))
        # same seal, same legend-name word in the text
        occ = [(i, t) for i, s in enumerate(S) if s['seals'] for t in s['toks'] if t in legend_names]
        seal = [s['seals'][0] if s['seals'] else None for s in S]
        tabs = np.array([i for i, _ in occ]); va = np.array([t for _, t in occ])

        sc = np.array([hash(seal[i]) for i in tabs])
        M = (sc[:, None] == sc[None, :]) & (tabs[:, None] != tabs[None, :])
        va = np.array([hash(x) for x in va])

        def ss(v):
            return int(((v[:, None] == v[None, :]) & M).sum() // 2)
        if len(occ) > 1:
            real = ss(va); nul = np.array([ss(rng.permutation(va)) for _ in range(200)])
            res['sameseal_name'].append(dict(n=len(occ), real=real, null=float(nul.mean()), p=float((nul >= real).mean())))
    summ = {}
    for nm in ('name', 'title'):
        summ[nm] = {m: float(np.nanmean([p[m] for p in res[nm]])) for m in ('v_per_n', 'top', 'loc_ratio', 'p_loc')}
        summ[nm]['n_draws'] = len(res[nm])
    summ['sameseal_name'] = dict(draws=len(res['sameseal_name']), pass_p05=sum(x['p'] < .05 for x in res['sameseal_name']),
                                 median_real=float(np.median([x['real'] for x in res['sameseal_name']])) if res['sameseal_name'] else None)
    return summ, res


def main():
    rng = np.random.default_rng(712)
    ur, raw = ur3_part(rng)
    print('UR3 control', json.dumps(ur), flush=True)
    a, b, c = pe_part(rng)
    keys = sorted(a, key=lambda s: -a[s]['v_per_n'])
    print('PE slot profiles (n>=15), sorted by V/n:')
    for s in keys:
        print('  ', s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a[s].items()})
    print('PE same value same seal', b)
    print('PE roles', json.dumps(c))
    json.dump(dict(ur3=ur, pe_profiles=a, same_seal=b, roles=c), open(os.path.join(CK, 'c2.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
