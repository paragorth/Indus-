"""pe80 cycle 2: TAPHONOMY AS A LIBRARY CARD.

Two physical afterlife measures per tablet, never written by the scribe:
  BREAK = share of lines that lost text (lacuna / dropped tokens)          [breakage]
  WEAR  = share of surviving tokens marked damaged '#' (signs + numerals)   [surface abrasion]
Each is residualised on: volume (publication = excavation / editor batch) fixed effects,
log area, thickness, log line count, missing-dimension flags, and (WEAR only) the expected
'#' rate from the sign mix (each sign's leave-tablet-out damaged rate), so hard-to-read
shapes do not count.
Random hypotheses = text features (single or random signed combinations of 2-4), computed
from surviving text as rates, so a broken tablet does not lose them mechanically.
Score = correlation of feature with residual damage; null = residual permuted within
volume x line-count tercile strata; FWER by max-statistic; half A train, half B replicate.
usage: python3 pe80_c2.py RUN SEED   RUN in pe | plant | shuf
"""
import sys, os, json, random, collections, math
import numpy as np
import pe80_common as pc
import common
import pe75_common as p75

NPERM = 300


def tablet_rows():
    T = common.load()
    cat = p75.catalogue()
    # per-sign damaged rate (global; leave-tablet-out applied below)
    sc = collections.Counter(); sd = collections.Counter()
    per_tab = []
    for t in T:
        c = collections.Counter(); d = collections.Counter()
        for l in t['lines']:
            for s, st in zip(l['signs'], l['sign_status']):
                if s == 'x' or st == 'dropped':
                    continue
                b = common.base(s)
                c[b] += 1; d[b] += st == 'damaged'
        per_tab.append((c, d)); sc.update(c); sd.update(d)
    rows = []
    for t, (c, d) in zip(T, per_tab):
        L = t['lines']
        if len(L) < 2:
            continue
        brk = np.mean([bool(l['lacuna'] or l.get('n_dropped_signs') or l.get('n_dropped_nums')) for l in L])
        toks = [st for l in L for s, st in zip(l['signs'], l['sign_status']) if s != 'x' and st != 'dropped']
        nst = [st for l in L for st in l['num_status'] if st != 'dropped']
        allst = toks + nst
        if len(allst) < 4:
            continue
        wear = np.mean([st == 'damaged' for st in allst])
        # expected sign-mix wear, leave-tablet-out
        exp = []
        for b, n in c.items():
            den = sc[b] - n
            r = (sd[b] - d[b]) / den if den > 0 else np.nan
            exp += [r] * n
        exp = [e for e in exp if not np.isnan(e)]
        expw = np.mean(exp) if exp else np.nan
        k = cat.get(t['id'], {})
        # features from surviving text (rates)
        ent = [l for l in L if l['numerals'] and l['signs'] and not l['lacuna'] and not l.get('n_dropped_signs')]
        if len(ent) < 3:
            continue
        lasts = [common.base(l['signs'][-1]) for l in ent if l['signs'][-1] != 'x']
        systems = [common.system_of(l['numerals']) for l in ent]
        hdr = common.header(t) or []
        hb = [common.base(s) for s in hdr if common.is_sign(s)]
        mags = [sum((n or 1) * (10 if cd == 'N14' else 1) for n, cd in l['numerals']) for l in ent]
        lens = [len(l['signs']) for l in ent]
        rows.append(dict(id=t['id'], vol=t.get('volume') or '?', site=t.get('site'), nl=len(L),
                         h=k.get('h'), w=k.get('w'), th=k.get('t'), pres=k.get('pres', ''),
                         brk=brk, wear=wear, expw=expw, lasts=lasts, systems=systems, hdr=hb,
                         mag=float(np.median(mags)) if mags else 0.0, lens=float(np.mean(lens)) if lens else 0.0,
                         rev=sum(l['surface'] == 'reverse' for l in L) / len(L),
                         comma=np.mean([l.get('has_comma', False) for l in ent]),
                         nsys=len(set(systems)), xshare=np.mean([s == 'x' for l in L for s in l['signs']] or [0])))
    return rows


def residualise(rows, y):
    vols = sorted({r['vol'] for r in rows})
    vi = {v: i for i, v in enumerate(vols)}
    X = []
    for r in rows:
        x = [0.0] * len(vols); x[vi[r['vol']]] = 1.0
        a = (r['h'] or 0) * (r['w'] or 0)
        x += [math.log(a) if a > 0 else 0.0, 1.0 if a <= 0 else 0.0,
              r['th'] or 0.0, 1.0 if not r['th'] else 0.0, math.log(r['nl'])]
        X.append(x)
    X = np.array(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def base_features(rows):
    cls = collections.Counter(s for r in rows for s in r['lasts'])
    top = [s for s, c in cls.most_common(40) if c >= 20]
    hcnt = collections.Counter(r['hdr'][0] for r in rows if r['hdr'])
    htop = [s for s, c in hcnt.most_common(12) if c >= 8]
    F, names = [], []
    def add(name, vals):
        v = np.array(vals, float)
        if v.std() > 0 and (v != 0).sum() >= 15:
            F.append(v); names.append(name)
    for s in top:
        add('last=' + s, [np.mean([x == s for x in r['lasts']]) if r['lasts'] else 0 for r in rows])
    for s in htop:
        add('hdr=' + s, [1.0 if r['hdr'] and r['hdr'][0] == s else 0.0 for r in rows])
    for sy in ['C', 'SDB', 'B', 'S-frac', 'C*', 'N23']:
        add('sys=' + sy, [np.mean([x == sy for x in r['systems']]) if r['systems'] else 0 for r in rows])
    for k in ['mag', 'lens', 'comma']:
        vals = [r[k] for r in rows]
        if k == 'mag':
            vals = [math.log1p(v) for v in vals]
        add(k, vals)
    F = np.array(F).T
    return F, names


def zstd(F, strata):
    G = F.copy().astype(float)
    for s in np.unique(strata):
        m = strata == s
        G[m] -= G[m].mean(0)
    sd = G.std(0); sd[sd == 0] = 1
    return G / sd


def make_hyps(nb, rng, nrand=3000):
    H = [((i,), (1.0,)) for i in range(nb)]
    for _ in range(nrand):
        k = int(rng.integers(2, 5))
        idx = tuple(sorted(rng.choice(nb, k, replace=False)))
        sg = tuple(float(x) for x in rng.choice([-1.0, 1.0], k))
        H.append((idx, sg))
    return H


def hyp_matrix(Fz, H):
    M = np.zeros((Fz.shape[0], len(H)))
    for j, (idx, sg) in enumerate(H):
        M[:, j] = (Fz[:, list(idx)] * np.array(sg)).sum(1)
    sd = M.std(0); sd[sd == 0] = 1
    return (M - M.mean(0)) / sd


def perm_strata(y, strata, R, rng):
    out = np.empty((len(y), R))
    for r in range(R):
        yy = y.copy()
        for s in np.unique(strata):
            m = np.flatnonzero(strata == s)
            yy[m] = y[rng.permutation(m)]
        out[:, r] = yy
    return out


def run(rows, target, rng, plant=None):
    y = np.array([r[target] for r in rows], float)
    if target == 'wear':
        e = np.array([r['expw'] for r in rows]); e[np.isnan(e)] = np.nanmean(e)
        y = y - e
    yres = residualise(rows, y)
    F, names = base_features(rows)
    if plant is not None:
        j = names.index(plant)
        f = F[:, j]
        yres = yres + float(os.environ.get('PE80_PLANT', 0.4)) * yres.std() * (f - f.mean()) / f.std()
    nl = np.array([r['nl'] for r in rows])
    terc = np.digitize(nl, np.quantile(nl, [1 / 3, 2 / 3]))
    vol = np.array([hash(r['vol']) % 100000 for r in rows])
    strata = vol * 10 + terc
    return yres, F, names, strata


def main():
    runname = sys.argv[1]; seed = int(sys.argv[2])
    rng = np.random.default_rng(seed + 77)
    rows = tablet_rows()
    rng0 = random.Random(seed)
    idx = list(range(len(rows))); rng0.shuffle(idx)
    half = [np.array(sorted(idx[: len(idx) // 2])), np.array(sorted(idx[len(idx) // 2:]))]
    out = dict(run=runname, seed=seed, n=len(rows))
    for target in ['brk', 'wear']:
        plant = 'hdr=M157' if runname == 'plant' else None
        yres, F, names, strata = run(rows, target, rng, plant)
        if runname == 'shuf':
            yres = perm_strata(yres, strata, 1, rng)[:, 0]
        H = make_hyps(F.shape[1], rng, int(os.environ.get('PE80_NRAND', 3000)))
        res = {}
        for hi, part in enumerate(half):
            Fz = zstd(F[part], strata[part])
            M = hyp_matrix(Fz, H)
            y = yres[part]; y = (y - y.mean()) / y.std()
            stat = M.T @ y / len(y)
            Y = perm_strata(y, strata[part], NPERM, rng)
            Y = (Y - Y.mean(0)) / Y.std(0)
            NS = M.T @ Y / len(y)        # (nH, R)
            maxnull = np.abs(NS).max(0)
            pfwer = (1 + (maxnull[None, :] >= np.abs(stat)[:, None]).sum(1)) / (NPERM + 1)
            pone = (1 + (np.sign(stat)[:, None] * NS >= np.abs(stat)[:, None]).sum(1)) / (NPERM + 1)
            res[hi] = (stat, pfwer, pone)
        sA, fA, _ = res[0]; sB, _, oB = res[1]
        surv = np.flatnonzero(fA < 0.05)
        repl = [j for j in surv if np.sign(sB[j]) == np.sign(sA[j]) and oB[j] < 0.01]
        def hname(j):
            idx, sg = H[j]
            return ' '.join(('+' if s > 0 else '-') + names[i] for i, s in zip(idx, sg))
        singles = [(names[j], round(float(sA[j]), 3), round(float(fA[j]), 3), round(float(sB[j]), 3), round(float(oB[j]), 3)) for j in range(len(names))]
        singles.sort(key=lambda x: x[2])
        out[target] = dict(n_hyp=len(H), n_survA=int(len(surv)), n_repl=len(repl),
                           repl=[(hname(j), round(float(sA[j]), 3), round(float(sB[j]), 3)) for j in repl[:40]],
                           singles_top=singles[:15],
                           plant_found=(plant in [names[j] for j in repl if len(H[j][0]) == 1]) if plant else None)
        print(runname, seed, target, 'hyps', len(H), 'survA', len(surv), 'repl', len(repl), 'plant', out[target]['plant_found'])
        for x in singles[:8]:
            print('   ', x)
    json.dump(out, open(os.path.join(pc.CK, 'c2_%s_%d.json' % (runname, seed)), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
