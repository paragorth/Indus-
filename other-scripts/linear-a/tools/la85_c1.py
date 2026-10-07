#!/usr/bin/env python3
"""la85 cycle 1, phase A (Linear A only): THE DAUGHTER'S CLOCK.
8,000 random value-free document statistics are scored for site stability inside Linear A (HT, KH, ZA, PH, other),
in units of their own Linear A sampling noise.  Massive guessing: survivors selected on HT vs KH alone are re-tested
on the held-out sites (ZA, PH, other) against non-survivors, against 20 site-label shuffles and against a planted
site-specific distortion.  The 300 most stable statistics (all LA sites) are then FROZEN with their LA values and
noise scales, and with predictions for Linear B (KN, PY, TH) and Ur III that have not been computed yet.
Usage: python3 la85_c1.py la      (LA-only phase; writes ckpt/c1_la.pkl and data/la85_frozen_c1.json + .sha256)
"""
import os, sys, json, pickle, random, hashlib
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la85_common as C

NSTAT, B, NSEL = 8000, 120, 300
GROUPS = ['HT', 'KH', 'ZA', 'PH', 'OT']


def boot_sd(args):
    specs, X, n, sd_seed = args
    rng = np.random.default_rng(sd_seed)
    V = np.full((B, len(specs)), np.nan)
    for b in range(B):
        idx = rng.choice(len(X), size=n, replace=False)
        V[b] = C.eval_all(specs, X[idx])
    return np.nanstd(V, axis=0)


def chunks(specs, k=2):
    m = (len(specs) + k - 1) // k
    return [specs[i:i + m] for i in range(0, len(specs), m)]


def sds_for(specs, X, n, tag, pool):
    parts = pool.map(boot_sd, [(c, X, n, C.seed(tag + str(i))) for i, c in enumerate(chunks(specs))])
    return np.concatenate(parts)


def stability(specs, X, grp, sd, pool=None):
    """z per group: (v_g - v_rest)/sd(n_g); selection z on HT vs KH; held-out RMS over ZA, PH, OT vs HT+KH."""
    grp = np.array(grp)
    v = {g: C.eval_all(specs, X[grp == g]) for g in GROUPS}
    rest = {g: C.eval_all(specs, X[grp != g]) for g in GROUPS}
    htkh = C.eval_all(specs, X[(grp == 'HT') | (grp == 'KH')])
    z = {g: (v[g] - rest[g]) / sd[g] for g in GROUPS}
    zsel = (v['HT'] - v['KH']) / np.sqrt(sd['HT'] ** 2 + sd['KH'] ** 2)
    zho = np.sqrt(np.nanmean(np.vstack([((v[g] - htkh) / sd[g]) ** 2 for g in ('ZA', 'PH', 'OT')]), axis=0))
    zall = np.sqrt(np.nanmean(np.vstack([z[g] ** 2 for g in GROUPS]), axis=0))
    return dict(v=v, zsel=zsel, zho=zho, zall=zall)


def heldout_gain(st, ok):
    """Massive-guessing re-test: survivors = |zsel| in the lowest 20 %; pass = zho < 1.5."""
    zs, zh = np.abs(st['zsel'][ok]), st['zho'][ok]
    cut = np.quantile(zs, 0.2)
    surv = zs <= cut
    p_s, p_n = np.mean(zh[surv] < 1.5), np.mean(zh[~surv] < 1.5)
    rho = np.corrcoef(np.argsort(np.argsort(zs)), np.argsort(np.argsort(zh)))[0, 1]
    return p_s, p_n, rho


def main():
    la = C.la_docs()
    X = C.feature_matrix(la)
    grp = [d['grp'] for d in la]
    rng = random.Random(C.seed('la85-c1-stats'))
    specs = C.random_stats(NSTAT, rng, X)
    v_la = C.eval_all(specs, X)
    ok = ~np.isnan(v_la)
    for g in GROUPS:
        ok &= ~np.isnan(C.eval_all(specs, X[np.array(grp) == g]))
    print('stats', len(specs), 'valid on every LA group', int(ok.sum()), flush=True)
    sizes = {g: grp.count(g) for g in GROUPS}
    sizes['n100'] = 100
    with Pool(2) as pool:
        sd = {k: sds_for(specs, X, n, 'la85-sd-' + k, pool) for k, n in sizes.items()}
    for k in sd:
        ok &= (sd[k] > 0) & ~np.isnan(sd[k])
    print('valid with nonzero noise', int(ok.sum()), flush=True)
    st = stability(specs, X, grp, sd)
    real = heldout_gain(st, ok)
    print('REAL held-out: survivors pass %.3f, non-survivors %.3f, rho(zsel,zho) %.3f' % real, flush=True)
    # control 1: site labels shuffled (20 runs)
    sh = []
    for r in range(20):
        g2 = list(grp); random.Random(C.seed('la85-shuf%d' % r)).shuffle(g2)
        sh.append(heldout_gain(stability(specs, X, g2, sd), ok))
        print('shuffle', r, '%.3f %.3f %.3f' % sh[-1], flush=True)
    # control 2: planted site-specific distortion (held-out sites write every number x7 and drop the first line)
    la_p = []
    for d in la:
        if d['grp'] in ('ZA', 'PH', 'OT'):
            lines = [[(t[0], t[1] * 7, t[2]) if t[0] == 'N' else t for t in l] for l in d['lines']]
            lines = lines[1:] if len(lines) > 2 else lines
            la_p.append(dict(d, lines=lines))
        else:
            la_p.append(d)
    Xp = C.feature_matrix(la_p)
    stp = stability(specs, Xp, grp, sd)
    pl = heldout_gain(stp, ok)
    numf = {C.FEATS.index(k) for k in ('lmean', 'ge10', 'ge100', 'dig0', 'one', 'nlines', 'headfree')}
    touched = np.array([s['f'] in numf or s.get('g') in numf for s in specs])
    print('PLANTED: survivors %.3f non %.3f rho %.3f' % pl, '| touched-feature stats held-out pass %.3f (real %.3f)' % (
        np.mean(stp['zho'][ok & touched] < 1.5), np.mean(st['zho'][ok & touched] < 1.5)), flush=True)
    # freeze: the NSEL most stable statistics on all LA sites, and NSEL least stable
    idx = np.where(ok)[0]
    order = idx[np.argsort(st['zall'][idx])]
    S, U = order[:NSEL].tolist(), order[-NSEL:].tolist()
    pickle.dump(dict(specs=specs, ok=ok, sd=sd, st=st, v_la=v_la, real=real, sh=sh, pl=pl, S=S, U=U,
                     touched=touched, stp_zho=stp['zho']), open(os.path.join(C.CK, 'c1_la.pkl'), 'wb'))
    frozen = {
        'loop': 'la85 cycle 1', 'date': '2026-10-07',
        'what': 'Linear A stable value-free statistics (lowest RMS site z), with LA pooled values and LA noise at n=100',
        'corpus': 'corpus_ra_v2 rd, tablets and lames with >= 2 lines and >= 1 number (302 documents)',
        'seed_stats': 'la85-c1-stats', 'nstat': NSTAT, 'boot': B,
        'S': [dict(i=i, spec=specs[i], v_la=float(v_la[i]), sd100=float(sd['n100'][i]), zall=float(st['zall'][i])) for i in S],
        'U': [dict(i=i, spec=specs[i], v_la=float(v_la[i]), sd100=float(sd['n100'][i]), zall=float(st['zall'][i])) for i in U],
        'distance': 'd_T(s) = |v_T(s) - v_LA(s)| / sd100(s); contrast(A,B) = median over a set of log((d_B+0.05)/(d_A+0.05))',
        'null': '2,000 random subsets of 300 valid statistics (same seed family la85-null)',
        'predictions': {
            'P1': 'contrast(KN, PY) over S > 0 and above the random-subset null (one-sided P < 0.05): stable Minoan conventions survive better in the Cretan daughter archive than on the mainland.',
            'P2': 'contrast(KN, UR3) over S > 0 and above the null: the stable laws are Aegean, not universal bureaucracy.',
            'P3': 'contrast(KN-RCT, KN-later) over S > 0 and above both the random-subset null and a size-matched KN-later subsample null (Room of the Chariot Tablets = earliest Linear B, metadata fetched from DAMOS only after this freeze).',
            'P4': 'contrast(TH, PY) has no stated direction (control).',
        },
        'kill': {'P1': 'contrast <= 0 or P >= 0.2', 'P2': 'contrast <= 0 or P >= 0.2', 'P3': 'contrast <= 0 or P >= 0.2'},
        'real_heldout': list(map(float, real)),
    }
    fp = os.path.join(C.D, 'la85_frozen_c1.json')
    s = json.dumps(frozen, sort_keys=True, indent=0)
    open(fp, 'w').write(s)
    h = hashlib.sha256(s.encode()).hexdigest()
    open(fp.replace('.json', '.sha256'), 'w').write(h + '  la85_frozen_c1.json\n')
    print('FROZEN', fp, h)


if __name__ == '__main__':
    main()
