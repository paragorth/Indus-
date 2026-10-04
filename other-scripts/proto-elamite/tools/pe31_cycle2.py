"""pe31 cycle 2: run the frozen statistic IX on the real PE name-like strings.
(1) pair list with search correction: each pair's IX against (a) the maximum IX over all pairs in
    within-string-shuffled corpora (family-wise) and (b) the pair's own shuffled values (BH-FDR);
    a planted-homophone corpus goes through the same pipeline to show the power.
(2) graphic variants: mean IX of same-base pairs (X vs X~a, X~a vs X~b) in the variant alphabet
    against frequency-matched random pairs and against the shuffle null."""
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe31_lib as L

MINC = 6
NSH = int(os.environ.get('NSH', 60))


def IX(data, alph, cnt):
    Sd = L.scores(data, alph)
    return L.freq_z(L.merge_evidence(data, alph), cnt, alph) + L.combo(Sd, cnt, alph, (1, 1, 1, 0)), Sd


def run(data, label, truth=None):
    alph, cnt = L.alphabet(data, MINC)
    n = len(alph); iu = np.triu_indices(n, 1)
    X, Sd = IX(data, alph, cnt)
    x = X[iu]
    rng = random.Random(7)
    mx, own = [], []
    for k in range(NSH):
        sh = L.shuffle_within(data, rng)
        Xs, _ = IX(sh, alph, cnt)
        own.append(Xs[iu]); mx.append(Xs[iu].max())
    own = np.array(own); mx = np.array(mx)
    p_own = (1 + (own >= x).sum(0)) / (NSH + 1)
    p_fw = (1 + (mx[:, None] >= x[None, :]).sum(0)) / (NSH + 1)
    # BH
    o = np.argsort(p_own); m = len(p_own)
    q = np.empty(m); q[o] = np.minimum.accumulate((p_own[o] * m / (np.arange(m) + 1))[::-1])[::-1]
    order = np.argsort(-x)
    top = []
    for t in order[:25]:
        a, b = alph[iu[0][t]], alph[iu[1][t]]
        top.append(dict(a=a, b=b, na=cnt[a], nb=cnt[b], IX=round(float(x[t]), 2), MP=int(Sd['MP'][iu[0][t], iu[1][t]]),
                        TAB=round(float(Sd['TAB'][iu[0][t], iu[1][t]]), 2), p_fw=round(float(p_fw[t]), 3), q=round(float(q[t]), 3),
                        same_base=L.base(a) == L.base(b)))
    res = dict(label=label, n_strings=len(data), n_types=len({w for w, _ in data}), n_signs=n, n_pairs=len(x),
               n_fw05=int((p_fw <= .05).sum()), n_q10=int((q <= .10).sum()), shuffle_max_mean=round(float(mx.mean()), 2),
               obs_max=round(float(x.max()), 2), top=top)
    if truth is not None:
        lab = np.array([tuple(sorted((alph[i], alph[j]))) in truth for i, j in zip(*iu)])
        res['truth'] = dict(n=int(lab.sum()), fw05=int((lab & (p_fw <= .05)).sum()), q10=int((lab & (q <= .1)).sum()),
                            auc=round(L.auc(x, lab), 3))
    return res, (alph, cnt, X, Sd)


def variant_test(alph, cnt, X, nperm=5000, seed=3, sel=None):
    """mean IX of same-base pairs vs frequency-matched random pairs"""
    n = len(alph); iu = np.triu_indices(n, 1)
    x = X[iu]
    sel = sel or (lambda a, b: L.base(a) == L.base(b))
    same = np.array([sel(alph[i], alph[j]) for i, j in zip(*iu)])
    lf = np.log([cnt[a] for a in alph]); qb = np.quantile(lf, [.2, .4, .6, .8]); b = np.digitize(lf, qb)
    key = np.minimum(b[iu[0]], b[iu[1]]) * 5 + np.maximum(b[iu[0]], b[iu[1]])
    rng = np.random.default_rng(seed)
    pools = {k: np.where((key == k) & ~same)[0] for k in np.unique(key)}
    ks = key[same]
    obs = x[same].mean()
    null = np.array([np.mean([x[rng.choice(pools[k])] for k in ks]) for _ in range(nperm)])
    pct = [float((x > v).mean()) for v in x[same]]
    pairs = sorted([(alph[i], alph[j], round(float(X[i, j]), 2)) for i, j, s in zip(*iu, same) if s], key=lambda t: -t[2])
    return dict(n_same=int(same.sum()), obs=round(float(obs), 3), null_mean=round(float(null.mean()), 3),
                p=float((1 + (null >= obs).sum()) / (nperm + 1)), median_pct_rank=round(float(np.median(pct)), 3), pairs=pairs)


def main():
    out = {}
    pe_b = L.pe_names(variants=False)
    pe_v = L.pe_names(variants=True)
    r, _ = run(pe_b, 'PE-base'); out['base'] = r; print(json.dumps({k: v for k, v in r.items() if k != 'top'}), flush=True)
    r, (alph, cnt, X, Sd) = run(pe_v, 'PE-variants'); out['variants'] = r
    print(json.dumps({k: v for k, v in r.items() if k != 'top'}), flush=True)
    out['variant_test'] = variant_test(alph, cnt, X); print(json.dumps({k: v for k, v in out['variant_test'].items() if k != 'pairs'}), flush=True)
    # within-string shuffle of the variant test
    rng = random.Random(11); sv = []
    for k in range(20):
        sh = L.shuffle_within(pe_v, rng); Xs, _ = IX(sh, alph, cnt)
        sv.append(variant_test(alph, cnt, Xs, nperm=300)['obs'])
    out['variant_test_shuffled_obs'] = sv
    # planted power, same pipeline
    a0, c0 = L.alphabet(pe_b, 1)
    cand = [s for s in a0 if 16 <= c0[s] <= 80]
    pw = []
    for mode in ('token', 'tablet'):
        rng = random.Random(500); ps = set(rng.sample(cand, 10))
        d = L.plant(pe_b, ps, mode, rng)
        r, _ = run(d, f'plant-{mode}', truth={tuple(sorted((s, s + "'"))) for s in ps})
        pw.append({k: v for k, v in r.items() if k != 'top'}); print(json.dumps(pw[-1]), flush=True)
    # planted graphic-variant test: same pipeline, plant pairs renamed X / X~p so the variant test sees them
    rng = random.Random(501); ps = set(rng.sample(cand, 10))
    d = L.plant(pe_v, ps, 'tablet', rng, suffix='~pl')
    a2, c2 = L.alphabet(d, MINC); X2, _ = IX(d, a2, c2)
    out['plant_variant_test'] = {k: v for k, v in variant_test(a2, c2, X2, sel=lambda a, b: a.endswith('~pl') != b.endswith('~pl') and a.replace('~pl', '') == b.replace('~pl', '')).items() if k != 'pairs'}
    out['plant_power'] = pw
    json.dump(out, open(os.path.join(L.CK, 'cycle2.json'), 'w'), indent=1)
    print(json.dumps(out['plant_variant_test']))


if __name__ == '__main__':
    main()
