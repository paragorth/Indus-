"""v48 cycle 2: is the placement stable, and what is it worth?

2a  family-identification power of the map: leave-one-out kNN family vote for every language, under the full
    feature set and 1,000 random half-feature subsets (the calibration for any family claim).
2b  the Voynich under the same 1,000 subsets: how often each family is its nearest; out-score distribution vs the
    languages' LOO out-scores and vs its own Markov-2 / shuffle nulls.
2c  feature groups alone (UNIT: entropy + V/C + syllable; WORD: length, morph, positional, affix; TEXT: reuse, Zipf,
    TTR): out-score and nearest corpora of the cleaned Voynich, its Markov-2, and of planted controls.
2d  slices: Currier A, Currier B, herbal A, B bio, B stars, hands 1-3, both transcriptions, after E3 + blind A/B.
"""
import sys, os, json, random
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v48_lib as L

OUT = 'v48_cycle2.txt'
GROUPS = {'UNIT': ['h1', 'h2', 'h2r', 'vshare', 'alt', 'bip', 'agree', 'stab', 'initV', 'finV', 'clus', 'clus2', 'VV', 'sylpw', 'skelH'],
          'WORD': ['wl', 'wlcv', 'wlsk', 'w1', 'wlong', 'morph', 'mpw', 'Hp1', 'Hp2', 'Hl1', 'Hl2', 'sufconc', 'sufpre'],
          'TEXT': ['burst', 'reuse', 'ttr', 'hapax', 'zipf', 'rep']}


def slice_job(args):
    src, inv, norm, sel = args
    C = L.voynich(src, norm, inv)
    meta = {}
    recs = json.load(open(os.path.join(L.DATA, 'derived', f'{src}_lines.json')))
    for r in recs:
        meta.setdefault(r['folio'], (r.get('lang'), r.get('hand'), r.get('illus')))
    def keep(d):
        lang, hand, ill = meta[d['id']]
        return dict(A=lang == 'A', B=lang == 'B', HA=(lang == 'A' and ill == 'H'), BB=(lang == 'B' and ill == 'B'),
                    BS=(lang == 'B' and ill == 'S'), H1=hand == '1', H2=hand == '2', H3=hand == '3', ALL=True)[sel]
    if sel in ('ALL',):
        C, rules = L.ab_blind(C)
    C = dict(C, docs=[d for d in C['docs'] if keep(d)])
    n = len(L.tokens(C))
    return dict(name=f'{src}/{inv}/{norm}/{sel}', n=n, fp=L.fp_corpus(C) if n >= 4000 else None,
                mk2=L.fp_corpus(L.null_markov2(C)) if n >= 4000 else None)


def main():
    c1 = L.load('c1_results.json'); s1 = L.load('c1_summary.json')
    lang = c1['lang']
    names = [r['name'] for r in lang]; fams = [r['fam'] for r in lang]
    F = L.FEATS
    X = np.array([[r['fp'][k] for k in F] for r in lang])
    best = s1['vrows'][0][0]
    vbest = next(r for r in c1['voy'] if r['name'] == best)
    vraw = next(r for r in c1['voy'] if r['name'] == 'ZL3b/GLY/E0/none')
    rng = np.random.default_rng(48)
    # ---------------- 2a/2b: subsets
    def nn_fam(Xl, x, feats_idx, exclude=None):
        mu, sd = Xl[:, feats_idx].mean(0), Xl[:, feats_idx].std(0) + 1e-9
        Z = (Xl[:, feats_idx] - mu) / sd; z = (x[feats_idx] - mu) / sd
        d = np.sqrt(((Z - z) ** 2).mean(1))
        if exclude is not None: d[exclude] = np.inf
        Zd = np.sqrt(((Z[:, None] - Z[None]) ** 2).mean(2)); np.fill_diagonal(Zd, np.inf)
        if exclude is not None: Zd[exclude] = np.inf; Zd[:, exclude] = np.inf
        ref = np.median(Zd.min(1)[np.isfinite(Zd.min(1))])
        o = np.argsort(d)
        return o, d[o[0]] / ref
    fam_count = Counter()
    def has_mate(i): return sum(f == fams[i] for f in fams) > 1
    loo_hits, loo_n = 0, 0
    subsets = [rng.choice(len(F), len(F) // 2, replace=False) for _ in range(1000)]
    full = np.arange(len(F))
    loo_full = []
    for i in range(len(names)):
        if not has_mate(i): continue
        o, _ = nn_fam(X, X[i], full, exclude=i)
        loo_full.append(fams[o[0]] == fams[i])
    for S in subsets[:300]:
        for i in range(len(names)):
            if not has_mate(i): continue
            o, _ = nn_fam(X, X[i], S, exclude=i)
            loo_hits += fams[o[0]] == fams[i]; loo_n += 1
    chance = np.mean([(sum(f == fams[i] for f in fams) - 1) / (len(fams) - 1) for i in range(len(fams)) if has_mate(i)])
    vb = np.array([vbest['fp'][k] for k in F]); vr = np.array([vraw['fp'][k] for k in F])
    vfam, vnn, vout, rout = Counter(), Counter(), [], []
    for S in subsets:
        o, out = nn_fam(X, vb, S); vfam[fams[o[0]]] += 1; vnn[names[o[0]]] += 1; vout.append(out)
        o2, out2 = nn_fam(X, vr, S); rout.append(out2)
    # LOO language out-score under subsets for comparison
    lout = []
    for S in subsets[:200]:
        for i in rng.choice(len(names), 5, replace=False):
            _, out = nn_fam(X, X[i], S, exclude=i); lout.append(out)
    s2a = (f'LOO family of the nearest corpus: full features {np.mean(loo_full):.2f}, random half subsets {loo_hits / loo_n:.2f} (chance {chance:.2f}; '
           f'{len(loo_full)} languages with a family mate)')
    s2b = (f'Voynich {best}: nearest family over 1,000 half-subsets ' + ', '.join(f'{k} {v / 10:.0f}%' for k, v in vfam.most_common(5)) +
           '; nearest corpus ' + ', '.join(f'{k} {v / 10:.0f}%' for k, v in vnn.most_common(5)) +
           f'; out-score median {np.median(vout):.2f} (raw E0 {np.median(rout):.2f}) vs languages LOO median {np.median(lout):.2f}, 95th {np.percentile(lout, 95):.2f}; '
           f'Voynich beyond the language 95th pct in {np.mean(np.array(vout) > np.percentile(lout, 95)):.2f} of subsets')
    print(s2a); print(s2b)
    # ---------------- 2c: feature groups
    gl = []
    for g, fs in GROUPS.items():
        S = np.array([F.index(k) for k in fs])
        o, out = nn_fam(X, vb, S)
        om, outm = nn_fam(X, np.array([vbest.get('mk2', vraw['mk2'])[k] for k in F]) if 'mk2' in vbest else np.array([vraw['mk2'][k] for k in F]), S)
        _, outr = nn_fam(X, vr, S)
        lo = []
        for i in range(len(names)):
            _, x = nn_fam(X, X[i], S, exclude=i); lo.append(x)
        gl.append(f'{g}: Voynich out {out:.2f} (raw {outr:.2f}; its Markov-2 {outm:.2f}; languages 95th {np.percentile(lo, 95):.2f}) NN ' +
                  ', '.join(f'{names[k]}' for k in o[:3]))
    s2c = ' | '.join(gl)
    print(s2c)
    # z profile of the best version, per group
    mu, sd = X.mean(0), X.std(0) + 1e-9
    z = (vb - mu) / sd
    prof = ', '.join(f'{F[i]} {z[i]:+.1f}' for i in np.argsort(-np.abs(z))[:8])
    print('extreme', prof)
    # ---------------- 2d: slices
    norm_inv = best.split('/')
    jobs = []
    for src in ('ZL3b', 'IT2a'):
        for sel in ('ALL', 'A', 'B', 'HA', 'BB', 'BS', 'H1', 'H2', 'H3'):
            jobs.append((src, norm_inv[1], norm_inv[2], sel))
    res = L.load('c2_slices.json')
    if res is None:
        with Pool(2) as p:
            res = p.map(slice_job, jobs, chunksize=1)
        L.save('c2_slices.json', res)
    M = L.Map([r['fp'] for r in lang], names, fams)
    sl = []
    for r in res:
        if r['fp'] is None: continue
        p = M.place(r['fp']); pm = M.place(r['mk2'])
        sl.append(f"{r['name'].split('/')[0][:2]}/{r['name'].split('/')[-1]} (n {r['n']}) out {p['out']:.2f} [mk2 {pm['out']:.2f}] NN {p['nn'][0][0]}, {p['nn'][1][0]}")
    s2d = '; '.join(sl)
    print(s2d)
    L.save('c2_summary.json', dict(s2a=s2a, s2b=s2b, s2c=s2c, s2d=s2d, prof=prof))
    L.row(OUT, 'V-48.2a', 'Calibration of family claims: leave-one-out nearest-corpus family for every language with a family mate, full feature set and 300 random half-feature subsets', s2a, 'The map can name a family only at this hit rate; any Voynich family claim is worth no more')
    L.row(OUT, 'V-48.2b', f'Stability: the best cleaned Voynich ({best}) and raw E0 placed under 1,000 random half-feature subsets; out-score against the languages\' own LOO out-scores under the same subsets', s2b + '. Most extreme features: ' + prof, 'see cycle verdict')
    L.row(OUT, 'V-48.2c', 'Feature groups alone (UNIT: entropy, Sukhotin/spectral V-C, syllable skeleton; WORD: length, Harris morphs, positional entropy, affix concentration; TEXT: reuse, Zipf, TTR); the Voynich Markov-2 resynthesis as the null that keeps unit statistics', s2c, 'see cycle verdict')
    L.row(OUT, 'V-48.2d', 'Slices after the best normalisation (+ blind A/B undo on ALL): Currier A, B, herbal A, B bio, B stars, hands 1-3, ZL3b and IT2a; each with its own Markov-2 null', s2d, 'see cycle verdict')


if __name__ == '__main__':
    main()
