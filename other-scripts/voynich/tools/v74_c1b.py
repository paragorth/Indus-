"""v74 cycle 1d: is the narrow space a property of the junction PAIR (P1's 'space-split pieces') or of the
first word alone? Tightest control: same first word AND same first glyph of the next word, other next word
(e.g. or.aiin vs or.al/or.ar/or.am). Stratified permutation of the target label inside (first word, next first
glyph) strata, 10,000 draws; logistic regression of (r < 1) on target + first-word glyph count + last glyph +
next first glyph + ZL separator type. Also the share for first words or/s/r/ar/o/ol against all other words of
equal glyph count. Out: data/v74_ckpt/c1b.json"""
import os, sys, json, collections
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v74_lib as X
import v74_c1 as C

RNG = np.random.default_rng(741)
if __name__ == '__main__':
    R = C.load(['gaps_V.json', 'gaps_VX.json'])
    for r, s in zip(R, C.seps_for(R)): r['sep'] = s
    out = {}
    for key in ('r', 'r0'):
        firsts = {a for a, _ in X.TARGETS}
        S = collections.defaultdict(list)
        for r in R:
            if r['a'] in firsts: S[(r['a'], C.first_g(r['b']))].append(r)
        S = {k: v for k, v in S.items() if any((r['a'], r['b']) in X.TSET for r in v)}
        tg = [r for v in S.values() for r in v if (r['a'], r['b']) in X.TSET]
        ct = [r for v in S.values() for r in v if (r['a'], r['b']) not in X.TSET]
        obs = np.mean([r[key] < 1 for r in tg])
        null = []
        for _ in range(10000):
            s = n = 0
            for v in S.values():
                nt = sum((r['a'], r['b']) in X.TSET for r in v)
                idx = RNG.choice(len(v), nt, replace=False); s += sum(v[i][key] < 1 for i in idx); n += nt
            null.append(s / n)
        o = dict(target=C.share([r[key] for r in tg]), tight_ctl=C.share([r[key] for r in ct]),
                 strata={f'{a}.{g}': (len(v), sum((r['a'], r['b']) in X.TSET for r in v)) for (a, g), v in S.items()},
                 perm_p=float(np.mean(np.array(null) >= obs)), null_mean=float(np.mean(null)),
                 p_mw=C.mw([r[key] for r in tg], [r[key] for r in ct]))
        # short-word effect: first words in the target set vs other first words of equal glyph count
        gl = {len(X.vglyphs(a)) for a in firsts}
        sw = [r for r in R if r['a'] in firsts]
        ow = [r for r in R if r['a'] not in firsts and len(X.vglyphs(r['a'])) in gl]
        o['short_target_words'] = C.share([r[key] for r in sw]); o['other_equal_length_words'] = C.share([r[key] for r in ow])
        o['p_short'] = C.mw([r[key] for r in sw], [r[key] for r in ow])
        # logistic regression
        try:
            import statsmodels.api as sm
            lg = collections.Counter(C.last_g(r['a']) for r in R); fg = collections.Counter(C.first_g(r['b']) for r in R)
            L = [g for g, n in lg.most_common(8)]; F = [g for g, n in fg.most_common(8)]
            Xm, y = [], []
            for r in R:
                if not np.isfinite(r[key]): continue
                Xm.append([1.0, float((r['a'], r['b']) in X.TSET), float(r['a'] in firsts), len(X.vglyphs(r['a'])),
                           float(r['sep'] == ',')] + [float(C.last_g(r['a']) == g) for g in L[1:]] + [float(C.first_g(r['b']) == g) for g in F[1:]])
                y.append(float(r[key] < 1))
            m = sm.Logit(np.array(y), np.array(Xm)).fit(disp=0)
            o['logit'] = dict(names=['const', 'target_pair', 'target_first_word', 'first_len', 'zl_uncertain'],
                              coef=[float(c) for c in m.params[:5]], p=[float(p) for p in m.pvalues[:5]])
        except Exception as e:
            o['logit'] = repr(e)
        out[key] = o
    X.jsave('c1b.json', out)
    print(json.dumps(out, indent=1, default=float))
