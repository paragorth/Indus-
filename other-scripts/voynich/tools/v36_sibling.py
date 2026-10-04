"""v36 cycle 4: FEATURAL SIBLINGS.  A sound change acts on a whole natural class, so if a rule
g -> g' helps fit X -> Y, the same rule applied to g's feature-sibling h (the glyph that differs
from g by the fewest features, with the output shifted by the same feature change when such a
glyph exists) should help too.  A spelling or key change is lexical/arbitrary: siblings do not help.
For each of the 30 learned rules and each glyph of its lhs that is changed:
  feature sibling  = the 2 most feature-similar glyphs h (excluding g)
  random sibling   = 2 random glyphs of g's frequency stratum, not among g's 4 nearest
Sibling rule = lhs with g -> h; rhs: if g' exists and a glyph h' with feats(h') = feats(h) - feats(g) + feats(g')
exists, g' -> h'; occurrences of g kept in rhs become h.  Score = sibling single-rule fit gain > 0 (helps) and
gain relative to the parent rule's gain.  Output: help rate featural vs random, and the difference.
Usage: python3 v36_sibling.py <name> [kind]
"""
import os, sys, json, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as L
import v25_shapes as S
import v36_fitnull as FN
from v36_fitnull import setup, feat
FN.K = 30
from v30_lib import Search


def featvec(F, g):
    i = F.ix[g]
    if F.kind in ('vhand', 'vimg'):
        return collections.Counter(S.VOYNICH[g])
    if F.kind == 'hangul':
        return collections.Counter(S.HANGUL[g])
    return collections.Counter({f: 1 for f in F.sets[i]})


def analog(F, g, g2, h):
    """glyph h' with feats(h') = feats(h) - feats(g) + feats(g2), or None."""
    if g2 not in F.ix:
        return None
    tgt = featvec(F, h) - featvec(F, g)
    tgt = collections.Counter({k: v for k, v in tgt.items() if v > 0})
    tgt.update(featvec(F, g2))
    for x in F.alph:
        if featvec(F, x) == tgt:
            return x
    return None


def siblings(rule, F, tok, rng, mode):
    lhs, rhs, lc, rc = rule
    a = tok(lhs); b = tok(rhs)
    out = []
    import difflib
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    amap = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        for t, i in enumerate(range(i1, i2)):
            amap[i] = (op, j1 + t if (j1 + t) < j2 else None)
    for i, g in enumerate(a):
        if g not in F.ix:
            continue
        gi = F.ix[g]
        sims = F.M[gi].copy(); sims[gi] = -1e9
        near = list(np.argsort(-sims))
        if mode == 'feat':
            hs = [F.alph[j] for j in near[:2]]
        else:
            st = F.strata[gi]
            pool = [F.alph[j] for j in range(len(F.alph)) if F.strata[j] == st and j != gi and j not in near[:4]]
            if not pool:
                pool = [F.alph[j] for j in near[4:]]
            hs = rng.sample(pool, min(2, len(pool)))
        op, j = amap.get(i, ('equal', None))
        for h in hs:
            na = list(a); na[i] = h
            nb = list(b)
            if op == 'equal' and j is not None:
                nb[j] = h
            elif op == 'replace' and j is not None:
                hp = analog(F, g, b[j], h)
                if hp is not None:
                    nb[j] = hp
            nl, nr = ''.join(na), ''.join(nb)
            if nl != nr:
                out.append((nl, nr, lc, rc))
    return out


def main(name, kind=None):
    xf, yf, rules, script = setup(name)
    kind = kind or {'voynich': 'vhand', 'hangul': 'hangul', 'latin': 'phon'}[script]
    F = feat(script, xf, kind)
    tok = L.tok_for('voynich' if script == 'voynich' else 'latin')
    Sr = Search(xf, yf, seed=11)
    base = Sr.score
    gain = lambda r: base - Sr.try_rules([r])[0]
    rng = random.Random(3)
    res = {'feat': [], 'rand': []}
    for r in rules:
        g0 = gain(r)
        if g0 <= 0:
            continue
        for mode, key in (('feat', 'feat'), ('rand', 'rand')):
            for s in siblings(r, F, tok, rng, mode):
                res[key].append((g0, gain(s)))
    out = dict(name=name, kind=kind)
    for k, v in res.items():
        v = np.array(v) if v else np.zeros((0, 2))
        out[k] = dict(n=len(v), help=float((v[:, 1] > 1e-5).mean()) if len(v) else np.nan,
                      rel=float(np.median(v[:, 1] / v[:, 0])) if len(v) else np.nan,
                      mean_rel=float(np.mean(np.clip(v[:, 1] / v[:, 0], -1, 1))) if len(v) else np.nan)
    # paired permutation: shuffle feat/rand labels
    a = [x[1] / x[0] for x in res['feat']]; b = [x[1] / x[0] for x in res['rand']]
    allv = np.clip(np.array(a + b), -1, 1); n = len(a)
    obs = np.mean(allv[:n]) - np.mean(allv[n:]) if n and len(b) else np.nan
    prng = np.random.default_rng(0); cnt = 0
    for _ in range(5000):
        p = prng.permutation(len(allv))
        if np.mean(allv[p[:n]]) - np.mean(allv[p[n:]]) >= obs:
            cnt += 1
    out['diff'] = float(obs); out['p'] = (cnt + 1) / 5001
    json.dump(out, open(os.path.join(L.CK, f'sib_{name}_{kind}.json'), 'w'), default=float)
    print(f"{name:14s} {kind:6s} feat n {out['feat']['n']} help {out['feat']['help']:.2f} rel {out['feat']['mean_rel']:+.3f} | "
          f"rand n {out['rand']['n']} help {out['rand']['help']:.2f} rel {out['rand']['mean_rel']:+.3f} | diff {obs:+.3f} p {out['p']:.4f}", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
