"""v36 cycle 4b: EXHAUSTIVE substitution matrix, no rule search.
Every single-glyph substitution a -> b (a != b) is scored alone, in 3 contexts (anywhere, word-initial,
word-final), by its fit gain on X -> Y (v30 distance, fit pages).  Voynich words are first re-encoded one
character per glyph (cth=T ckh=K cph=P cfh=F ch=C sh=S) so a rule never cuts a glyph.
A sound change acts on a natural class with ONE feature change: the substitutions that help should share
a feature delta (feats(b) - feats(a)).  Statistics on the positive gains:
  ETA   share of gain variance explained by the feature-delta group (groups with >= 2 members)
  PAR   among the top-12 substitutions, the number that share their delta with another top-12 one
  SUBR  Spearman of gain with feature similarity of a and b (minimal change)
Null = 1,000 frequency-stratified glyph-label permutations (features reassigned among same-frequency glyphs).
Usage: python3 v36_parallel.py <name> [kind]
"""
import os, sys, json, random, collections
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as L
import v36_fitnull as FN
import v25_shapes as S
from v30_lib import Search

ENC = [('cth', 'T'), ('ckh', 'K'), ('cph', 'P'), ('cfh', 'F'), ('ch', 'C'), ('sh', 'S')]
DEC = {b: a for a, b in ENC}


def enc(w):
    for a, b in ENC:
        w = w.replace(a, b)
    return w


def fvec(F, g):
    if F.kind in ('vhand', 'vimg'):
        return collections.Counter(S.VOYNICH[g])
    if F.kind == 'hangul':
        return collections.Counter(S.HANGUL[g])
    return collections.Counter({f: 1 for f in F.sets[F.ix[g]]})


def delta_key(va, vb):
    ks = set(va) | set(vb)
    return tuple(sorted((k, vb.get(k, 0) - va.get(k, 0)) for k in ks if vb.get(k, 0) != va.get(k, 0)))


def scores(pairs_gain, F, labels):
    """pairs_gain: list of (ia, ib, ctx, gain); labels: permutation mapping glyph index -> feature donor index."""
    V = [fvec(F, F.alph[i]) for i in labels]
    keys = [(delta_key(V[a], V[b]), c) for a, b, c, _ in pairs_gain]
    g = np.array([x[3] for x in pairs_gain]); gp = np.clip(g, 0, None)
    cnt = collections.Counter(keys)
    m = np.array([cnt[k] >= 2 for k in keys])
    if m.sum() > 2 and gp[m].var() > 0:
        grp = collections.defaultdict(list)
        for k, x, ok in zip(keys, gp, m):
            if ok:
                grp[k].append(x)
        mu = gp[m].mean()
        eta = sum(len(v) * (np.mean(v) - mu) ** 2 for v in grp.values()) / (((gp[m] - mu) ** 2).sum())
    else:
        eta = np.nan
    top = np.argsort(-g)[:12]
    tk = collections.Counter(keys[i] for i in top)
    par = sum(1 for i in top if tk[keys[i]] >= 2)
    sim = np.array([F.M[labels[a], labels[b]] for a, b, _, _ in pairs_gain])
    subr = spearmanr(sim, g).correlation
    return eta, par, subr


def main(name, kind=None, nperm=1000):
    xf, yf, rules, script = FN.setup(name)
    if script == 'voynich':
        xf = [enc(w) for w in xf]; yf = [enc(w) for w in yf]
    kind = kind or {'voynich': 'vhand', 'hangul': 'hangul', 'latin': 'phon'}[script]
    fr0 = collections.Counter(c for w in xf for c in w)
    if script == 'voynich':
        fr = collections.Counter({DEC.get(c, c): n for c, n in fr0.items()})
        alph = [g for g in S.VOYNICH if fr[g] >= 30]
        code = {g: dict((a, b) for a, b in ENC).get(g, g) for g in alph}
    elif script == 'hangul':
        fr = fr0; alph = [g for g in S.HANGUL if fr[g] >= 30]; code = {g: g for g in alph}
    else:
        fr = fr0; alph = [g for g, c in fr.most_common() if c >= 30][:40]; code = {g: g for g in alph}
    F = L.Feat(kind, alph, fr)
    Sr = Search(xf, yf, seed=11)
    base = Sr.score
    PG = []
    for ia, a in enumerate(alph):
        for ib, b in enumerate(alph):
            if a == b:
                continue
            for c, (lc, rc) in enumerate([(None, None), ('#', None), (None, '#')]):
                g = base - Sr.try_rules([(code[a], code[b], lc, rc)])[0]
                PG.append((ia, ib, c, g))
    ident = np.arange(len(alph))
    obs = scores(PG, F, ident)
    rng = np.random.default_rng(0)
    null = np.array([scores(PG, F, F.perm(rng)) for _ in range(nperm)], float)
    res = dict(name=name, kind=kind, n=len(alph))
    for j, k in enumerate(('ETA', 'PAR', 'SUBR')):
        a = null[:, j]; a = a[~np.isnan(a)]
        res[k] = dict(obs=float(obs[j]), null=float(a.mean()), z=float((obs[j] - a.mean()) / (a.std() + 1e-12)),
                      p=float(((a >= obs[j]).sum() + 1) / (len(a) + 1)))
    top = sorted(PG, key=lambda x: -x[3])[:12]
    res['top'] = [(alph[a], alph[b], '*IF'[c], round(g, 4)) for a, b, c, g in top]
    json.dump(res, open(os.path.join(L.CK, f'par_{name}_{kind}.json'), 'w'), default=float, ensure_ascii=False)
    print(f"{name:14s} {kind:6s} " + ' '.join(f"{k} {res[k]['obs']:.3f} z{res[k]['z']:+.2f} p{res[k]['p']:.3f}" for k in ('ETA', 'PAR', 'SUBR'))
          + ' top ' + ' '.join(f'{a}>{b}{c}' for a, b, c, _ in res['top'][:8]), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
