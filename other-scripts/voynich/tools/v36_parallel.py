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
    if F.kind == 'hangul2':
        return collections.Counter(L.hangul2(g))
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
    # row-controlled: remove each (source glyph, context) mean gain, then ETA of the residual by delta group
    rowm = collections.defaultdict(list)
    for (a, b, c, x) in pairs_gain:
        rowm[(a, c)].append(max(x, 0))
    rmu = {k: np.mean(v) for k, v in rowm.items()}
    res_ = np.array([max(x, 0) - rmu[(a, c)] for a, b, c, x in pairs_gain])
    if m.sum() > 2 and res_[m].var() > 0:
        grp = collections.defaultdict(list)
        for k, x, ok in zip(keys, res_, m):
            if ok:
                grp[k].append(x)
        mu = res_[m].mean()
        eta_r = sum(len(v) * (np.mean(v) - mu) ** 2 for v in grp.values()) / (((res_[m] - mu) ** 2).sum())
    else:
        eta_r = np.nan
    # row coherence: are the source glyphs whose replacement helps a feature class?
    n = len(labels); rv = np.zeros(n)
    for (a, c), v in rmu.items():
        rv[a] += v
    W = np.outer(rv, rv); np.fill_diagonal(W, 0)
    Mp = F.M[np.ix_(labels, labels)]
    rowcoh = (W * Mp).sum() / W.sum() if W.sum() > 0 else np.nan
    top = np.argsort(-g)[:12]
    tk = collections.Counter(keys[i] for i in top)
    par = sum(1 for i in top if tk[keys[i]] >= 2)
    sim = np.array([F.M[labels[a], labels[b]] for a, b, _, _ in pairs_gain])
    subr = spearmanr(sim, g).correlation
    return eta, par, subr, eta_r, rowcoh


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
    for j, k in enumerate(('ETA', 'PAR', 'SUBR', 'ETAR', 'ROWCOH')):
        a = null[:, j]; a = a[~np.isnan(a)]
        res[k] = dict(obs=float(obs[j]), null=float(a.mean()), z=float((obs[j] - a.mean()) / (a.std() + 1e-12)),
                      p=float(((a >= obs[j]).sum() + 1) / (len(a) + 1)))
    V = [fvec(F, g) for g in alph]
    grp = collections.defaultdict(list)
    for a, b, c, g in PG:
        grp[(delta_key(V[a], V[b]), '*IF'[c])].append((max(g, 0), alph[a] + '>' + alph[b]))
    gl = sorted(((np.mean([x for x, _ in v]), k, [n for _, n in v]) for k, v in grp.items() if len(v) >= 2), key=lambda t: -t[0])[:6]
    res['top_groups'] = [(round(float(m_), 4), str(k), ms[:6]) for m_, k, ms in gl]
    res['gains'] = [(alph[a], alph[b], c, g) for a, b, c, g in PG]
    top = sorted(PG, key=lambda x: -x[3])[:12]
    res['top'] = [(alph[a], alph[b], '*IF'[c], round(g, 4)) for a, b, c, g in top]
    sd = os.environ.get('V36_SEED', '0')
    json.dump(res, open(os.path.join(L.CK, f'par_{name}_{kind}_s{sd}.json'), 'w'), default=float, ensure_ascii=False)
    print(f"{name:14s} {kind:6s} " + ' '.join(f"{k} {res[k]['obs']:.3f} z{res[k]['z']:+.2f} p{res[k]['p']:.3f}" for k in ('ETA', 'PAR', 'SUBR', 'ETAR', 'ROWCOH'))
          + ' | groups ' + ' ; '.join(f'{k} {m_}' for m_, k, _ in res['top_groups'][:2]) + ' | top ' + ' '.join(f'{a}>{b}{c}' for a, b, c, _ in res['top'][:8]), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
