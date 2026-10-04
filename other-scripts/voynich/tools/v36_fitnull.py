"""v36 cycle 3: FIT-MATCHED random rule sets.
For a pair X -> Y (same fit split as the learned rules), 4,000 random single rules are drawn from the
v30 proposal distribution (random + guided) and each is scored alone by its fit gain on X -> Y.
A null rule set = for each of the top-10 learned rules, a random pool rule whose single-rule gain is
within a factor 1.5 of that rule's own single-rule gain (gain-profile matched; ties broken at random).
2,000 null sets give the distribution of COH / CTX / SUB; the learned set is placed in it.
This asks: among rules that fit X -> Y equally well, are the ones the search keeps more featural?
Usage: python3 v36_fitnull.py <name> [kind]   (name = v30 pair or plant_*)
"""
import os, sys, json, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as L
import v36_plant as P
from v30_lib import Search, split_pages
from v30_ladder import make_split, PAIRS
import v25_shapes as S

K = 10


def setup(name):
    if name.startswith('plant_'):
        pn = name[6:]
        X, Y, _ = P.make(pn)
        xf, xh, _ = split_pages(X, 3000, 1500, 1); yf, yh, _ = split_pages(Y, 3000, 1500, 2)
        d = json.load(open(os.path.join(L.CK, f'{name}.json')))
        script = {'P': 'voynich', 'H': 'hangul', 'G': 'latin'}[pn[0]]
        seed = 0
    else:
        seed = 0
        xf, xh, yf, yh = make_split(name, seed)
        d = json.load(open(os.path.join(L.CK30, f'g3_{name}_s{seed}.json')))
        script = 'voynich' if name.startswith(('V_', 'S_AB', 'N_AA', 'N_BB')) else 'latin'
    rules = [tuple(r) for r in d['path'][-1]['rules']][:K]
    return xf, yf, rules, script


def feat(script, xwords, kind):
    tok = L.tok_for(script)
    fr = L.glyph_freq(collections.Counter(xwords), tok)
    if script == 'voynich':
        alph = [g for g in S.VOYNICH if fr[g] >= 20]
    elif script == 'hangul':
        alph = [g for g in S.HANGUL if fr[g] >= 20]
    else:
        alph = [g for g, c in fr.most_common() if c >= 20]
    return L.Feat(kind, alph, fr)


def main(name, kind=None, npool=4000, nsets=2000):
    xf, yf, rules, script = setup(name)
    kind = kind or {'voynich': 'vhand', 'hangul': 'hangul', 'latin': 'phon'}[script]
    out = os.path.join(L.CK, f'fit_{name}_{kind}.json')
    Sr = Search(xf, yf, seed=11); Sr.guided = True; Sr.refresh_guide()
    base = Sr.score
    gain = lambda r: base - Sr.try_rules([r])[0]
    learned = [(r, gain(r)) for r in rules]
    pool = {}
    while len(pool) < npool:
        r = Sr.propose()
        if r in pool:
            continue
        pool[r] = gain(r)
    pool = [(r, g) for r, g in pool.items() if g > 0]
    tok = L.tok_for('voynich' if script == 'voynich' else 'latin')
    types = collections.Counter(xf)
    F = feat(script, xf, kind)
    eff_cache = {}

    def eff(r):
        e = eff_cache.get(r)
        if e is None:
            e = L.rule_effect(r, types, tok); eff_cache[r] = e
        return e
    obs_eff = [eff(r) for r, _ in learned if eff(r)['mass'] > 0]
    obs = L.stats(obs_eff, F, nperm=1000)
    rng = random.Random(5)
    gains = np.array([g for _, g in pool])
    cand_lists = []
    for r, g in learned:
        idx = [i for i in range(len(pool)) if g / 1.5 <= gains[i] <= g * 1.5 and pool[i][0] != r]
        if len(idx) < 5:  # widen to the 10 nearest gains
            idx = list(np.argsort(np.abs(np.log(gains + 1e-9) - np.log(max(g, 1e-9))))[:10])
        cand_lists.append(idx)
    null = collections.defaultdict(list)
    for _ in range(nsets):
        rs = [pool[rng.choice(c)][0] for c in cand_lists]
        E = [eff(r) for r in rs]; E = [e for e in E if e['mass'] > 0]
        vt = L.pooled(E, 'targets', F.ix); vc = L.pooled(E, 'ctx', F.ix)
        null['COH'].append(L.coh(vt, F.M)); null['CTX'].append(L.coh(vc, F.M)); null['SUB'].append(L.sub_sim(E, F))
    res = dict(name=name, kind=kind, npool=len(pool), learned_gain=[g for _, g in learned], obs=obs)
    for m in ('COH', 'CTX', 'SUB'):
        a = np.array(null[m], float); a = a[~np.isnan(a)]
        o = obs[m]['obs']
        res[m] = dict(obs=o, null_mean=float(a.mean()), null_sd=float(a.std()), pct=float((a < o).mean()),
                      z=float((o - a.mean()) / (a.std() + 1e-12)))
    json.dump(res, open(out, 'w'), default=float)
    print(name, kind, 'pool', len(pool), ' '.join(f"{m} obs {res[m]['obs']:.3f} null {res[m]['null_mean']:.3f} pct {res[m]['pct']:.2f}" for m in ('COH', 'CTX', 'SUB')), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
