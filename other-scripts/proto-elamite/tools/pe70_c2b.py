"""pe70 cycle 2b: is a seal dossier more than a find lot?

(a) For every tablet in a recurring-seal group: idf-weighted Jaccard to its seal-mates minus to its K=2
    nearest publication-number neighbours in the same volume that do not share the seal; paired sign-flip
    null (10,000); also seal-mate vs nearest SEALED neighbours.
(b) Cross-seal vocabulary: signs carried by tablets of >= 2 different recurring seals but rare elsewhere
    (|M153+M342| type); null: sign columns permuted among tablets within volume x band (5,000).
(c) |M153+M342| and M340 with the PES0329 tablets removed: sealed share vs volume x band null.
usage: pe70_c2b.py -> data/pe70_ckpt/c2b.json
"""
import json, os, collections, math
import numpy as np
from pe70_common import get, CK, DATA
from pe70_c2 import groups_of, idf, wjac
from pe70_feats import strata, perm_within


def main():
    rng = np.random.default_rng(705)
    R = get('pe')
    meta = json.load(open(os.path.join(DATA, 'pe8_meta.json')))
    num = []
    for r in R:
        m = meta.get(r['id'], {})
        try:
            num.append(float(m.get('pub_num')))
        except Exception:
            num.append(np.nan)
    num = np.array(num)
    G = groups_of(R)
    df, w = idf(R)
    S = [set(r['toks']) for r in R]
    vol = np.array([r['vol'] for r in R])
    sealed = np.array([r['sealed'] for r in R])
    res = {}
    for nm, cond in (('any', np.ones(len(R), bool)), ('sealed', sealed)):
        diffs, rows = [], []
        for s, ix in G.items():
            for i in ix:
                mates = [j for j in ix if j != i]
                if np.isnan(num[i]):
                    continue
                cand = [j for j in np.where((vol == vol[i]) & cond)[0] if j not in ix and not np.isnan(num[j])]
                if len(cand) < 2:
                    continue
                cand = sorted(cand, key=lambda j: abs(num[j] - num[i]))[:2]
                a = np.mean([wjac(S[i], S[j], w) for j in mates]); b = np.mean([wjac(S[i], S[j], w) for j in cand])
                diffs.append(a - b); rows.append((R[i]['id'], s, round(a, 3), round(b, 3), [abs(num[j] - num[i]) for j in cand]))
        d = np.array(diffs)
        nul = np.array([(d * rng.choice([-1, 1], len(d))).mean() for _ in range(10000)])
        res['adj_' + nm] = dict(n=len(d), mean_diff=float(d.mean()), share_pos=float((d > 0).mean()), p=float((nul >= d.mean()).mean()),
                                rows=rows)
        print('adjacent', nm, {k: v for k, v in res['adj_' + nm].items() if k != 'rows'}, flush=True)
    gaps = [abs(num[a] - num[b]) for ix in G.values() for a in ix for b in ix if a < b and vol[a] == vol[b]
            and not np.isnan(num[a]) and not np.isnan(num[b])]
    res['mate_gap_median'] = float(np.median(gaps)) if gaps else None
    # (b) cross-seal vocabulary
    seal_of = {}
    for s, ix in G.items():
        for i in ix:
            seal_of.setdefault(i, s)
    tok = sorted({t for r in R for t in r['toks']})
    ti = {t: k for k, t in enumerate(tok)}
    X = np.zeros((len(R), len(tok)), bool)
    for i, r in enumerate(R):
        for t in r['toks']:
            X[i, ti[t]] = True
    gi = np.array(sorted(seal_of)); glab = np.array([seal_of[i] for i in gi])

    def cross(Xm):
        out = []
        dfa = Xm.sum(0)
        for k in np.where((Xm[gi].sum(0) >= 2) & (dfa <= 40))[0]:
            seals = set(glab[Xm[gi, k]])
            if len(seals) >= 2:
                out.append((tok[k], len(seals), int(Xm[gi, k].sum()), int(dfa[k])))
        return out
    real = cross(X)
    stt = strata(R, 'volband')
    score = lambda c: sum(s - 1 for _, s, _, _ in c)
    nul = []
    cols = np.where(X.sum(0) <= 40)[0]
    SI = [np.where(stt == s)[0] for s in np.unique(stt)]
    for _ in range(1000):
        Xp = X.copy()
        for ii in SI:
            sub = X[np.ix_(ii, cols)]
            o = np.argsort(rng.random(sub.shape), axis=0)
            Xp[np.ix_(ii, cols)] = np.take_along_axis(sub, o, axis=0)
        nul.append(score(cross(Xp)))
    nul = np.array(nul)
    res['cross'] = dict(real=sorted(real, key=lambda x: -x[1]), score=score(real), null=float(nul.mean()),
                        p=float((nul >= score(real)).mean()))
    print('cross-seal', res['cross'], flush=True)
    # (c) without PES0329
    keep = np.array([i for i in range(len(R)) if 'PES0329' not in R[i]['seals']])
    st = strata([R[i] for i in keep], 'volband')
    y = sealed[keep].astype(int)
    for t in ('|M153+M342|', 'M340', '|M153+X|'):
        x = X[keep, ti[t]].astype(int)
        obs = int((x * y).sum())
        nl = np.array([(x * perm_within(y, st, rng)).sum() for _ in range(5000)])
        res['no329_' + t] = dict(n=int(x.sum()), sealed=obs, expected=float(nl.mean()), p=float((nl >= obs).mean()))
        print('no PES0329', t, res['no329_' + t], flush=True)
    json.dump(res, open(os.path.join(CK, 'c2b.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
