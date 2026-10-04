"""v45 cycle 1: does the residual (the word choices the rule model cannot predict) carry content?

(a) Calibration of the residual: PIT uniformity and surprisal minus model entropy per corpus.
(b) Pair classes: residual-usage similarity of pages that share a label (section, hand, quire, bifolio;
    Latin chapter) against pages that do not, compared with raw tf-idf similarity. Null: labels permuted
    within section (within book for Latin). Same pipeline on the generator (GEN), the richer generator (PL)
    and the planted Latin encodings (LAw, LAl).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v45_lib import *

FN = 'v45_cycle1.txt'
NP = 1000


def calib(name):
    C = get_corpus(name); R = residual(name)
    rec = [r for p in C if p['id'] in R for r in R[p['id']]['recs']]
    u = np.array([r[8] for r in rec if r[8] >= 0])
    oov = np.mean([r[8] < 0 for r in rec])
    ks = float(np.max(np.abs(np.sort(u) - (np.arange(len(u)) + 0.5) / len(u))))
    hist = np.histogram(u, bins=10, range=(0, 1))[0] / len(u)
    ex = np.mean([-r[5] - r[6] for r in rec if r[8] >= 0])
    return dict(n=len(rec), oov=float(oov), ks=ks, hist=hist.round(3).tolist(), excess=float(ex),
                surpr=float(np.mean([-r[5] for r in rec])), H=float(np.mean([r[6] for r in rec])))


def pairtest(S, labels, strata, rng, mask_extra=None):
    """mean S over same-label pairs minus different-label pairs, both inside the same stratum."""
    labels = [None if x is None or x == (None, None) else str(x) for x in labels]
    strata = [str(x) for x in strata]
    n = len(labels)
    iu = np.triu_indices(n, 1)
    st = np.array(strata)
    same_st = (st[:, None] == st[None, :])[iu]
    keep = same_st & (np.ones(len(iu[0]), bool) if mask_extra is None else mask_extra[iu])
    s = S[iu][keep]

    def stat(lab):
        lab = np.array(lab + [None], dtype=object)[:-1]
        eq = (lab[:, None] == lab[None, :])[iu][keep]
        ok = np.array([x is not None for x in lab])
        okp = (ok[:, None] & ok[None, :])[iu][keep]
        a, b = s[eq & okp], s[~eq & okp]
        if len(a) < 3 or len(b) < 3: return np.nan, 0
        return a.mean() - b.mean(), len(a)

    obs, na = stat(labels)
    null = []
    for _ in range(NP):
        lab = list(labels)
        for v in set(strata):
            ix = [i for i in range(n) if strata[i] == v]
            perm = rng.permutation(ix)
            for a, b in zip(ix, perm): lab[a] = labels[b]
        null.append(stat(lab)[0])
    null = np.array(null)
    return dict(d=float(obs), n=int(na), z=float((obs - np.nanmean(null)) / (np.nanstd(null) + 1e-12)))


def topic_tests(name, rng, M):
    C = get_corpus(name); R = residual(name)
    ids, voc, Om, Em, Rm = resid_matrix(C, R)
    pg = {p['id']: p for p in C}
    sims = dict(raw=cos(tfidf(Om)), resid=cos(Rm))
    out = {}
    if name.startswith('LA'):
        secs = [pg[i]['sec'] for i in ids]
        chap = [pg[i]['chap'] for i in ids]
        num = np.array([int(i[2:]) for i in ids])
        nonadj = np.abs(num[:, None] - num[None, :]) > 1
        for k, S in sims.items():
            out[k + '|chapter'] = pairtest(S, chap, secs, rng)
            out[k + '|chapter_nonadj'] = pairtest(S, chap, secs, rng, nonadj)
            out[k + '|section'] = pairtest(S, secs, ['all'] * len(ids), rng)
    else:
        secs = [pg[i]['sec'] for i in ids]
        md = [M.get(i, {}) for i in ids]
        labs = dict(section=(secs, ['all'] * len(ids)),
                    hand=([m.get('hand') for m in md], secs),
                    quire=([m.get('quire') for m in md], secs),
                    bifolio=([(m.get('quire'), m.get('bifolio')) if m.get('bifolio') else None for m in md], secs),
                    illus=([m.get('illus') for m in md], [m.get('lang') for m in md]))
        for k, S in sims.items():
            for lk, (lab, st) in labs.items():
                out[k + '|' + lk] = pairtest(S, lab, st, rng)
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(451)
    M = meta()
    names = sys.argv[1:] or ['V', 'VI', 'GEN0', 'GEN1', 'PL', 'LAw', 'LAl']
    res = {}
    for nm in names:
        if not os.path.exists(os.path.join(CK, f'resid_{nm}.json')): print('missing', nm); continue
        c = calib(nm); t = topic_tests(nm, rng, M)
        res[nm] = dict(calib=c, topic=t); jsave(f'c1_{nm}.json', res[nm])
        print(nm, json.dumps(c), flush=True)
        for k, v in t.items(): print('  ', k, '%+.4f n=%d z %+.1f' % (v['d'], v['n'], v['z']), flush=True)
