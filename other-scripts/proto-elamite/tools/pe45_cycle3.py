"""pe45 cycle 3: do the sign classes and kin tablets replicate out of sample?
(a) Split-half: tablets split at random into halves (20 splits); the cycle-2 classifier is applied
    to each half separately; Spearman rho of P(marker) and P(god) across signs present (freq >= 3)
    in both halves.  Null: the same on PE with names resampled across tablets (kills tablet
    structure) -- a marker class that is real must replicate better than in the null.
(b) Kin tablets, size-stratified: AUC of the kin score for Ur III 'A dumu B' tablets vs other
    tablets of the same size; PE tablets against the resampled null of the same size.
(c) Prediction: signs flagged as markers in half A must co-occur on half-B tablets with the same
    partner signs more than frequency-matched non-markers ('families travel together').
usage: python3 pe45_cycle3.py -> data/pe45_ckpt/cycle3.json"""
import os, sys, json, random, pickle, collections
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe45_common import encode, load_corpora, CK  # noqa
from pe45_abc import null_across  # noqa
from pe45_signs import apply, kin_scores, auc  # noqa


def halves(C, rng):
    idx = list(range(len(C)))
    rng.shuffle(idx)
    h = len(idx) // 2
    return [C[i] for i in idx[:h]], [C[i] for i in idx[h:]]


def split_rho(clf, C, rng, nsplit=20):
    rm, rg = [], []
    for _ in range(nsplit):
        A, B = halves(C, rng)
        ka, _, _, ga, ma = apply(clf, A)
        kb, _, _, gb, mb = apply(clf, B)
        da, db = dict(zip(ka, zip(ga, ma))), dict(zip(kb, zip(gb, mb)))
        com = [e for e in da if e in db]
        if len(com) < 10:
            continue
        rg.append(spearmanr([da[e][0] for e in com], [db[e][0] for e in com])[0])
        rm.append(spearmanr([da[e][1] for e in com], [db[e][1] for e in com])[0])
    return float(np.mean(rm)), float(np.mean(rg))


def cotravel_pred(clf, C, rng, nsplit=20):
    """marker signs from half A: on half B, fraction of their tablets where another name shares a
    rare partner sign seen with them in A.  Compare to non-markers matched by frequency."""
    out = []
    for _ in range(nsplit):
        A, B = halves(C, rng)
        ka, F, _, _, ma = apply(clf, A)
        pm = dict(zip(ka, ma))
        freqA = collections.Counter(e for t in A for n in t for e in n)
        partA = collections.defaultdict(set)
        for t in A:
            es = {e for n in t for e in n}
            for e in es:
                partA[e] |= es - {e}
        tabsB = collections.defaultdict(list)
        for ti, t in enumerate(B):
            for n in t:
                for e in n:
                    tabsB[e].append(ti)
        def hit(e):
            ts = set(tabsB.get(e, []))
            if not ts:
                return None
            h = 0
            for ti in ts:
                es = {x for n in B[ti] for x in n} - {e}
                h += bool(es & partA[e])
            return h / len(ts)
        cand = [e for e in pm if e in tabsB]
        if len(cand) < 20:
            continue
        q = np.quantile([pm[e] for e in cand], 0.8)
        mk = [e for e in cand if pm[e] >= q]
        other = [e for e in cand if pm[e] < q]
        # frequency matching: for each marker, nearest-frequency non-marker
        osort = sorted(other, key=lambda e: freqA[e])
        fo = np.array([freqA[e] for e in osort])
        hm, ho = [], []
        for e in mk:
            j = int(np.clip(np.searchsorted(fo, freqA[e]), 0, len(osort) - 1))
            a, b = hit(e), hit(osort[j])
            if a is not None and b is not None:
                hm.append(a)
                ho.append(b)
        out.append(float(np.mean(hm) - np.mean(ho)))
    return float(np.mean(out)), float(np.std(out))


def strat_kin(scores_pos_neg_by_n):
    num = den = 0.0
    for n, (p, q) in scores_pos_neg_by_n.items():
        if p and q:
            a = auc(p, q)
            num += a * len(p) * len(q)
            den += len(p) * len(q)
    return num / den if den else float('nan')


def family_control(rng):
    """real Ur III families from seal patronymics (father + 2-4 sons, opaque syllables, {d} deleted)
    vs tablets of the same size drawn from the same names at random."""
    d = json.load(open(os.path.join(HERE, '..', 'data', 'pe7_corpora.json')))
    fa = collections.defaultdict(set)
    for p in d['UR3_PAT']:
        fa[tuple(p['father'])].add(tuple(p['son']))
    op = lambda n: tuple(x for x in n if x != '{d}')
    fams = [[op(f)] + [op(s) for s in sorted(v)] for f, v in fa.items() if 2 <= len(v) <= 4]
    alln = [n for f in fams for n in f]
    rnd = [[alln[rng.randrange(len(alln))] for _ in f] for f in fams]
    C, inv = encode(fams + rnd)
    lab = [1] * len(fams) + [0] * len(rnd)
    clf = pickle.load(open(os.path.join(CK, 'clf_UR3.pkl'), 'rb'))
    k, _, _, _, m = apply(clf, C)
    ks = kin_scores(C, dict(zip(k, m)))
    def share(t):
        u = list(dict.fromkeys(t)); p = h = 0
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                p += 1; h += bool(set(u[i]) & set(u[j]))
        return h / p if p else 0
    sh = [share(t) for t in C]
    return dict(n_fam=len(fams), auc_kinscore=auc([s for s, l in zip(ks, lab) if l and s is not None], [s for s, l in zip(ks, lab) if not l and s is not None]),
                auc_rawshare=auc([s for s, l in zip(sh, lab) if l], [s for s, l in zip(sh, lab) if not l]),
                share_fam=float(np.mean([s for s, l in zip(sh, lab) if l])), share_rnd=float(np.mean([s for s, l in zip(sh, lab) if not l])))


def main():
    D = load_corpora()
    rng = random.Random(4546)
    res = {'UR3_FAMILY': family_control(rng)}
    print('UR3_FAMILY', res['UR3_FAMILY'], flush=True)
    for which in ('PE', 'UR3'):
        clf = pickle.load(open(os.path.join(CK, 'clf_%s.pkl' % which), 'rb'))
        C, inv = encode([t['names'] for t in D[which]])
        r = {}
        r['split_rho_mark'], r['split_rho_god'] = split_rho(clf, C, rng)
        nulls = [split_rho(clf, null_across(C, rng), rng, nsplit=5) for _ in range(8)]
        r['null_split_rho_mark'] = [x[0] for x in nulls]
        r['null_split_rho_god'] = [x[1] for x in nulls]
        r['cotravel_pred'] = cotravel_pred(clf, C, rng)
        r['cotravel_pred_null'] = [cotravel_pred(clf, null_across(C, rng), rng, nsplit=5)[0] for _ in range(8)]
        # kin, size-stratified
        k, _, _, _, m = apply(clf, C)
        ks = kin_scores(C, dict(zip(k, m)))
        if which == 'UR3':
            byn = collections.defaultdict(lambda: ([], []))
            for t, s, raw in zip(C, ks, D[which]):
                if s is None:
                    continue
                n = min(len(set(t)), 8)
                (byn[n][0] if raw['kin'] else byn[n][1]).append(s)
            r['ur3_kin_auc_sizestrat'] = strat_kin(byn)
            r['ur3_kin_n'] = sum(len(v[0]) for v in byn.values())
        else:
            null_by_n = collections.defaultdict(list)
            for _ in range(30):
                Cn = null_across(C, rng)
                kn, _, _, _, mn = apply(clf, Cn)
                for t, s in zip(Cn, kin_scores(Cn, dict(zip(kn, mn)))):
                    if s is not None:
                        null_by_n[min(len(set(t)), 8)].append(s)
            tabs = []
            for t, s, raw in zip(C, ks, D[which]):
                if s is None:
                    continue
                n = min(len(set(t)), 8)
                nv = np.array(null_by_n[n])
                tabs.append(dict(id=raw['id'], n=len(set(t)), score=float(s), p=float(((nv >= s).sum() + 1) / (len(nv) + 1))))
            ps = np.array([x['p'] for x in tabs])
            r['pe_kin_n'] = len(tabs)
            r['pe_kin_p01'] = int((ps <= 0.01).sum())
            r['pe_kin_expected_p01'] = 0.01 * len(tabs)
            r['pe_kin_p05'] = int((ps <= 0.05).sum())
            # BH at q=0.1
            o = np.sort(ps)
            m_ = len(o)
            kk = [i for i in range(m_) if o[i] <= 0.1 * (i + 1) / m_]
            r['pe_kin_BH10'] = (kk[-1] + 1) if kk else 0
            r['pe_kin_top'] = sorted(tabs, key=lambda x: x['p'])[:25]
            # which signs drive the top kin tablets
            pm = dict(zip(k, m))
            drv = collections.Counter()
            for x in r['pe_kin_top'][:15]:
                t = C[[raw['id'] for raw in D[which]].index(x['id'])]
                u = list(dict.fromkeys(t))
                cc = collections.Counter(e for nn in u for e in set(nn))
                for e, c in cc.items():
                    if c >= 2:
                        drv[inv[e]] += pm.get(e, 0) * (c - 1) / (len(u) - 1)
            r['pe_kin_drivers'] = drv.most_common(15)
        res[which] = r
        print(which, json.dumps({a: b for a, b in r.items() if a != 'pe_kin_top'}), flush=True)
    json.dump(res, open(os.path.join(CK, 'cycle3.json'), 'w'))


if __name__ == '__main__':
    main()
