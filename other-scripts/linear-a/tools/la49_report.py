#!/usr/bin/env python3
"""LA-49 analysis of a dissected model population. usage: la49_report.py TAG [corpora...]
Per corpus: prediction gain, F3 sum circuit (consistency, drivers, ablation specificity),
F1/F2 number head (existence across seeds, attractor words), F4 geometry (RSA consistency,
consensus clusters, truth agreement on controls). Shuffled Linear A defines null thresholds."""
import sys, os, json, glob, collections, itertools, random
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la49_common as L
import la45_common as C45

TAG = sys.argv[1]
MINN = 3


def load(corp):
    fs = sorted(glob.glob(os.path.join(L.CK, TAG, corp + '_*.json')))
    return [json.load(open(f)) for f in fs if os.path.basename(f).split('_')[0] == corp]


def norm_driver(m, key='pre'):
    mu, sd = m['sens_mean'], max(m['sens_sd'], 1e-6)
    return {w: (s / n - mu) / sd for w, (s, n) in m[key].items() if n >= MINN}


def consistency(vecs, maxpairs=300):
    """Mean pairwise Spearman over shared keys."""
    pairs = list(itertools.combinations(range(len(vecs)), 2))
    random.Random(1).shuffle(pairs)
    rs = []
    for i, j in pairs[:maxpairs]:
        ks = sorted(set(vecs[i]) & set(vecs[j]))
        if len(ks) < 8:
            continue
        r = spearmanr([vecs[i][k] for k in ks], [vecs[j][k] for k in ks]).correlation
        if r == r:
            rs.append(r)
    return float(np.mean(rs)) if rs else float('nan'), len(rs)


def consensus(vecs):
    acc = collections.defaultdict(list)
    for v in vecs:
        for k, x in v.items():
            acc[k].append(x)
    return {k: (float(np.mean(x)), len(x)) for k, x in acc.items() if len(x) >= max(2, len(vecs) // 2)}


def num_head(m):
    best, bs = None, -9
    for h, v in m['heads'].items():
        d = v['d']
        dn = d.get('outn', 0)
        other = max(d.get('outw', 0), d.get('outl', 0))
        s = dn - other
        if s > bs:
            bs, best = s, h
    v = m['heads'][best]['d']
    spec = v.get('outn', 0) > 0.05 and v.get('outn', 0) > 2 * max(v.get('outw', 0), v.get('outl', 0), 0.0)
    return best, spec, v


def attractor(m, h):
    a = {w: s / n for w, (s, n) in m['attn'][h].items() if n >= MINN}
    if not a:
        return {}
    mu = np.mean(list(a.values()))
    return {w: x - mu for w, x in a.items()}


def sum_head(m):
    h = max(m['heads'], key=lambda k: m['heads'][k]['sensdrop'])
    d = m['heads'][h]['d']
    return h, m['heads'][h]['sensdrop'], d.get('inn', 0), d.get('outn', 0)


def rsa(models):
    keys = sorted(set.intersection(*[set(m['vec']) for m in models])) if models else []
    mats = []
    for m in models:
        X = np.array([m['vec'][k] for k in keys])
        X = X - X.mean(0)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        mats.append(X @ X.T)
    iu = np.triu_indices(len(keys), 1)
    rs = []
    for i, j in list(itertools.combinations(range(len(mats)), 2))[:300]:
        rs.append(np.corrcoef(mats[i][iu], mats[j][iu])[0, 1])
    return keys, (np.mean(mats, 0) if mats else None), (float(np.mean(rs)) if rs else float('nan'))


def clusters(keys, S, k):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    Dm = 1 - S
    np.fill_diagonal(Dm, 0)
    Dm = (Dm + Dm.T) / 2
    Z = linkage(squareform(np.clip(Dm, 0, None), checks=False), 'average')
    return dict(zip(keys, fcluster(Z, k, 'maxclust')))


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


def truth_for(corp):
    if corp.startswith('LB'):
        return C45.lb_truth(L.corpus(corp))
    if corp.startswith('UR'):
        return C45.ur_truth(L.corpus(corp))
    return {}


def analyse(corp, nullthr=None):
    ms = load(corp)
    if not ms:
        return None
    out = {'corpus': corp, 'n_models': len(ms), 'n_tf': sum(m['cfg']['arch'] == 'tf' for m in ms)}
    out['loss_n'] = float(np.mean([m['loss'].get('n', np.nan) for m in ms]))
    out['loss_w'] = float(np.mean([m['loss'].get('w', np.nan) for m in ms]))
    # F3
    for key in ('pre', 'post'):
        vs = [norm_driver(m, key) for m in ms]
        c, npair = consistency(vs)
        cons = consensus(vs)
        out['F3_' + key + '_cons'] = c
        out['F3_' + key + '_top'] = sorted(((round(v, 3), w, n) for w, (v, n) in cons.items()), reverse=True)[:15]
        out['F3_' + key + '_all'] = {w: v for w, (v, n) in cons.items()}
    tfs = [m for m in ms if m['cfg']['arch'] == 'tf']
    # F3 ablation specificity: sum head hurts top (total-like) held-out positions more than other numbers
    sh = [sum_head(m) for m in tfs]
    out['F3_abl_spec'] = float(np.mean([(a > 0.05 and a > 2 * max(b, 0)) for _, _, a, b in sh])) if sh else float('nan')
    out['F3_abl_in'] = float(np.mean([a for _, _, a, _ in sh])) if sh else float('nan')
    out['F3_abl_out'] = float(np.mean([b for _, _, _, b in sh])) if sh else float('nan')
    out['F3_sensdrop_frac'] = float(np.mean([d / max(m['top_mean'], 1e-6) for (_, d, _, _), m in zip(sh, tfs)])) if sh else float('nan')
    # F1/F2
    nh = [num_head(m) for m in tfs]
    out['F2_numhead_frac'] = float(np.mean([s for _, s, _ in nh])) if nh else float('nan')
    out['F2_numhead_dn'] = float(np.mean([v.get('outn', 0) for _, _, v in nh])) if nh else float('nan')
    av = [attractor(m, h) for m, (h, s, _) in zip(tfs, nh)]
    c, _ = consistency(av)
    out['F1_cons'] = c
    cons = consensus(av)
    out['F1_top'] = sorted(((round(v, 3), w, n) for w, (v, n) in cons.items()), reverse=True)[:15]
    out['F1_all'] = {w: v for w, (v, n) in cons.items()}
    # F4
    keys, S, r = rsa(ms)
    out['F4_rsa'] = r
    out['F4_ntypes'] = len(keys)
    if S is not None and len(keys) > 10:
        for k in (4, 6, 8):
            cl = clusters(keys, S, k)
            out['F4_k%d' % k] = cl
        tr = truth_for(corp)
        if tr:
            lab = [w for w in keys if w in tr]
            res = {}
            for k in (4, 6, 8):
                cl = out['F4_k%d' % k]
                a = ari([tr[w] for w in lab], [cl[w] for w in lab])
                rng = random.Random(2)
                null = []
                for _ in range(500):
                    t2 = [tr[w] for w in lab]
                    rng.shuffle(t2)
                    null.append(ari(t2, [cl[w] for w in lab]))
                res[k] = (round(a, 3), round(float(np.mean(np.array(null) >= a)), 4), len(lab))
            out['F4_truth_ari'] = res
            # people vs goods: PC1 of consensus similarity, AUC COMMODITY vs PERSON
            ev, evec = np.linalg.eigh(S - S.mean())
            for pc in (1, 2):
                v = evec[:, -pc]
                cm = [v[i] for i, w in enumerate(keys) if tr.get(w) == 'COMMODITY']
                pe = [v[i] for i, w in enumerate(keys) if tr.get(w) == 'PERSON']
                if cm and pe:
                    auc = np.mean([[x > y for y in pe] for x in cm])
                    out['F4_pc%d_auc_comm_person' % pc] = (round(float(max(auc, 1 - auc)), 3), len(cm), len(pe))
        del out['F4_k4'], out['F4_k8']
    return out


def rank_of(allv, words):
    order = sorted(allv, key=lambda w: -allv[w])
    return {w: (order.index(w) + 1 if w in allv else None, len(order)) for w in words}


if __name__ == '__main__':
    corps = sys.argv[2:] or ['LA', 'LASHUF0', 'LASHUF1', 'LASHUF2', 'LB', 'LB4', 'UR', 'PLANT']
    R = {}
    for c in corps:
        a = analyse(c)
        if a:
            R[c] = a
    json.dump(R, open(os.path.join(L.CK, TAG, 'report.json'), 'w'), default=str)
    for c, a in R.items():
        print('=' * 20, c, 'models', a['n_models'], 'tf', a['n_tf'], 'loss_n %.2f loss_w %.2f' % (a['loss_n'], a['loss_w']))
        print(' F3 sum: cons pre %.3f post %.3f | sum-head spec %.2f (dIn %.3f dOut %.3f) sensdrop %.2f' % (
            a['F3_pre_cons'], a['F3_post_cons'], a['F3_abl_spec'], a['F3_abl_in'], a['F3_abl_out'], a['F3_sensdrop_frac']))
        print('   PRE top', a['F3_pre_top'][:10])
        print('   POST top', a['F3_post_top'][:8])
        print(' F2 number head frac %.2f dn %.3f | F1 attractor cons %.3f' % (a['F2_numhead_frac'], a['F2_numhead_dn'], a['F1_cons']))
        print('   F1 top', a['F1_top'][:10])
        print(' F4 rsa %.3f ntypes %d' % (a['F4_rsa'], a['F4_ntypes']), {k: v for k, v in a.items() if k.startswith('F4_') and k not in ('F4_k6', 'F4_rsa', 'F4_ntypes')})
        if c.startswith('LB'):
            print('   truth ranks PRE', rank_of(a['F3_pre_all'], ['to-so', 'to-sa', 'to-so-de']))
            F1 = a['F1_all']; lg = [v for w, v in F1.items() if w.startswith('L:')]; wd = [v for w, v in F1.items() if not w.startswith('L:')]
            print('   F1 logogram AUC', round(float(np.mean([[x > y for y in wd] for x in lg])), 3) if lg and wd else None, len(lg))
        if c.startswith('UR'):
            print('   truth ranks PRE', rank_of(a['F3_pre_all'], ['szunigin']), 'POST', rank_of(a['F3_post_all'], ['szunigin']))
            tr = C45.UR_TRUTH
            F1 = a['F1_all']; lg = [v for w, v in F1.items() if tr.get(w) in ('COMMODITY', 'UNIT')]; wd = [v for w, v in F1.items() if tr.get(w) not in ('COMMODITY', 'UNIT')]
            print('   F1 commodity/unit AUC', round(float(np.mean([[x > y for y in wd] for x in lg])), 3) if lg and wd else None, len(lg))
        if c == 'PLANT':
            _, P = L.planted()
            print('   planted', P)
            print('   TOTAL ranks PRE', rank_of(a['F3_pre_all'], P['TOTAL']))
            print('   BIND ranks F1', rank_of(a['F1_all'], P['BIND']))
        if c == 'LA':
            print('   KU-RO ranks PRE', rank_of(a['F3_pre_all'], ['KU-RO', 'PO-TO-KU-RO', 'KI-RO']))
    # null threshold from shuffled corpora: 99th pct of consensus driver scores
    sh = [R[c] for c in R if c.startswith('LASHUF')]
    if sh:
        for key in ('F3_pre_all', 'F1_all'):
            allv = [v for a in sh for v in a[key].values()]
            print('NULL', key, '95%% %.3f 99%% %.3f max %.3f' % (np.quantile(allv, .95), np.quantile(allv, .99), max(allv)))
