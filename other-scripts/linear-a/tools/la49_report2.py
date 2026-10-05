#!/usr/bin/env python3
"""LA-49 cycle 2 report: running-sum circuit by entry duplication (position level and by transplant),
and function-vector geometry. usage: la49_report2.py TAG"""
import sys, os, json, glob, collections, itertools, random
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la49_common as L
import la45_common as C45
from la49_report import consistency, consensus, clusters, ari, rank_of

TAG = sys.argv[1]
NB = len(L.BUCKETS)
# fv layout per direction: NB log-probs, scale sens, dup sens  -> 15; next then prev
IDX = {'next_scale': NB, 'next_dup': NB + 1, 'prev_scale': 2 * NB + 2, 'prev_dup': 2 * NB + 3}


def load(corp):
    fs = sorted(glob.glob(os.path.join(L.CK, TAG, corp + '_*.json')))
    return [json.load(open(f)) for f in fs if os.path.basename(f).split('_')[0] == corp]


def zdict(d):
    v = np.array(list(d.values()))
    mu, sd = v.mean(), v.std() + 1e-9
    return {k: (x - mu) / sd for k, x in d.items()}


def pos_driver(m, key):
    mu, sd = m['dup_mean'], max(m['dup_sd'], 1e-6)
    return {w: (s / n - mu) / sd for w, (s, n) in m[key].items() if n >= 3}


def truth_words(corp):
    if corp.startswith('LB'):
        return {'pre': ['to-so', 'to-sa'], 'post': []}
    if corp.startswith('UR'):
        return {'pre': [], 'post': ['szunigin']}
    if corp == 'PLANT':
        return {'pre': L.planted()[1]['TOTAL'], 'post': []}
    if corp.startswith('LA') and not corp.startswith('LASHUF'):
        return {'pre': ['KU-RO', 'PO-TO-KU-RO', 'KI-RO'], 'post': []}
    return {'pre': [], 'post': []}


def analyse(corp):
    ms = load(corp)
    if not ms:
        return None
    out = {'n': len(ms), 'loss_n': float(np.mean([m['loss'].get('n', np.nan) for m in ms]))}
    tw = truth_words(corp)
    # position-level duplication driver
    for key, tk in (('dpre', 'pre'), ('dpost', 'post')):
        vs = [pos_driver(m, key) for m in ms]
        out[key + '_cons'] = consistency(vs)[0]
        cs = consensus(vs)
        out[key + '_all'] = {w: v for w, (v, n) in cs.items()}
        out[key + '_top'] = sorted(((round(v, 2), w) for w, (v, n) in cs.items()), reverse=True)[:12]
        out[key + '_truth'] = rank_of(out[key + '_all'], tw[tk])
    # transplant: per-word dup sensitivity on fixed hosts
    def comp_val(v, comp):
        if comp.endswith('_E'):
            o = 0 if comp.startswith('next') else NB + 2
            p = np.exp(np.array(v[o:o + NB])); p /= p.sum()
            return float(p @ np.array(L.CENT))
        return v[IDX[comp]]
    for comp in ('next_dup', 'prev_dup', 'next_scale', 'next_E', 'prev_E'):
        vs = [zdict({w: comp_val(v, comp) for w, v in m['fv'].items()}) for m in ms]
        out['tp_' + comp + '_cons'] = consistency(vs)[0]
        cs = consensus(vs)
        out['tp_' + comp + '_all'] = {w: v for w, (v, n) in cs.items()}
        out['tp_' + comp + '_top'] = sorted(((round(v, 2), w) for w, (v, n) in cs.items()), reverse=True)[:12]
        out['tp_' + comp + '_truth'] = rank_of(out['tp_' + comp + '_all'], tw['pre'] if comp.startswith('next') else tw['post'])
    # function-vector geometry
    keys = sorted(set.intersection(*[set(m['fv']) for m in ms]))
    mats = []
    for m in ms:
        X = np.array([m['fv'][k] for k in keys])
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        mats.append(X @ X.T)
    iu = np.triu_indices(len(keys), 1)
    rs = [np.corrcoef(mats[i][iu], mats[j][iu])[0, 1] for i, j in list(itertools.combinations(range(len(mats)), 2))[:300]]
    out['fv_rsa'] = float(np.mean(rs)) if rs else float('nan')
    S = np.mean(mats, 0)
    out['fv_keys'] = keys
    if len(keys) > 10:
        cl = clusters(keys, S, 6)
        out['fv_k6'] = cl
        tr = C45.lb_truth(L.corpus(corp)) if corp.startswith('LB') else (C45.ur_truth(L.corpus(corp)) if corp.startswith('UR') else {})
        if tr:
            lab = [w for w in keys if w in tr]
            a = ari([tr[w] for w in lab], [cl[w] for w in lab])
            rng = random.Random(3); null = []
            for _ in range(500):
                t2 = [tr[w] for w in lab]; rng.shuffle(t2); null.append(ari(t2, [cl[w] for w in lab]))
            out['fv_truth_ari'] = (round(a, 3), float(np.mean(np.array(null) >= a)), len(lab))
            ev, evec = np.linalg.eigh(S - S.mean())
            best = None
            for pc in (1, 2, 3):
                v = evec[:, -pc]
                cm = [v[i] for i, w in enumerate(keys) if tr.get(w) == 'COMMODITY']
                pe = [v[i] for i, w in enumerate(keys) if tr.get(w) == 'PERSON']
                if cm and pe:
                    auc = float(np.mean([[x > y for y in pe] for x in cm]))
                    out['fv_pc%d_auc' % pc] = (round(max(auc, 1 - auc), 3), len(cm), len(pe))
        # nearest neighbours of the total word(s) in consensus function space
        anchors = tw['pre'] + tw['post']
        nn = {}
        for a in anchors:
            if a in keys:
                i = keys.index(a)
                order = np.argsort(-S[i])
                nn[a] = [(keys[j], round(float(S[i, j]), 2)) for j in order[1:9]]
        out['fv_nn'] = nn
        # sizes
        out['fv_k6_sizes'] = collections.Counter(cl.values()).most_common()
    return out


if __name__ == '__main__':
    corps = sys.argv[2:] or ['LA', 'LASHUF0', 'LASHUF1', 'LB4', 'UR4', 'PLANT']
    R = {}
    for c in corps:
        a = analyse(c)
        if a:
            R[c] = a
    json.dump(R, open(os.path.join(L.CK, TAG, 'report2.json'), 'w'), default=str)
    for c, a in R.items():
        print('=' * 16, c, 'n', a['n'], 'loss_n %.2f' % a['loss_n'])
        print(' dup PRE cons %.3f top %s' % (a['dpre_cons'], a['dpre_top'][:8]))
        print('   truth', a['dpre_truth'])
        print(' dup POST cons %.3f top %s' % (a['dpost_cons'], a['dpost_top'][:8]))
        print('   truth', a['dpost_truth'])
        for comp in ('next_dup', 'prev_dup', 'next_scale', 'next_E', 'prev_E'):
            print(' transplant %s cons %.3f top %s' % (comp, a['tp_' + comp + '_cons'], a['tp_' + comp + '_top'][:8]))
            print('   truth', a['tp_' + comp + '_truth'])
        print(' fv rsa %.3f' % a['fv_rsa'], {k: v for k, v in a.items() if k.startswith('fv_') and k not in ('fv_rsa', 'fv_keys', 'fv_k6', 'fv_nn')})
        print(' fv nn', a.get('fv_nn'))
    sh = [R[c] for c in R if c.startswith('LASHUF')]
    for key in ('dpre_all', 'dpost_all', 'tp_next_dup_all', 'tp_prev_dup_all', 'tp_next_E_all', 'tp_prev_E_all'):
        allv = [v for a in sh for v in a[key].values()]
        if allv:
            print('NULL', key, '95%% %.2f 99%% %.2f max %.2f' % (np.quantile(allv, .95), np.quantile(allv, .99), max(allv)))
