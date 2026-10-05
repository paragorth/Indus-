#!/usr/bin/env python3
"""LA-49 cycle 3b: replicate the word geometry across two independent model populations.
Population 1 = cycle-1 models (64-token window, transformers + GRUs); population 2 = cycle-2 transformers
(128-token window, other seeds and sizes; type vectors recomputed from saved weights). Per population:
consensus similarity of word-type states after regressing out simple profiles (log count, logogram flag,
P(next is number), P(prev is number), P(first)) and document co-occurrence (Jaccard). Clusters found in
population 1 (average linkage, k = 12) are tested for cohesion in population 2 (2,000 random groups of the
same size). Controls: shuffled LA (nothing should replicate), PLANT (planted totals must), LB4.
usage: LA49_MAXLEN=128 la49_c3b.py CORPUS..."""
import sys, os, json, glob, collections
os.environ.setdefault('LA49_MAXLEN', '128')
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la49_common as L
import la49_report as R
R.TAG = 'c1'
from la49_geom import profile, cooc, cohesion
from la49_report import clusters

torch.set_num_threads(1)


def pop2_vecs(corp, docs):
    voc = L.Vocab(docs)
    ch = L.chunks(docs, voc)
    out = []
    for f in sorted(glob.glob(os.path.join(L.CK, 'c2', corp + '_tf_*.json'))):
        cfg = json.load(open(f))['cfg']
        m = L.TF(len(voc.itos), cfg['d'], cfg['h'], cfg['nl'], 0.0)
        m.load_state_dict(torch.load(f.replace('.json', '.pt')))
        m.eval()
        out.append({k: v for k, v in L.type_vectors(m, ch, voc).items()})
    return out


def resid_consensus(vecs, keys, A, iu):
    res = []
    for v in vecs:
        X = np.array([v[k] for k in keys]); X = X - X.mean(0)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        s = (X @ X.T)[iu]
        b, *_ = np.linalg.lstsq(A, s, rcond=None)
        res.append(s - A @ b)
    rr = [np.corrcoef(res[i], res[j])[0, 1] for i in range(len(res)) for j in range(i + 1, len(res))]
    Rm = np.zeros((len(keys), len(keys))); Rm[iu] = np.mean(res, 0); Rm = Rm + Rm.T
    return Rm, float(np.mean(rr)) if rr else float('nan')


out = {}
for corp in sys.argv[1:]:
    docs = L.corpus(corp)
    v1 = [{k: np.array(x) for k, x in m['vec'].items()} for m in R.load(corp)]
    v2 = pop2_vecs(corp, docs)
    keys = sorted(set.intersection(*[set(v) for v in v1 + v2]))
    P = profile(docs)
    X = np.array([P[k] for k in keys])
    iu = np.triu_indices(len(keys), 1)
    Cm = cooc(docs, keys)
    A = np.column_stack([np.ones(len(iu[0]))] + [np.abs(X[:, j][:, None] - X[:, j][None, :])[iu] for j in range(X.shape[1])] + [Cm[iu]])
    R1, rsa1 = resid_consensus(v1, keys, A, iu)
    R2, rsa2 = resid_consensus(v2, keys, A, iu)
    cross = float(np.corrcoef(R1[iu], R2[iu])[0, 1])
    rng = np.random.default_rng(1)
    S1 = (R1 - R1.min()) / (R1.max() - R1.min()); np.fill_diagonal(S1, 1)
    cl = clusters(keys, S1, 12)
    grp = collections.defaultdict(list)
    for w in keys:
        grp[cl[w]].append(w)
    tested = []
    for g, ws in grp.items():
        if 3 <= len(ws) <= 40:
            c = cohesion(R2, keys, ws, rng)
            tested.append((c[1], c[0], len(ws), sorted(ws, key=lambda w: -P[w][0])[:14]))
    tested.sort()
    nt = max(1, len(tested))
    rep = [t for t in tested if t[0] * nt < 0.01]
    res = {'n1': len(v1), 'n2': len(v2), 'types': len(keys), 'resid_rsa_pop1': round(rsa1, 3), 'resid_rsa_pop2': round(rsa2, 3),
           'cross_pop_r': round(cross, 3), 'n_clusters_tested': len(tested), 'n_replicated_bonf01': len(rep), 'replicated': rep,
           'all_tested': tested}
    if corp == 'PLANT':
        tot = L.planted()[1]['TOTAL']
        res['planted_total_clusters'] = {str(k): v for k, v in collections.Counter(int(cl[w]) for w in tot if w in cl).items()}
        res['planted_total_pop2'] = cohesion(R2, keys, tot, rng)
    if corp == 'LA':
        res['la45_heading_pop2'] = cohesion(R2, keys, ['*301', 'KA', 'KU', 'SI', 'RO', 'ZE'], rng)
    res['groups'] = {str(k): v for k, v in grp.items()}
    out[corp] = res
    print('==', corp, {k: v for k, v in res.items() if k not in ('all_tested', 'replicated')})
    for t in rep:
        print('   replicated P %.4f coh %.3f n %d' % t[:3], t[3])
json.dump(out, open(os.path.join(L.CK, 'c3', 'c3b.json'), 'w'), default=str)
