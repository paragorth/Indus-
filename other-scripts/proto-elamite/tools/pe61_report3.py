#!/usr/bin/env python3
"""PE-61 cycle 3 report: the parliament (each administration's own experts) on Proto-Elamite."""
import os, sys, json, pickle, collections
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe61_common as P
L = P.L
TAG = sys.argv[1] if len(sys.argv) > 1 else 'c3'
MINOCC = 6
OUT = os.path.join(P.CK, TAG)
D = pickle.load(open(os.path.join(OUT, 'votes.pkl'), 'rb'))
V, voters = D['V'], D['voters']
CL = P.pe59_classes(); CL['MEAS+CNT'] = CL['MEASURED'] | CL['COUNTED']
TGT = {'COM': 'MEAS+CNT', 'PER': 'PERSON', 'PLA': 'PERSON', 'TRA': 'PREFIX', 'TOT': 'M288', 'HDR': 'PREFIX', 'UNI': 'MEAS+CNT'}
SH = {s: ['S%d_%d' % (int(s[1]), j) for j in range(6)] for s in ('S1', 'S2', 'S3')}
lines = []


def out(s):
    print(s, flush=True); lines.append(s)


def tab(role, src, rep, key):
    t = V[(role, src, rep)]['tabs'].get(key, {})
    return {s: v for s, (v, c) in t.items() if c >= MINOCC}


def verdict(role, key, vs, rep=0):
    tabs = [tab(role, s, rep, key) for s in vs]
    signs = sorted(set.intersection(*[set(t) for t in tabs]))
    M = np.array([[t[s] for t in tabs] for s in signs])
    top = M >= np.quantile(M, 0.9, axis=0)[None, :]
    rho = np.mean([spearmanr(M[:, i], M[:, j])[0] for i in range(len(vs)) for j in range(i + 1, len(vs))])
    return signs, M, top.sum(1), rho


res = {}
for role in L.ROLES:
    vs = voters[role]
    c = TGT[role]
    out('=== %s voters %s (pe59 target class %s)' % (role, vs, c))
    for s in vs:
        r = P.auc_sets(tab(role, s, 0, 'REAL'), CL[c])
        nl = [P.auc_sets(tab(role, s, rep, 'REAL'), CL[c]) for rep in (1, 2)]
        sh = {k: np.median([P.auc_sets(tab(role, s, 0, x), CL[c]) for x in v]) for k, v in SH.items()}
        pl = V[(role, s, 0)]['plant']; pc = V[(role, s, 0)]['pcpe']
        out('  %-3s AUC(%s) %.3f | null voters %s | S1 %.3f S2 %.3f S3 %.3f | PLANT %s | PC-in-PE-order %s' % (
            s, c, r, ' '.join('%.2f' % x for x in nl), sh['S1'], sh['S2'], sh['S3'],
            '%.2f' % np.median(pl) if pl else '-', '%.2f' % np.median(pc) if pc and s != 'PC' else ('(train)' if s == 'PC' else '-')))
    signs, M, votes, rho = verdict(role, 'REAL', vs)
    nmaj = int((votes > len(vs) / 2).sum())
    shv = {k: [verdict(role, x, vs) for x in v] for k, v in SH.items()}
    nul = [verdict(role, 'REAL', vs, rep) for rep in (1, 2)]
    out('  inter-voter rho %.3f | null voters %s | S1 %.3f S2 %.3f S3 %.3f' % (
        rho, ' '.join('%.3f' % n[3] for n in nul), *[np.median([x[3] for x in shv[k]]) for k in ('S1', 'S2', 'S3')]))
    out('  signs with a majority top-decile vote %d of %d | null voters %s | S1 med %.0f S2 med %.0f max %d S3 med %.0f max %d' % (
        nmaj, len(signs), ' '.join(str(int((n[2] > len(vs) / 2).sum())) for n in nul),
        np.median([(x[2] > len(vs) / 2).sum() for x in shv['S1']]), np.median([(x[2] > len(vs) / 2).sum() for x in shv['S2']]),
        max((x[2] > len(vs) / 2).sum() for x in shv['S2']), np.median([(x[2] > len(vs) / 2).sum() for x in shv['S3']]),
        max((x[2] > len(vs) / 2).sum() for x in shv['S3'])))
    vs_np = [s for s in vs if s != 'PC']
    s2, M2, v2, _ = verdict(role, 'REAL', vs_np)
    maj_np = {s2[i] for i in range(len(s2)) if v2[i] > len(vs_np) / 2}
    o = np.lexsort((-M.mean(1), -votes))
    def cls(t):
        return '/'.join(k[:4] for k in ('MEASURED', 'COUNTED', 'PERSON', 'PREFIX', 'M376', 'M288') if t in CL[k]) or '-'
    out('  verdict: ' + '; '.join('%s %d/%d%s [%s]' % (signs[i], votes[i], len(vs), '' if signs[i] in maj_np else ' (noPC-)', cls(signs[i]))
                                  for i in o[:16] if votes[i] > len(vs) / 2))
    res[role] = dict(majority=[signs[i] for i in o if votes[i] > len(vs) / 2], majority_noPC=sorted(maj_np), rho=rho, nmaj=nmaj)
json.dump(res, open(os.path.join(OUT, 'report.json'), 'w'), default=float)
open(os.path.join(OUT, 'report.txt'), 'w').write('\n'.join(lines) + '\n')
