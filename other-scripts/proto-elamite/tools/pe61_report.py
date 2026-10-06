#!/usr/bin/env python3
"""PE-61 report for a votes.pkl (cycle 1 arm A or cycle 2 arm B): role votes on Proto-Elamite signs, agreement
with the pe59 frozen classes, against label-permuted training nulls, S1/S2/S3 shuffles and a dumb baseline
(the single feature t_adjN: how often the sign stands next to a numeral).
Usage: pe61_report.py tag [minocc]"""
import os, sys, json, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe61_common as P
L = P.L
TAG = sys.argv[1]
MINOCC = int(sys.argv[2]) if len(sys.argv) > 2 else 6
OUT = os.path.join(P.CK, TAG)
V = pickle.load(open(os.path.join(OUT, 'votes.pkl'), 'rb'))
R = V['R']
CL = P.pe59_classes()
CL['MEAS+CNT'] = CL['MEASURED'] | CL['COUNTED']
PE = pickle.load(open(os.path.join(P.CK, 'pe_feats.pkl'), 'rb'))
lines = []


def out(s):
    print(s, flush=True); lines.append(s)


def table(key, rep, role):
    ms = [R[k][key] for k in R if k[0] == role and k[2] == rep and R[k] is not None and key in R[k]]
    if not ms:
        return None, None
    signs = [t for t, (v, c) in ms[0].items() if c >= MINOCC]
    M = np.array([[m[t][0] for m in ms] for t in signs])        # signs x models
    return signs, M


def stats(signs, M):
    tab = dict(zip(signs, M.mean(1)))
    s = {}
    for c in ('MEAS+CNT', 'MEASURED', 'COUNTED', 'PERSON', 'PREFIX'):
        s[c] = P.auc_sets(tab, CL[c])
    rk = {t: r for r, t in enumerate(sorted(tab, key=lambda t: -tab[t]))}
    for c in ('M376', 'M288'):
        s[c] = 1 - rk['M376' if c == 'M376' else 'M288'] / max(1, len(rk) - 1) if c in rk else float('nan')
    s['sd'] = float(np.std(M.mean(1)))
    top = M >= np.quantile(M, 0.9, axis=0)[None, :]
    s['maj'] = float(np.mean(top.mean(1) >= 0.5))
    return s, tab, top.mean(1)


def baseline(key):
    bl = PE[key]
    acc, cnt = collections.defaultdict(float), collections.Counter()
    for b in bl:
        for t, v in zip(b['types'], b['X'][:, L.FI['t_adjN']]):
            acc[t] += v; cnt[t] += 1
    tab = {t: acc[t] / cnt[t] for t in acc if cnt[t] >= MINOCC}
    return {c: P.auc_sets(tab, CL[c]) for c in ('MEAS+CNT', 'MEASURED', 'COUNTED', 'PERSON', 'PREFIX')}


roles = sorted({k[0] for k in R}, key=lambda r: L.ROLES.index(r))
nrep = max(k[2] for k in R)
out('PE-61 %s report (signs with >= %d occurrences over 3 partitions)' % (TAG, MINOCC))
b = baseline('REAL')
out('dumb baseline t_adjN on PE: ' + ' '.join('%s %.3f' % kv for kv in b.items()))
summary = {}
cols = ['MEAS+CNT', 'MEASURED', 'COUNTED', 'PERSON', 'PREFIX', 'M376', 'M288']
for role in roles:
    signs, M = table('REAL', 0, role)
    if signs is None:
        continue
    s, tab, agree = stats(signs, M)
    nul = [stats(*table('REAL', r, role))[0] for r in range(1, nrep + 1) if table('REAL', r, role)[0]]
    sh = {tag: [stats(*table('%s_%d' % (tag, j), 0, role))[0] for j in range(P.NSHUF)] for tag in ('S1', 'S2', 'S3')}
    pl = [a for k in R if k[0] == role and k[2] == 0 and R[k] for a in R[k]['PLANT']]
    out('=== %s: %d models (systems %s) | PLANT AUC median %.3f' % (role, M.shape[1], V['elig'][role], np.median(pl) if pl else float('nan')))
    for c in cols:
        nv = [x[c] for x in nul]
        out('  %-8s real %.3f | label-perm null med %.3f max %.3f | S1 med %.3f q95 %.3f | S2 med %.3f q95 %.3f | S3 med %.3f q95 %.3f' % (
            c, s[c], np.median(nv) if nv else np.nan, max(nv) if nv else np.nan,
            np.median([x[c] for x in sh['S1']]), np.quantile([x[c] for x in sh['S1']], 0.95),
            np.median([x[c] for x in sh['S2']]), np.quantile([x[c] for x in sh['S2']], 0.95),
            np.median([x[c] for x in sh['S3']]), np.quantile([x[c] for x in sh['S3']], 0.95)))
    for nm in ('PC', 'PCPE'):
        if any(nm in (R[k] or {}) for k in R if k[0] == role):
            def ens(rep):
                v = [np.mean(R[k][nm]) for k in R if k[0] == role and k[2] == rep and R[k]]
                return np.median(v) if v else np.nan, (np.mean(np.array(v) >= 0.6) if v else np.nan)
            nv = [ens(r)[0] for r in range(1, nrep + 1)]
            out('  held-out %-4s per-model AUC median %.3f (share >= 0.6 %.2f) | label-perm null medians %s' % (
                nm, ens(0)[0], ens(0)[1], ' '.join('%.3f' % x for x in nv)))
    out('  spread sd %.4f (S1 med %.4f, S2 %.4f, S3 %.4f) | share signs with majority top-decile %.3f (S1 %.3f S2 %.3f S3 %.3f)' % (
        s['sd'], np.median([x['sd'] for x in sh['S1']]), np.median([x['sd'] for x in sh['S2']]), np.median([x['sd'] for x in sh['S3']]),
        s['maj'], np.median([x['maj'] for x in sh['S1']]), np.median([x['maj'] for x in sh['S2']]), np.median([x['maj'] for x in sh['S3']])))
    o = np.argsort(-M.mean(1))
    def cls(t):
        return '/'.join(c[:4] for c in ('MEASURED', 'COUNTED', 'PERSON', 'PREFIX', 'M376', 'M288') if t in CL[c]) or '-'
    out('  top: ' + '; '.join('%s %.3f ag%.2f [%s]' % (signs[i], M[i].mean(), agree[i], cls(signs[i])) for i in o[:14]))
    summary[role] = dict(real=s, null=nul, sh={k: v for k, v in sh.items()}, plant=pl,
                         table={t: [float(tab[t]), float(a)] for t, a in zip(signs, agree)})
json.dump(summary, open(os.path.join(OUT, 'report.json'), 'w'), default=float)
open(os.path.join(OUT, 'report.txt'), 'w').write('\n'.join(lines) + '\n')
