"""pe35 cycle 2: the frozen pe33 leads, tested on NEW tablets only (pre-registered in loops/pe35_cycle1.txt).
H1 caprid-only vs bovid-only largest bare count; H2 M139 on predation-sealed tablets; H3 M054 on boat-sealed tablets;
H4 predation / boat seals share vocabulary across seals (pe33.2 statistic, seal-block null)."""
import json, os, sys, collections
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact
from pe35_common import load, CK, DATA
from pe33_cycle4 import maxcount
from pe33_cycle2 import vecs, run
from pe33_common import perm_seal

rows, T = load('new')
corp = {t['id']: t for t in json.load(open(os.path.join(DATA, 'pe_corpus.json')))}
out = dict(n_new=len(rows), seals=len({r['seal'] for r in rows}))
mc = {r['id']: maxcount(r, corp[r['id']]) for r in rows}
cap = [mc[r['id']] for r in rows if 'CAPRID' in r['motif'] and 'BOVID' not in r['motif']]
bov = [mc[r['id']] for r in rows if 'BOVID' in r['motif'] and 'CAPRID' not in r['motif']]
h1 = dict(n_caprid=len(cap), n_bovid=len(bov), med_caprid=float(np.median(cap)) if cap else None,
          med_bovid=float(np.median(bov)) if bov else None)
if cap and bov:
    h1['pairs_in_order'] = float(np.mean([c > b for c in cap for b in bov]) + 0.5 * np.mean([c == b for c in cap for b in bov]))
    h1['p_mw'] = float(mannwhitneyu(cap, bov, alternative='greater').pvalue)
out['H1'] = h1


def has(r, s):
    return ('S:' + s) in r['content']


for key, mot, sign in (('H2', 'PREDATION', 'M139'), ('H3', 'WATER', 'M054')):
    a = [r for r in rows if mot in r['motif']]; b = [r for r in rows if mot not in r['motif']]
    k1, k2 = sum(has(r, sign) for r in a), sum(has(r, sign) for r in b)
    res = dict(n_motif=len(a), with_sign=k1, n_other=len(b), other_with_sign=k2, ids=[r['id'] for r in a],
               seals=sorted({r['seal'] for r in a}))
    if a and b:
        res['p_fisher'] = float(fisher_exact([[k1, len(a) - k1], [k2, len(b) - k2]], alternative='greater')[1])
    out[key] = res
# H4 cross-seal vocabulary sharing on new tablets
rng = np.random.default_rng(352)
mot = [m for m in ('PREDATION', 'WATER', 'BOVID', 'CAPRID', 'FELINE', 'ANTHRO')
       if len({r['seal'] for r in rows if m in r['motif']}) >= 2 and sum(m in r['motif'] for r in rows) <= len(rows) - 2]
if mot:
    S = vecs([r['content'] for r in rows])
    real, p, pp, _ = run(rows, S, mot, 2000, rng, lambda Y: perm_seal(Y, rows, rng))
    out['H4'] = dict(motifs=mot, real=[float(x) for x in real], p=[float(x) for x in p], pooled_p=pp)
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(f'{CK}/cycle2.json', 'w'), indent=1, default=str)
