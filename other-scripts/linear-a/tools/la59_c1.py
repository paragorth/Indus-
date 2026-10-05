#!/usr/bin/env python3
"""LA-59 cycle 1: fit the human-confusion kernel to the sign alternation graph.
usage: la59_c1.py LA|LB|LBs [nnull]
Graph = consensus over 1,000 random extraction settings (min length, position class, grouping, frequency filter,
80 % document subsample). Fit = Gibbs annealing of every sign over 18 consonant x 5 vowel states (8 restarts).
Nulls: (a) degree-preserving rewired graphs, re-fitted; (b) keys with shuffled off-diagonal values (human
structure destroyed, same values), re-fitted on the real graph. Planted control (LA): hidden random states,
graph drawn from the kernel with LA's degrees and weight (share 1.0 and 0.5). LB: values hidden in the fit;
agreement with true consonant / vowel / features scored afterwards against a permutation of the fitted states."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la59_common import *

tag = sys.argv[1]; nnull = int(sys.argv[2]) if len(sys.argv) > 2 else 12
rng = np.random.default_rng({'LA': 591, 'LB': 592, 'LBs': 593}[tag])
SC = consonant_key(); SV, _ = vowel_key(); K = kernel(SC, SV, eps=0.0)
t0 = time.time()
if tag == 'LA':
    words = la_words()
elif tag == 'LB':
    words = lb_words()
else:   # LB at LA size: random documents until LA's word count
    allw = lb_words(); docs = sorted({r['doc'] for r in allw}); rng.shuffle(docs)
    keep = set(); n = 0
    by = collections.Counter(r['doc'] for r in allw)
    for dd in docs:
        if n >= len(la_words()):
            break
        keep.add(dd); n += by[dd]
    words = [r for r in allw if r['doc'] in keep]
W = ensemble_graph(words, 1000, rng)
signs, A = graph_from_W(W, min_deg=1.0)
print(tag, len(words), 'words', len(signs), 'signs', round(A.sum() / 2, 1), 'weight', round(time.time() - t0), 's', flush=True)

def fit(A_, K_, restarts=8, sweeps=30):
    res = []
    for r in range(restarts):
        s, g, l = anneal(A_, K_, rng, sweeps)
        res.append((s, g, l))
    res.sort(key=lambda x: -x[1])
    return res

out = dict(tag=tag, signs=signs, A=A.tolist(), W={'|'.join(k): v for k, v in W.items()})
R = fit(A, K)
out['real'] = [dict(s=r[0].tolist(), g=r[1], lam=r[2]) for r in R]
print('real gains', [round(r[1], 4) for r in R], 'lam', R[0][2], round(time.time() - t0), 's', flush=True)

# massive random guessing: 200,000 random assignments (distribution of the raw score)
G = np.concatenate([gain_batch(A, rng.integers(0, 90, (20000, len(signs))), K) for _ in range(10)])
out['random'] = dict(n=len(G), max=float(G.max()), q999=float(np.quantile(G, 0.999)), mean=float(G.mean()))
print('random 200k: max', G.max(), round(time.time() - t0), 's', flush=True)

# null (b): shuffled keys
nb = []
for k in range(nnull):
    SCs = SC.copy(); SCs[:16, :16] = shuffle_offdiag(SC[:16, :16], rng)
    SVs = shuffle_offdiag(SV, rng)
    Rs = fit(A, kernel(SCs, SVs, 0.0), restarts=4)
    nb.append(Rs[0][1])
out['null_key'] = nb
print('shuffled-key gains', np.round(nb, 4), round(time.time() - t0), 's', flush=True)
# null (b2): consonant key shuffled only (vowel key real) and vowel key shuffled only
nbc, nbv = [], []
for k in range(max(4, nnull // 2)):
    SCs = SC.copy(); SCs[:16, :16] = shuffle_offdiag(SC[:16, :16], rng)
    nbc.append(fit(A, kernel(SCs, SV, 0.0), restarts=4)[0][1])
    nbv.append(fit(A, kernel(SC, shuffle_offdiag(SV, rng), 0.0), restarts=4)[0][1])
out['null_keyC'] = nbc; out['null_keyV'] = nbv
print('C-shuffled', np.round(nbc, 4), 'V-shuffled', np.round(nbv, 4), round(time.time() - t0), 's', flush=True)
# null (a): rewired graphs
na = []
for k in range(nnull):
    B = rewire(A, rng)
    na.append(fit(B, K, restarts=4)[0][1])
out['null_rewire'] = na
print('rewired gains', np.round(na, 4), round(time.time() - t0), 's', flush=True)

# planted control on this graph's degrees and weight
d = A.sum(1); ne = int(round(A.sum() / 2))
pl = []
for share in (1.0, 0.5, 0.25):
    for rep in range(3):
        st = rng.integers(0, 90, len(signs))
        B = sample_planted(d, st, K, ne, rng, share)
        Rp = fit(B, K, restarts=4)
        sf = Rp[0][0]
        tc = [C_LABELS[x // 5] for x in st]; tv = [VOWELS[x % 5] for x in st]
        pl.append(dict(share=share, g=Rp[0][1], lam=Rp[0][2], g_true=gain(B, st, K),
                       acc_c=float(np.mean([C_LABELS[x // 5] == c for x, c in zip(sf, tc)])),
                       acc_v=float(np.mean([VOWELS[x % 5] == v for x, v in zip(sf, tv)])),
                       co_c=co_assign_agreement(sf, tc, lambda x: x // 5),
                       co_v=co_assign_agreement(sf, tv, lambda x: x % 5)))
        print('planted', pl[-1], round(time.time() - t0), 's', flush=True)
out['planted'] = pl
json.dump(out, open(os.path.join(CK, f'c1_{tag}.json'), 'w'))
print('done', round(time.time() - t0), 's')
