"""v64 cycle 3: wheel and table rules between neighbouring words.

With each corpus's climbed concept alphabet (cycle 2; for the controls also the true
alphabet), measure along each line:
  WIN3   mean number of distinct concepts in 3 consecutive words (a tabula column
         keeps a run of words inside one small concept set -> low)
  ROT    for adjacent words with the same number of concepts (>=2), at each slot the
         step rank(next)-rank(prev) mod k in the canonical order; ROT = share of the
         commonest non-zero step pooled over slots (a turning wheel -> high)
  SHARE  mean Jaccard of the concept sets of adjacent words
Each is compared with a within-line word shuffle null (200 shuffles): z = (obs-null)/sd.
Voynich ring and circular texts (ltypes R, C) are measured the same way, with
paragraph text as comparison, and the ring texts' own within-ring shuffle as null.
"""
import sys, json, random, os, time, math
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v64_lib as V


def stats(lines, parse, rank, k):
    win, nwin = 0, 0
    steps = Counter(); nst = 0
    share, nsh = 0, 0
    for L in lines:
        S = [parse(w) for w in L['words']]
        for i in range(len(S) - 2):
            u = set(S[i]) | set(S[i + 1]) | set(S[i + 2])
            if u:
                win += len(u); nwin += 1
        for a, b in zip(S, S[1:]):
            if a and b:
                sa, sb = set(a), set(b)
                share += len(sa & sb) / len(sa | sb); nsh += 1
            if len(a) == len(b) >= 2:
                for j, (x, y) in enumerate(zip(a, b)):
                    st = (rank[y] - rank[x]) % k
                    steps[(j, st)] += 1; nst += 1
    nz = Counter()
    for (j, st), c in steps.items():
        if st:
            nz[st] += c
    rot = (max(nz.values()) / nst) if nst and nz else 0
    return {'WIN3': win / max(1, nwin), 'ROT': rot, 'SHARE': share / max(1, nsh),
            'nwin': nwin, 'nst': nst}


def shuffled(lines, rng):
    out = []
    for L in lines:
        ws = list(L['words']); rng.shuffle(ws); out.append({'words': ws})
    return out


def test(lines, alpha, rng, nsh=200):
    parse = V.make_parser(alpha); k = len(alpha)
    rank = V.canonical_rank(lines, parse, k)
    obs = stats(lines, parse, rank, k)
    null = [stats(shuffled(lines, rng), parse, rank, k) for _ in range(nsh)]
    res = {'obs': obs}
    for key in ('WIN3', 'ROT', 'SHARE'):
        v = [n[key] for n in null]; m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** .5 or 1e-9
        res[key] = (round(obs[key], 4), round(m, 4), round((obs[key] - m) / sd, 2))
    return res


if __name__ == '__main__':
    D = V.all_corpora(); C = D['corpora']
    rng = random.Random(64)
    out = {}
    for k in ['ars', 'med', 'lat', 'voy', 'voyit', 'voy_mk2', 'voy_gshuf', 'voy_sc']:
        c2 = json.load(open(os.path.join(V.CK, 'c2_%s.json' % k)))
        out[k] = {'climbed': test(C[k], c2['alpha'], rng)}
        if k in D['codes']:
            out[k]['truth'] = test(C[k], list(D['codes'][k].values()), rng)
        print(k, out[k], flush=True)
    # ring / circular texts
    alpha = json.load(open(os.path.join(V.CK, 'c2_voy.json')))['alpha']
    rings = V.voynich('ZL3b', ltypes=('R', 'C'))
    out['voy_rings'] = test(rings, alpha, rng)
    # a paragraph sample of the same size for comparison
    par = C['voy']; n = sum(len(L['words']) for L in rings)
    samp, tot = [], 0
    for L in random.Random(1).sample(par, len(par)):
        samp.append(L); tot += len(L['words'])
        if tot >= n:
            break
    out['voy_par_sample'] = test(samp, alpha, rng)
    print('rings', out['voy_rings'], '\npar', out['voy_par_sample'])
    json.dump(out, open(os.path.join(V.CK, 'c3.json'), 'w'))
