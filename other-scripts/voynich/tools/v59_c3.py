import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
"""v59 cycle 3: the drawings as the Rosetta stone (image-pivot word translation, A herbal <-> B herbal).

Each word type on a side is placed at the centroid of the plant-drawing embeddings (DINOv2, from
v38) of the pages it occurs on. If A and B write the same language about the same kinds of plants,
a word shared by A and B (or an A word and its B translation) should sit at similar drawing
centroids on both sides. Spelling never enters, so an opaque encoding of one side changes nothing:
the control is Gerard's Herball (1636, OCR text + woodcuts, same network) split into two page sets
at the Voynich page counts. Null: drawings permuted among side-2 pages.
"""
import sys, json, re, collections, random
import numpy as np
from v59_lib import *

DER = os.path.join(ROOT, 'data', 'derived')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/gerard/djvu.xml'
NET = sys.argv[1] if len(sys.argv) > 1 else 'dinov2'
NPERM = 1000


def gerard_pages():
    cache = os.path.join(CK, 'gerard_words.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    import v38_lib
    vis = json.load(open(os.path.join(DER, 'v38_vis_gerard.json')))
    keys = sorted([k for k in vis if vis[k]['area'] >= 0.06], key=int)
    ocr = v38_lib.gerard_ocr(SCR, set(int(k) - 1 for k in keys))
    out = {}
    for k in keys:
        ws = [re.sub(r'[^a-z]', '', w[0].lower()) for w in ocr[int(k) - 1][2]]
        ws = [w for w in ws if len(w) >= 3]
        if len(ws) >= 30:
            out[k] = ws
    json.dump(out, open(cache, 'w'))
    return out


def emb(name):
    d = json.load(open(os.path.join(DER, f'v38_vis2_{name}.json')))
    return {k: np.array(v[NET], float) for k, v in d.items()}


def word_pages(words_by_page, keys, minpg):
    wp = collections.defaultdict(set)
    for i, k in enumerate(keys):
        for w in set(words_by_page[k]):
            wp[w].add(i)
    return {w: sorted(s) for w, s in wp.items() if len(s) >= minpg and len(s) <= 0.6 * len(keys)}


def centroids(E, wp, words):
    return np.array([E[wp[w]].mean(0) for w in words])


def prep_E(E):
    E = E - E.mean(0)
    u, s, vt = np.linalg.svd(E, full_matrices=False)
    E = u[:, :16] * s[:16]
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def mrr_identity(E1, E2, wp1, wp2, shared, cand2):
    C1 = centroids(E1, wp1, shared); C2 = centroids(E2, wp2, cand2)
    C1 = C1 - C1.mean(0); C2 = C2 - C2.mean(0)
    C1 /= np.linalg.norm(C1, axis=1, keepdims=True) + 1e-9
    C2 /= np.linalg.norm(C2, axis=1, keepdims=True) + 1e-9
    S = C1 @ C2.T
    pos = {w: j for j, w in enumerate(cand2)}
    own = np.array([S[i, pos[w]] for i, w in enumerate(shared)])
    rank = (S > own[:, None]).sum(1)
    return float(np.mean(1 / (1 + rank))), S


def test_pair(tag, W1, keys1, E1, W2, keys2, E2, minpg=3, pairs=None, rng=None):
    wp1 = word_pages(W1, keys1, minpg); wp2 = word_pages(W2, keys2, minpg)
    E1 = prep_E(np.array([E1[k] for k in keys1])); E2r = np.array([E2[k] for k in keys2])
    E2 = prep_E(E2r)
    cand2 = sorted(wp2)
    if pairs is None:
        shared = sorted(set(wp1) & set(wp2))
        src, tgt = shared, shared
    else:
        pr = [(a, b) for a, b in pairs if a in wp1 and b in wp2]
        src, tgt = [a for a, _ in pr], [b for _, b in pr]
    if len(src) < 5:
        return {'tag': tag, 'n': len(src)}
    def score(E2x):
        C1 = centroids(E1, wp1, src); C2 = centroids(E2x, wp2, cand2)
        C1 = C1 - C1.mean(0); C2 = C2 - C2.mean(0)
        C1 /= np.linalg.norm(C1, axis=1, keepdims=True) + 1e-9
        C2 /= np.linalg.norm(C2, axis=1, keepdims=True) + 1e-9
        S = C1 @ C2.T
        pos = {w: j for j, w in enumerate(cand2)}
        own = np.array([S[i, pos[w]] for i, w in enumerate(tgt)])
        rank = (S > own[:, None]).sum(1)
        return float(np.mean(1 / (1 + rank))), float(np.mean(rank < max(1, len(cand2) // 10)))
    real = score(E2)
    null = []
    for _ in range(NPERM):
        null.append(score(E2[rng.permutation(len(E2))]))
    null = np.array(null)
    z = (real[0] - null[:, 0].mean()) / (null[:, 0].std() + 1e-12)
    p = float((null[:, 0] >= real[0]).mean())
    return {'tag': tag, 'n': len(src), 'n_cand': len(cand2), 'mrr': real[0], 'top10pct': real[1],
            'null_mrr': float(null[:, 0].mean()), 'null_top10': float(null[:, 1].mean()), 'z': float(z), 'p': p}


def main():
    rng = np.random.default_rng(59)
    out = []
    # ---- Gerard control at Voynich page counts (91 vs 27) and at halves
    G = gerard_pages(); EG = emb('gerard')
    gk = [k for k in sorted(G, key=int) if k in EG]
    print('gerard pages', len(gk), flush=True)
    for r in range(6):
        perm = list(rng.permutation(len(gk)))
        k1 = [gk[i] for i in sorted(perm[:91])]; k2 = [gk[i] for i in sorted(perm[91:118])]
        res = test_pair(f'G_91v27_r{r}', G, k1, EG, G, k2, EG, rng=rng); out.append(res); print(res, flush=True)
    # opaque-encoded side 2 gives identical numbers (the pivot never reads spelling); one check:
    perm = list(rng.permutation(len(gk)))
    k1 = [gk[i] for i in sorted(perm[:91])]; k2 = [gk[i] for i in sorted(perm[91:118])]
    enc = {c: chr(0x4e00 + i) * (1 + i % 2) for i, c in enumerate('abcdefghijklmnopqrstuvwxyz')}
    G2 = {k: [''.join(enc[c] for c in w) for w in G[k]] for k in k2}
    pr = [(w, ''.join(enc[c] for c in w)) for w in set(x for k in k1 for x in G[k])]
    res = test_pair('G_91v27_opaque', G, k1, EG, G2, k2, EG, pairs=pr, rng=rng); out.append(res); print(res, flush=True)
    # unrelated-text control: Gerard words vs Voynich B drawings (no shared words possible) -> use
    # cross-drawing shuffle: Gerard side-1 words with side-2 pages whose drawings are re-assigned at random
    # ---- Voynich
    for tr in ('ZL3b', 'IT2a'):
        EV = emb('voynich')
        L = json.load(open(os.path.join(DER, tr + '_lines.json')))
        W = collections.defaultdict(list); lang = {}
        for l in L:
            if l['ltype'] == 'P' and str(l['illus']) == 'H':
                W[l['folio']] += [w for w in l['words'] if w and '?' not in w and re.fullmatch(r'[a-z]+', w)]
                lang[l['folio']] = l['lang']
        kA = [k for k in EV if lang.get(k) == 'A' and len(W[k]) >= 20]
        kB = [k for k in EV if lang.get(k) == 'B' and len(W[k]) >= 20]
        print(tr, 'A pages', len(kA), 'B pages', len(kB), flush=True)
        res = test_pair(f'V_{tr}_A_B_shared', W, kA, EV, W, kB, EV, rng=rng); out.append(res); print(res, flush=True)
        for r in range(4):
            perm = list(rng.permutation(len(kA)))
            a1 = [kA[i] for i in sorted(perm[:len(kA) // 2])]; a2 = [kA[i] for i in sorted(perm[len(kA) // 2:])]
            res = test_pair(f'V_{tr}_A_A_r{r}', W, a1, EV, W, a2, EV, rng=rng); out.append(res); print(res, flush=True)
            perm = list(rng.permutation(len(kA)))
            a1 = [kA[i] for i in sorted(perm[:min(64, len(kA) - 27)])]; a2 = [kA[i] for i in sorted(perm[-27:])]
            res = test_pair(f'V_{tr}_A_A27_r{r}', W, a1, EV, W, a2, EV, rng=rng); out.append(res); print(res, flush=True)
        # context-aligner pairs (cycle 2b recurrent non-identical pairs) through the image pivot
        f = os.path.join(CK, 'c2b_V_AB.json' if tr == 'ZL3b' else 'c2b_V_AB_IT.json')
        if os.path.exists(f):
            rec = [(a, b) for a, b, c in json.load(open(f))['recur'] if a != b]
            res = test_pair(f'V_{tr}_ctxpairs_nonid', W, kA, EV, W, kB, EV, minpg=2, pairs=rec, rng=rng); out.append(res); print(res, flush=True)
    json.dump(out, open(os.path.join(CK, f'c3_{NET}.json'), 'w'))


if __name__ == '__main__':
    main()
