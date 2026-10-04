"""pe31 cycle 4: CLOSED-FRAME SWAPS.  Observation from the controls (Ur III, OB, Linear B): a
spelling-variant swap happens inside a long, specific frame (few other signs ever fill it, not the
final slot), while slot-mate swaps (lu2/ur, suen/utu) happen in short open frames that take many
fillers.  Score CF(X,Y) = sum over shared frames of 1/(fillers-1) * [string length >= 3] *
[slot not final].  Calibrate (AUC, top-K hits) on the same truth sets; z against frequency-matched
pairs; within-string shuffle null for the count of pairs with CF >= the planted median; then run PE."""
import json, os, random, sys
from collections import defaultdict
from itertools import combinations
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe31_lib as L

MINC = 4


def cf(data, alph):
    A = set(alph); idx = {g: i for i, g in enumerate(alph)}
    fr = defaultdict(set)
    for w in {w for w, _ in data}:
        if len(w) < 3:
            continue
        for i, s in enumerate(w[:-1]):
            if s in A:
                fr[w[:i] + ('_',) + w[i + 1:]].add(s)
    M = np.zeros((len(alph), len(alph)))
    for v in fr.values():
        if len(v) > 1:
            wt = 1.0 / (len(v) - 1)
            for a, b in combinations(v, 2):
                M[idx[a], idx[b]] += wt; M[idx[b], idx[a]] += wt
    return M


def evaluate(data, T, label):
    alph, cnt = L.alphabet(data, MINC)
    n = len(alph); iu = np.triu_indices(n, 1)
    M = cf(data, alph)
    lab = np.array([tuple(sorted((alph[i], alph[j]))) in T for i, j in zip(*iu)])
    x = M[iu] + np.random.default_rng(0).random(len(iu[0])) * 1e-6
    K = max(10, int(lab.sum()))
    top = np.argsort(-x)[:K]
    nz = x > 1e-3
    return dict(corpus=label, n_truth=int(lab.sum()), auc=round(L.auc(x, lab), 3), hits_topK=int(lab[top].sum()), K=K,
                precision_nonzero=round(float(lab[nz].mean()), 4) if nz.any() else 0, base_rate=round(float(lab.mean()), 4),
                truth_nonzero=int((lab & nz).sum()), n_nonzero=int(nz.sum())), (alph, cnt, M)


def main():
    out = []
    for nm in ('UR3_SEAL', 'OB_SEAL', 'LINB'):
        d = L.control(nm); alph, _ = L.alphabet(d, MINC)
        r, _ = evaluate(d, L.truth_pairs(nm, alph), nm); out.append(r); print(json.dumps(r), flush=True)
    pe = L.pe_names(False)
    a0, c0 = L.alphabet(pe, 1); cand = [s for s in a0 if 16 <= c0[s] <= 80]
    for mode in ('token', 'tablet'):
        for rep in range(3):
            rng = random.Random(900 + rep); ps = set(rng.sample(cand, 10))
            r, _ = evaluate(L.plant(pe, ps, mode, rng), {tuple(sorted((s, s + "'"))) for s in ps}, f'plant-{mode}-{rep}')
            out.append(r); print(json.dumps(r), flush=True)
    # real PE, base and variant alphabets, with within-string shuffle null of the top score mass
    real = {}
    for nm, d in (('PE_base', pe), ('PE_var', L.pe_names(True))):
        alph, cnt = L.alphabet(d, MINC); M = cf(d, alph); n = len(alph); iu = np.triu_indices(n, 1); x = M[iu]
        rng = random.Random(5); sh_top, sh_nz = [], []
        for _ in range(200):
            Ms = cf(L.shuffle_within(d, rng), alph)[iu]
            sh_top.append(np.sort(Ms)[-20:].sum()); sh_nz.append((Ms > 1e-3).sum())
        # per-pair search-corrected: count of shuffles whose max >= the pair's value
        mx = []
        rng = random.Random(6)
        for _ in range(200):
            mx.append(cf(L.shuffle_within(d, rng), alph)[iu].max())
        mx = np.array(mx)
        order = np.argsort(-x)[:20]
        pairs = [dict(a=alph[iu[0][t]], b=alph[iu[1][t]], na=cnt[alph[iu[0][t]]], nb=cnt[alph[iu[1][t]]], CF=round(float(x[t]), 2),
                      p_fw=round(float((1 + (mx >= x[t]).sum()) / 201), 3), same_base=L.base(alph[iu[0][t]]) == L.base(alph[iu[1][t]]))
                 for t in order]
        same = np.array([L.base(alph[i]) == L.base(alph[j]) for i, j in zip(*iu)])
        real[nm] = dict(n_signs=n, nonzero=int((x > 1e-3).sum()), top20_sum=round(float(np.sort(x)[-20:].sum()), 2),
                        shuffle_top20_sum=round(float(np.mean(sh_top)), 2), p_top20=float((1 + (np.array(sh_top) >= np.sort(x)[-20:].sum()).sum()) / 201),
                        shuffle_nonzero=round(float(np.mean(sh_nz)), 1),
                        same_base_share_of_nonzero=round(float(same[x > 1e-3].mean()), 4), same_base_base_rate=round(float(same.mean()), 4),
                        pairs=pairs)
        print(nm, json.dumps({k: v for k, v in real[nm].items() if k != 'pairs'}), flush=True)
    json.dump(dict(calib=out, real=real), open(os.path.join(L.CK, 'cycle4.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
