"""pe31 cycle 1: calibration.  Which interchangeability statistic recovers KNOWN
homophone / doublet pairs (Ur III and Old Babylonian seal names: same sign value
with a different index; Linear B doublets) and PLANTED homophones in the PE
middles (one sign split in two, per token or per tablet)?  The statistic is
frozen here, before any real PE pair is looked at."""
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe31_lib as L

MINC = 6
WEIGHTS = {'MP': (1, 0, 0, 0), 'CTX': (0, 1, 0, 0), 'POS': (0, 0, 1, 0), 'TAB-': (0, 0, 0, -1),
           'MP+CTX': (1, 1, 0, 0), 'MP+CTX+POS': (1, 1, 1, 0), 'ALL': (1, 1, 1, -1)}


def evaluate(data, truth_fn, label):
    alph, cnt = L.alphabet(data, MINC)
    Sd = L.scores(data, alph)
    T = truth_fn(alph)
    n = len(alph); iu = np.triu_indices(n, 1)
    lab = np.array([tuple(sorted((alph[i], alph[j]))) in T for i, j in zip(*iu)])
    res = dict(corpus=label, n_signs=n, n_truth=int(lab.sum()), n_pairs=len(lab))
    for k, w in WEIGHTS.items():
        Z = L.combo(Sd, cnt, alph, w)[iu]
        top = np.argsort(-Z)[:max(10, int(lab.sum()))]
        res[k] = dict(auc=round(L.auc(Z, lab), 3), hits_topK=int(lab[top].sum()),
                      K=len(top), mean_rank_pct=round(float(np.mean([(Z > z).mean() for z in Z[lab]])) if lab.any() else -1, 3))
    return res


def main():
    out = []
    for name in ('UR3_SEAL', 'OB_SEAL', 'LINB'):
        d = L.control(name)
        r = evaluate(d, lambda A, nm=name: L.truth_pairs(nm, A), name)
        out.append(r); print(json.dumps(r), flush=True)
    pe = L.pe_names(variants=False)
    alph, cnt = L.alphabet(pe, 1)
    cand = [s for s in alph if 16 <= cnt[s] <= 80]
    for mode in ('token', 'tablet'):
        for rep in range(3):
            rng = random.Random(100 + rep)
            ps = set(rng.sample(cand, 10))
            d = L.plant(pe, ps, mode, rng)
            r = evaluate(d, lambda A, ps=ps: {tuple(sorted((s, s + "'"))) for s in ps if s + "'" in A and s in A},
                         f'PE-plant-{mode}-{rep}')
            out.append(r); print(json.dumps(r), flush=True)
    # Same-name test in the controls: are the known homophone pairs even visible as minimal pairs?
    json.dump(out, open(os.path.join(L.CK, 'cycle1.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
