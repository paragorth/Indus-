"""pe30 cycle 3b: power of the per-feature role test (cycle 3a) on a planted
last-entry exclusion (half of the baseline-closing tablets), 200 within-tablet
permutations. Prints real / null-mean / p for P_last and the top features."""
from pe30_cycle3 import *

rng = np.random.default_rng(31)
C = load_cases(2); fl, _ = feature_list(C)
sc0 = Scorer(C, fl, pairs=False)
clos = [k for k in range(len(C)) if sc0.M[0, k]]
out = {}
for npl in (16, 8):
    pick = sorted(rng.choice(clos, npl, replace=False).tolist())
    Cp = list(C)
    for k, c in zip(pick, plant([C[k] for k in pick], lambda x: 'P_last' in x)[0]):
        Cp[k] = c
    scp = Scorer(Cp, fl, pairs=False)
    fp = [k for k in range(len(Cp)) if not scp.M[0, k]]
    pf = rescued(Cp, fp, fl)[0]
    nf = []
    for r in range(200):
        D = permute_within([Cp[k] for k in fp], rng); Df = list(Cp)
        for j, k in enumerate(fp):
            Df[k] = D[j]
        nf.append(rescued(Df, fp, fl)[0])
    nf = np.array(nf)
    tested = [i for i in range(len(fl)) if pf[i] >= 2]
    res = sorted([(fl[i], int(pf[i]), round(float(nf[:, i].mean()), 2),
                   float((1 + (nf[:, i] >= pf[i]).sum()) / 201)) for i in tested], key=lambda x: x[3])
    i = fl.index('P_last')
    out[npl] = {'P_last': (int(pf[i]), float(nf[:, i].mean()), float((1 + (nf[:, i] >= pf[i]).sum()) / 201)),
                'n_tested': len(tested), 'top': res[:5]}
    print(npl, out[npl], flush=True)
json.dump(out, open(os.path.join(DATA, 'pe30_cycle3b.json'), 'w'), indent=1)
