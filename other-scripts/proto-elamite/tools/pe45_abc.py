"""pe45 cycle 1: rejection ABC over the naming-grammar bank.
Fit on block A statistics (frequency, slot concentration, recurrence, same/cross-tablet sharing);
check the survivors on HELD-OUT block B (per-sign slot fixity, positional sharing, cliques,
co-travel, dispersion, shared-sign network).
Targets: PE; PE null 1 (signs shuffled within names); PE null 2 (names resampled across tablets);
UR3 (opaque Ur III names, own bank); 40 planted grammars drawn from the prior (truth known).
usage: python3 pe45_abc.py  -> data/pe45_ckpt/abc.json"""
import os, sys, glob, json, random, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe45_common import (prior, simulate, stats, encode, load_corpora, PNAMES, SNAMES, SA, SB, CK)  # noqa

NP = len(PNAMES)
IA = [SNAMES.index(k) for k in SA]
IB = [SNAMES.index(k) for k in SB]


def load_bank(which):
    fs = sorted(glob.glob(os.path.join(CK, 'bank_' + which, 'c*.npy')))
    fs = [f for f in fs if not f.endswith('.tmp.npy')]
    M = np.concatenate([np.load(f) for f in fs]).astype(np.float64)
    M = M[np.all(np.isfinite(M), axis=1)]
    return M[:, :NP], M[:, NP:]


def scale(S):
    med = np.median(S, 0)
    mad = np.median(np.abs(S - med), 0) * 1.4826
    mad[mad < 1e-6] = np.std(S, 0)[mad < 1e-6] + 1e-6
    return med, mad


def abc(P, S, obs, k=200, exclude=None):
    med, mad = scale(S)
    Z = (S - med) / mad
    zo = (obs - med) / mad
    dA = np.sqrt(((Z[:, IA] - zo[IA]) ** 2).sum(1))
    if exclude is not None:
        dA[exclude] = np.inf
    acc = np.argsort(dA)[:k]
    # held-out: observed B vs accepted B
    ZB = Z[acc][:, IB]
    pct = [(float((ZB[:, j] < zo[IB][j]).mean())) for j in range(len(IB))]
    dB_obs = float(np.sqrt(((np.median(ZB, 0) - zo[IB]) ** 2).sum()))
    # reference: how far an accepted sim's own B lies from the accepted median (posterior-predictive spread)
    dB_ref = np.sqrt(((ZB - np.median(ZB, 0)) ** 2).sum(1))
    post = {PNAMES[j]: [float(np.percentile(P[acc, j], q)) for q in (5, 50, 95)] for j in range(NP)}
    return dict(acc=acc.tolist(), dA_max=float(dA[acc[-1]]), post=post,
                heldout_pct=dict(zip(SB, pct)), dB_obs=dB_obs,
                dB_ref_p=float((dB_ref >= dB_obs).mean()), dB_ref_med=float(np.median(dB_ref)))


def null_within(C, rng):
    out = []
    for t in C:
        tt = []
        for n in t:
            n = list(n)
            rng.shuffle(n)
            tt.append(tuple(n))
        out.append(tt)
    return out


def null_across(C, rng):
    allm = [n for t in C for n in t]
    rng.shuffle(allm)
    out, k = [], 0
    for t in C:
        out.append(allm[k:k + len(t)])
        k += len(t)
    return out


def main():
    D = load_corpora()
    res = {}
    P, S = load_bank('PE')
    print('PE bank', len(P), flush=True)
    C, _ = encode([t['names'] for t in D['PE']])
    sizes = [len(t) for t in C]
    lens = [len(n) for t in C for n in t]
    obs = stats(C)
    res['PE'] = abc(P, S, obs)
    res['PE']['obs'] = obs.tolist()
    rng = random.Random(45)
    for nm, f in (('NULL_within', null_within), ('NULL_across', null_across)):
        reps = []
        for r in range(5):
            o = stats(f(C, rng), seed=r)
            reps.append(abc(P, S, o))
        res[nm] = reps
    # planted: grammars from the prior, simulated on the PE template, fitted against the same bank
    pl = []
    for r in range(40):
        th = prior(rng)
        Cs, _, _ = simulate(th, sizes, lens, rng)
        a = abc(P, S, stats(Cs, seed=r))
        a['truth'] = th
        del a['acc']
        pl.append(a)
    res['PLANT'] = pl
    PU, SU = load_bank('UR3')
    print('UR3 bank', len(PU), flush=True)
    CU, _ = encode([t['names'] for t in D['UR3']])
    o = stats(CU)
    res['UR3'] = abc(PU, SU, o)
    res['UR3']['obs'] = o.tolist()
    res['UR3_NULL_across'] = [abc(PU, SU, stats(null_across(CU, rng), seed=r)) for r in range(5)]
    json.dump(res, open(os.path.join(CK, 'abc.json'), 'w'))


def report():
    R = json.load(open(os.path.join(CK, 'abc.json')))
    def line(nm, a):
        p = a['post']
        keys = ['p_god', 'ng', 'p_mark', 'kin_frac', 'p_inh', 'p_pool', 'ln_lin', 'god_slot', 'mark_slot', 'a', 'lVf']
        return nm + ' ' + ' '.join('%s=%.2f[%.2f,%.2f]' % (k, p[k][1], p[k][0], p[k][2]) for k in keys) + \
            ' | heldout dB=%.2f p=%.3f' % (a['dB_obs'], a['dB_ref_p'])
    print(line('PE', R['PE']))
    print({k: round(v, 2) for k, v in R['PE']['heldout_pct'].items()})
    for nm in ('NULL_within', 'NULL_across'):
        for a in R[nm][:2]:
            print(line(nm, a))
    print(line('UR3', R['UR3']))
    print({k: round(v, 2) for k, v in R['UR3']['heldout_pct'].items()})
    for a in R['UR3_NULL_across'][:2]:
        print(line('UR3null', a))
    # planted recovery: correlation truth vs posterior median, coverage of 90% interval
    for k in PNAMES:
        t = np.array([a['truth'][k] for a in R['PLANT']], float)
        m = np.array([a['post'][k][1] for a in R['PLANT']])
        cov = np.mean([a['post'][k][0] <= a['truth'][k] <= a['post'][k][2] for a in R['PLANT']])
        r = np.corrcoef(t, m)[0, 1] if np.std(t) > 0 and np.std(m) > 0 else float('nan')
        print('PLANT %-10s r=%.2f cover90=%.2f' % (k, r, cov))
    print('PLANT heldout p (fraction <0.05):', np.mean([a['dB_ref_p'] < 0.05 for a in R['PLANT']]))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        main()
