"""v87 cycle 3: (i) encoder-prior sensitivity: a second bank under a different encoder prior ('wide'); cross-prior
calibration (fit on base, plants from wide) and the Voynich posterior under each prior.
(ii) out-of-scale prediction: the accepted (source, scheme) pairs are re-run at 10,000 tokens and must predict the
Voynich's type count and hapax share at 10,000 tokens (never used in fitting). Calibrated on planted texts and
compared with prior-predictive and meaningless trigram-resynthesis predictions."""
import os, json, random, math
from collections import Counter
from multiprocessing import Pool
import numpy as np
import v87_lib as L
import v87_common as C
import v87_bank as B

Rb, Fb, Pb = C.load_bank('base'); Rw, Fw, Pw = C.load_bank('wide')
sb = np.array([r['sid'] for r in Rb]); sw = np.array([r['sid'] for r in Rw])
scb = L.scale_of(Fb[:, C.TI]); scw = L.scale_of(Fw[:, C.TI])
out = {'bank_base': len(Rb), 'bank_wide': len(Rw)}
BIG = 10000


def auc(score, y):
    o = np.argsort(score); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calib(Ftr, Ptr, str_, sc, Fte, Pte, ste, n=400, seed=0):
    rs = np.random.default_rng(seed); t = rs.choice(len(Fte), n, replace=False)
    po = [L.abc(Ftr[:, C.TI], Ptr, Fte[i, C.TI], sc, mask=(str_ != ste[i])) for i in t]
    return dict(auc_list=auc(np.array([p['is_list'] for p in po]), Pte[t, 0].astype(int)),
                r_ptt=float(np.corrcoef([p['ptt'] for p in po], Pte[t, 1])[0, 1]),
                r_pwl=float(np.corrcoef([p['pwl'] for p in po], Pte[t, 2])[0, 1]))


# ---- (i) prior sensitivity
out['calib_base_on_base'] = calib(Fb, Pb, sb, scb, Fb, Pb, sb)
out['calib_wide_on_wide'] = calib(Fw, Pw, sw, scw, Fw, Pw, sw)
out['calib_base_on_wide'] = calib(Fb, Pb, sb, scb, Fw, Pw, sw)   # encoder outside the fitting prior
out['calib_wide_on_base'] = calib(Fw, Pw, sw, scw, Fb, Pb, sb)
print(json.dumps({k: v for k, v in out.items() if k.startswith('calib')}))
for name in ['ZL3b', 'IT2a']:
    ch = L.voy_chunks(L.voy_lines(name))['all']; vecs = [C.fvec(c) for c in ch]
    pb_ = C.summarize(C.run_abc(Fb, Pb, vecs, scb)); pw_ = C.summarize(C.run_abc(Fw, Pw, vecs, scw))
    out['voy_' + name] = dict(base=pb_, wide=pw_, prior_wide=dict(is_list=float(Pw[:, 0].mean()), ptt=float(Pw[:, 1].mean()),
                                                                 pwl=float(Pw[:, 2].mean())))
    print(name, json.dumps(out['voy_' + name]))


# ---- (ii) out-of-scale prediction
def big_stats(lines):
    toks = [t for Lw in lines for t in Lw][:BIG]
    tc = Counter(toks)
    return [len(tc) / len(toks), sum(1 for v in tc.values() if v == 1) / len(tc)]


def rerun(seed):
    rng = random.Random(seed); sid = rng.choice(B.SIDS); sc = L.random_scheme(rng)
    ents, lines, n = L.simulate(B.S[sid], sc, rng, ntok=BIG)
    if n < BIG * 0.95: return None
    return big_stats(lines)


def predict(vec2k, excl=set(), k=60):
    m = ~np.isin(sb, list(excl))
    po = L.abc(Fb[:, C.TI], Pb, vec2k[C.TI], scb, mask=m)
    top = po['idx'][:k]
    return [int(Rb[i]['seed']) for i in top]


if __name__ == '__main__':
    with Pool(2) as pool:
        # calibration plants: 40 planted texts at 10k tokens; fit their first 2k chunk; predict their 10k stats
        rs = random.Random(5); plants = []
        for k in range(40):
            seed = 990000 + k
            rng = random.Random(seed); sid = rng.choice(B.SIDS); sc = L.random_scheme(rng)
            ents, lines, n = L.simulate(B.S[sid], sc, rng, ntok=BIG)
            if n < BIG * 0.95: continue
            # first 2,000 tokens
            first = []; t = 0
            for Lw in lines:
                if t >= L.NTOK: break
                first.append(Lw[:L.NTOK - t]); t += len(first[-1])
            plants.append((sid, big_stats(lines), C.fvec(first), C.fvec(C.trigram_resynth(first))))
        pc = []
        for sid, truth, v2, vr in plants:
            pred = [x for x in pool.map(rerun, predict(v2, {sid})) if x]
            predr = [x for x in pool.map(rerun, predict(vr, {sid})) if x]
            pc.append(dict(truth=truth, pred=np.median(pred, 0).tolist(), lo=np.percentile(pred, 10, 0).tolist(),
                           hi=np.percentile(pred, 90, 0).tolist(), pred_resynth=np.median(predr, 0).tolist()))
        prior_pred = [x for x in pool.map(rerun, [int(r['seed']) for r in Rb[:300]]) if x]
        pp = np.median(prior_pred, 0)
        T = np.array([p['truth'] for p in pc]); Pm = np.array([p['pred'] for p in pc]); Pr_ = np.array([p['pred_resynth'] for p in pc])
        lo = np.array([p['lo'] for p in pc]); hi = np.array([p['hi'] for p in pc])
        out['scale_calib'] = dict(n=len(pc),
            ttr_mae=float(np.abs(Pm[:, 0] - T[:, 0]).mean()), ttr_mae_prior=float(np.abs(pp[0] - T[:, 0]).mean()),
            ttr_mae_resynth=float(np.abs(Pr_[:, 0] - T[:, 0]).mean()), ttr_r=float(np.corrcoef(Pm[:, 0], T[:, 0])[0, 1]),
            ttr_cov80=float(((T[:, 0] >= lo[:, 0]) & (T[:, 0] <= hi[:, 0])).mean()),
            hap_mae=float(np.abs(Pm[:, 1] - T[:, 1]).mean()), hap_mae_prior=float(np.abs(pp[1] - T[:, 1]).mean()),
            hap_cov80=float(((T[:, 1] >= lo[:, 1]) & (T[:, 1] <= hi[:, 1])).mean()))
        print('scale calib', out['scale_calib'])
        # Voynich: predict from each 2k chunk inside the first 10k-token stretch of several 10k windows
        for name in ['ZL3b', 'IT2a']:
            vl = L.voy_lines(name)
            wins = L.voy_chunks(vl, ntok=BIG)['all']
            res = []
            for w in wins:
                truth = big_stats(w)
                first = []; t = 0
                for Lw in w:
                    if t >= L.NTOK: break
                    first.append(Lw[:L.NTOK - t]); t += len(first[-1])
                pred = [x for x in pool.map(rerun, predict(C.fvec(first))) if x]
                predr = [x for x in pool.map(rerun, predict(C.fvec(C.trigram_resynth(first)))) if x]
                res.append(dict(truth=truth, pred=np.median(pred, 0).tolist(), lo=np.percentile(pred, 10, 0).tolist(),
                                hi=np.percentile(pred, 90, 0).tolist(), pred_resynth=np.median(predr, 0).tolist()))
            out['scale_' + name] = dict(windows=res, prior_pred=pp.tolist())
            print(name, json.dumps(res))
    json.dump(out, open(os.path.join(L.CK, 'c3.json'), 'w'), indent=1, default=float)
