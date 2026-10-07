"""LA-79 cycle 3: backcasting and a forecast lottery.

(a) Backcasting: treat the 1945 Hagia Triada corpus as the unknown. Random role hypotheses
    (la79_c2 generator) are fitted on one site group and scored on another, for every ordered
    pair (transfer matrix), and fitted on all post-1950 finds and scored on the <= 1950 corpus.
    Question: is HT the odd archive (fails both ways) or only a bad teacher (fails forward only)?
    Controls: planted universal grammar (transfer must be symmetric and positive); labels
    shuffled within each site (transfer must be ~0).
(b) Forecast lottery: 100,000 random sign sets, each 'forecasting' that its share of signs will
    rise (or fall) in what is found next. Scored at the 1950 and 1976 cuts; survivors re-tested
    on the finds published after 1988 (held out). Control: 200 shuffled-date runs.
usage: la79_c3.py back N seed | lottery
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la79_c2 as C2
import la78_engine as E
from la79_common import CK, SITE2G

GRP = np.array([SITE2G.get(s, 'OTH') for s in C2.site])
GS = ['HT', 'KH', 'ZA', 'PH', 'KN', 'OTH']


def transfer(N, seed, y, tag):
    rng = np.random.default_rng(seed)
    Y = [(y, 4)]
    p50 = C2.year0 <= 1950
    M = np.zeros((N, len(GS), len(GS)))
    fb = np.zeros((N, 2))
    for i in range(N):
        f, K = C2.gen(rng, C2.FAMS[i % len(C2.FAMS)])
        for a, ga in enumerate(GS):
            for b, gb in enumerate(GS):
                if a != b:
                    M[i, a, b] = E.score(f, Y, GRP == ga, GRP == gb)
        fb[i] = [E.score(f, Y, p50, ~p50), E.score(f, Y, ~p50, p50)]
    np.savez(os.path.join(CK, 'c3_back_%s_%d.npz' % (tag, seed)), M=M, fb=fb)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'back':
        N, seed = int(sys.argv[2]), int(sys.argv[3])
        E.MODE['mode'] = 'prior'
        transfer(N, seed, C2.y1, 'real')
        # within-site label shuffle
        rng = np.random.default_rng(seed + 7)
        ys = C2.y1.copy()
        for g in GS:
            m = np.where(GRP == g)[0]; ys[m] = ys[rng.permutation(m)]
        transfer(N // 2, seed, ys, 'lshuf')
        # planted universal grammar with the real site label mix
        LP = np.zeros((len(ys), 4))
        for g in GS:
            c = np.bincount(C2.y1[GRP == g], minlength=4) + 1.0; LP[GRP == g] = np.log(c / c.sum())
        part = rng.integers(0, 3, len(C2.signs)); ft = part[C2.last]; Em = rng.normal(0, 1, (3, 4))
        lg = LP + Em[ft]; p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        yp = (p.cumsum(1) > rng.random((len(ys), 1))).argmax(1)
        transfer(N // 2, seed, yp, 'plant')
    else:
        # forecast lottery on sign shares per document set
        from la79_common import load
        D = load()
        signs = sorted({s for d in D for w in d['words'] for s in w['s']})
        S = {s: i for i, s in enumerate(signs)}
        X = np.zeros((len(D), len(signs)))
        for i, d in enumerate(D):
            for w in d['words']:
                for s in w['s']:
                    X[i, S[s]] += 1
        yr = np.array([d['year'] for d in D])
        freq = X.sum(0)
        ok = freq >= 10
        rng = np.random.default_rng(79)
        NS = 100000
        sets = np.zeros((NS, len(signs)), bool)
        for k in range(NS):
            m = int(rng.integers(3, 31))
            idx = rng.choice(np.where(ok)[0], size=m, replace=False)
            sets[k, idx] = True
        direction = rng.choice([-1, 1], size=NS)

        def share(mask):
            v = X[mask].sum(0); return v / max(v.sum(), 1)

        def scores(yr):
            out = []
            for a, b in ((1950, 1976), (1976, 1988), (1988, 3000)):
                past, fut = share(yr <= a), share((yr > a) & (yr <= b))
                dp = sets @ (fut - past)
                out.append(direction * dp)
            return np.stack(out, 1)
        R = scores(yr)
        surv = (R[:, 0] > 0) & (R[:, 1] > 0)
        held = R[:, 2] > 0
        res = dict(n=NS, surv=int(surv.sum()), held_given_surv=float(held[surv].mean()), held_all=float(held.mean()))
        null = []
        for r in range(200):
            p = rng.permutation(len(D)); Rs = scores(yr[p])
            sv = (Rs[:, 0] > 0) & (Rs[:, 1] > 0)
            null.append([int(sv.sum()), float((Rs[:, 2] > 0)[sv].mean())])
        null = np.array(null)
        res['null_surv_mean'] = float(null[:, 0].mean()); res['null_held_given_surv'] = [float(null[:, 1].mean()), float(np.percentile(null[:, 1], 95))]
        # per-sign drift consensus from survivors (signed membership)
        cons = (sets[surv] * direction[surv, None]).sum(0) / np.maximum(sets[surv].sum(0), 1)
        res['rising'] = [signs[i] for i in np.argsort(-cons) if ok[i]][:15]
        res['falling'] = [signs[i] for i in np.argsort(cons) if ok[i]][:15]
        # consensus re-scored on the held-out window (>1988): share change of top-15 rising vs falling
        past, fut = share(yr <= 1988), share(yr > 1988)
        ri = [S[s] for s in res['rising']]; fa = [S[s] for s in res['falling']]
        res['heldout_rising_minus_falling'] = float((fut - past)[ri].sum() - (fut - past)[fa].sum())
        res['cons'] = {signs[i]: float(cons[i]) for i in range(len(signs)) if ok[i]}
        json.dump(res, open(os.path.join(CK, 'c3_lottery.json'), 'w'))
        print(json.dumps({k: v for k, v in res.items() if k != 'cons'}))
