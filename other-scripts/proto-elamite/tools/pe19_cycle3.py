"""pe19 cycle 3: VARIANT-CHOICE-ONLY seriation (content-neutral by construction).

Cycles 1-2 found the content (genre) axis, not time: a planted 40-trait drift
was not recovered. Here a tablet is described ONLY by which variant it chose for
bases written in >= 2 ways (bare form counted as variant '~0'). The same base
sign means the same thing, so the choice should carry hand / school / date, not
commodity.
Controls: (1) planted variant switching (15 bases; each tablet's form chosen by a
sigmoid of hidden t; strengths k = 4, 10, 30); (2) proto-cuneiform Uruk IV vs III
(Uruk only), same encoding. Never reads the hidden labels.
"""
import json, os, re, sys, time
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import spearmanr
from pe19_common import *


def forms_of(t):
    out = []
    for l in t['lines']:
        for s in l['signs']:
            s = s.strip('|')
            if s.lower() == 'x' or s.startswith('X') or '+' in s:
                continue
            b = base_of(s)
            m = re.search(r'~([a-z0-9]+)', s)
            out.append((b, m.group(1) if m else '0'))
    return out


def build_V(T, min_tab=3, min_traits=2, plant=None):
    use = {}
    for t in T:
        use[t['id']] = forms_of(t)
    if plant:
        use = plant(use)
    cnt = Counter()
    for p, fl in use.items():
        for f in set(fl):
            cnt[f] += 1
    ok = {f for f, c in cnt.items() if c >= min_tab}
    bases = Counter(b for b, v in ok)
    keep = sorted(f for f in ok if bases[f[0]] >= 2)
    idx = {f: i for i, f in enumerate(keep)}
    ids = [p for p, fl in use.items() if len({f for f in fl if f in idx}) >= min_traits]
    X = np.zeros((len(ids), len(keep)), np.float32)
    for i, p in enumerate(ids):
        for f in use[p]:
            if f in idx:
                X[i, idx[f]] = 1
    return ids, X, ['%s~%s' % f for f in keep], use


def seriate_V(X, elen, seed=0, uni=True):
    res = {'CA': orient(ca_scores(X), elen), 'SPEC': orient(spectral_scores(X, k=8), elen)}
    if uni:
        best = None
        for r, ini in enumerate([ca_scores(X), np.random.default_rng(seed).normal(size=len(X))]):
            x, L = unimodal_fit(X, ini, steps=400, seed=seed + r)
            if best is None or L < best[1]:
                best = (x, L)
        res['UNI'] = orient(best[0], elen)
    res['CONS'] = np.mean([rankdata(v) for v in res.values()], axis=0)
    return res


def elen_of(T, ids):
    d = {}
    for t in T:
        el = [len([s for s in l['signs'] if s.lower() != 'x']) for l in t['lines'] if l['numerals'] and l['signs']]
        d[t['id']] = np.mean(el) if el else 0.0
    return np.array([d[p] for p in ids])


def make_plant(T, k, nb, seed):
    rng = np.random.default_rng(seed)
    base_forms = defaultdict(Counter)
    for t in T:
        for b, v in forms_of(t):
            base_forms[b][v] += 1
    cand = [b for b, c in base_forms.items() if len(c) >= 2 and sum(c.values()) >= 15]
    chosen = list(rng.choice(cand, min(nb, len(cand)), replace=False))
    tt = {t['id']: rng.uniform() for t in T}
    cs = {b: rng.uniform(0.25, 0.75) for b in chosen}
    two = {b: [v for v, _ in base_forms[b].most_common(2)] for b in chosen}

    def plant(use):
        out = {}
        for p, fl in use.items():
            new = []
            pr = {b: 1 / (1 + np.exp(-k * (tt[p] - cs[b]))) for b in chosen}
            for b, v in fl:
                if b in pr:
                    v = two[b][1] if rng.uniform() < pr[b] else two[b][0]
                new.append((b, v))
            out[p] = new
        return out
    return plant, tt


def main():
    t0 = time.time()
    out = {}
    T = load_pe()
    ids, X, names, use = build_V(T)
    el = elen_of(T, ids)
    out['PE'] = {'n': len(ids), 'm': len(names)}
    res = seriate_V(X, el)
    out['PE']['hash'] = {m: freeze('C3_V_%s' % m, ids, v) for m, v in res.items()}
    print('PE', out['PE'], round(time.time() - t0), flush=True)
    for r in range(10):
        rng = np.random.default_rng(300 + r)
        Xs = np.array([rng.permutation(c) for c in X.T]).T
        ok = Xs.sum(1) >= 1
        freeze('NULLSHUF_C3_%d' % r, [ids[i] for i in np.where(ok)[0]], orient(ca_scores(Xs[ok]), el[ok]))

    pl = {}
    for k in (4, 10, 30):
        for nb in (15, 60):
            for rep in range(2):
                plant, tt = make_plant(T, k, nb, seed=10 * k + nb + rep)
                pids, PX, pn, _ = build_V(T, plant=plant)
                pel = elen_of(T, pids)
                r_ = seriate_V(PX, pel, uni=False)
                tv = np.array([tt[p] for p in pids])
                pl['k%d_nb%d_%d' % (k, nb, rep)] = {m: abs(float(spearmanr(v, tv)[0])) for m, v in r_.items()}
                print('plant', k, nb, rep, pl['k%d_nb%d_%d' % (k, nb, rep)], flush=True)
    out['planted'] = pl

    PC = load_pc()
    per = {t['id']: t['period'] for t in PC}
    pids, PX, pn, _ = build_V(PC)
    pel = elen_of(PC, pids)
    r_ = seriate_V(PX, pel)
    pcout = {'n': len(pids), 'm': len(pn), 'nIV': sum(per[p] == 'Uruk IV' for p in pids)}
    for m, v in r_.items():
        pos = rankdata(v) / len(v)
        a = auc([pos[i] for i, p in enumerate(pids) if per[p] == 'Uruk IV'],
                [pos[i] for i, p in enumerate(pids) if per[p] == 'Uruk III'])
        pcout[m] = a
        pcout[m + '_unoriented'] = max(a, 1 - a)
    # null for the control: periods shuffled
    rng = np.random.default_rng(5)
    v = r_['CA']; pos = rankdata(v) / len(v)
    lab = np.array([per[p] == 'Uruk IV' for p in pids])
    nul = []
    for _ in range(1000):
        lp = rng.permutation(lab)
        a = auc(pos[lp], pos[~lp]); nul.append(max(a, 1 - a))
    pcout['null_unoriented_95'] = float(np.percentile(nul, 95))
    out['uruk_control'] = pcout
    print('uruk', pcout, flush=True)
    json.dump(out, open(os.path.join(CK, 'c3_fit.json'), 'w'), indent=1)
    print('done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
