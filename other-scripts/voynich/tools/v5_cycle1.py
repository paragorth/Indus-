"""v5 cycle 1 -- cadence and 'mode final' fingerprints of notation.

T1a  unit-specificity of the line-final symbol: excess MI(unit; symbol) over a null that deals lines
     to random units (keeps unit sizes), for four positions in the line: first symbol, last symbol,
     last symbol of the middle word ('word-final, mid-line'), first symbol of the middle word.
     Notation fingerprint: LAST >> mid-line word-final (each piece/paragraph keeps one final).
T1b  closing final: does the last line of a unit end on the unit's habitual final (the modal final
     of its other lines)? null = a random non-last line in the same unit plays 'last'.
T4   'the final sets the scale': predict a unit's closing symbol from the unit's interior symbol
     make-up (edge symbols removed), nearest centroid, leave-one-out; null = permuted labels.
     Compared with the same for the unit's opening symbol.
"""
import sys, os, math, random, json
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_corpora as vc, vlib

R = 200


def mi_lines(labels, syms):
    n = len(labels)
    cj = Counter(zip(labels, syms)); ca = Counter(labels); cb = Counter(syms)
    return sum(v / n * math.log2(v * n / (ca[a] * cb[b])) for (a, b), v in cj.items())


def pos_symbol(L, pos):
    if pos == 'first':
        return L[0][0]
    if pos == 'last':
        return L[-1][-1]
    w = L[len(L) // 2] if len(L) >= 3 else None
    if w is None:
        return None
    return w[-1] if pos == 'mid_wordfinal' else w[0]


def t1a(units, stratified=False, seed=0):
    rng = random.Random(seed)
    out = {}
    for pos in ['first', 'last', 'mid_wordfinal', 'mid_wordinitial']:
        lab, sym, strat = [], [], []
        for ui, u in enumerate(units):
            if len(u['lines']) < 3:
                continue
            for L in u['lines']:
                s = pos_symbol(L, pos)
                if s is None:
                    continue
                lab.append(ui); sym.append(s); strat.append(u['stratum'])
        obs = mi_lines(lab, sym)
        nulls = []
        for _ in range(R):
            if stratified:
                by = defaultdict(list)
                for i, st in enumerate(strat):
                    by[st].append(i)
                perm = list(lab)
                for st, idx in by.items():
                    vals = [lab[i] for i in idx]; rng.shuffle(vals)
                    for i, v in zip(idx, vals):
                        perm[i] = v
            else:
                perm = list(lab); rng.shuffle(perm)
            nulls.append(mi_lines(perm, sym))
        mu, sd = np.mean(nulls), np.std(nulls)
        H = -sum(v / len(sym) * math.log2(v / len(sym)) for v in Counter(sym).values())
        out[pos] = {'n': len(sym), 'excess': obs - mu, 'z': (obs - mu) / sd if sd else 0,
                    'excess_over_H': (obs - mu) / H}
    out['ratio_last_vs_midwordfinal'] = out['last']['excess'] / max(out['mid_wordfinal']['excess'], 1e-4)
    return out


def t1b(units, seed=0):
    rng = random.Random(seed)
    hit = tot = 0; nh = []
    els = [u for u in units if len(u['lines']) >= 4]
    for u in els:
        fins = [L[-1][-1] for L in u['lines']]
        mode = Counter(fins[:-1]).most_common(1)[0][0]
        hit += fins[-1] == mode; tot += 1
    for _ in range(R):
        h = 0
        for u in els:
            fins = [L[-1][-1] for L in u['lines']]
            k = rng.randrange(len(fins) - 1)
            rest = fins[:k] + fins[k + 1:]
            h += fins[k] == Counter(rest).most_common(1)[0][0]
        nh.append(h)
    mu, sd = np.mean(nh), np.std(nh)
    return {'units': tot, 'obs_rate': hit / max(tot, 1), 'null_rate': mu / max(tot, 1), 'z': (hit - mu) / sd if sd else 0}


def t4(units, which='close', seed=0, min_class=8, stratified=False):
    rng = random.Random(seed)
    els = [u for u in units if len(u['lines']) >= 3]
    labs = [(u['lines'][-1][-1][-1] if which == 'close' else u['lines'][0][0][0]) for u in els]
    cnt = Counter(labs)
    keep = [i for i, l in enumerate(labs) if cnt[l] >= min_class]
    if len(set(labs[i] for i in keep)) < 2:
        return None
    alpha = sorted({s for u in els for L in u['lines'] for w in L for s in w})
    ai = {s: k for k, s in enumerate(alpha)}
    X = np.zeros((len(keep), len(alpha)))
    for r, i in enumerate(keep):
        for L in els[i]['lines']:
            seq = [s for w in L for s in w][1:-1]
            for s in seq:
                X[r, ai[s]] += 1
    X = X / np.maximum(X.sum(1, keepdims=True), 1)
    X = X - X.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    y = [labs[i] for i in keep]
    strat = [els[i]['stratum'] for i in keep]

    def acc(y):
        ys = sorted(set(y)); yi = np.array([ys.index(v) for v in y])
        S = np.zeros((len(ys), X.shape[1])); nn = np.zeros(len(ys))
        for r, k in enumerate(yi):
            S[k] += X[r]; nn[k] += 1
        correct = 0
        for r, k in enumerate(yi):
            C = S.copy(); C[k] -= X[r]; cn = nn.copy(); cn[k] -= 1
            C = C / np.maximum(cn[:, None], 1)
            correct += int(np.argmax(C @ X[r]) == k)
        return correct / len(yi)
    obs = acc(y)
    nulls = []
    for _ in range(R // 2):
        if stratified:
            by = defaultdict(list)
            for i, st in enumerate(strat):
                by[st].append(i)
            p = list(y)
            for idx in by.values():
                v = [y[i] for i in idx]; rng.shuffle(v)
                for i, vv in zip(idx, v):
                    p[i] = vv
        else:
            p = list(y); rng.shuffle(p)
        nulls.append(acc(p))
    mu, sd = np.mean(nulls), np.std(nulls)
    return {'units': len(y), 'classes': len(set(y)), 'acc': obs, 'null': mu,
            'z': (obs - mu) / sd if sd else 0, 'majority': Counter(y).most_common(1)[0][1] / len(y)}


if __name__ == '__main__':
    C = vc.all_corpora()
    res = {}
    for k, units in C.items():
        r = {'t1a': t1a(units), 't1b': t1b(units), 't4_close': t4(units, 'close'), 't4_open': t4(units, 'open')}
        if k.startswith('V-'):
            r['t1a_strat'] = t1a(units, stratified=True)
            r['t4_close_strat'] = t4(units, 'close', stratified=True)
        res[k] = r
        a = r['t1a']
        print(f"\n== {k}")
        for pos in ['first', 'last', 'mid_wordfinal', 'mid_wordinitial']:
            print(f"  T1a {pos:16s} n={a[pos]['n']:5d} excessMI={a[pos]['excess']:.4f} z={a[pos]['z']:6.1f} /H={a[pos]['excess_over_H']:.4f}")
        print(f"  T1a ratio last/mid-wordfinal = {a['ratio_last_vs_midwordfinal']:.2f}")
        if 't1a_strat' in r:
            s = r['t1a_strat']
            print('  T1a within-section: ' + ' '.join(f"{p}:{s[p]['excess']:.4f}(z{s[p]['z']:.1f})" for p in ['first', 'last', 'mid_wordfinal', 'mid_wordinitial']))
        print(f"  T1b closing final = habitual final: {r['t1b']}")
        print(f"  T4 close: {r['t4_close']}")
        print(f"  T4 open : {r['t4_open']}")
        if 't4_close_strat' in r:
            print(f"  T4 close within-section: {r['t4_close_strat']}")
    vlib.save('v5_cycle1', res)
