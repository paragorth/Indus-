"""LA-35 cycle 3: the bare form and the extended form. For every word W (>= 1 sign) that also
occurs as W+X (one or two extra final signs), compare the numbers after W with the numbers after
W+X. A plural/dual/counting suffix predicts W+X before larger numbers (or before 1 if the bare
form is the plural). Statistic per pair: mean log(n) of W+X minus mean log(n) of W, and the share
of '1' entries; summed over pairs (signed, pooled), and per suffix X across >= 2 stems.
Also the 'substitution' form: W+X vs W+Y (same stem, different final sign).
Nulls: numbers permuted within (tablet, commodity) [N1], within (site, commodity) [N2],
and across the corpus within commodity [N3]; 2,000 each. Controls: LB at LA size and full;
planted suffix in LA (W+X before >= 2, applied to 30% of the eligible entries)."""
import json, sys, math, random
import numpy as np
from la35_common import *
from la35_c1 import lb_sub


def pairs_of(E, mode):
    words = Counter(e['word'] for e in E)
    P = []
    ws = set(words)
    for w in ws:
        if mode == 'ext':
            for k in (1, 2):
                if len(w) - k >= 2 and w[:-k] in ws:
                    P.append((w[:-k], w, '-'.join(w[-k:])))
        else:  # substitution: same all-but-last, different last sign; ordered pairs once
            for v in ws:
                if v != w and len(v) == len(w) and len(w) >= 3 and v[:-1] == w[:-1] and v < w:
                    P.append((v, w, v[-1] + '/' + w[-1]))
    return P


def stat(E, P, vals):
    idx = defaultdict(list)
    for i, e in enumerate(E): idx[e['word']].append(i)
    lv = np.log(vals); one = (vals == 1).astype(float)
    dl, d1, w = [], [], []
    for a, b, x in P:
        ia, ib = idx[a], idx[b]
        dl.append(lv[ib].mean() - lv[ia].mean()); d1.append(one[ib].mean() - one[ia].mean())
    dl, d1 = np.array(dl), np.array(d1)
    # pooled absolute (any direction per pair) and signed per suffix across >= 2 stems
    absl = float(np.abs(dl).mean()) if len(dl) else 0.0
    bysuf = defaultdict(list)
    for (a, b, x), v in zip(P, dl): bysuf[x].append(v)
    cons = sum(abs(sum(v)) / math.sqrt(len(v)) for v in bysuf.values() if len(v) >= 2)
    return dict(signed=float(dl.mean()) if len(dl) else 0.0, abs=absl, one=float(d1.mean()) if len(d1) else 0.0,
                cons=float(cons)), {x: (len(v), float(np.mean(v))) for x, v in bysuf.items()}


def run(name, E, nperm, seed, verbose=False):
    rng = np.random.default_rng(seed)
    vals = np.array([e['v'] for e in E], dtype=float)
    out = dict(name=name)
    for mode in ('ext', 'sub'):
        P = pairs_of(E, mode)
        real, suf = stat(E, P, vals)
        r = dict(npairs=len(P), real=real)
        for nl, key in (('N1', lambda e: (e['doc'], e['com'])), ('N2', lambda e: (e['site'], e['com'])),
                        ('N3', lambda e: e['com'])):
            g = strata(E, key)
            sims = [stat(E, P, permute_vals(vals, g, rng))[0] for _ in range(nperm)]
            r[nl] = {k: dict(mean=round(float(np.mean([s[k] for s in sims])), 4),
                             p_hi=round((sum(s[k] >= real[k] for s in sims) + 1) / (nperm + 1), 4),
                             p_lo=round((sum(s[k] <= real[k] for s in sims) + 1) / (nperm + 1), 4))
                     for k in real}
        if verbose:
            r['suffixes'] = sorted(suf.items(), key=lambda kv: -kv[1][0])[:12]
            r['pairs'] = [('-'.join(a), '-'.join(b)) for a, b, x in P][:40]
        out[mode] = r
    print(json.dumps(out), flush=True)
    return out


def plant_suffix(E, p, seed):
    rng = random.Random(seed)
    fin = Counter(e['word'][-1] for e in E)
    X = rng.choice([s for s, _ in fin.most_common(20)][5:])
    out = []
    for e in E:
        e = dict(e)
        if e['v'] >= 2 and rng.random() < p: e['word'] = e['word'] + (X,)
        out.append(e)
    return out, X


if __name__ == '__main__':
    NP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    LA = la_entries(); LB = lb_entries()
    res = [run('LA', LA, NP, 1, verbose=True)]
    for s in range(3):
        P, X = plant_suffix(LA, 0.3, 300 + s)
        res.append(run('PLANT0.3 %s' % X, P, NP // 4, 10 + s))
    for s in range(6):
        res.append(run('LBsub%d' % s, lb_sub(LB, len(LA), s), NP // 4, 20 + s))
    res.append(run('LBfull', LB, NP // 4, 40, verbose=True))
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), indent=1)
