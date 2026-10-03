"""v8 cycle 1: is there a 1-D writing clock (vocabulary drift) in the Voynich pages?

Spectral seriation (Fiedler vector of Hellinger affinity) + open TSP/2-opt.
Statistics:
  split-half |rho|  : seriate even lines and odd lines of each page separately; agreement of the two orders.
  truth |rho|       : (controls only) agreement of the recovered order with the true page order.
  TSP gain          : (null path length - real path length) / null, null = tokens redealt.
Controls (same page and line lengths as the Voynich, cut from Isidore's Etymologiae, Latin):
  plain Latin pages (topic drift only), Latin + slowly drifting homophonic cipher (positive),
  tokens redealt across pages (negative), and a within-stratum redeal for the Voynich.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from v8_lib import *

rng = random.Random(1)
V = voynich_pages()
print('Voynich pages', len(V), 'tokens', sum(ntok(p) for p in V))
LAT = latin_words()
print('Latin words', len(LAT))
Lp = text_to_pages(LAT, V, offset=2000)
Lc = drift_cipher(Lp, seed=3)
true = np.arange(len(V))


def report(name, pages, truth=True):
    X, voc = matrix(pages, 5)
    f = spectral_order(X)
    sh = split_half(pages)
    tr = absrho(f, true) if truth else float('nan')
    D = dist(X)
    path, Ln = tsp_order(D, restarts=2)
    trt = absrho(rankpos(path), true) if truth else float('nan')
    return dict(name=name, n=len(pages), types=len(voc), split=sh, truth=tr, tsp_truth=trt, L=Ln, f=f, path=path)


res = {}
for name, pg in [('Latin plain', Lp), ('Latin drift-cipher', Lc),
                 ('Latin redeal', redeal(Lp, seed=1)), ('Latin cipher redeal', redeal(Lc, seed=1)),
                 ('Voynich', V), ('Voynich redeal', redeal(V, seed=1))]:
    r = report(name, pg, truth=True)
    res[name] = r
    print(f"{name:22s} types={r['types']:5d} split-half={r['split']:.3f} fiedler~order={r['truth']:.3f} tsp~order={r['tsp_truth']:.3f} TSP L={r['L']:.2f}")

# TSP gain relative to redeal null (5 seeds)
def tsp_gain(pages, seeds=5):
    X, _ = matrix(pages, 5); L = tsp_order(dist(X), restarts=2)[1]
    nulls = []
    for s in range(seeds):
        Xn, _ = matrix(redeal(pages, seed=10 + s), 5)
        nulls.append(tsp_order(dist(Xn), restarts=2)[1])
    return L, np.mean(nulls), 1 - L / np.mean(nulls)

for name, pg in [('Latin plain', Lp), ('Latin drift-cipher', Lc), ('Voynich', V)]:
    L, Ln, g = tsp_gain(pg, 3)
    res[name]['gain'] = g
    print(f'TSP gain {name}: real {L:.2f} null {Ln:.2f} gain {g:.3f}')

# ---------- strata: is there drift inside a homogeneous group? ----------
def strata_test(pages, label, seeds=10):
    sh = split_half(pages)
    nulls = [split_half(redeal(pages, seed=100 + s)) for s in range(seeds)]
    mu, sd = np.mean(nulls), np.std(nulls) + 1e-9
    return sh, mu, sd, (sh - mu) / sd

strata = {
    'Currier A (all)': lambda p: p['lang'] == 'A',
    'Currier B (all)': lambda p: p['lang'] == 'B',
    'A herbal': lambda p: p['lang'] == 'A' and p['illus'] == 'H',
    'B herbal': lambda p: p['lang'] == 'B' and p['illus'] == 'H',
    'B bio (Q M)': lambda p: p['quire'] == 'M',
    'B stars (illus S, lang B)': lambda p: p['lang'] == 'B' and p['illus'] == 'S',
    'pharma (illus P)': lambda p: p['illus'] == 'P',
}
srows = []
for lab, fn in strata.items():
    sub = [p for p in V if fn(p)]
    if len(sub) < 10:
        continue
    sh, mu, sd, z = strata_test(sub, lab)
    # matched controls: contiguous block of the Latin plain / cipher pages of the same size
    k = len(sub)
    st = rng.randrange(0, len(V) - k)
    lp = [dict(Lp[i]) for i in range(st, st + k)]
    lc = [dict(Lc[i]) for i in range(st, st + k)]
    # use the same line templates as the stratum
    lp = text_to_pages(LAT, sub, offset=2000 + sum(ntok(p) for p in V[:st]))
    lc_full = drift_cipher(text_to_pages(LAT, V, offset=2000), seed=3)
    lc = lc_full[st:st + k]
    lpz = strata_test(lp, 'lat', 5)
    lcz = strata_test(lc, 'cip', 5)
    # recovered order vs binding order inside stratum
    Xs, _ = matrix(sub, 3)
    fs = spectral_order(Xs)
    bind = absrho(fs, [p['order'] for p in sub])
    # recovered order vs truth for cipher control
    Xc, _ = matrix(lc, 3)
    ctruth = absrho(spectral_order(Xc), np.arange(k))
    srows.append((lab, k, sh, mu, z, bind, lpz[0], lpz[3], lcz[0], lcz[3], ctruth))
    print(f'{lab:28s} n={k:3d} split={sh:.3f} null={mu:.3f} z={z:5.1f} | ~binding={bind:.3f} | Latin plain split={lpz[0]:.3f} z={lpz[3]:.1f} | Latin cipher split={lcz[0]:.3f} z={lcz[3]:.1f} truth={ctruth:.3f}')

# ---------- quire contiguity along the full Voynich seriation ----------
f = res['Voynich']['f']
rk = np.argsort(np.argsort(f))
def mean_same(key):
    ids = [i for i, p in enumerate(V) if key(p) is not None]
    pairs = [(i, j) for a, i in enumerate(ids) for j in ids[a + 1:] if key(V[i]) == key(V[j])]
    real = np.mean([abs(rk[i] - rk[j]) for i, j in pairs])
    labs = [key(V[i]) for i in ids]
    nulls = []
    for s in range(300):
        rng.shuffle(labs)
        lab = dict(zip(ids, labs))
        pp = [(i, j) for a, i in enumerate(ids) for j in ids[a + 1:] if lab[i] == lab[j]]
        nulls.append(np.mean([abs(rk[i] - rk[j]) for i, j in pp]))
    return real, np.mean(nulls), (real - np.mean(nulls)) / np.std(nulls)
qc = mean_same(lambda p: p['quire'])
qcA = mean_same(lambda p: p['quire'] if p['lang'] == 'A' else None)
qcB = mean_same(lambda p: p['quire'] if p['lang'] == 'B' else None)
print('quire contiguity all', qc, 'within A', qcA, 'within B', qcB)
# where do A and B sit along the axis
langs = [p['lang'] for p in V]
print('Fiedler~lang(B=1)', absrho(f, [1 if l == 'B' else 0 for l in langs]))
print('Fiedler~binding', absrho(f, true))

rows = []
R = res
rows.append(('V-1.1', 'Positive/negative calibration of spectral seriation: Isidore Etymologiae (Latin) cut into pages with the exact Voynich page/line lengths (%d pages); plain, with a slowly drifting homophonic letter cipher (23 letters switch variant at random times, logistic width 0.08), and with tokens redealt across pages. Stats: split-half |rho| (even vs odd lines seriated separately) and |rho| of recovered order vs true order' % len(V),
             'plain: split %.2f, ~truth %.2f (TSP %.2f); drift cipher: split %.2f, ~truth %.2f (TSP %.2f); redeal: split %.2f / %.2f' % (R['Latin plain']['split'], R['Latin plain']['truth'], R['Latin plain']['tsp_truth'], R['Latin drift-cipher']['split'], R['Latin drift-cipher']['truth'], R['Latin drift-cipher']['tsp_truth'], R['Latin redeal']['split'], R['Latin cipher redeal']['split']),
             'Method calibrated: a drifting key is recovered as a 1-D order; redeal kills it'))
rows.append(('V-1.2', 'Same seriation on Voynich (ZL3b, P/C/R lines, pages >= 30 tokens, words with ? dropped); null = tokens redealt; TSP path gain vs redeal null; recovered order vs binding order',
             'split %.2f (redeal %.2f); TSP gain %.3f (Latin plain %.3f, cipher %.3f); Fiedler~binding |rho| %.2f; Fiedler~Currier B %.2f' % (R['Voynich']['split'], R['Voynich redeal']['split'], R['Voynich']['gain'], R['Latin plain']['gain'], R['Latin drift-cipher']['gain'], absrho(f, true), absrho(f, [1 if l == 'B' else 0 for l in langs])),
             'A strong, reproducible 1-D axis exists; at whole-book level it is the A/B axis (see V-1.3 for whether it is more than that)'))
s = '; '.join(f'{r[0]} n={r[1]} split {r[2]:.2f} (null {r[3]:.2f}, z {r[4]:.1f}) ~binding {r[5]:.2f}' for r in srows)
c = '; '.join(f'{r[0]}: Latin plain {r[6]:.2f} (z {r[7]:.1f}), cipher {r[8]:.2f} (z {r[9]:.1f}, ~truth {r[10]:.2f})' for r in srows)
rows.append(('V-1.3', 'Drift inside homogeneous strata (language x illustration): split-half |rho| vs within-stratum token redeal (10 seeds); matched Latin plain and drift-cipher blocks of the same size and line template', s + ' || controls: ' + c, 'see final verdict line'))
rows.append(('V-1.4', 'Quire contiguity along the whole-book seriation: mean |rank gap| of same-quire page pairs vs quire labels permuted (300x), all pages and within A / within B',
             'all: %.1f vs %.1f (z %.1f); within A: %.1f vs %.1f (z %.1f); within B: %.1f vs %.1f (z %.1f)' % (qc + qcA + qcB),
             'quires are compact in vocabulary time if z << 0'))
import pickle
pickle.dump(dict(rows=rows, srows=srows, f=f, ids=[p['id'] for p in V]), open(os.path.join(DATA, 'results', 'v8_cycle1.pkl'), 'wb'))
for r in rows:
    print(r)
