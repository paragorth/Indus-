"""v71 cycle 2 report: fit scrambling-writer hypotheses (text x f1 x f2 x f3) to fingerprints.
Multilinear surrogate over the real corner runs; midpoints check the surrogate; chunk-1 corner
runs of two texts are planted targets with known fractions; generators are the null."""
import os, sys, json, glob, random, itertools
import numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v71_lib as L
import v71_c2 as C2

def loadfp(cyc):
    out = {}
    for f in glob.glob(os.path.join(L.CK, cyc, '*.json')):
        d = json.load(open(f)); out[d['meta']['name']] = d
    return out

def main():
    c1, c2 = loadfp('c1'), loadfp('c2')
    keys = sorted(k for k in next(iter(c1.values()))['fp'] if not k.startswith('S0|'))
    vec = lambda d: np.array([d['fp'].get(k, 0.0) for k in keys])
    refs = [d for d in c1.values() if d['meta']['kind'] == 'ref']
    Xr = np.array([vec(d) for d in refs]); mu, sd = Xr.mean(0), Xr.std(0) + 1e-9
    z = lambda d: (vec(d) - mu) / sd
    corners = list(itertools.product([0, 1], repeat=3))
    # corner tables per text (chunk 0)
    CT = {}
    for k in C2.TEXTS:
        F = {}
        for c in corners:
            nm = 'ref__%s__0' % k if c == (0, 0, 0) else 'scr__%s__0__%g_%g_%g' % (k, *c)
            src = c1 if c == (0, 0, 0) else c2
            if nm in src: F[c] = z(src[nm])
        if len(F) == 8: CT[k] = F
    out = []
    P = lambda s: (print(s), out.append(s))
    P('texts with full corner sets: %s; %d features (S0 excluded)' % (list(CT), len(keys)))
    def surrogate(k, f):
        F = CT[k]; v = 0
        for c in corners:
            w = np.prod([fi if ci else 1 - fi for fi, ci in zip(f, c)]); v = v + w * F[c]
        return v
    # surrogate check on midpoints
    errs, spans = [], []
    for k in CT:
        for f in [(0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0.5)]:
            nm = 'scr__%s__0__%g_%g_%g' % (k, *f)
            if nm not in c2: continue
            real = z(c2[nm]); sur = surrogate(k, f)
            errs.append(np.sqrt(((real - sur) ** 2).mean())); spans.append(np.sqrt(((CT[k][(0, 0, 0)] - CT[k][(1, 1, 1)]) ** 2).mean()))
    if errs: P('SURROGATE CHECK: rms(real midpoint - surrogate) %.3f vs corner span %.3f (ratio %.2f, n %d)' % (np.mean(errs), np.mean(spans), np.mean(errs) / np.mean(spans), len(errs)))
    # hypotheses
    rng = np.random.default_rng(71)
    H = [(k, tuple(rng.random(3))) for k in CT for _ in range(1500)] + [(k, c) for k in CT for c in corners]
    HS = np.array([surrogate(k, f) for k, f in H])
    half = rng.permutation(len(keys)); A, B = half[: len(keys) // 2], half[len(keys) // 2:]
    def fit(t):
        dA = np.sqrt(((HS[:, A] - t[A]) ** 2).mean(1)); i = int(dA.argmin())
        dB = np.sqrt(((HS[i, B] - t[B]) ** 2).mean())
        # best by B alone for comparison and the B-distance of the best unscrambled text
        d0 = min(np.sqrt(((CT[k][(0, 0, 0)][B] - t[B]) ** 2).mean()) for k in CT)
        return H[i], float(dA[i]), float(dB), float(d0)
    # scale of a good fit: same text, other chunk, unscrambled
    same = []
    for k in CT:
        for j in (1, 2, 3):
            nm = 'ref__%s__%d' % (k, j)
            if nm in c1: same.append(np.sqrt(((z(c1[nm])[B] - CT[k][(0, 0, 0)][B]) ** 2).mean()))
    P('GOOD-FIT SCALE: same text, other chunk, unscrambled: held-out-half distance median %.3f, max %.3f (n %d)' % (np.median(same), np.max(same), len(same)))
    P('PLANTED (chunk 1 scrambled at known corners; surrogate built on chunk 0):')
    ok_t = ok_f = n = 0
    for nm, d in sorted(c2.items()):
        m = d['meta']
        if m['chunk'] != 1: continue
        (k, f), dA, dB, d0 = fit(z(d)); n += 1
        fr = np.round(f, 2); ok_t += k == m['text']; ok_f += all(abs(a - b) < 0.5 for a, b in zip(f, m['f']))
        P('  %-34s true %s %s -> %s %s dB %.3f' % (nm, m['text'], m['f'], k, list(fr), dB))
    for k in ('forme_of_cury', 'culpeper'):
        nm = 'ref__%s__1' % k
        if nm in c1:
            (kk, f), dA, dB, d0 = fit(z(c1[nm])); n += 1; ok_t += kk == k; ok_f += all(x < 0.5 for x in f)
            P('  %-34s true %s [0,0,0] -> %s %s dB %.3f' % (nm, k, kk, list(np.round(f, 2)), dB))
    P('  planted: text recovered %d/%d, fractions within 0.5 on all three scales %d/%d' % (ok_t, n, ok_f, n))
    P('GENERATORS (null) and VOYNICH:')
    rows = defaultdict(list)
    for nm, d in sorted(c1.items()):
        m = d['meta']
        if m['kind'] not in ('gen', 'voy'): continue
        (k, f), dA, dB, d0 = fit(z(d))
        lab = ('gen_' + m['gen'] + ('_V' if m['src'].startswith('V') else '_ref')) if m['kind'] == 'gen' else 'V_%s_%s' % (m['tr'][:2], m['sec'])
        rows[lab].append((k, f, dA, dB, d0))
        P('  %-30s -> %-15s f(S1,S2,S3) %s  dA %.3f dB %.3f (best unscrambled %.3f)' % (nm, k, list(np.round(f, 2)), dA, dB, d0))
    P('SUMMARY by group: mean held-out distance dB, gain over the best unscrambled text, best texts, mean fractions')
    for lab in sorted(rows):
        R = rows[lab]
        P('  %-22s n %d dB %.3f (unscr %.3f) texts %s f %s' % (lab, len(R), np.mean([r[3] for r in R]), np.mean([r[4] for r in R]),
          dict(zip(*np.unique([r[0] for r in R], return_counts=True))), list(np.round(np.mean([r[1] for r in R], 0), 2))))
    json.dump({'lines': out}, open(os.path.join(L.CK, 'report_c2.json'), 'w'))

if __name__ == '__main__':
    main()
