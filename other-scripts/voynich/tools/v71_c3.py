"""v71 cycle 3: time's arrow at every scale.  Instead of shuffling, ask whether a held-out
classifier can tell the written order of two adjacent units from the reversed order:
 W  adjacent words in a line, L  adjacent lines in a paragraph, Lm  the same without the
 paragraph's first and last lines, P  adjacent paragraphs in a page.  Null for each scale: the same
 chunk with the units shuffled inside their parent (arrow destroyed), same configurations.
Pair features: phi(A) - phi(B) (bags of hashed word classes, length) plus a hashed junction
(last word of A, first word of B) minus the reversed junction; L2 logistic regression without
intercept; 10 random feature configurations per scale, at most 600 pairs per scale; folds by page.  Arrow = held-out accuracy.
Chunks are the cycle-1 chunks (same seeds).  Results data/v71_ckpt/c3/<job>.json."""
import os, sys, json, random, zlib
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v71_lib as L
import v71_c1 as C1
from sklearn.linear_model import LogisticRegression

OUTD = os.path.join(L.CK, 'c3'); os.makedirs(OUTD, exist_ok=True)
SC = ['W', 'L', 'Lm', 'P']
NULLSC = {'W': 'S1', 'L': 'S2', 'Lm': 'S2', 'P': 'S3'}

def configs(n=10, seed=73):
    rng = random.Random(seed); out = []
    for _ in range(n):
        out.append({'rep': rng.choice(['h16', 'h64', 'h256', 'pre1', 'pre2', 'suf1', 'suf2']), 'seed': rng.randrange(10 ** 6),
                    'junc': rng.random() < 0.5, 'len': rng.random() < 0.5, 'C': rng.choice([0.03, 0.1, 0.3, 1.0])})
    return out
CFG = configs()

def wclass(w, c):
    r = c['rep']
    if r.startswith('h'): return zlib.crc32(('%d|%s' % (c['seed'], w)).encode()) % int(r[1:])
    k = int(r[3]); s = w[:k] if r.startswith('pre') else w[-k:]
    return zlib.crc32(('%d|%s|%s' % (c['seed'], r, s)).encode()) % 256

def units(pages, sc):
    """list of (fold, A, B) with A, B lists of words, A written before B."""
    out = []
    for gi, pg in enumerate(pages):
        f = gi % 4
        if sc == 'W':
            for pa in pg:
                for l in pa:
                    for a, b in zip(l, l[1:]): out.append((f, [a], [b]))
        elif sc in ('L', 'Lm'):
            for pa in pg:
                ls = pa if sc == 'L' else pa[1:-1]
                for a, b in zip(ls, ls[1:]): out.append((f, a, b))
        elif sc == 'P':
            ps = [[w for l in pa for w in l] for pa in pg]
            for a, b in zip(ps, ps[1:]): out.append((f, a, b))
    if sc == 'G':
        ps = [[w for pa in pg for l in pa for w in l] for pg in pages]
        for gi, (a, b) in enumerate(zip(ps, ps[1:])): out.append((gi % 4, a, b))
    return out

def featurize(U, c):
    D = 256 + 2 + (1024 if c['junc'] else 0)
    X = np.zeros((len(U), D))
    for i, (_, a, b) in enumerate(U):
        for w in a: X[i, wclass(w, c)] += 1.0 / len(a)
        for w in b: X[i, wclass(w, c)] -= 1.0 / len(b)
        if c['len']:
            X[i, 256] = np.log(len(a)) - np.log(len(b))
            X[i, 257] = np.mean([len(w) for w in a]) - np.mean([len(w) for w in b])
        if c['junc']:
            X[i, 258 + (wclass(a[-1], c) * 31 + wclass(b[0], c)) % 1024] += 1
            X[i, 258 + (wclass(b[-1], c) * 31 + wclass(a[0], c)) % 1024] -= 1
    return X

def arrow(pages, null=False):
    res = {}
    for sc in SC:
        P = L.shuffle(pages, NULLSC[sc], random.Random(sc))[0] if null else pages
        U = units(P, sc)
        if len(U) > 600: U = random.Random(len(U)).sample(U, 600)     # cap pairs per scale (CPU budget)
        folds = np.array([u[0] for u in U])
        if len(U) < 16 or len(set(folds)) < 2:
            res[sc] = None; continue
        accs = []
        for c in CFG:
            X = featurize(U, c)
            rng = np.random.default_rng(c['seed'])
            sgn = rng.choice([-1, 1], size=len(U))      # half of the pairs presented reversed
            Xs, y = X * sgn[:, None], (sgn > 0).astype(int)
            cor = n = 0
            for f in sorted(set(folds)):
                tr, te = folds != f, folds == f
                if len(set(y[tr])) < 2: continue
                m = LogisticRegression(C=c['C'], fit_intercept=False, max_iter=500).fit(Xs[tr], y[tr])
                cor += (m.predict(Xs[te]) == y[te]).sum(); n += te.sum()
            accs.append(cor / max(1, n))
        res[sc] = {'n': len(U), 'acc': [round(a, 4) for a in accs]}
    return res

def run(job):
    name, meta, ch = job
    out = os.path.join(OUTD, name + '.json')
    if os.path.exists(out): return name, 'skip'
    r = arrow(ch); r0 = arrow(ch, null=True)
    meta = dict(meta); meta['name'] = name
    json.dump({'meta': meta, 'arrow': r, 'null': r0}, open(out, 'w'))
    return name, 'ok'

if __name__ == '__main__':
    J = C1.jobs()
    # planted control: a Voynich-layout Markov text (no arrow above the word) whose lines are
    # re-ordered within each paragraph by a hidden key (lines sorted by first-word class): an arrow at L by construction
    extra = []
    for name, meta, ch in J:
        if name.startswith('gen__markov_VherbalA__0') or name.startswith('gen__markov_Vstars__0'):
            P = [[sorted(pa, key=lambda l: zlib.crc32(l[0].encode()) % 7) for pa in pg] for pg in ch]
            extra.append(('plant__sorted_' + name.split('__')[1] + '__0', {'kind': 'plant'}, P))
    J = J + extra
    print('jobs', len(J), flush=True)
    with Pool(2) as p:
        for name, st in p.imap_unordered(run, J):
            print(name, st, flush=True)
    print('DONE', flush=True)
