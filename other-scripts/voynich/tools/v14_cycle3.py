"""v14 cycle 3: three ways to rescue the dice.
(a) 'Noisy dice': fit devices only to the relative frequencies of the top-k fillers
    of each slot (k = 4..8, the tail treated as scribal noise and dropped), effect
    size as total-variation distance; search-corrected over devices x k against
    jittered nulls (sigma 0.1, 0.3; 40 each).  Positive control: planted dice with
    30% of tokens replaced by Voynich tokens.
(b) 'A die picks the table': latent-class model = mixture of C independent slot
    tables (C = 1,2,3,4,6,8,12,16), EM, held-out lines.  A scribe switching among T
    dice tables saturates at C = T and leaves no residual within-class slot MI.
    Positive control: planted corpus with 3 tables chosen by a die.
(c) Other rolled quantities: word length (glyphs) and words per line fitted by devices."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from v14_lib import *
from v14_cycle1 import get_tokens, v7model
from multiprocessing import Pool

def topk_fit(counts, k, dev):
    c = np.array([n for _, n in counts.most_common(k)], float)
    g, assign = fit_device(c, dev)
    if not np.isfinite(g): return np.inf, np.inf
    at = DEVICES[dev]
    if np.allclose(at, at[0]): q = assign / len(at)
    else: q = np.bincount(assign, weights=at, minlength=len(c))
    q = q / q.sum()
    tv = 0.5 * float(np.abs(c / c.sum() - q).sum())
    return (g - (k - 1)) / math.sqrt(2 * (k - 1)), tv

KS3 = range(4, 9)
def noisy_scores(counts):
    z = {}; tv = {}
    for d in DEV_ORDER:
        for k in KS3:
            if k > len(counts) or k > len(DEVICES[d]): continue
            z[(d, k)], tv[(d, k)] = topk_fit(counts, k, d)
    return z, tv

def identify_noisy(z, alpha_z=3.1):
    """least flexible device accepted (z < ~chi2 0.999) at the largest k it can reach"""
    for d in DEV_ORDER:
        ks = [k for (dd, k) in z if dd == d]
        if ks and z[(d, max(ks))] < alpha_z: return d, max(ks)
    return None, None

def job_noisy(name):
    if load('c3a_' + name) is not None: return
    t = time.time(); rng = np.random.default_rng(3)
    if name == 'planted_noisy30':
        L, truth = planted_dice(design='mixed', seed=21)
        vt = [w for l in voy_lines('ZL3b') for w in l['words']]
        r2 = random.Random(4)
        for l in L:
            l['words'] = [w if r2.random() > 0.3 else r2.choice(vt) for w in l['words']]
        toks = parse(L, learn_model(L, 4, 'planted_noisy30'))
    elif name.startswith('voynich_ZL3b_'):
        lang = name[-1]
        toks = [x for x in parse(voy_lines('ZL3b'), v7model('voynich_ZL3b_K4')) if x['lang'] == lang]
    else:
        toks, _, _ = get_tokens(name)
    out = {'slots': []}
    for k in range(4):
        c = Counter(x['f'][k] for x in toks); N = sum(c.values())
        z, tv = noisy_scores(c)
        rec = {'z': z, 'tv': tv, 'ident': identify_noisy(z), 'top': c.most_common(8), 'null': {}}
        for s in (0.1, 0.3):
            nl = []
            for _ in range(40):
                zn, tvn = noisy_scores(jitter_counts(c, s, N, rng))
                nl.append((min(zn.values()), min(tvn.values()), identify_noisy(zn)))
            rec['null'][s] = nl
        out['slots'].append(rec)
    save('c3a_' + name, out); print('c3a', name, round(time.time() - t), flush=True)

# ---------------------------------------------------------------- latent class
def em_lc(Xtr, Xte, C, ncat, iters=150, seed=0):
    rng = np.random.default_rng(seed)
    n, K = Xtr.shape
    pi = np.full(C, 1 / C)
    th = [rng.dirichlet(np.ones(ncat), C) for _ in range(K)]
    def ll(X):
        L = np.log(pi)[None, :] + sum(np.log(th[k][:, X[:, k]].T + 1e-300) for k in range(K))
        m = L.max(1, keepdims=True)
        return L, m[:, 0] + np.log(np.exp(L - m).sum(1))
    prev = -np.inf
    for it in range(iters):
        L, lt = ll(Xtr)
        R = np.exp(L - lt[:, None])
        pi = R.mean(0) + 1e-12; pi /= pi.sum()
        for k in range(K):
            M = np.zeros((C, ncat))
            for v in range(ncat):
                M[:, v] = R[Xtr[:, k] == v].sum(0)
            th[k] = (M + 0.1) / (M + 0.1).sum(1, keepdims=True)
        cur = lt.mean()
        if cur - prev < 1e-6: break
        prev = cur
    _, lte = ll(Xte)
    # residual within-class MI (train), averaged over pairs, from hard assignment
    L, lt = ll(Xtr); z = L.argmax(1)
    res = []
    for i, j in itertools.combinations(range(K), 2):
        tot = 0
        for c in range(C):
            ix = z == c
            if ix.sum() > 50: tot += ix.mean() * mi(Xtr[ix, i], Xtr[ix, j], ncat, ncat)
        res.append(tot)
    return float(lt.mean() / math.log(2)), float(lte.mean() / math.log(2)), float(np.mean(res))

def planted_tables(n=35000, seed=31, T=3):
    rng = np.random.default_rng(seed); r2 = random.Random(seed)
    tabs = [[(d, [r2.choice(P_SLOTS[k]) for _ in range(len(DEVICES[d]))]) for k, d in
             enumerate(['d6', 'coin+d6', 'd6', '2d6sum'])] for _ in range(T)]
    which = rng.integers(0, T, n)
    words = []
    for t in which:
        w = ''.join(fl[roll(d, rng, 1)[0]] for d, fl in tabs[t])
        words.append(w or 'y')
    return chop(words, r2)

CS = (1, 2, 3, 4, 6, 8, 12, 16)
def job_lc(name):
    if load('c3b_' + name) is not None: return
    t = time.time()
    if name == 'planted_3tables':
        L = planted_tables(); toks = parse(L, learn_model(L, 4, 'planted_3tables'))
    else:
        toks, _, _ = get_tokens(name)
    X = code_cols(toks, 4).T
    line = np.array([x['line'] for x in toks])
    te = (line % 5) == 0
    out = {}
    for C in CS:
        best = None
        for s in range(3):
            r = em_lc(X[~te], X[te], C, 10, seed=s)
            if best is None or r[0] > best[0]: best = r
        out[C] = best
        print('c3b', name, C, [round(x, 4) for x in best], flush=True)
    save('c3b_' + name, out); print('c3b', name, 'done', round(time.time() - t), flush=True)

def job_other(name):
    if load('c3c_' + name) is not None: return
    if name.startswith('voynich'):
        L = voy_lines('ZL3b')
    elif name == 'latin_verbose': L = latin_verbose()
    elif name == 'italian_verbose': L = italian_verbose()
    elif name == 'planted_mixed': L, _ = planted_dice()
    rng = np.random.default_rng(5); out = {}
    for q, cnt in (('wordlen', Counter(len(w) for l in L for w in l['words'])),
                   ('wpl', Counter(len(l['words']) for l in L))):
        sc = slot_scores(cnt)
        N = sum(cnt.values())
        nl = [parsimony(slot_scores(jitter_counts(cnt, 0.1, N, rng))) for _ in range(40)]
        out[q] = {'counts': sorted(cnt.items()), 'ident': parsimony(sc), 'z_full': {d: zof(sc, d, 'full') for d in DEV_ORDER},
                  'null_ident': Counter(nl)}
    save('c3c_' + name, out); print('c3c', name, flush=True)

JOBS = ([(job_lc, n) for n in ('planted_3tables', 'voynich_ZL3b', 'latin_verbose', 'italian_verbose', 'planted_mixed')] +
        [(job_noisy, n) for n in ('planted_noisy30', 'voynich_ZL3b', 'voynich_ZL3b_A', 'voynich_ZL3b_B', 'latin_verbose', 'italian_verbose')] +
        [(job_other, n) for n in ('voynich', 'latin_verbose', 'italian_verbose', 'planted_mixed')])

def runjob(j): j[0](j[1])

if __name__ == '__main__':
    with Pool(2) as p:
        list(p.imap_unordered(runjob, JOBS))
