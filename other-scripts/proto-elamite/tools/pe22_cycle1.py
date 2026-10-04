"""PE-22 cycle 1: the census. How many signs, sign variants, entry strings and sign pairs did
the Proto-Elamite scribes use, of which we see only a sample?

Units: base signs (variants merged), graphs (variants kept), entry strings (whole entries of
>= 2 signs, numerals dropped), sign bigrams inside entries.
Estimators: Chao1, ACE, Chao2 (tablets as sampling units), jack2, Chao-Jost coverage,
Zipf-Mandelbrot ABC (4,000 random communities per unit), tablet bootstrap of Chao1 (200).
Controls:
  planted  - Zipf-Mandelbrot repertoires of KNOWN size, written on PE-shaped tablets with a
             per-tablet topic (Dirichlet, alpha 30), estimators applied blind: recovery ratio;
  PC       - proto-cuneiform admin, full and cut to PE size;
  endemism - share of recurrent sign types found at one site only, against 100 token
             shuffles across tablets (site sizes kept).
"""
import sys, json, collections, random, time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from pe22_common import *

rng = np.random.default_rng(2201)
prng = random.Random(2201)
t0 = time.time()
res = {}
PE = load_pe()
PE = [d for d in PE if d['site'] != 'Larsa']
PC = load_pc()
KEYS = ('signs', 'graphs', 'words', 'bigr')


def census(docs, name, abc_keys=('signs', 'words'), boot=200):
    out = {}
    for k in KEYS:
        e = estimates(docs, k)
        A = list(collections.Counter(w for d in docs for w in d[k]).values())
        if k in abc_keys:
            post = abc_fit(A, rng)
            e['abc_S'] = iv([S for _, S, _, _ in post])
        bs = []
        D = [d for d in docs if d[k]]
        for b in range(boot):
            smp = [D[i] for i in rng.integers(0, len(D), len(D))]
            A2 = list(collections.Counter(w for d in smp for w in d[k]).values())
            bs.append(chao1(A2))
        e['chao1_boot'] = iv(bs)
        out[k] = e
        print(name, k, json.dumps(e, default=float), round(time.time() - t0), flush=True)
    return out


res['PE_all'] = census(PE, 'PE_all')
tr, ho = pe_split(PE)
res['PE_train'] = census(tr, 'PE_train', abc_keys=(), boot=60)
res['PC_all'] = census(PC, 'PC_all', abc_keys=('signs',), boot=60)
nPE = sum(len(d['signs']) for d in PE)
PCs = PC[:]
prng.shuffle(PCs)
cut, t = [], 0
for d in PCs:
    if t >= nPE:
        break
    cut.append(d)
    t += len(d['signs'])
res['PC_PEsize'] = census(cut, 'PC_PEsize', abc_keys=('signs',), boot=60)
res['PC_PEsize']['note'] = 'PC tablets drawn at random until the PE sign-token count is reached; truth = PC_all S'

# ---------------- planted repertoires of known size on PE-shaped tablets
sizes = [len(d['signs']) for d in PE if d['signs']]
A_pe = list(collections.Counter(w for d in PE for w in d['signs']).values())
post = abc_fit(A_pe, rng)
pl = []
for S_true in (600, 1200, 2400, 4800):
    for r in range(12):
        _, _, a, q = post[r % len(post)]
        p = zm(S_true, a, q)
        toks = []
        for m in sizes:
            th = rng.dirichlet(30 * p + 1e-9) if r % 2 else p
            toks.append(rng.multinomial(m, th))
        X = np.sum(toks, 0)
        A = X[X > 0]
        inc = (np.array(toks) > 0).sum(0)
        inc = inc[inc > 0]
        row = dict(S_true=S_true, clumped=bool(r % 2), S_obs=int(len(A)), chao1=chao1(A), ace=ace(A),
                   chao2=chao2(inc, len(sizes)), jack2=jack2(inc, len(sizes)))
        if r < 4:
            pa = abc_fit(list(A), rng, n_sims=2000, keep=80)
            row['abc'] = float(np.median([S for _, S, _, _ in pa]))
        pl.append(row)
    print('planted', S_true, round(time.time() - t0), flush=True)
summ_pl = {}
for S_true in (600, 1200, 2400, 4800):
    for cl in (False, True):
        rr = [x for x in pl if x['S_true'] == S_true and x['clumped'] == cl]
        summ_pl['%d_%s' % (S_true, 'clumped' if cl else 'flat')] = {
            k: float(np.median([x[k] / S_true for x in rr if k in x])) for k in ('S_obs', 'chao1', 'ace', 'chao2', 'jack2', 'abc')
            if any(k in x for x in rr)}
res['planted_recovery_ratio'] = summ_pl
print(json.dumps(summ_pl), flush=True)

# ---------------- site endemism of sign types
def endemism(docs, key='signs'):
    st = collections.defaultdict(set)
    ab = collections.Counter()
    for d in docs:
        for w in d[key]:
            st[w].add(d['site'])
            ab[w] += 1
    rec = [w for w in ab if ab[w] >= 2]
    one = sum(1 for w in rec if len(st[w]) == 1)
    nonsusa_only = sum(1 for w in rec if 'Susa' not in st[w])
    per = collections.Counter()
    for w in ab:
        if len(st[w]) == 1:
            per[next(iter(st[w]))] += 1
    return dict(n_rec=len(rec), one_site=one / len(rec), nonsusa_only=nonsusa_only, endemic_by_site=dict(per))


real = endemism(PE)
null = [endemism(L.token_shuffle(PE, prng, key='signs')) for _ in range(100)]
res['endemism'] = dict(real=real, null_one_site=iv([x['one_site'] for x in null]),
                       null_nonsusa_only=iv([x['nonsusa_only'] for x in null]),
                       null_endemic_by_site={s: iv([x['endemic_by_site'].get(s, 0) for x in null])
                                             for s in real['endemic_by_site']})
print(json.dumps(res['endemism']), flush=True)
json.dump(res, open(os.path.join(CK, 'cycle1.json'), 'w'), indent=1, default=float)
print('done', round(time.time() - t0))
