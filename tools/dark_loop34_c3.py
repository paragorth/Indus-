"""S-DARK-34 cycle 3: NUMERIC checksums. Is the stroke numeral (S204 value) a function of the rest of the text?
Texts (all object types, complete, direction recorded) with exactly one valued numeral token (W1, W2 and W31 are
markers, S234/S-DARK-15, excluded); target = its value (1-9), features = the rest of the text.
 (a) classifier (tree / forest / kNN) as in cycles 1-2, train Mohenjo-daro + Harappa, test other sites + IM77-only,
     with the planted checksum (value := 3 + (sum of other sign indices) mod 6), planted class sign and label shuffle;
 (b) simple functions: n signs, n+1, n distinct, n fish, n non-frame signs, position of the numeral, sum of sign
     indices mod 10/mod 7 - exact-match rate and Spearman rho against a permutation null (values permuted across
     texts within site x object type, 2,000 reps);
 (c) annealed sign-to-value map (0-9 on the 60 commonest non-numeral signs, value = sum over the rest), 20 restarts,
     fitted on the train set, scored on the held-out sets, against the same anneal on a corpus whose numeral values
     were permuted (the null) and on a planted corpus (value := clip(sum of a random map), the power check).
Usage: python3 tools/dark_loop34_c3.py <seq_raw|seq_strong|seq_all>"""
import sys, json
sys.argv = [sys.argv[0], '3'] + sys.argv[1:]
from dark_loop34 import *
from scipy.stats import spearmanr

EXCLN = {1, 2, 31}
def num_pos(seq):
    ps = [i for i, t in enumerate(seq) if t in NUMER and t not in EXCLN]
    return ps[0] if len(ps) == 1 else None
TARGETS['numeral_value'] = (lambda o: (VAL[o['seq'][num_pos(o['seq'])]], num_pos(o['seq'])),
                            lambda o: num_pos(o['seq']) is not None and len(o['seq']) >= 3)
ALL = [o for o in OBJ]
train = [o for o in ALL if o['big'] and TARGETS['numeral_value'][1](o)]
sites = [o for o in ALL if not o['big'] and TARGETS['numeral_value'][1](o)]
im77 = [o for o in NEW if TARGETS['numeral_value'][1](o)]
log = open(SP + f'loop34_c3_{LV}.txt', 'w')
def P(s): print(s); log.write(s + '\n'); log.flush()
vals = lambda L: collections.Counter(VAL[o['seq'][num_pos(o['seq'])]] for o in L)
P(f'# S-DARK-34 cycle 3, level {LV}: one-numeral texts train {len(train)} {sorted(vals(train).items())}, '
  f'held-out sites {len(sites)} {sorted(vals(sites).items())}, IM77-only {len(im77)} {sorted(vals(im77).items())}')
P(f'  train by type {collections.Counter(o["ot"] for o in train).most_common()}; seals only {sum(o["ot"]=="SEAL" for o in train)}')

# (a) classifier
P('--- (a) classifier, target = numeral value (classes 1-9)')
tests = {'sites': sites, 'im77': im77}
for tag, kw in (('TEXT-ONLY ', dict(use_facts=False)), ('TEXT+FACTS', dict(use_facts=True)),
                ('PLANT-CK  ', dict(use_facts=False, plant='checksum', report_feats=False)),
                ('PLANT-CS  ', dict(use_facts=False, plant='classsign', report_feats=False)),
                ('SHUFFLED  ', dict(use_facts=False, shuffle=True, report_feats=False))):
    r, _ = evaluate('numeral_value', train, tests if 'facts' not in tag else {'sites': sites}, **kw); P(tag + ' ' + fmt(r))
# seals only
tr_s = [o for o in train if o['ot'] == 'SEAL']; te_s = {'sites': [o for o in sites if o['ot'] == 'SEAL'], 'im77': [o for o in im77 if o['ot'] == 'SEAL']}
if len(tr_s) >= 40:
    r, _ = evaluate('numeral_value', tr_s, te_s, use_facts=False); P('SEALS-ONLY ' + fmt(r))

# (b) simple functions
P('--- (b) simple functions of the rest of the text vs the numeral value (exact match rate; Spearman rho) with permutation null')
def rest_of(o):
    p = num_pos(o['seq']); return [t for i, t in enumerate(o['seq']) if i != p], p
FRAME = OPEN | MARK | SUF | set(CLOSERS)
FUNCS = {
    'n_signs': lambda r, p: len(r), 'n_signs+1': lambda r, p: len(r) + 1, 'n_distinct': lambda r, p: len(set(r)),
    'n_fish': lambda r, p: sum(t in FISH for t in r), 'n_nonframe': lambda r, p: sum(t not in FRAME for t in r),
    'position': lambda r, p: p, 'position_from_end': lambda r, p: len(r) - p,
    'sumW_mod10': lambda r, p: sum(abs(t) for t in r) % 10, 'sumW_mod7': lambda r, p: sum(abs(t) for t in r) % 7,
    'sumrank_mod9+1': lambda r, p: sum(RANK.get(t, 120) for t in r) % 9 + 1,
}
def strata_perm(objs, y, reps, rng):
    groups = collections.defaultdict(list)
    for i, o in enumerate(objs): groups[(o['big'], o['ot'])].append(i)
    out = []
    y = np.array(y)
    for _ in range(reps):
        yp = y.copy()
        for idx in groups.values():
            idx = np.array(idx); yp[idx] = y[rng.permutation(idx)]
        out.append(yp)
    return out
rng = np.random.default_rng(34)
for setname, objs in (('train', train), ('sites', sites), ('im77', im77), ('all', train + sites + im77)):
    if len(objs) < 15: continue
    y = [VAL[o['seq'][num_pos(o['seq'])]] for o in objs]
    perms = strata_perm(objs, y, 2000, rng)
    for fname, f in FUNCS.items():
        fx = [f(*rest_of(o)) for o in objs]
        em = float(np.mean(np.array(fx) == np.array(y)))
        rho = spearmanr(fx, y).correlation if len(set(fx)) > 1 else 0.0
        em0 = np.array([np.mean(np.array(fx) == yp) for yp in perms])
        rho0 = np.array([spearmanr(fx, yp).correlation if len(set(fx)) > 1 else 0.0 for yp in perms[:500]])
        p_em = float(np.mean(em0 >= em)); p_rho = float(np.mean(np.abs(rho0) >= abs(rho)))
        flag = ' <==' if (p_em < 0.01 or p_rho < 0.01) else ''
        P(f'  {setname:5s} n={len(objs):3d} {fname:16s} exact {em:.3f} (null {em0.mean():.3f}, P={p_em:.3f}) rho {rho:+.3f} (null |rho| 95% {np.quantile(np.abs(rho0),0.95):.3f}, P={p_rho:.3f}){flag}')

# (c) annealed sign-to-value map
P('--- (c) annealed additive map: value = sum of per-sign values over the rest (0-9 on 60 commonest signs, +other)')
cnt = collections.Counter(t for o in train for t in rest_of(o)[0])
SIG = [s for s, _ in cnt.most_common(60)]; SI = {s: i for i, s in enumerate(SIG)}
def design(objs):
    X = np.zeros((len(objs), 61))
    for i, o in enumerate(objs):
        for t in rest_of(o)[0]: X[i, SI.get(t, 60)] += 1
    return X
def anneal(X, y, restarts=20, steps=3000, seed=0):
    rng = np.random.default_rng(seed); best = (-2, None)
    for r in range(restarts):
        w = rng.integers(0, 10, 61); cur = spearmanr(X @ w, y).correlation or 0; T = 0.3
        for s in range(steps):
            w2 = w.copy(); w2[rng.integers(61)] = rng.integers(0, 10)
            c2 = spearmanr(X @ w2, y).correlation or 0
            if c2 > cur or rng.random() < math.exp((c2 - cur) / T): w, cur = w2, c2
            T *= 0.999
        if cur > best[0]: best = (cur, w.copy())
    return best
Xtr = design(train); ytr = np.array([VAL[o['seq'][num_pos(o['seq'])]] for o in train])
Xs = design(sites); ys = np.array([VAL[o['seq'][num_pos(o['seq'])]] for o in sites])
Xi = design(im77); yi = np.array([VAL[o['seq'][num_pos(o['seq'])]] for o in im77])
def score(w, X, y): return (spearmanr(X @ w, y).correlation if len(y) > 5 else float('nan'))
for label, ytrain in (('REAL', ytr), ('PERMUTED-1', rng.permutation(ytr)), ('PERMUTED-2', rng.permutation(ytr)), ('PERMUTED-3', rng.permutation(ytr))):
    tr_rho, w = anneal(Xtr, ytrain, restarts=12, steps=2000, seed=1)
    P(f'  {label:10s} train rho {tr_rho:.3f} -> held-out sites rho {score(w, Xs, ys):+.3f} (n={len(ys)}), IM77 rho {score(w, Xi, yi):+.3f} (n={len(yi)}); all-ones rule sites {score(np.ones(61), Xs, ys):+.3f}')
# planted power check
wt = rng.integers(0, 4, 61); yp_tr = np.clip(np.round((Xtr @ wt) / 2 + rng.normal(0, 1.0, len(ytr))), 1, 9); yp_s = np.clip(np.round((Xs @ wt) / 2 + rng.normal(0, 1.0, len(ys))), 1, 9)
tr_rho, w = anneal(Xtr, yp_tr, restarts=12, steps=2000, seed=2)
P(f'  PLANTED    truth train rho {spearmanr(Xtr @ wt, yp_tr).correlation:.3f}, truth sites {spearmanr(Xs @ wt, yp_s).correlation:.3f}; recovered train {tr_rho:.3f} -> sites {score(w, Xs, yp_s):+.3f}; value recovery rho(w, truth) {spearmanr(w, wt).correlation:.2f}')
