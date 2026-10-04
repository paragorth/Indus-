"""pe7 cycle 3: (A) 'mutation clock' / find-group test and (B) which elements behave as inherited.

A. Random samples of 200 distinct names; tree as in cycles 1-2.  Labels that are
   properties of the find context: PE publication volume (excavation campaign),
   Sb museum-number block of 100 (accession batch), site; both permuted at the
   tablet level with same-tablet pairs excluded.  Mantel: name distance vs |Sb
   number difference| (Spearman), tablet-level permutation.
   Positive: Ur III and OB seal owners by provenience; Linear B by site (KN/PY).
   Negative: random strings carrying the PE labels.
B. Whole-corpus element transmission (no trees):
   Ur III / OB father-son: share of pairs sharing >= 1 element vs fathers permuted
   (10,000x); per-element lift P(son has e | father has e) / P(son has e).
   PE / Linear B: names on the same tablet vs names permuted among tablet slots
   (tablet sizes kept, 2,000x); per-sign co-tablet lift with permutation p.
   Per-character retention index on best trees (PE real vs shuffled-element null,
   Ur III pairs) -> does tree-homoplasy grading recover the truly transmitted
   Ur III elements?  (validation of the grading)"""
import sys, random, json, os, time
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe7_common import *  # noqa

N, NREP = 200, 6


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def mantel_sb(D, meta, rng, nperm=999):
    tab = [x['tablets'][0] for x in meta]
    sb = {t: x['sb'][0] for t, x in zip(tab, meta)}
    idx = [i for i, t in enumerate(tab) if sb[t] is not None]
    if len(idx) < 30:
        return None
    ut = sorted({tab[i] for i in idx})
    pairs = [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:] if tab[i] != tab[j]]
    I = np.array([p[0] for p in pairs]); J = np.array([p[1] for p in pairs])
    d = D[I, J]
    from scipy.stats import rankdata
    rd = rankdata(d)

    def stat(m):
        x = np.abs(np.array([m[tab[i]] for i in I]) - np.array([m[tab[j]] for j in J]))
        return float(np.corrcoef(rd, rankdata(x))[0, 1])
    obs = stat(sb)
    null = []
    vals = [sb[t] for t in ut]
    for _ in range(nperm):
        p = rng.permutation(vals)
        null.append(stat(dict(zip(ut, p))))
    null = np.array(null)
    return {'rho': obs, 'null': float(null.mean()), 'p': float((1 + (null >= obs).sum()) / (nperm + 1)), 'pairs': len(pairs)}


def part_a(job):
    kind, rep = job
    key = 'c3a_%s_%d' % (kind, rep)
    r = ckpt(key)
    if r:
        return key, r
    C = corpora()
    rng = random.Random(9000 + rep * 13 + len(kind))
    trng = random.Random(rep * 11 + 1)
    nrng = np.random.default_rng(rep + 77)
    t0 = time.time()
    base = kind.split('_')[0] if kind.startswith(('PE', 'RAND')) else kind
    if kind in ('PE', 'PE_shuf', 'RAND'):
        names, meta = sample_corpus(C, 'PE', N, rng)
        if kind == 'PE_shuf':
            names = shuffle_elements(names, rng)
        if kind == 'RAND':
            names = random_strings(names, rng)
        tab = [x['tablets'][0] for x in meta]
        same = np.array([[tab[i] == tab[j] for j in range(N)] for i in range(N)])
        labs = {'volume': ([{x['vol'][0]} for x in meta], tab, same),
                'sbblock': ([{x['sb'][0]} if x['sb'][0] is not None else set() for x in meta], tab, same),
                'site': ([{'Susa' if x['vol'][0].startswith(('MDP', 'TCL')) else 'other'} for x in meta], tab, same)}
    elif kind in ('UR3_SEAL', 'OB_SEAL'):
        names, meta = sample_corpus(C, kind, N, rng)
        labs = {'prov': ([set(x['prov'][:1]) - {''} for x in meta], None, None)}
    elif kind == 'LINB':
        names, meta = sample_corpus(C, 'LINB', N, rng)
        labs = {'site': ([set(x['site'][:1]) for x in meta], None, None)}
    elif kind == 'UR3_PAT':
        names, meta = sample_corpus(C, 'UR3_PAT', N, rng)
        labs = {}
    out, t, D, em, P = tree_stats(names, trng, restarts=6)
    out.pop('_co')
    out['rates'] = [float(x) for x in out['rates']]
    res = {'tree': out}
    for lab, (sets, grp, excl) in labs.items():
        if len({x for s in sets for x in s}) < 2:
            continue
        res[lab] = {'PS': label_assoc(t, sets, rng), 'dist': pair_test(D, sets, nrng, groups=grp, exclude=excl)}
    if kind in ('PE', 'RAND'):
        res['mantel_sb'] = mantel_sb(D, meta, nrng)
    steps = P.per_char(t)
    cnt = Counter(s for w in names for s in set(w))
    res['perchar'] = {str(P.chars[i]): [steps[i], cnt[P.chars[i]]] for i in range(P.m)}
    res['sec'] = time.time() - t0
    save_ckpt(key, res)
    return key, res


# ------------------------------------------------------------------ part B
def pat_transmission(C, name, nperm=10000, seed=1):
    pool = [x for x in C[name] if x['son'] and x['father']]
    S = [set(x['son']) for x in pool]
    F = [set(x['father']) for x in pool]
    rng = np.random.default_rng(seed)
    obs = np.mean([bool(a & b) for a, b in zip(S, F)])
    null = []
    for _ in range(min(nperm, 2000)):
        p = rng.permutation(len(F))
        null.append(np.mean([bool(S[i] & F[j]) for i, j in enumerate(p)]))
    null = np.array(null)
    # per element lift
    n = len(pool)
    sh = Counter(); fa = Counter(); so = Counter()
    for a, b in zip(S, F):
        for e in b:
            fa[e] += 1
        for e in a:
            so[e] += 1
        for e in a & b:
            sh[e] += 1
    el = []
    for e in fa:
        if fa[e] >= 8 and so[e] >= 8:
            exp = fa[e] * so[e] / n
            el.append((e, sh[e], round(exp, 2), round(sh[e] / exp, 2) if exp else None, fa[e], so[e]))
    el.sort(key=lambda x: -(x[1] - x[2]))
    return {'pairs': n, 'share_any': float(obs), 'null': float(null.mean()),
            'z': float((obs - null.mean()) / (null.std() + 1e-12)), 'p': float((1 + (null >= obs).sum()) / (len(null) + 1)),
            'elements': el[:25], 'elements_all': el}


def cotablet(names_by_tab, nperm=2000, seed=2, minc=8):
    """names_by_tab: list of lists of names (tuples; one list per tablet, >= 2 names).
    Statistics: share of co-tablet pairs sharing >= 1 element, the same first element,
    the same last element; per-element shared counts."""
    rng = np.random.default_rng(seed)
    slots = [len(x) for x in names_by_tab]
    seqs = [tuple(s) for x in names_by_tab for s in x]
    flat = [set(s) for s in seqs]
    fst = [s[0] for s in seqs]
    lst = [s[-1] for s in seqs]
    cnt = Counter(e for s in flat for e in s)

    def stat(order):
        k = 0; anyshare = 0; npair = 0
        per = Counter()
        for L in slots:
            ix = order[k:k + L]
            grp = [flat[i] for i in ix]
            k += L
            for a in range(L):
                for b in range(a + 1, L):
                    npair += 1
                    if fst[ix[a]] == fst[ix[b]]:
                        per['<FIRST>'] += 1
                    if lst[ix[a]] == lst[ix[b]]:
                        per['<LAST>'] += 1
                    inter = grp[a] & grp[b]
                    if inter:
                        anyshare += 1
                    for e in inter:
                        per[e] += 1
        return anyshare / npair, per, npair
    obs, per, npair = stat(np.arange(len(flat)))
    nulls = []
    pernull = defaultdict(list)
    els = [e for e in cnt if cnt[e] >= minc] + ['<FIRST>', '<LAST>']
    cnt['<FIRST>'] = cnt['<LAST>'] = len(flat)
    for _ in range(nperm):
        v, pn, _ = stat(rng.permutation(len(flat)))
        nulls.append(v)
        for e in els:
            pernull[e].append(pn[e])
    nulls = np.array(nulls)
    rows = []
    for e in els:
        a = np.array(pernull[e])
        rows.append((e, per[e], round(float(a.mean()), 2), round(per[e] / a.mean(), 2) if a.mean() > 0 else None,
                     float((1 + (a >= per[e]).sum()) / (nperm + 1)), cnt[e]))
    rows.sort(key=lambda x: x[4])
    return {'tablets': len(slots), 'names': len(flat), 'pairs': npair, 'share_any': obs,
            'null': float(nulls.mean()), 'z': float((obs - nulls.mean()) / (nulls.std() + 1e-12)),
            'p': float((1 + (nulls >= obs).sum()) / (nperm + 1)), 'elements': rows}


def part_b():
    key = 'c3b'
    r = ckpt(key)
    if r:
        return r
    C = corpora()
    res = {}
    for nm in ('UR3_PAT', 'OB_PAT'):
        res[nm] = pat_transmission(C, nm)
    # PE: distinct names per tablet (a name counted once per tablet)
    by = defaultdict(list)
    for x in C['PE']:
        for t in x['tablets']:
            by[t].append(tuple(x['seq']))
    res['PE_cotablet'] = cotablet([v for v in by.values() if len(v) >= 2])
    # PE excluding herd office tablets
    from pe7_build import HERD
    res['PE_cotablet_noherd'] = cotablet([v for t, v in by.items() if len(v) >= 2 and t not in HERD])
    by = defaultdict(list)
    for x in C['LINB']:
        if len(x['seq']) >= 2:
            for t in x['tablets']:
                by[t].append(tuple(x['seq']))
    res['LINB_cotablet'] = cotablet([v for v in by.values() if len(v) >= 2])
    # Ur III families as 'tablets' of 2 (positive for the co-tablet machinery)
    fam = [[tuple(x['son']), tuple(x['father'])] for x in C['UR3_PAT']]
    res['UR3_family_as_tablet'] = cotablet(fam, nperm=500)
    fam = [[tuple(x['son']), tuple(x['father'])] for x in C['OB_PAT']]
    res['OB_family_as_tablet'] = cotablet(fam, nperm=500)
    # Ur III administrative names on one tablet (Drehem 'ki PN-ta' / 'giri3 PN', pe4 controls): not kin
    ctl = json.load(open(os.path.join(DATA, 'pe4_controls.json')))['names']
    by = defaultdict(set)
    for r in ctl:
        if len(r['attr']) >= 2:
            by[r['t']].add(tuple(r['attr']))
    res['UR3_admin_cotablet'] = cotablet([sorted(v) for v in by.values() if len(v) >= 2], nperm=500)
    save_ckpt(key, res)
    return res


if __name__ == '__main__':
    kinds = ['PE', 'PE_shuf', 'RAND', 'UR3_SEAL', 'OB_SEAL', 'LINB', 'UR3_PAT']
    jobs = [(k, r) for r in range(NREP) for k in kinds]
    res = {}
    with Pool(2) as pool:
        b = pool.apply_async(part_b)
        for key, out in pool.imap_unordered(part_a, jobs):
            res[key] = out
            brief = {l: (v['PS']['p'], v['dist'] and round(v['dist']['z'], 2)) for l, v in out.items() if isinstance(v, dict) and 'PS' in v}
            print(key, round(out['sec']), brief, out.get('mantel_sb') and round(out['mantel_sb']['p'], 3), flush=True)
        res['B'] = b.get()
    json.dump(res, open(os.path.join(DATA, 'pe7_cycle3.json'), 'w'))
