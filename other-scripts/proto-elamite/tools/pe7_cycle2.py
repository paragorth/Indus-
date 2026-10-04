"""pe7 cycle 2: do tree clades map onto tablets, headers, offices or the herd office?

Per replicate: a cluster sample of 200 distinct names (whole tablets), edit-model
distances, best parsimony tree (NJ + 5 random restarts, SPR), MCMC pair
co-membership (clades <= 8 leaves).  For each label:
  PS   : Fitch parsimony score of the label on the best tree vs 499 leaf permutations
  dist : mean edit distance of label-sharing pairs minus other pairs vs 999 permutations
  co   : the same with posterior co-membership (1 - co as a distance)
Tablet-level labels (header, volume) are permuted at the tablet level and same-tablet
pairs are excluded, so they are not a re-run of the tablet test.
Positive controls: Ur III / OB father-son pairs (label = family), Chinese surnames.
Negative: Linear B personnel (label = tablet), random strings carrying PE labels.
Recovery: planted families (4 names from one founder, 1-3 mutations each under the
fitted PE edit rates) replace 15 PE tablets' worth of names (60/200)."""
import sys, random, json, os, time
import numpy as np
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe7_common import *  # noqa

N, NREP = 200, 8


def mutate(w, em, rng, pool_el, wts):
    pm, ps, pi, pd = em.rates
    w = list(w)
    z = ps + pi + pd
    r = rng.random() * z
    if r < ps and w:
        w[rng.randrange(len(w))] = rng.choices(pool_el, wts)[0]
    elif r < ps + pi or len(w) <= 1:
        w.insert(rng.randrange(len(w) + 1), rng.choices(pool_el, wts)[0])
    else:
        w.pop(rng.randrange(len(w)))
    return tuple(w)


def plant(C, rng, m, nfam=15, size=4):
    names, meta = sample_corpus(C, 'PE_tab', N - nfam * size, rng)
    labs = [set(x['tablets'][:1]) for x in meta]
    allpe = [tuple(x['seq']) for x in C['PE']]
    em = EditModel(allpe)
    em.set_rates(0.40, 0.30, 0.15, 0.15)   # fitted PE rates (cycle 1)
    f = Counter(s for w in allpe for s in w)
    el, wt = zip(*f.items())
    used = set(names)
    for k in range(nfam):
        founder = rng.choice(allpe)
        fam = set()
        tries = 0
        while len(fam) < size and tries < 200:
            tries += 1
            w = founder
            for _ in range(m):
                w = mutate(w, em, rng, el, wt)
            if len(w) >= 2 and w not in used and w not in fam:
                fam.add(w)
        for w in fam:
            names.append(w)
            labs.append({'PLANT%d' % k})
            used.add(w)
    return names, labs


def tests(names, labsets, rng, tree_rng, groups_for=None):
    out, t, D, em, P = tree_stats(names, tree_rng, restarts=6)
    co = out.pop('_co')
    out['rates'] = [float(x) for x in out['rates']]
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    res = {'tree': out}
    for lab, (sets, grp, excl) in labsets.items():
        if sum(1 for s in sets if s) < 4:
            continue
        r = {'PS': label_assoc(t, sets, rng)}
        r['dist'] = pair_test(D, sets, nrng, groups=grp, exclude=excl)
        r['co'] = pair_test(1.0 - co, sets, nrng, groups=grp, exclude=excl)
        res[lab] = r
    return res


def pe_labels(meta, names):
    n = len(names)
    tab = [x['tablets'][0] for x in meta]
    same = np.array([[tab[i] == tab[j] for j in range(n)] for i in range(n)])
    L = {}
    L['tablet'] = ([set(x['tablets']) for x in meta], None, None)
    L['class'] = ([set(c for c in x['cls'] if c != '-') for x in meta], None, None)
    L['system'] = ([set(x['sys']) for x in meta], None, None)
    L['header'] = ([{x['hdr'][0]} if x['hdr'][0] else set() for x in meta], tab, same)
    L['volume'] = ([{x['vol'][0]} for x in meta], tab, same)
    return L


def run(job):
    kind, rep = job
    key = 'c2_%s_%d' % (kind, rep)
    r = ckpt(key)
    if r:
        return key, r
    C = corpora()
    rng = random.Random(5000 + rep * 17 + hash(kind) % 97)
    trng = random.Random(rep * 7 + 3)
    t0 = time.time()
    if kind == 'PE_tab':
        names, meta = sample_corpus(C, 'PE_tab', N, rng)
        res = tests(names, rng=rng, tree_rng=trng, labsets=pe_labels(meta, names))
    elif kind == 'PE_herd':
        herd = [x for x in C['PE'] if x['herd']]
        other = [x for x in C['PE'] if not x['herd']]
        meta = herd + rng.sample(other, N - len(herd))
        names = [tuple(x['seq']) for x in meta]
        res = tests(names, rng=rng, tree_rng=trng,
                    labsets={'herd': ([{'H'} if x['herd'] else {'O'} for x in meta], None, None),
                            'herdpair': ([{'H'} if x['herd'] else set() for x in meta], None, None)})
    elif kind in ('UR3_PAT', 'OB_PAT', 'CN_FULL'):
        names, meta = sample_corpus(C, kind, N, rng)
        res = tests(names, rng=rng, tree_rng=trng, labsets={'family': ([set(x['fam']) for x in meta], None, None)})
    elif kind == 'LINB_tab':
        names, meta = sample_corpus(C, 'LINB_tab', N, rng)
        res = tests(names, rng=rng, tree_rng=trng,
                    labsets={'tablet': ([set(x['tablets']) for x in meta], None, None),
                             'series': ([set(x['series']) for x in meta], None, None)})
    elif kind == 'RAND_tab':
        names, meta = sample_corpus(C, 'PE_tab', N, rng)
        names = random_strings(names, rng)
        res = tests(names, rng=rng, tree_rng=trng, labsets=pe_labels(meta, names))
    elif kind.startswith('PLANT'):
        m = int(kind[-1])
        names, labs = plant(C, rng, m)
        res = tests(names, rng=rng, tree_rng=trng, labsets={'tablet': (labs, None, None)})
    res['sec'] = time.time() - t0
    save_ckpt(key, res)
    return key, res


if __name__ == '__main__':
    kinds = ['PE_tab', 'PE_herd', 'UR3_PAT', 'OB_PAT', 'CN_FULL', 'LINB_tab', 'RAND_tab', 'PLANT1', 'PLANT2', 'PLANT3']
    jobs = [(k, r) for r in range(NREP) for k in kinds]
    res = {}
    with Pool(2) as pool:
        for key, out in pool.imap_unordered(run, jobs):
            res[key] = out
            brief = {l: (v['PS']['p'], v['dist'] and round(v['dist']['z'], 2), v['co'] and round(v['co']['z'], 2))
                     for l, v in out.items() if isinstance(v, dict) and 'PS' in v}
            print(key, round(out['sec']), brief, flush=True)
    json.dump(res, open(os.path.join(DATA, 'pe7_cycle2.json'), 'w'))
