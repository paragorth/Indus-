"""pe42 cycle 3: WHICH ENTRIES WERE GROUPED IN THE HEAD?

If a scribe adds a list in mental chunks (runs of neighbouring lines, a
column, the part written before he turned the tablet), his slips drop or
repeat whole CHUNKS; if he adds line by line, a two-line slip hits any two
lines.  Families (applied to failing tablets only):
  RUN2 / RUN3     omit 2 / 3 adjacent entries          (chunk)
  PAIR2           omit 2 non-adjacent entries          (comparator, no chunk)
  PREFIX          total = sum of the first k entries (k = 1..n-2; stopped early)
  SUFFIX          total = sum of the last k entries   (started late)
  DRUN2           2 adjacent entries counted twice
  DPAIR2          2 non-adjacent entries counted twice
Score per family = failing tablets explained; normalised by instance counts
(chunk share expected from the number of adjacent vs non-adjacent pairs).
Nulls: totals reassigned (200), noise (200). Calibration: planted RUN2 and
planted PAIR2 must be told apart (ratio statistic). Corpora PE, PC, MES."""
import collections, itertools, json, random, sys
from multiprocessing import Pool
from pe42_common import *

FAM3 = ['RUN2', 'RUN3', 'PAIR2', 'PREFIX', 'SUFFIX', 'DRUN2', 'DPAIR2']


def inst3(E, S):
    n = len(E)
    out = []
    for i in range(n - 1):
        out.append(('RUN2', S - E[i] - E[i + 1]))
        out.append(('DRUN2', S + E[i] + E[i + 1]))
    for i in range(n - 2):
        out.append(('RUN3', S - E[i] - E[i + 1] - E[i + 2]))
    for i, j in itertools.combinations(range(n), 2):
        if j > i + 1:
            out.append(('PAIR2', S - E[i] - E[j]))
            out.append(('DPAIR2', S + E[i] + E[j]))
    acc = 0
    for k in range(1, n - 1):
        acc += E[k - 1]
        out.append(('PREFIX', acc))
    acc = 0
    for k in range(1, n - 1):
        acc += E[n - k]
        out.append(('SUFFIX', acc))
    return out


def explain3(case, corp):
    if closes(case, corp):
        return None
    ex = set(); N = collections.Counter()
    for h in case['hyps']:
        for mset in mapsets(h, corp):
            pvs = [pair_vals(p, corp.maps[mn]) for p, mn in zip(h, mset)]
            if any(pv is None for pv in pvs):
                continue
            ok = [pv[0] == sum(pv[1]) for pv in pvs]
            for k, pv in enumerate(pvs):
                if ok[k] or not all(ok[j] for j in range(len(h)) if j != k):
                    continue
                T, E = pv
                for fam, pred in inst3(E, sum(E)):
                    N[fam] += 1
                    if pred == T:
                        ex.add(fam)
    # single-entry explanations (OMIT/DOUBLE) take precedence: a tablet fixed by
    # one omission is not evidence for a chunk
    _, ex1, _ = explain(case, corp, None)
    single = bool(ex1.get('OMITH') or ex1.get('OMITO') or ex1.get('DOUBLE'))
    return {'ex': sorted(ex), 'N': dict(N), 'single': single}


def count3(C, corp):
    c = collections.Counter(); Nadj = Nnon = 0
    for case in C:
        r = explain3(case, corp)
        if r is None:
            continue
        c['_fail'] += 1
        if r['single']:
            c['_single'] += 1
            continue
        for f in r['ex']:
            c[f] += 1
    return c


def plant_multi(C, corp, rng, fam, nmax=30):
    out = []
    ks = [k for k, c in enumerate(C) if closes(c, corp)]
    rng.shuffle(ks)
    for k in ks:
        if len(out) >= nmax:
            break
        c = C[k]; h = c['hyps'][0]
        mset = next((ms for ms in mapsets(h, corp)
                     if all((pv := pair_vals(p, corp.maps[mn])) is not None and pv[0] == sum(pv[1])
                            for p, mn in zip(h, ms))), None)
        if mset is None:
            continue
        p = h[0]; m = corp.maps[mset[0]]
        T, E = pair_vals(p, m)
        if len(E) < 4:
            continue
        cand = [pred for f, pred in inst3(E, sum(E)) if f == fam and pred > 0]
        if not cand:
            continue
        nm = canon(rng.choice(cand), m)
        if nm is None:
            continue
        c2 = json.loads(json.dumps(c)); c2['hyps'] = [json.loads(json.dumps(h))]
        c2['hyps'][0][0]['T']['nums'] = nm
        if not closes(c2, corp):
            out.append(c2)
    return out


def job(a):
    name, kind, seed = a
    rng = random.Random(seed)
    C = load_cases(name); corp = CORP[name]
    if kind == 'real':
        return (name, kind, seed, dict(count3(C, corp)))
    if kind == 'totals':
        return (name, kind, seed, dict(count3(null_totals(C, corp, rng), corp)))
    if kind == 'noise':
        return (name, kind, seed, dict(count3(null_noise(C, corp, rng), corp)))
    fam = kind.split(':')[1]
    P = plant_multi(C, corp, rng, fam)
    return (name, kind, seed, dict(count3(P, corp)))


if __name__ == '__main__':
    NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    tasks = []
    for name in ('PE', 'PC', 'UR3'):
        tasks.append((name, 'real', 0))
        tasks += [(name, k, s) for k in ('totals', 'noise') for s in range(NREP)]
        tasks += [(name, 'plant:' + f, 500 + s) for f in ('RUN2', 'PAIR2', 'PREFIX') for s in range(5)]
    with Pool(2) as pool:
        res = pool.map(job, tasks, chunksize=4)
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'))
    print('done')
