"""pe5 cycle 2: tablet PAIRS that share several quantities (copies, transfers, summaries).

Item sets per tablet:
  'num'  = distinct (system, value) with value >= VMIN, any role
  'lab'  = distinct (label, value): PE label = full sign string of a multi-sign entry;
           Ur III label = first word after the numeral
Pair score = # shared items.  Statistic = # tablet pairs sharing >= k items (k = 2, 3, 4).
Null = values shuffled across records within strata (system x tablet-scale bin x role),
labels and tablet layout kept; 200x.  Controls: Ur III Drehem/Umma/Girsu PE-sized
subsamples (random and one-period), proto-cuneiform, synthetic planted copies.
Then: for PE pairs above null, do they share header / commodity class / site?
"""
import json, math, os, pickle, random
from collections import Counter, defaultdict
from itertools import combinations
from pe5_common import *  # noqa

REPS = 200
VMIN = 5
DFMAX = 30          # ignore items present on more than DFMAX tablets (uninformative)
OUT = {}


def label(r, kind):
    if kind == 'num':
        return (r['sys'], r['val'])
    s = r.get('signs') or []
    if 'nums' in r:                       # PE: 'lab' = final (class) sign, 'str' = full string
        if not s:
            return None
        return (s[-1], r['val']) if kind == 'lab' else (tuple(s), r['val'])
    return (s[0] if s else '', r['val'])


def pair_counts(recs, kind):
    items = defaultdict(set)
    for r in recs:
        if r['val'] is None or r['val'] < VMIN:
            continue
        L = label(r, kind)
        if L is not None:
            items[L].add(r['tab'])
    pc = Counter()
    for L, tabs in items.items():
        if 2 <= len(tabs) <= DFMAX:
            for a, b in combinations(sorted(tabs), 2):
                pc[(a, b)] += 1
    return pc


def strata(recs):
    scale = defaultdict(list)
    for r in recs:
        if r['val']:
            scale[r['tab']].append(r['val'])
    tb = {t: int(math.log2(sorted(v)[len(v) // 2] + 1)) for t, v in scale.items()}
    st = defaultdict(list)
    for k, r in enumerate(recs):
        if r['val'] is not None:
            st[(r['sys'], tb.get(r['tab']), r['role'])].append(k)
    return st


def shuffled(recs, st, rng):
    new = [dict(r) for r in recs]
    for idx in st.values():
        vals = [recs[k]['val'] for k in idx]
        rng.shuffle(vals)
        for k, v in zip(idx, vals):
            new[k]['val'] = v
    return new


def test(name, recs, rng, kinds=('num', 'lab')):
    st = strata(recs)
    res = {}
    for kind in kinds:
        obs = pair_counts(recs, kind)
        o = {k: sum(1 for v in obs.values() if v >= k) for k in (2, 3, 4)}
        nl = {k: [] for k in (2, 3, 4)}
        for _ in range(REPS):
            pc = pair_counts(shuffled(recs, st, rng), kind)
            for k in (2, 3, 4):
                nl[k].append(sum(1 for v in pc.values() if v >= k))
        res[kind] = {}
        for k in (2, 3, 4):
            mu = sum(nl[k]) / REPS
            sd = (sum((x - mu) ** 2 for x in nl[k]) / REPS) ** .5 or 1
            res[kind]['k%d' % k] = {'obs': o[k], 'null': round(mu, 1), 'z': round((o[k] - mu) / sd, 1),
                                    'p': round((1 + sum(x >= o[k] for x in nl[k])) / (1 + REPS), 4)}
        res[kind]['top'] = [(a, b, v) for (a, b), v in obs.most_common(15)]
    OUT[name] = res
    print(name, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'top'} for k, v in res.items()}), flush=True)
    return res


def plant(recs, rng, npairs=20, nitems=3):
    """Copy nitems quantity lines (with labels) of tablet A onto tablet B, npairs times."""
    new = [dict(r) for r in recs]
    bytab = defaultdict(list)
    for k, r in enumerate(new):
        if r['val'] is not None and r['val'] >= VMIN:
            bytab[r['tab']].append(k)
    tabs = [t for t, v in bytab.items() if len(v) >= nitems]
    for _ in range(npairs):
        a, b = rng.sample(tabs, 2)
        src = rng.sample(bytab[a], nitems)
        dst = rng.sample(bytab[b], nitems)
        for s, d in zip(src, dst):
            new[d]['val'] = new[s]['val']
            new[d]['sys'] = new[s]['sys']
            if 'nums' in new[s]:
                new[d]['signs'] = new[s]['signs']
    return new


def main():
    rng = random.Random(52)
    pe, pm = pe_records('D3', 'NOT')
    ntab = len(pm)
    test('PE', pe, rng, kinds=('num', 'lab', 'str'))
    test('PE_planted20x3', plant(pe, rng), rng, kinds=('num', 'lab', 'str'))
    test('PE_cnt_only', [r for r in pe if r['sys'] == 'S'], rng, kinds=('lab',))
    test('PE_cap_only', [r for r in pe if r['sys'] == 'C'], rng, kinds=('lab',))
    for prov in (('Puzri', 'Umma', 'Girsu') if not os.environ.get('PE5_ONLYPE') else ()):
        u, um = pickle.load(open(os.path.join(SCRATCH, 'ur3_%s.pkl' % prov), 'rb'))
        pick = set(rng.sample(sorted(um), ntab))
        test('UR3_%s_PEsize' % prov, [r for r in u if r['tab'] in pick], rng)
        yr = Counter()
        for t, m in um.items():
            k = '.'.join(m.get('date', '').split('.')[:2])
            if not k.startswith(('00', '--')) and k:
                yr[k] += 1
        top = {y for y, _ in yr.most_common(3)}
        tabs = sorted(t for t, m in um.items() if '.'.join(m.get('date', '').split('.')[:2]) in top)
        pick = set(rng.sample(tabs, min(ntab, len(tabs))))
        test('UR3_%s_PEsize_3yrs' % prov, [r for r in u if r['tab'] in pick], rng)
    pc, pcm = pc_records()
    test('PC_all', pc, rng, kinds=('num',))
    json.dump(OUT, open(os.path.join(PEDATA, 'pe5_cycle2.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
