"""pe5 cycle 1: do written totals reappear as entries on OTHER tablets?

Statistic H = # written totals (value >= minval) whose (system, value) occurs as an entry
on another tablet.  Null A = same-shape perturbation of every total (one digit +-1,
digit stays in range), 500x.  Control B (peakiness) = the same test run on random
entries used as pseudo-totals (matched in number and magnitude).  Ledger signal =
totals' excess over null A clearly above entries' excess.
Corpora: PE (D3 / NOT values), Ur III Drehem/Umma/Girsu (full and PE-sized subsamples),
proto-cuneiform (all), synthetic negative (entries shuffled across tablets, totals = sums),
synthetic positive (negative + k planted chains).
"""
import json, os, pickle, random, sys
from collections import Counter, defaultdict
from pe5_common import *  # noqa

R = 300
OUT = {}


def excess(recs, totals, rng, minval, vs=None, cs=None, reps=R):
    idx = index_vals(recs)
    obs = len(match_stat(totals, idx, minval))
    nulls = []
    for _ in range(reps):
        pt = []
        for r in totals:
            if r['val'] is None or r['val'] < minval:
                continue
            v = perturb(r, rng, vs, cs)
            if v is None:
                continue
            q = dict(r)
            q['val'] = v
            pt.append(q)
        nulls.append(len(match_stat(pt, idx, 0)))
    mu = sum(nulls) / len(nulls)
    sd = (sum((x - mu) ** 2 for x in nulls) / len(nulls)) ** .5 or 1
    p = (1 + sum(x >= obs for x in nulls)) / (1 + len(nulls))
    n = sum(1 for r in totals if r['val'] is not None and r['val'] >= minval)
    return {'n': n, 'obs': obs, 'null': round(mu, 1), 'z': round((obs - mu) / sd, 2),
            'ratio': round(obs / mu, 2) if mu else None, 'p': round(p, 4)}


def pseudo_totals(recs, totals, rng):
    """Entries matched to totals by magnitude bin (log2), one per total, different tablets."""
    import math
    bins = defaultdict(list)
    for r in recs:
        if r['role'] == 'E' and r['val']:
            bins[(r['sys'], int(math.log2(r['val'] + 1)))].append(r)
    out = []
    for t in totals:
        if not t['val']:
            continue
        b = bins.get((t['sys'], int(math.log2(t['val'] + 1))))
        if b:
            q = dict(rng.choice(b))
            q['role'] = 'P'
            out.append(q)
    return out


def run(name, recs, rng, vs=None, cs=None, minvals=(1, 20)):
    totals = [r for r in recs if r['role'] == 'T']
    res = {}
    for mv in minvals:
        res['tot_min%d' % mv] = excess(recs, totals, rng, mv, vs, cs)
        # entries as pseudo-totals: the pseudo entry itself must not count as its own match
        ps = pseudo_totals(recs, totals, rng)
        res['ent_min%d' % mv] = excess(recs, ps, rng, mv, vs, cs)
    OUT[name] = res
    print(name, json.dumps(res), flush=True)
    return res


def subsample(recs, meta, ntab, rng, tabs=None):
    pool = sorted(tabs if tabs is not None else meta)
    pick = set(rng.sample(pool, min(ntab, len(pool))))
    return [r for r in recs if r['tab'] in pick]


def synthetic(recs, rng, plant=0):
    """Negative: shuffle entries across tablets (system kept), totals = sum of new entries.
    Positive: additionally copy `plant` totals into an entry slot of another tablet."""
    ents = [r for r in recs if r['role'] == 'E' and r['val'] is not None]
    by_sys = defaultdict(list)
    for r in ents:
        by_sys[r['sys']].append(r['val'])
    for s in by_sys:
        rng.shuffle(by_sys[s])
    ptr = Counter()
    new = []
    sums = defaultdict(float)
    for r in recs:
        if r['role'] == 'E' and r['val'] is not None:
            q = dict(r)
            q['val'] = by_sys[r['sys']][ptr[r['sys']]]
            q.pop('nums', None)
            ptr[r['sys']] += 1
            new.append(q)
            if r['obv']:
                sums[(r['tab'], r['sys'])] += q['val']
    tots = []
    for r in recs:
        if r['role'] == 'T':
            q = dict(r)
            q['val'] = round(sums.get((r['tab'], r['sys']), 0), 4) or None
            q.pop('nums', None)
            if q['val']:
                tots.append(q)
    if plant:
        targets = rng.sample(tots, plant)
        ent_new = [q for q in new]
        for t in targets:
            e = rng.choice([q for q in ent_new if q['tab'] != t['tab'] and q['sys'] == t['sys']])
            e['val'] = t['val']
    return new + tots


def main():
    rng = random.Random(5)
    pe, pm = pe_records('D3', 'NOT')
    print('PE records', len(pe), 'totals', sum(r['role'] == 'T' for r in pe))
    run('PE_D3', pe, rng, VSETS['D3'], CSETS['NOT'])
    for vs in ('DEC', 'SEX', 'SEXr'):
        r2, _ = pe_records(vs, 'NOT')
        run('PE_' + vs, r2, rng, VSETS[vs], CSETS['NOT'], minvals=(20,))
    r2, _ = pe_records('D3', 'PCS')
    run('PE_D3_PCS', r2, rng, VSETS['D3'], CSETS['PCS'], minvals=(20,))
    # notation-only match (value free): integer-ize notation key
    ntab = len(pm)
    for prov in ('Puzri', 'Umma', 'Girsu'):
        u, um = pickle.load(open(os.path.join(SCRATCH, 'ur3_%s.pkl' % prov), 'rb'))
        run('UR3_%s_full' % prov, u, rng, minvals=(20,))
        # PE-sized: random tablets
        run('UR3_%s_PEsize' % prov, subsample(u, um, ntab, rng), rng, minvals=(1, 20))
        # PE-sized, time concentrated: tablets of the most common reign-year
        yr = Counter()
        for t, m in um.items():
            d = m.get('date', '')
            k = '.'.join(d.split('.')[:2])
            if d and not k.startswith(('00', '--')):
                yr[k] += 1
        top = [y for y, _ in yr.most_common(3)]
        tabs = [t for t, m in um.items() if '.'.join(m.get('date', '').split('.')[:2]) in top]
        run('UR3_%s_PEsize_3yrs' % prov, subsample(u, um, ntab, rng, tabs), rng, minvals=(1, 20))
    pc, pcm = pc_records()
    run('PC_all', pc, rng, minvals=(1, 20))
    neg = synthetic(pe, rng)
    run('NEG_PE_shuffled', neg, rng, minvals=(1, 20))
    pos = synthetic(pe, rng, plant=30)
    run('POS_PE_planted30', pos, rng, minvals=(1, 20))
    json.dump(OUT, open(os.path.join(PEDATA, 'pe5_cycle1.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
