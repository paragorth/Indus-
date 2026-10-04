"""pe5 cycle 3: block copies between tablets, family consistency, and cross-tablet sums.

(a) Block copy: longest run of consecutive identical lines (label, value) shared by two
    tablets.  Statistic: # pairs with run >= 3 / >= 4.  Null: values shuffled within
    strata (labels kept), 200x.  Controls: Ur III one-period PE-sized (Drehem, Girsu, Umma).
(b) Family: for PE pairs with a block >= 3, share of pairs with the same header sign and
    the same publication volume, vs all pairs of tablets with >= 3 lines.
(c) Cross-tablet sums: totals that use N45/N34 (or capacity totals) equal to the sum of
    two other tablets' totals/obverse sums, under each value set; null = same-shape
    perturbation of the target total, 500x.
"""
import json, os, pickle, random
from collections import Counter, defaultdict
from itertools import combinations
from pe5_common import *  # noqa
from pe5_cycle2 import strata, shuffled

REPS = 200
OUT = {}


def seqs(recs, kind):
    by = defaultdict(list)
    for r in sorted(recs, key=lambda r: (r['tab'], r['i'])):
        if r['val'] is None:
            continue
        s = r.get('signs') or []
        if 'nums' in r or kind == 'pe':
            if not s:
                continue
            tok = (tuple(s), r['val'])
        else:
            tok = (s[0] if s else '', r['val'])
        by[r['tab']].append(tok)
    return by


def runs(by, minrun=3):
    """pairs -> longest common contiguous run (>= 2 via bigram index, then extend)."""
    big = defaultdict(set)
    for t, s in by.items():
        for k in range(len(s) - 1):
            if s[k][1] >= 1 and s[k + 1][1] >= 1:
                big[(s[k], s[k + 1])].add((t, k))
    best = Counter()
    for key, occ in big.items():
        if len(occ) < 2 or len(occ) > 40:
            continue
        occ = sorted(occ)
        for (ta, ka), (tb, kb) in combinations(occ, 2):
            if ta == tb:
                continue
            sa, sb = by[ta], by[tb]
            n = 0
            while ka + n < len(sa) and kb + n < len(sb) and sa[ka + n] == sb[kb + n]:
                n += 1
            pair = (ta, tb) if ta < tb else (tb, ta)
            best[pair] = max(best[pair], n)
    return best


def block_test(name, recs, rng, kind):
    st = strata(recs)
    obs = runs(seqs(recs, kind))
    o = {k: sum(v >= k for v in obs.values()) for k in (3, 4, 6)}
    nl = {k: [] for k in o}
    for _ in range(REPS):
        b = runs(seqs(shuffled(recs, st, rng), kind))
        for k in o:
            nl[k].append(sum(v >= k for v in b.values()))
    res = {}
    for k in o:
        mu = sum(nl[k]) / REPS
        sd = (sum((x - mu) ** 2 for x in nl[k]) / REPS) ** .5 or 1
        res['run%d' % k] = {'obs': o[k], 'null': round(mu, 2), 'z': round((o[k] - mu) / sd, 1),
                            'p': round((1 + sum(x >= o[k] for x in nl[k])) / (1 + REPS), 4)}
    res['top'] = [(a, b, v) for (a, b), v in obs.most_common(25)]
    OUT[name] = res
    print(name, json.dumps({k: v for k, v in res.items() if k != 'top'}), flush=True)
    return obs


def family(obs, meta, minrun=3):
    vol = {t: re.sub(r',.*', '', m['design']) for t, m in meta.items()}
    chain = [(a, b) for (a, b), v in obs.items() if v >= minrun]
    def rates(pairs):
        h = sum(1 for a, b in pairs if meta[a]['hdr'] and meta[a]['hdr'] == meta[b]['hdr'])
        hb = sum(1 for a, b in pairs if meta[a]['hdr'] and meta[b]['hdr'])
        v = sum(1 for a, b in pairs if vol[a] == vol[b])
        return {'n': len(pairs), 'same_hdr': h, 'both_hdr': hb, 'same_vol': v}
    rng = random.Random(9)
    tabs = [t for t in meta]
    base = [tuple(rng.sample(tabs, 2)) for _ in range(20000)]
    out = {'chain': rates(chain), 'random': rates(base),
           'pairs': [(a, meta[a]['design'], meta[a]['hdr'], b, meta[b]['design'], meta[b]['hdr'], obs[(a, b)])
                     for a, b in chain]}
    return out


def tablet_sums(recs):
    s = defaultdict(float)
    for r in recs:
        if r['role'] == 'E' and r['obv'] and r['val'] is not None:
            s[(r['tab'], r['sys'])] += r['val']
    return s


def sums_test(vs, cs, rng, reps=500):
    recs, meta = pe_records(vs, cs)
    tots = [r for r in recs if r['role'] == 'T' and r['val']]
    tsum = tablet_sums(recs)
    pool = defaultdict(list)          # per system: (tab, value) of totals and obverse sums
    for r in tots:
        pool[r['sys']].append((r['tab'], r['val']))
    for (t, s), v in tsum.items():
        pool[s].append((t, round(v, 4)))
    pairsums = {}
    for s, L in pool.items():
        d = defaultdict(set)
        L = list({x for x in L})
        for (ta, va), (tb, vb) in combinations(L, 2):
            if ta != tb:
                d[round(va + vb, 4)].add((ta, tb))
        pairsums[s] = d
    targets = [r for r in tots if (r['sys'] == 'S' and any(c in ('N45', 'N34') for _, c in r['nums']))
               or (r['sys'] == 'C' and r['val'] >= 120)]

    def hits(vals):
        h = 0
        for r, v in vals:
            if v is None:
                continue
            ok = [p for p in pairsums[r['sys']].get(round(v, 4), ()) if r['tab'] not in p]
            h += bool(ok)
        return h
    obs = hits([(r, r['val']) for r in targets])
    nl = [hits([(r, perturb(r, rng, VSETS[vs], CSETS[cs])) for r in targets]) for _ in range(reps)]
    mu = sum(nl) / reps
    return {'targets': len(targets), 'S_targets': sum(r['sys'] == 'S' for r in targets),
            'obs': obs, 'null': round(mu, 2),
            'p': round((1 + sum(x >= obs for x in nl)) / (1 + reps), 4)}


def main():
    rng = random.Random(53)
    pe, pm = pe_records('D3', 'NOT')
    obs = block_test('PE_blocks', pe, rng, 'pe')
    OUT['PE_family'] = family(obs, pm)
    print('family', json.dumps({k: v for k, v in OUT['PE_family'].items() if k != 'pairs'}))
    for p in OUT['PE_family']['pairs']:
        print('  ', p)
    ntab = len(pm)
    for prov in ('Puzri', 'Girsu', 'Umma'):
        u, um = pickle.load(open(os.path.join(SCRATCH, 'ur3_%s.pkl' % prov), 'rb'))
        yr = Counter()
        for t, m in um.items():
            k = '.'.join(m.get('date', '').split('.')[:2])
            if k and not k.startswith(('00', '--')):
                yr[k] += 1
        top = {y for y, _ in yr.most_common(3)}
        tabs = sorted(t for t, m in um.items() if '.'.join(m.get('date', '').split('.')[:2]) in top)
        pick = set(rng.sample(tabs, min(ntab, len(tabs))))
        block_test('UR3_%s_PEsize_3yrs' % prov, [r for r in u if r['tab'] in pick], rng, 'ur')
    for vs in ('D3', 'DEC', 'SEX', 'SEXr'):
        for cs in ('NOT', 'PCS'):
            r = sums_test(vs, cs, rng)
            OUT['sums_%s_%s' % (vs, cs)] = r
            print('sums', vs, cs, r, flush=True)
    json.dump(OUT, open(os.path.join(PEDATA, 'pe5_cycle3.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
