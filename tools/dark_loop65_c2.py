"""S-DARK-65 cycle 2: tests A (shared middle = one firm), B (fixed head pairing = roles), C (frame-only stamp +
long designation = office counter-signature) against nulls, on the impression pairs of cycle 1.
Null N1: partner impression replaced by a random distinct sealing impression of the same site (any sealing, other object).
Null N1L: as N1 but length-matched (same sign count, +-1 fallback).
Null N2: both impressions replaced by two random seal texts of the same site (pooled 'other' when < 10 seals).
Usage: python3 tools/dark_loop65_c2.py <seq_raw|seq_strong|seq_all> [nperm]"""
import sys, collections, itertools, random
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop65 import *
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'; NP = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rnd = random.Random(65)
objs, S = load_sealings(LV)
seals = seal_texts(objs)
allseqs = [f['seq'] for o in objs.values() for f in o['faces'] if f['seq']]
parse = make_parser(learn_qual(allseqs)); middle = middle_fn(parse); name_only = middle_fn(parse, ('NAME',))
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
P(f'== S-DARK-65 cycle 2 ({LV}, nperm {NP}): two-party tests on impression pairs')
multi = [o for o in S.values() if len(o['imps']) >= 2]
def substr(a, b):
    a, b = list(a), list(b)
    if len(a) > len(b): a, b = b, a
    return any(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1))
pairs_all = [(o, a, b) for o in multi for a, b in itertools.combinations(o['imps'], 2)]
pairs_strict = [(o, a, b) for o, a, b in pairs_all if len(a['seq']) >= 2 and len(b['seq']) >= 2 and not substr(a['seq'], b['seq'])]
# pools
pool_tag = collections.defaultdict(list)      # site -> [(oid, face)]
for o in S.values():
    for f in o['imps']: pool_tag[o['site']].append((o['oid'], f))
pool_seal = collections.defaultdict(list)
for o in objs.values():
    if o['ot'] == 'seal':
        for f in o['faces']:
            if f['seq']: pool_seal[o['site']].append(f)
other_seals = [f for s, L in pool_seal.items() if len(L) < 10 for f in L]
def seal_pool(site):
    L = pool_seal.get(site, [])
    return L if len(L) >= 10 else other_seals + L
def draw_tag(o, f, lenmatch=False):
    cand = [g for oid, g in pool_tag[o['site']] if oid != o['oid']]
    if lenmatch:
        c2 = [g for g in cand if len(g['seq']) == len(f['seq'])]
        if len(c2) < 3: c2 = [g for g in cand if abs(len(g['seq']) - len(f['seq'])) <= 1]
        cand = c2 or cand
    return rnd.choice(cand) if cand else None
KEYS = ['shared_any', 'shared_mid', 'shared_name', 'same_mid', 'ident', 'jacc', 'same_head', 'both_jar', 'lendiff', 'short_long',
        'frame_only_one', 'frame_only_both', 'opener_one', 'opener_both', 'match_one', 'match_both', 'match_none']
MEAN = {'jacc', 'lendiff'}
def stats(a, b, fa, fb): return pair_stats(a, b, middle, name_only, head_of, seals, fa, fb)
def run(name, pairs):
    n = len(pairs)
    if n < 4: P(f'\n-- {name}: {n} pairs, too few'); return
    obs = collections.Counter(); hp = collections.Counter()
    for o, a, b in pairs:
        st = stats(a['seq'], b['seq'], a, b)
        for k in KEYS: obs[k] += st[k]
        hp[st['heads']] += 1
    nulls = {}
    for nm, mode in (('N1 same-site sealing impressions', 'tag'), ('N1L length-matched', 'tagL'), ('N2 same-site seal pairs', 'seal')):
        null = collections.defaultdict(list); hpn = collections.Counter()
        for _ in range(NP):
            c = collections.Counter()
            for o, a, b in pairs:
                if mode == 'seal':
                    pl = seal_pool(o['site'])
                    fa, fb = rnd.choice(pl), rnd.choice(pl)
                else:
                    fa = a; fb = draw_tag(o, b, lenmatch=(mode == 'tagL'))
                if fb is None: continue
                st = stats(fa['seq'], fb['seq'], fa, fb)
                for k in KEYS: c[k] += st[k]
                hpn[st['heads']] += 1
            for k in KEYS: null[k].append(c[k])
        nulls[nm] = (null, hpn)
    P(f'\n-- {name}: {len(set(o["oid"] for o, _, _ in pairs))} sealings, {n} impression pairs')
    P(f'   {"statistic":16s} {"obs":>9s} | ' + ' | '.join(f'{nm[:26]:>26s}' for nm in nulls))
    for k in KEYS:
        o_ = obs[k]
        cells = []
        for nm, (null, _) in nulls.items():
            nl = null[k]; mu = sum(nl) / NP
            cells.append(f'{mu/n:6.3f} P+={pval(o_, nl):.3f} P-={pval(o_, nl, "lo"):.3f}')
        P(f'   {k:16s} {o_/n:6.3f} ({o_ if k not in MEAN else round(o_,1):>3}) | ' + ' | '.join(cells))
    # head x head table against N1 expectation
    null, hpn = nulls['N1 same-site sealing impressions']
    P('   head x head pairs (obs vs N1 expected):')
    for hpair, c in sorted(hp.items(), key=lambda x: -x[1]):
        P(f'      {hpair[0]:>11s} + {hpair[1]:<11s} {c:3d}  vs {hpn[hpair]/NP:5.2f}')
    # chi-square-like independence statistic against N1
    exp = {k: v / NP for k, v in hpn.items()}
    chi = sum((hp[k] - exp.get(k, 0)) ** 2 / exp[k] for k in hp if exp.get(k, 0) > 0)
    # null distribution of chi: redraw
    chin = []
    for _ in range(min(NP, 300)):
        c = collections.Counter()
        for o, a, b in pairs:
            fb = draw_tag(o, b)
            if fb is None: continue
            c[stats(a['seq'], fb['seq'], a, fb)['heads']] += 1
        chin.append(sum((c[k] - exp.get(k, 0)) ** 2 / exp[k] for k in c if exp.get(k, 0) > 0))
    P(f'   head-pair chi2 vs N1 table: {chi:.2f}, null mean {sum(chin)/len(chin):.2f}, P = {pval(chi, chin):.3f}')
run('all pairs', pairs_all)
run('strict pairs (both >= 2 signs, neither a substring of the other)', pairs_strict)
run('Lothal', [p for p in pairs_all if p[0]['site'] == 'Lothal'])
run('non-Lothal', [p for p in pairs_all if p[0]['site'] != 'Lothal'])
run('Lothal strict', [p for p in pairs_strict if p[0]['site'] == 'Lothal'])
# which signs are shared across impressions, and the shared-middle examples
P('\n-- pairs sharing a middle sign (all pairs):')
for o, a, b in pairs_all:
    st = stats(a['seq'], b['seq'], a, b)
    if st['shared_mid']: P(f'   {o["cisi"] or o["oid"]:8s} {fmt(a["seq"]):30s} | {fmt(b["seq"]):30s} shared {sorted(set(middle(a["seq"])) & set(middle(b["seq"])))}')
P('\n-- pairs where exactly one impression matches a known seal text (all pairs):')
for o, a, b in pairs_all:
    ma, mb = matches_seal(a, seals), matches_seal(b, seals)
    if (ma is None) != (mb is None):
        P(f'   {o["cisi"] or o["oid"]:8s} {fmt(a["seq"]):30s} [{ma or "-"}] | {fmt(b["seq"]):30s} [{mb or "-"}]')
open(OUT + f'loop65_c2_{LV}.txt', 'w').write('\n'.join(out) + '\n')
