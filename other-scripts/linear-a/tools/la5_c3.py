#!/usr/bin/env python3
"""LA-5 cycle 3: statistics tied to the sign VALUES (the only items a code cannot fake by accident).
A code that uses syllabic signs as marks has no reason to respect the 5-vowel / consonant-series partition that the
values (conventional LB-derived transcription) put on the signs. Language written with a CV syllabary does:
  echo   share of adjacent valued pairs inside a word with the SAME vowel, over the mean after position-class shuffles
         of the signs (20 per draw; keeps slot profiles, removes adjacency) -> ratio (1 = no vowel structure)
  cecho  same with the consonant series (pure vowels excluded)
  vinit  P(word-initial | pure vowel A E I O U) / P(word-initial | other valued sign), over the same ratio for random
         5-sign sets drawn from the valued signs (100 draws) -> ratio of ratios
  kober  type pairs (>= 3 signs) identical except the final sign: share whose two finals share the consonant series,
         over random pairs of distinct signs from the same pool of alternating finals (inflection-grid signature)
Half-samples of types (m = 371 or half the population), 40 draws, as cycle 1. Controls: LB, LB_KN, LB_PY, LBpers
(language), planted codes LA_randID / LA_slot / LB_randID / LB_slot (same signs, same values, code sequences),
LA without the types that also occur in Linear B (LA_noLB), LA without *-signs words (LA_valued).
Output ../data/la5_c3.json, ../data/la5_c3.txt
"""
import sys, os, json, random, collections, math, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la5_common as C

ND = int(sys.argv[1]) if len(sys.argv) > 1 else 40
NP = 100
rnd = random.Random(53)
CONS = re.compile(r'^(D|J|K|M|N|P|Q|R|S|T|W|Z|DW|NW|TW|PH)?([AEIOU])[23]?$')
def cons(s):
    m = CONS.match(s); return (m.group(1) or '') if m else None

def echo_stats(types, r, nsh=20):
    """echo / cecho: observed same-vowel (same-consonant) share of adjacent valued pairs over the mean share after
    P-shuffles (signs permuted across words within position class: keeps every slot profile, removes adjacency).
    vinit: initial-rate ratio of pure vowels over random 5-sign sets. kober: consonant sharing of alternating finals
    over random pairs of distinct signs drawn from the same pool of alternating finals."""
    def shares(tt):
        v = c = nv = nc = 0
        for t in tt:
            for i in range(len(t) - 1):
                x, y = t[i], t[i + 1]
                if C.vowel(x) and C.vowel(y):
                    nv += 1; v += C.vowel(x) == C.vowel(y)
                    cx, cy = cons(x), cons(y)
                    if cx and cy: nc += 1; c += cx == cy
        return (v / nv if nv else float('nan')), (c / nc if nc else float('nan')), nv
    o_v, o_c, npairs = shares(types)
    sv = sc = 0.0
    for _ in range(nsh):
        a, b, _ = shares(C.shuffle_pos(types, r)); sv += a; sc += b
    n_v, n_c = sv / nsh, sc / nsh
    tok = collections.Counter(); ini = collections.Counter()
    for t in types:
        for i, a in enumerate(t):
            if C.vowel(a): tok[a] += 1; ini[a] += (i == 0)
    def vr(S):
        a = sum(ini[x] for x in S) / max(1, sum(tok[x] for x in S)); rest = [x for x in tok if x not in S]
        b = sum(ini[x] for x in rest) / max(1, sum(tok[x] for x in rest)); return a / b if b else float('nan')
    PV = [x for x in ('A', 'E', 'I', 'O', 'U') if x in tok]
    o_i = vr(PV); n_i = sum(vr(r.sample(list(tok), len(PV))) for _ in range(NP)) / NP
    by = collections.defaultdict(set)
    for t in types:
        if len(t) >= 3 and cons(t[-1]): by[t[:-1]].add(t[-1])
    alt = [(x, y) for fs in by.values() for x in fs for y in fs if x < y]
    if alt:
        o_k = sum(1 for x, y in alt if cons(x) == cons(y)) / len(alt)
        pool = [z for p in alt for z in p]; hit = n = 0
        for _ in range(2000):
            x, y = r.choice(pool), r.choice(pool)
            if x != y: n += 1; hit += cons(x) == cons(y)
        n_k = hit / n if n else float('nan')
    else: o_k = n_k = float('nan')
    rat = lambda a, b: a / b if b and not math.isnan(a) and not math.isnan(b) else float('nan')
    return dict(echo=rat(o_v, n_v), echo_obs=o_v, cecho=rat(o_c, n_c), vinit=rat(o_i, n_i), vinit_obs=o_i,
                kober=rat(o_k, n_k), n_alt=len(alt), n_pairs=npairs)

la = C.la_docs(); lb = C.lb_docs()
wa = C.words_of(la); wb = C.words_of(lb)
T = lambda ws, f=lambda s: True: sorted(set(w for _, s, w in ws if f(s)))
pops = {'LA': T(wa), 'LA_HT': T(wa, lambda s: s == 'Haghia Triada'), 'LA_nonHT': T(wa, lambda s: s != 'Haghia Triada'),
        'LB': T(wb), 'LB_KN': T(wb, lambda s: s == 'KN'), 'LB_PY': T(wb, lambda s: s == 'PY'),
        'LBpers': C.comparators()['LBpers']}
LBset = set(pops['LB']) | set(pops['LBpers'])
pops['LA_noLB'] = [t for t in pops['LA'] if t not in LBset]
pops['LA_valued'] = [t for t in pops['LA'] if all(C.vowel(a) for a in t)]
r0 = random.Random(7)
for base in ('LA', 'LB'):
    pops[base + '_randID'] = sorted(set(C.random_id_code(pops[base], r0).values()))
    pops[base + '_slot'] = sorted(set(C.slot_code(pops[base], r0).values()))
M = len(pops['LA']) // 2
res = {}; lines = [f'LA-5 cycle 3: value-tied statistics; m = {M} (or half the population), {ND} draws, {NP} label permutations',
                   'types: ' + ', '.join(f'{k} {len(v)}' for k, v in pops.items())]
for name, types in pops.items():
    m = M if len(types) >= M * 1.2 else len(types) // 2
    acc = collections.defaultdict(list)
    for d in range(ND):
        for k, v in echo_stats(rnd.sample(types, m), rnd).items(): acc[k].append(v)
    full = echo_stats(types, random.Random(1))
    res[name] = dict(m=m, half={k: C.summ(v) for k, v in acc.items()}, full=full)
    lines.append(f'\n{name} (m={m}; full n_pairs {full["n_pairs"]}, n_alt {full["n_alt"]}): ' +
                 '  '.join(f'{k} {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]' for k, v in res[name]['half'].items() if k not in ('n_alt', 'n_pairs')) +
                 '\n   full: ' + '  '.join(f'{k} {v:.3f}' for k, v in full.items()))
    print(lines[-1], flush=True)

# full-set test: observed same-vowel / same-consonant share vs 2,000 P-shuffles of the full type set (one-sided both ways)
def perm_p(types, key, nperm=2000):
    r = random.Random(11)
    def sh(tt):
        k = n = 0
        for t in tt:
            for i in range(len(t) - 1):
                x, y = t[i], t[i + 1]
                lx, ly = (C.vowel(x), C.vowel(y)) if key == 'v' else (cons(x), cons(y))
                if lx and ly and C.vowel(x) and C.vowel(y): n += 1; k += lx == ly
        return k / n, n
    o, n = sh(types); nn = [sh(C.shuffle_pos(types, r))[0] for _ in range(nperm)]
    mu = sum(nn) / nperm; sd = (sum((x - mu) ** 2 for x in nn) / nperm) ** 0.5
    hi = (sum(1 for x in nn if x >= o) + 1) / (nperm + 1); lo = (sum(1 for x in nn if x <= o) + 1) / (nperm + 1)
    return dict(obs=o, null=mu, ratio=o / mu, z=(o - mu) / sd, p_hi=hi, p_lo=lo, n=n)
lines.append('\nfull-set tests (2,000 position-class shuffles; v = same vowel, c = same consonant series):')
for name in ('LA', 'LA_HT', 'LA_nonHT', 'LA_noLB', 'LA_valued', 'LB', 'LB_KN', 'LB_PY', 'LBpers', 'LA_randID', 'LA_slot', 'LB_randID', 'LB_slot'):
    for key in ('v', 'c'):
        x = perm_p(pops[name], key, 2000 if len(pops[name]) < 2000 else 300); res[name]['perm_' + key] = x
        lines.append(f'  {name} {key}: obs {x["obs"]:.4f} null {x["null"]:.4f} ratio {x["ratio"]:.3f} z {x["z"]:+.2f} p_hi {x["p_hi"]:.4f} p_lo {x["p_lo"]:.4f} pairs {x["n"]}')
        print(lines[-1], flush=True)
json.dump(res, open(C.LAD + '/la5_c3.json', 'w'), indent=1)
open(C.LAD + '/la5_c3.txt', 'w').write('\n'.join(lines) + '\n')
