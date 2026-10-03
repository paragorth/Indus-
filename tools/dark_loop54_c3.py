"""Loop 54 cycle 3: aggregate cycles 1-2 into the re-graded table.
Inputs: loop54_c1_<level>_big.json (3 levels), loop54_c1_<level>_heldout.json (seq_raw, seq_all), loop54_c2_im77.json.
Rules (strict, as set for this loop):
  - an 'adjacency' claim is chain-explained by construction (first-order chain reproduces bigrams in expectation);
  - a 'position' / 'distance' / 'trigram' / 'corpus' claim is 'beyond chain' at MD+H if it lies outside the Markov-1 AND Markov-2
    2.5-97.5% bands at all three merge levels (merge-dependent if only at some);
  - it 'replicates' if it is also outside both bands on the held-out sites (either level) or on the IM77-only set, in the same direction;
  - entry grade: A kept only if some non-adjacency claim is beyond both chains at every level AND replicates;
    A -> B if beyond chains at MD+H but not replicated out of sample (or only at some merge levels);
    A/B -> C if every claim is chain-explained, unless an outside fact is recorded: then B at most;
    B kept if beyond chains (replication not required for B), else C (B- with outside fact);
    C entries and grammar statements: marked chain-explained / beyond chain / untestable only.
Output: loop54_regrade.csv, loop54_c3_table.txt"""
import json, csv, collections
from dark_loop54_common import DARK, fmt

LEV = ['seq_raw', 'seq_strong', 'seq_all']
big = {L: json.load(open(DARK + f'loop54_c1_{L}_big.json')) for L in LEV}
held = {L: json.load(open(DARK + f'loop54_c1_{L}_heldout.json')) for L in ('seq_raw', 'seq_all')}
im = json.load(open(DARK + 'loop54_c2_im77.json'))
n = len(big['seq_all'])
def key(r): return (r['entry'], r['claim'])
def beyond(r):
    """None = untestable; True = outside both M1 and M2 bands; False otherwise. Returns (flag, direction)."""
    m1, m2 = r['M1'], r['M2']
    if m1['inside'] is None or m2['inside'] is None or r['obs'] != r['obs']: return None, ''
    return (not m1['inside']) and (not m2['inside']), m1['dir'] or m2['dir']

rows = []; per_entry = collections.defaultdict(list)
for i in range(n):
    r = big['seq_all'][i]; k = key(r)
    assert all(key(big[L][i]) == k for L in LEV)
    kind = r['kind']
    bs = {L: beyond(big[L][i]) for L in LEV}
    flags = [bs[L][0] for L in LEV]
    if kind == 'adjacency': md = 'adjacency (chain by construction)'
    elif all(f is None for f in flags): md = 'untestable'
    elif all(f for f in flags): md = 'beyond chains (' + bs['seq_all'][1] + ')'
    elif any(f for f in flags): md = 'beyond chains only at ' + '/'.join(L for L in LEV if bs[L][0]) + ' (merge-dependent)'
    else: md = 'chain-explained'
    # held-out sites
    hs = {L: beyond(held[L][i]) for L in held}
    hflag = any(v[0] for v in hs.values() if v[0] is not None) and all(v[1] == bs['seq_all'][1] or not v[0] for v in hs.values())
    hdir = '/'.join(f"{L[4:]}:{'beyond' if v[0] else ('untestable' if v[0] is None else 'inside')}" for L, v in hs.items())
    # im77
    ir = im[i]; ib = beyond(ir)
    iflag = bool(ib[0]) and ib[1] == bs['seq_all'][1]
    idir = 'beyond' if ib[0] else ('untestable/no tokens' if ib[0] is None else 'inside')
    repl = hflag or iflag
    g = r.get('G', {})
    gflag = '' if not g or g.get('inside') is None else ('in' if g['inside'] else g['dir'])
    rec = {'entry': r['entry'], 'grade': r['grade'], 'claim': r['claim'], 'kind': kind, 'obs_all': r['obs'],
           'M1': f"{fmt(r['M1']['med'])} [{fmt(r['M1']['lo'])}-{fmt(r['M1']['hi'])}]", 'M2': f"{fmt(r['M2']['med'])} [{fmt(r['M2']['lo'])}-{fmt(r['M2']['hi'])}]",
           'M1E': ('in' if r['M1E']['inside'] else r['M1E']['dir']) if r['M1E']['inside'] is not None else '-', 'G': gflag,
           'mdh': md, 'obs_raw': big['seq_raw'][i]['obs'], 'obs_strong': big['seq_strong'][i]['obs'],
           'heldout': f"obs {fmt(held['seq_all'][i]['obs'])} vs M1 {fmt(held['seq_all'][i]['M1']['med'])} [{fmt(held['seq_all'][i]['M1']['lo'])}-{fmt(held['seq_all'][i]['M1']['hi'])}] ({hdir})",
           'im77': f"obs {fmt(ir['obs'])} vs M1 {fmt(ir['M1']['med'])} [{fmt(ir['M1']['lo'])}-{fmt(ir['M1']['hi'])}] ({idir})",
           'beyond_all_levels': all(f for f in flags) if kind != 'adjacency' else False, 'beyond_some': any(f for f in flags) if kind != 'adjacency' else False,
           'replicated': repl if kind != 'adjacency' else False, 'outside': r['outside'], 'srow': r['srow']}
    rows.append(rec); per_entry[r['entry']].append(rec)

def propose(entry, recs):
    g = recs[0]['grade']
    base = g.split()[0].strip('/')
    outside = any(x['outside'] for x in recs)
    strong = sorted([x for x in recs if x['beyond_all_levels']], key=lambda x: x['M1E'] == 'in')   # M1E-robust claims first
    some = [x for x in recs if x['beyond_some']]
    rep = sorted([x for x in strong if x['replicated']], key=lambda x: x['M1E'] == 'in')
    if g == 'grammar' or base.startswith('C'):
        if strong: return g if g == 'grammar' else base, 'beyond chains' + (' and replicated' if rep else '') + ': ' + strong[0]['claim']
        if some: return base if base.startswith('C') else g, 'beyond chains at some merge levels only: ' + some[0]['claim']
        return base if base.startswith('C') else g, 'chain-explained' + (' (outside fact: ' + next(x['outside'] for x in recs if x['outside']) + ')' if outside else '')
    if base.startswith('A'):
        if rep: return 'A', 'beats both chains at all levels and replicates: ' + rep[0]['claim']
        if strong: return 'B', 'beats both chains at MD+H but not replicated out of sample (power or absent): ' + strong[0]['claim']
        if some: return 'B', 'beyond chains only at some merge levels: ' + some[0]['claim']
        if outside: return 'B', 'sequence claims chain-explained; kept at B on the outside fact: ' + next(x['outside'] for x in recs if x['outside'])
        return 'C', 'every sequence claim is chain-explained; no outside fact'
    # B
    if strong: return 'B', 'beats both chains' + (' and replicates' if rep else ' (MD+H only)') + ': ' + strong[0]['claim']
    if some: return 'B-', 'beyond chains only at some merge levels: ' + some[0]['claim']
    if outside: return 'B-', 'sequence claims chain-explained; outside fact only: ' + next(x['outside'] for x in recs if x['outside'])
    return 'C', 'every sequence claim is chain-explained; no outside fact'

def propose2(e, recs):
    pg, why = propose(e, recs)
    dec = [x for x in recs if x['claim'] in why]
    if dec and dec[0]['M1E'] == 'in' and dec[0]['kind'] != 'adjacency': why += ' [end-state fact: an order-1 chain with END reproduces it]'
    return pg, why
OVERRIDE = {
 'W817/W861 opener': ('A', 'opener-W400 exclusion beats M1, M2 and the END-state chain at all levels (0.025 vs 0.053 [0.033-0.073]); opener-before-closer at a distance replicates on held-out sites and IM77-only (0.37 vs 0.23; 0.62 vs 0.30); the exclusion itself is the same way but inside wide bands out of sample (0.007 vs 0.015; 0.038 vs 0.055); W817/861 initial is a start-state fact; outside fact: seals vs tablets (S29) and W400 on tablets 6.8x (loop 54 c4)'),
 'W400 suffix': ('A', 'raw-final 0.88 vs 0.42 replicates on held-out sites (0.76 vs 0.50) and IM77-only (0.86 vs 0.43) [end-state fact]; the opener exclusion beats every chain at MD+H (0.044 vs 0.073 [0.048-0.098]) and is the same way out of sample (0.040 vs 0.083; 0.080 vs 0.094, inside); outside fact: tablets 6.8x, sealings 1% (object type)'),
 'W740 jar closer': ('A', 'mutual exclusion with the other closers beats M1, M2 and the END-state chain (0.042 vs 0.164 [0.143-0.187]) and replicates on held-out sites (0.040 vs 0.138) and IM77-only (0.073 vs 0.20); jar never doubled (13 vs 126 per 1000); finality 0.87 vs 0.47 is an end-state fact; the gloss stays C'),
 'W2 connective': ('A', 'the once-rule beats every chain (3.5 vs 47 [32-66] per 1000; END-state and S366 generator too) and replicates on held-out sites (5.6 vs 37 [17-67]); underpowered on IM77-only (2 of 40 texts repeat it, 50 vs 43 [0-97]); second position and marked-jar exclusion beat the chains at MD+H only; initial rate chain-explained'),
 'W820 wheel opener': ('B', 'as a member of the opener set it shares the W400 exclusion (beyond all chains); every claim that distinguishes it (initial 0.74 vs 0.72, final 0.11 vs 0.10 [S286], followed by W2/W60) is inside the chain bands; S288 alternation with W595 chain-explained'),
 'W920+W60(+741) opener unit': ('C', 'W920 initial 0.60 vs 0.60 [0.52-0.70] (start-state fact); the run 920-60-741 beats M1 (12.9 vs 2.9) but not M2 (9.7 [band includes it]): an order-2 fact; no outside fact'),
 'name-initial elements W692/575/125/416/413/920/495': ('C', 'first-in-middle share 0.21 vs M1 0.20 [0.16-0.24], text-initial 0.55 vs 0.51 [0.47-0.56]; inside the chain bands on MD+H, held-out sites and IM77-only; S313 replicated against a shuffle, which a chain beats anyway'),
 'name-final elements W840/460/435/440/717/70/35/690': ('C', 'last-before-closer share 0.36 vs M1 0.35 [0.30-0.40]; inside the chain bands on all three sets; S311/S313 were shuffle-null results'),
 'credential model (sec. 11)': ('C', 'nesting 0.163 vs M1 0.156 [0.141-0.171] (inside) and below M2 (0.257); on IM77-only 0.058 vs 0.19 [0.15-0.24], i.e. reversed; confirms S-DARK-41.3'),
 'W575 seven-X': ('C', "'always 7' is a bigram (modal numeral share 0.93 vs M1 0.91 [0.60-1.00]); a chain fitted to the texts has it by construction; no outside fact"),
 'W585 seven-X': ('C', "'always 7' is a bigram (0.85 vs 0.85 [0.59-1.00]); no outside fact"),
 'W632 two-X': ('C', 'merged into W630 at seq_strong/seq_all (no tokens); at seq_raw the fixed numeral is a bigram; no outside fact'),
}
prop = {e: propose2(e, recs) for e, recs in per_entry.items()}
for e, v in OVERRIDE.items():
    if e in prop: prop[e] = v
with open(DARK + 'loop54_regrade.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['sign', 'current grade', 'claim', 'statistic (obs seq_all; seq_raw; seq_strong)', 'Markov-1 null (med [band])', 'Markov-2 null', 'held-out (sites; IM77-only)', 'proposed grade', 'reason', 'kind', 'MD+H verdict', 'M1E', 'S366 gen', 'outside fact', 'S-rows'])
    for e, recs in per_entry.items():
        pg, why = prop[e]
        for x in recs:
            w.writerow([e, x['grade'], x['claim'], f"{fmt(x['obs_all'])}; {fmt(x['obs_raw'])}; {fmt(x['obs_strong'])}", x['M1'], x['M2'], x['heldout'] + ' | IM77 ' + x['im77'], pg, why, x['kind'], x['mdh'], x['M1E'], x['G'], x['outside'], x['srow']])
out = ['# Loop 54 cycle 3: re-graded table (one line per entry; decisive claim in the reason). Grades: proposed vs current.', '| entry | current | proposed | reason | claims beyond chains at all 3 levels | replicated out of sample |', '|---|---|---|---|---|---|']
changes = []
for e, recs in per_entry.items():
    pg, why = prop[e]; cur = recs[0]['grade']
    nb = sum(x['beyond_all_levels'] for x in recs); nr = sum(x['replicated'] for x in recs if x['beyond_all_levels'])
    out.append(f'| {e} | {cur} | {pg} | {why} | {nb}/{len(recs)} | {nr} |')
    if cur != 'grammar' and pg != cur.split()[0].strip('/').rstrip('-') and not cur.startswith(pg + ' '): changes.append((e, cur, pg, why))
out.append('\n## Grade changes')
for e, cur, pg, why in changes: out.append(f'- {e}: {cur} -> {pg}: {why}')
out.append('\n## GRAMMAR statements')
for e, recs in per_entry.items():
    if recs[0]['grade'] == 'grammar' or e.startswith('GRAMMAR'):
        for x in recs: out.append(f"- {e}: {x['claim']}: obs {fmt(x['obs_all'])} vs M1 {x['M1']} M2 {x['M2']}; {x['mdh']}; held-out {x['heldout']}; IM77 {x['im77']}; S366 gen {x['G'] or '-'}")
open(DARK + 'loop54_c3_table.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
