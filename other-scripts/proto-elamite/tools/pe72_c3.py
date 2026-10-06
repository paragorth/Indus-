"""pe72 cycle 3: (a) full graded glosses of 20 randomly drawn held-out tablets under the frozen v2 reading,
written for a specialist to check (data/pe72_sample_glosses.txt); (b) one refreshed file of every frozen
outside-corpus prediction with kill lines and a hash (data/pe72_frozen_predictions.json).
"""
import sys, os, json, random, hashlib
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], 'PE']
import pe72_lib as L
import pe72_c1 as c1
P = L.P

T, tr, ho, maps = c1.setup()
r1, r2, rsha, wres = c1.frozen_roles(T, tr, ho, maps)
C2 = L.Ctx('PE', r2, maps, tr)
d1 = json.load(open(os.path.join(L.CK, 'c1_PE.json')))
dpc = json.load(open(os.path.join(L.CK, 'c1_PC.json')))

# ---------------------------------------------------------------- (a) sample glosses
elig = [t for t in ho if L.decode(t, C2) is not None]
rng = random.Random(P.seed('pe72sample'))
sample = sorted(rng.sample(elig, 20), key=lambda t: t['id'])
GR_FIX = {'weight sign': 'C- (weights failed held-out: 58% = decoy rate)'}
out = ['# pe72 sample glosses: 20 held-out tablets drawn at random (seed "pe72sample") from the %d decodable held-out '
       'tablets, decoded with the FROZEN v2 reading (data/pe72_reading_frozen.json, sha %s).' % (len(elig), rsha[:16]),
       '# Each element: text | grade (A established, B likely, C guess, C+ survived a decoy-controlled kill test, '
       'C- failed to gain support) | check (PASS / FAIL / - = no check).',
       '# Line and tablet verdicts come from the frozen v2 reading WITHOUT its weight element (the weights failed the held-out '
       'test, Q10); the weight notes are still printed, graded C-. The frozen-with-weights verdict is in [brackets].',
       '# Values: capacity in N39C (ladder N39C 1, N30D 2, N30C 4, N24 12, N39B 24, N01 120, N14 720; B-); litres C.',
       '# Counts in the sexagesimal count map (N14 = 10, N34 = 60). Nothing here is a word reading: glosses name '
       'roles and quantities, not meanings.', '']
summ = []
r2nw = dict(r2); r2nw['wdir'] = {}
C2nw = L.Ctx('PE', r2nw, maps, tr)
for t in sample:
    r = L.decode(t, C2, want_gloss=True)
    rn = L.decode(t, C2nw, want_gloss=True)
    stn = {txt.split(':')[0]: st for txt, g, st in rn['gloss'] if txt.startswith('entry ')}
    r['gloss'] = [(txt, g, stn.get(txt.split(':')[0], st) if txt.startswith('entry ') else st) for txt, g, st in r['gloss']]
    r['frozen'] = (r['dense'], r['full'], r['npass'], r['nfail'])
    r.update({k: rn[k] for k in ('dense', 'full', 'npass', 'nfail', 'tfail')})
    out.append('## %s (%s)%s  -- tablet verdict: %s; lines passing %d, failing %d of %d clean entries' %
               (t['id'], t['site'], ' sealed' if t.get('sealed') else '',
                'fully explained (dense)' if r['dense'] else 'no conflict (lax)' if r['full'] else
                'CONFLICT (' + ', '.join(r['tfail'] + (['%d line(s)' % r['nfail']] if r['nfail'] else [])
                                      + (['total does not close'] if r['C3'] is False else [])) + ')',
                r['npass'], r['nfail'], r['nlines']) + '  [frozen with weights: dense %s, no-conflict %s, pass %d, fail %d]' % r['frozen'])
    for txt, g, st in r['gloss']:
        for k, v in GR_FIX.items():
            if k in txt:
                g = v
        out.append('  %-100s | %-6s | %s' % (txt, g, st))
    out.append('')
    summ.append({'id': t['id'], 'full': r['full'], 'dense': r['dense'], 'pass': r['npass'], 'fail': r['nfail'],
                 'lines': r['nlines'], 'C3': r['C3']})
open(os.path.join(L.DATA, 'pe72_sample_glosses.txt'), 'w').write('\n'.join(out) + '\n')

# ---------------------------------------------------------------- (b) predictions
def fsha(fn):
    d = json.load(open(os.path.join(L.DATA, fn)))
    return d.get('sha256', d.get('sha256_16', P.sha(d)))[:16]


comp = d1['v2']['comp']
yahya = [t for t in T if 'Yahya' in t.get('msite', '')]
ys = ['M056', 'M044', 'M219', 'M136']
yrate = {s: sum(any(s in l['signs'] for l in t['lines']) for t in yahya) for s in ys}
yany = sum(any(s in l['signs'] for l in t['lines'] for s in ys) for t in yahya)
susa = [t for t in T if t['site'].startswith('Susa')]
sany = sum(any(s in l['signs'] for l in t['lines'] for s in ys) for t in susa)
# sealed-type observed rates (whole corpus, pe70 definitions approximated on pe59 tablets)
def hdr157(t):
    h = [l for l in t['lines'] if l['role'] == 'H' and l['signs']]
    return bool(h) and h[0]['signs'][0] == 'M157'
def capl(t):
    return any(l['role'] == 'E' and l['numclean'] and P.ncls(l['nums']) == 'CAP' for l in t['lines'])
sl = [t for t in T if t.get('sealed')]; us = [t for t in T if not t.get('sealed')]
seal_rates = {'sealed_n': len(sl), 'sealed_M157_header': sum(map(hdr157, sl)), 'sealed_with_capacity_line': sum(map(capl, sl)),
              'unsealed_n': len(us), 'unsealed_M157_header': sum(map(hdr157, us)), 'unsealed_with_capacity_line': sum(map(capl, us))}
hs = [x for x in T for l in x['lines'] for s in l.get('rsigns', []) if s == 'M005~a']
p68 = json.load(open(os.path.join(L.DATA, 'pe68_frozen_broken_predictions.json')))
p69 = json.load(open(os.path.join(L.DATA, 'pe69_frozen_predictions.json')))
p64 = json.load(open(os.path.join(L.DATA, 'pe64_ladder_frozen.json')))

PR = []
def add(**k):
    PR.append(k)

add(id='Q1', grade='C', what='capacity unit sizes: N39C about 0.6-0.8 l (N30C 2.4-3.2 l, N24 7.2-9.6 l, N01 72-96 l), '
    'alias x9-12 (N39C 5-7 l)', src='pe16, pe55, pe59 P1; pe57: untestable on open data',
    would_support='Proto-Elamite-period standard vessels (Susa III, Malyan Banesh, Yahya IVC) with capacity modes near '
    '0.7 l, 2.8 l and 8.4 l (or 5-7 l), beating random vessel sets (pe57 calibration: random sets pass 33-50%)',
    would_kill='a measured PE-period vessel series with no mode in 0.5-1.0 l, 2-3.5 l or 4-9 l')
add(id='Q2', grade='C+', what='Yahya set: new Tepe Yahya tablets carry M056 / M044 / M219 / M136 and Susa tablets '
    'rarely carry two of them', src='pe48, pe66 F4',
    corpus={'yahya_tablets': len(yahya), 'yahya_with_any': yany, 'per_sign': yrate, 'susa_tablets': len(susa),
            'susa_with_any': sany},
    would_support='>= 1 of the 4 on >= 30% of >= 20 new Yahya tablets (corpus 16/27), above the Susa rate (corpus 222/1502 = 15%), '
    'and new Susa tablets rarely carry two of them (pe66: 5 vs 19.7 expected)',
    would_kill='40+ new Yahya tablets with none of the four, or a new-Yahya rate at or below the Susa rate')
add(id='Q3', grade='B (type) / C (meaning)', what='sealed-type rules: new sealed tablets are short, headed (M157 more '
    'often than unsealed), carry M288 lines and fewer capacity lines; meaning = signed-for per-head disbursement (C); '
    '|M153+X| compounds mark the sealing party (C)', src='pe70 F1, F5', corpus=seal_rates,
    would_support='on >= 40 new sealed tablets: M157-header share above and capacity-line share below the unsealed '
    'rates of the same archive; a new |M153+...| tablet with a PES0329/0333/0334 impression',
    would_kill='>= 40 new sealed tablets with header M157 and capacity lines at unsealed rates (15.8% / 38%); '
    '>= 3 new |M153+...| tablets with well-preserved unsealed surfaces')
add(id='Q4', grade='B-', what="per-head rule: a bare 'M288 n' line directly after a count line ending in a PERSON-class "
    "sign holds 60 N39C = 2(N39B) 1(N24) per unit (sometimes 120); 'signs + M288' lines follow no per-unit rule",
    src='pe27, pe63 D3, pe66 F6, pe69 F2; pe72 held-out', heldout={'pass': comp.get('allot_ok', 0), 'fail': comp.get('allot_bad', 0)},
    pe64_L1=p64['predictions'][0] if isinstance(p64.get('predictions'), list) and p64['predictions'] else p64.get('predictions'),
    would_support='>= 50% of >= 15 new such pairs at 60k or 120k N39C (random-unit twins: 0%)',
    would_kill='< 1 in 6 of >= 15 new pairs (pe64 L1), or a 30 or 20 per-unit rung on >= 10 new tablets (pe64 L2)')
add(id='Q5', grade='C', what='broken-tablet completions: lost M288 values, lost team counts, total gaps and join '
    'candidates', src='pe68 frozen file (sha %s), pe69 regrade (sha %s)' % (fsha('pe68_frozen_broken_predictions.json'),
                                                                        fsha('pe69_frozen_predictions.json')),
    counts={'pe68_total_gaps': len(p68.get('total_gaps', [])), 'pe68_lost_M288': len(p68.get('team_sum_M288', [])),
            'pe68_join_candidates': len(p68.get('join_candidates_top40', [])),
            'pe69_lost_M288_values': len(p69.get('lost_M288_values', [])), 'pe69_lost_counts': len(p69.get('lost_counts', []))},
    note='pe69: team-count predictions withdrawn (team rule killed); the per-line rule fails on damaged runs (4/14 vs 8.4), '
    'so every completion is C',
    would_support='>= 50% of restored values match on the first 10 joins or collations',
    would_kill='kill lines as frozen in the pe68/pe69 files (e.g. >= 3 of the first 5 restored M288 values off-rule)')
add(id='Q6', grade='B ([M327+M342]) / C+ (M005~a)', what='header slot: |M327+M342| and M005~a stand in the header line',
    src='pe66 F9, F16; pe72 held-out', heldout={'hslot_pass_tablets': comp.get('t_hslot_ok', 0),
                                                'hslot_conflict_tablets': comp.get('t_hslot_bad', 0)},
    corpus={'M005~a_occurrences': len(hs)},
    would_support='>= 80% of new |M327+M342| in the header line; M005~a header share > 3x plain M005',
    would_kill='< 50% of >= 10 new |M327+M342| in the header line; M005~a header share <= plain M005 on >= 10 new')
add(id='Q7', grade='B', what='MEASURED-class final signs (incl. grain-office C+ M081 M265 M266 M296 M112 M286 M248) take '
    'capacity numerals; COUNTED-class never', src='pe59 P4/P5, pe66 F10',
    signs={'MEASURED': sorted(r2['sets']['MEASURED']), 'COUNTED': sorted(r2['sets']['COUNTED'])},
    heldout_role_checks={'pass': comp.get('role_ok'), 'fail': comp.get('role_bad')},
    would_support='>= 80% / >= 95% on >= 50 new entries', would_kill='MEASURED <= base capacity rate; COUNTED < 85% non-capacity')
add(id='Q8', grade='A/B', what='totals: a single reverse numeric line equals the entry sum in the tablet system; no sign '
    'marks totals', src='pe38, pe59 P6/P7', heldout={'v1_close': d1['v1']['C3'], 'v2_close': d1['v2']['C3'],
                                                       'eligible': d1['v2']['C3_n']},
    would_support='>= 10% of new clean totalled tablets close', would_kill='< 4% close; a sign on > 40% of new total lines')
add(id='Q9', grade='B', what='dossier rule: a new copy of a pe63 template series keeps the series header and number system',
    src='pe63; pe72 held-out', heldout={'header_pass': comp.get('t_dossier_header_ok', 0),
                                        'header_fail': comp.get('t_dossier_header_bad', 0)},
    series=L.DOSSIERS, would_support='>= 4 of 5 new copies keep header and system', would_kill='<= 2 of 5')
add(id='Q10', grade='C- (demoted by pe72)', what='class-sign weights (pe52/pe58 directions): an entry with the sign lies '
    'above (+) / below (-) its same-system siblings', src='pe52, pe58; pe72 held-out',
    heldout={'pass': comp.get('w_ok'), 'fail': comp.get('w_bad')},
    note='held-out 58%, the same as frequency-matched decoy signs (58%); pe52 support line was >= 70%',
    would_support='>= 70% on >= 20 new cases AND above decoy signs', would_kill='<= decoy rate on >= 20 new cases')
add(id='Q11', grade='B', what='edge tag: top-edge 1(N34) is commoner on M157-headed tablets and never a sum',
    src='pe23, pe59 P8', would_support='rate on M157 tablets >= 2x other', would_kill='equal rates on >= 100 new tablets')
add(id='Q12', grade='B', what='decoding new Susa tablets with the frozen v2 reading gives the held-out rates +- 10 points',
    heldout={'n': d1['v2']['n'], 'no_conflict': d1['v2']['full'], 'dense': d1['v2']['dense'],
             'line_pass': d1['v2']['pass'], 'line_fail': d1['v2']['fail']},
    would_support='rates within +- 10 points on >= 50 new tablets (e.g. the 88 Tehran Susa tablets once transliterated)',
    would_kill='dense rate below the shuffled-role twin rate (%.1f%%)' % (100 * d1['twins']['ALL'][0]['dense'] / d1['v2']['n']))
add(id='Q13', grade='C', what='independent offices, not a Susa-centred state; outposts write about local goods only as '
    '"local to one site"', src='pe65, pe67 (sha %s)' % fsha('pe67_frozen_predictions.json'),
    would_kill='a new outpost archive predicted from Susa with held-out gain >= 0.2 (pe65)')
outp = {'made': '6 Oct 2026, pe72', 'reading_v2_sha': rsha, 'reading_v1_sha': json.load(open(os.path.join(L.DATA, 'pe59_reading_frozen.json')))['sha256'],
        'supersedes': ['pe59_frozen_predictions.json (P1-P10)'],
        'still_frozen_elsewhere': {f: fsha(f) for f in ('pe46_frozen_predicted_codes.json', 'pe48_frozen_forecasts.json',
                                                         'pe22_outofcorpus_predictions.json', 'pe64_ladder_frozen.json',
                                                         'pe67_frozen_predictions.json', 'pe68_frozen_broken_predictions.json',
                                                         'pe69_frozen_predictions.json')},
        'predictions': PR}
outp['sha256'] = P.sha(outp['predictions'])
json.dump(outp, open(os.path.join(L.DATA, 'pe72_frozen_predictions.json'), 'w'), indent=1, default=str)
json.dump({'sample': summ, 'sha': outp['sha256']}, open(os.path.join(L.CK, 'c3.json'), 'w'), indent=1)
print('sample', Counter((s['dense'], s['full']) for s in summ), 'pred sha', outp['sha256'][:16], len(PR))
