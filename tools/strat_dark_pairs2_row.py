"""Summarise one S-DARK-2 cycle (real + scrambled-fact control) into data/derived/dark/loop2_cycle<N>_row.txt."""
import json, sys, collections
N = int(sys.argv[1])
R = json.load(open('data/derived/dark/loop2_cycle%d.json' % N))
try: K = json.load(open('data/derived/dark/loop2_cycle%d_control.json' % N))
except FileNotFoundError: K = None

def counts(D):
    ps = D['pair_surv']; ss = D['set_surv']
    return dict(pair_fired=len(D['pair_arrows']), pair_testable=sum(1 for a in D['pair_arrows'] if a['train']),
                pair_hits=len(ps), pair_rep=sum(s['ok'] for s in ps), pair_bonf=sum(s['bonf'] for s in ps),
                pair_strict=sum(s['strict_ok'] for s in ps), pair_all=sum(s['ok'] and s['bonf'] and s['strict_ok'] for s in ps),
                set_fired=len(D['set_arrows']), set_testable=sum(1 for a in D['set_arrows'] if a['train']),
                set_hits=len(ss), set_rep=sum(s['ok'] for s in ss), set_bonf=sum(s.get('bonf', False) for s in ss))
c = counts(R); k = counts(K) if K else None
new_rels = [r for r in R['active'] if r in {'complementary','samemid_diffclose','prefix1','suffix1','numeq','numdiff','summod','sameopen_diffmid','sharek','revmatch','sameclose_diffmid','bothopen','bothjar','lendiff','ident'} and N == 1] \
    or [r for r in R['active'] if r in {'closerclass','sharemidsign','second','lcs2','numsum','revbigram','sharerare','anypos','orderkept','freqbin','bothrare'} and N == 2] \
    or [r for r in R['active'] if r in {'openunit','bothnum','closepair','midlen','midjacc','samelast2','sharenumsign','rotation','frameclass','singlediff'} and N == 3] \
    or (['combo (random and/or/xor of two boolean relations)'] if N == 4 else [])
rep = [s for s in R['pair_surv'] if s['ok']]
fam = collections.Counter((s['kind'], s['fact']) for s in rep)
full = [s for s in rep if s['bonf'] and s['strict_ok']]
srep = [s for s in R['set_surv'] if s['ok']]
lines = []
lines.append('cycle %d seed %d | new relations: %s' % (N, R['seed'], ', '.join(new_rels)))
lines.append('PAIR real: %s' % c); lines.append('PAIR control (facts scrambled within site, room within site x type): %s' % k)
lines.append('replicated pair arrows (held-out sites p<0.01 AND seq_all p<0.01):')
for s in rep:
    lines.append('  %s %s x %s | train MI %.4f p %.4f n %d | fine p %s | strict(site x type) p %s | held MI %.4f p %.4f n %d | seq_all p %.4f | bonf=%s strict=%s' % (
        s['kind'], s['p'], s['fact'], s['train'][0], s['train'][1], s['train'][2], ('%.5f' % s['fine'][1]) if s['fine'] else 'na',
        ('%.4f' % s['strict'][1]) if s['strict'] else 'na', s['held'][0], s['held'][1], s['held'][2], s['all'][1], s['bonf'], s['strict_ok']))
lines.append('replicated set arrows (Harappa same direction p<0.05 AND seq_all p<0.01):')
for s in srep:
    lines.append('  %s' % {kk: vv for kk, vv in s.items() if kk not in ('kind',)})
pf = 'PAIR %d fired (%d testable), %d train hits p<0.002 (%d replicated on held-out sites + seq_all, %d also Bonferroni at %d arrows AND within-site-x-type null)' % (
    c['pair_fired'], c['pair_testable'], c['pair_hits'], c['pair_rep'], c['pair_all'], c['pair_fired'])
if k: pf += '; scrambled-fact control: %d train hits, %d replicated, %d full' % (k['pair_hits'], k['pair_rep'], k['pair_all'])
sf = 'SET %d fired (%d testable), %d MD hits p<0.004 (%d replicated at Harappa + seq_all)' % (c['set_fired'], c['set_testable'], c['set_hits'], c['set_rep'])
if k: sf += '; control: %d hits, %d replicated' % (k['set_hits'], k['set_rep'])
verdict = sys.argv[2] if len(sys.argv) > 2 else 'VERDICT-TODO'
row = '| S-DARK-2.%d | Arrow-in-the-dark, pair+set machine (tools/strat_dark_pairs2.py, seed %d). Pair arrows: random text-pair relation (%d families incl. new: %s) x random object-pair fact (%s); MI statistic; null = object-level permutation of the fact within site (and within site x type as a second null); train MD+Harappa, replicate on held-out sites (p<0.01) and seq_all; Bonferroni over arrows fired. Set arrows: find-groups (site, area, block, room) >= 3 objects, %d set statistics x size band x type filter x dedup; null = re-deal objects among groups of same sizes within site x type; train MD, replicate Harappa (same direction) + seq_all. Control: whole machine on facts scrambled within site | %s. %s | %s |' % (
    N, R['seed'], len(R['active']), ', '.join(new_rels), 'type, emblem, material, shape, boss, area, room, time, period, size h/v, depth, cult, color, completeness, direction, condition, sides',
    len(set(a['stat'] for a in R['set_arrows'])), pf, sf, verdict)
lines.append(''); lines.append('READY-TO-PASTE ROW:'); lines.append(row)
open('data/derived/dark/loop2_cycle%d_row.txt' % N, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
