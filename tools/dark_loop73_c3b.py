"""Loop 73 cycle 3b: relative placement of the Indus middle between the PE middle (t = 0) and the nearest name list
(t = 1) for every length-matched statistic on which PE and the name lists separate (gap > 2 pooled CI half-widths
and PE outside the Ur III .. Linear B range); t from the median, interval from the Indus CI ends. Non-diagnostic
statistics (PE inside the name-list range) listed separately. Reads loop73_c3.json, loop73_c1b.json, loop73_c2.json.
Usage: python3 tools/dark_loop73_c3b.py
"""
import sys, json
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop73_common import DARK, P, LOG, save
R = json.load(open(DARK + 'loop73_c3.json'))
IND = ['Indus_seq_raw', 'Indus_seq_strong', 'Indus_seq_all', 'Indus_im77', 'Indus_seq_raw_seals', 'Indus_im77_seals']
out = {}
P('== Loop 73 cycle 3b: Indus placement t (0 = PE middle, 1 = nearest name list), length-matched draws')
for s, v in R['PE_mid'].items():
    if s == 'shortfall' or not isinstance(v, list): continue
    pe = v; ur = R['UrIII_names'][s]; lb = R['LinB_names'][s]
    hw = lambda x: (x[2] - x[1]) / 2
    lo_n, hi_n = min(ur[0], lb[0]), max(ur[0], lb[0])
    inside = lo_n <= pe[0] <= hi_n
    near = ur if abs(ur[0] - pe[0]) < abs(lb[0] - pe[0]) else lb
    gap = abs(near[0] - pe[0]); diag = (not inside) and gap > 2 * max(hw(pe), hw(near))
    row = dict(pe=pe, ur=ur, lb=lb, diagnostic=diag)
    ts = []
    for lv in IND:
        iv = R[lv][s]
        t = (iv[0] - pe[0]) / (near[0] - pe[0]) if near[0] != pe[0] else float('nan')
        ta = (iv[1] - pe[0]) / (near[0] - pe[0]); tb = (iv[2] - pe[0]) / (near[0] - pe[0])
        ts.append((lv, t, min(ta, tb), max(ta, tb)))
    row['t'] = ts; out[s] = row
    tag = 'DIAGNOSTIC' if diag else ('non-diagnostic: PE inside the name range' if inside else 'non-diagnostic: gap < 2 CI')
    P(f'  {s:11s} PE {pe[0]:.3f}  UrIII {ur[0]:.3f}  LinB {lb[0]:.3f}  [{tag}]  t: ' + '; '.join(f'{lv.replace("Indus_","")} {t:.2f} [{a:.2f},{b:.2f}]' for lv, t, a, b in ts))
D = [s for s in out if out[s]['diagnostic']]
P(f'\n  diagnostic statistics: {D}')
for lv in IND:
    tv = [dict(out[s]['t'])[lv] if False else [x for x in out[s]['t'] if x[0] == lv][0][1] for s in D]
    P(f'  {lv}: mean t over diagnostic statistics {sum(tv)/len(tv):.2f}; t < 0.5 (nearer PE) on {sum(1 for x in tv if x < 0.5)} of {len(tv)}')
# non-length-matched cross-checks from cycles 1b and 2
B = json.load(open(DARK + 'loop73_c1b.json')); C2 = json.load(open(DARK + 'loop73_c2.json'))
P('\n  designation share of entry bits (cycle 1b): PE all {:.2f} / mid>=2 {:.2f}; Indus raw {:.2f} strong {:.2f} all {:.2f} IM77 {:.2f}; Ur III {:.2f}; Linear B {:.2f}'.format(
    B['PE_entries']['des_share_all'], B['PE_entries_mid2']['des_share_all'], B['Indus_seq_raw']['des_share_all'], B['Indus_seq_strong']['des_share_all'],
    B['Indus_seq_all']['des_share_all'], B['Indus_im77']['des_share_all'], B['UrIII_legends']['des_share_all'], B['LinB_lines']['des_share_all']))
P('  recurrence in another document (cycle 2): PE {:.3f} (x{:.2f}); Indus seals raw {:.3f} (x{:.2f}) all {:.3f}; Ur III {:.3f} (x{:.2f}); Linear B {:.3f} (x{:.2f})'.format(
    C2['rec_PE']['obs'], C2['rec_PE']['ratio'], C2['rec_Indus_seq_raw_seals']['obs'], C2['rec_Indus_seq_raw_seals']['ratio'], C2['rec_Indus_seq_all_seals']['obs'],
    C2['rec_UrIII']['obs'], C2['rec_UrIII']['ratio'], C2['rec_LinB']['obs'], C2['rec_LinB']['ratio']))
save('loop73_c3b', out)
