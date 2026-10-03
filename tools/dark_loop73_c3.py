"""Loop 73 cycle 3: the ladder at MATCHED LENGTHS, with bootstrap CIs, and a per-statistic verdict.
All designations >= 2 elements are drawn to the PE middle length distribution (lengths 2-8; the PE middle is the
shortest), n = 700 per draw, 30 draws (tokens for the frequency statistics, distinct strings for the rest).
Corpora: PE middles (all; Susa only; non-class-stripped variant), Indus seq_raw / seq_strong / seq_all / IM77 middles
(all objects; seals only), Ur III owner names, Linear B personnel names.
Verdict per statistic: Indus 'WITH PE' if every Indus level's median is nearer PE than either name list and within
2 pooled CI half-widths of PE; 'WITH NAMES' if nearer a name list than PE; else 'BETWEEN'. PE itself is classed
'PE ~ NAMES' if PE is within 2 half-widths of a name list.
Usage: python3 tools/dark_loop73_c3.py [ndraw]
"""
import sys
NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 30
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop73_common import *

N = 700
R = {}
P(f'== Loop 73 cycle 3: length-matched ladder, n = {N}, {NDRAW} draws')
E = pe_entries()
pe2 = [e['mid'] for e in E if len(e['mid']) >= 2]
L0 = collections.Counter(len(m) for m in pe2)
P('  PE middle length distribution (target):', sorted(L0.items()))
TOK = {'PE_mid': pe2, 'PE_mid_Susa': [e['mid'] for e in E if len(e['mid']) >= 2 and e['prov'].startswith('Susa')]}
# variant: keep the class sign in the middle (only the prefix stripped)
TOK['PE_mid_keepclass'] = [tuple(list(e['mid']) + ([e['cls']] if e['cls'] else [])) for e in E if len(e['mid']) + (1 if e['cls'] else 0) >= 2]
for LV in ['seq_raw', 'seq_strong', 'seq_all', 'im77']:
    O = indus_objs(LV)
    TOK['Indus_' + LV] = [o['mid'] for o in O if len(o['mid']) >= 2]
    TOK['Indus_' + LV + '_seals'] = [o['mid'] for o in O if len(o['mid']) >= 2 and o['ot'] == 'seal']
TOK['UrIII_names'] = [t for t in ur3_names_tokens() if len(t) >= 2]
TOK['LinB_names'] = [x['name'] for x in linb_names_tab() if len(x['name']) >= 2]

def lm_draw(pool, n, r, dedup):
    if dedup: pool = sorted(set(pool))
    byL = collections.defaultdict(list)
    for s in pool: byL[len(s)].append(s)
    tot = sum(L0.values()); out = []; short = 0
    for L, c in L0.items():
        want = int(round(n * c / tot)); have = byL.get(L, [])
        if len(have) >= want: out += r.sample(have, want)
        else: out += have; short += want - len(have)
    return out, short

FK = ['uniq', 'gini_id', 'zipf_id', 'heaps_id', 'top10']
OKs = ['el_ttr', 'el_gini', 'el_heaps', 'pos_excess', 'pos_bound', 'big_reuse', 'mi_ex', 'pair_pred', 'substr']
EK = ['init_ex', 'fin_ex', 'fin_H']
BK = ['bind_share']
ALLK = FK + OKs + EK + BK
for k, pool in TOK.items():
    runs = []; shorts = []
    for b in range(NDRAW):
        r = random.Random(7500 + b)
        t, s1 = lm_draw(pool, N, r, False); m = freq_metrics(t, r)
        d, s2 = lm_draw(pool, N, r, True); shorts.append(s2)
        m.update(D56.metrics(d, r, nnull=10)); m.update(edge_metrics(d, r, nrand=10)); m.update(bind_metrics(d, r, nperm=60))
        runs.append(m)
    R[k] = {s: (q([x[s] for x in runs], 0.5), q([x[s] for x in runs], 0.025), q([x[s] for x in runs], 0.975)) for s in ALLK}
    R[k]['shortfall'] = q(shorts, 0.5)
    P(f'  {k:22s} (dedup shortfall {q(shorts,0.5)}) ' + ' '.join(f'{s} {R[k][s][0]:.3f} [{R[k][s][1]:.3f},{R[k][s][2]:.3f}]' for s in ALLK))

P('\n=== verdict per statistic (length-matched); Indus levels: seq_raw, seq_strong, seq_all, im77')
V = {}
IND = ['Indus_seq_raw', 'Indus_seq_strong', 'Indus_seq_all', 'Indus_im77']
for s in ALLK:
    pe = R['PE_mid'][s]; ur = R['UrIII_names'][s]; lb = R['LinB_names'][s]
    hw = lambda x: (x[2] - x[1]) / 2
    cls = []
    for lv in IND:
        iv = R[lv][s]; dpe = abs(iv[0] - pe[0]); dn = min(abs(iv[0] - ur[0]), abs(iv[0] - lb[0]))
        tol = 2 * max(hw(iv), hw(pe), 1e-3)
        cls.append('WITH_PE' if dpe < dn and dpe <= tol else 'WITH_NAMES' if dn < dpe else 'BETWEEN')
    c = collections.Counter(cls).most_common(1)[0][0] if len(set(cls[:3])) == 1 else 'MIXED:' + '/'.join(cls)
    pen = min(abs(pe[0] - ur[0]), abs(pe[0] - lb[0])) <= 2 * max(hw(pe), hw(ur), hw(lb), 1e-3)
    V[s] = dict(indus=cls, verdict=c if len(set(cls[:3])) == 1 else c, pe_like_names=pen)
    P(f'  {s:11s} PE {pe[0]:.3f} [{pe[1]:.3f},{pe[2]:.3f}]  Indus ' + ' / '.join(f'{R[lv][s][0]:.3f}' for lv in IND) +
      f'  UrIII {ur[0]:.3f}  LinB {lb[0]:.3f}  -> Indus {"/".join(cls)}; PE~names {pen}')
R['_verdict'] = V
save('loop73_c3', R)
