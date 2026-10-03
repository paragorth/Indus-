"""Loop 73 cycle 1: the Indus-middle battery run on Proto-Elamite entry middles and header designations.
(a) identifier-frequency statistics on designation tokens (S-DARK-53): uniqueness, Gini, Zipf slope, Heaps, top-10 share
(b) element / order statistics on deduplicated designations >= 2 elements (S-DARK-56): element TTR, Gini, Zipf, Heaps,
    positional excess, adjacent MI excess, pair predictability, bigram reuse, substring share
(c) closed edge sets (S-DARK-56.3d): top-10 initial / final coverage minus frequency-matched random strings
(d) slot binding inside the designation (S-DARK-61.2): penultimate -> final MI excess, loyalty
(e) the one-element entry (S-DARK-58.3 position-class code)
Populations drawn to common n (without replacement, NDRAW draws): PE entry middles, PE header designations (small),
Indus middles seq_raw / seq_strong / seq_all / IM77, Ur III owner names (one per legend), Linear B personnel names.
Usage: python3 tools/dark_loop73_c1.py [ndraw]
"""
import sys
NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 30
sys.argv = ['x', '1000']
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop73_common import *
import dark_loop58_calib as C58
C58.P = P

R = {}
P(f'== Loop 73 cycle 1: PE middles vs Indus middles, identical battery; ndraw {NDRAW}')
P(f'PE frame sets fixed from test a (z >= 3): PREFIX {sorted(PREFIX)}; CLASS {sorted(CLASS)}')
E = pe_entries()
HD = pe_headers()
pe_tok1 = [e['mid'] for e in E if len(e['mid']) >= 1]
pe_tok2 = [e['mid'] for e in E if len(e['mid']) >= 2]
hd_tok1 = [h['des'] for h in HD if len(h['des']) >= 1]
P(f'PE clean entries {len(E)}: middle length {sorted(collections.Counter(len(e["mid"]) for e in E).items())}; class sign present {sum(1 for e in E if e["cls"])}, prefix {sum(1 for e in E if e["prefix"])}')
P(f'PE header designations (opener removed) with >= 1 sign: {len(hd_tok1)}; >= 2: {sum(1 for h in hd_tok1 if len(h)>=2)}; commonest {collections.Counter(hd_tok1).most_common(8)}')
pops_tok = {'PE_mid': pe_tok1, 'PE_header': hd_tok1}
pops_ded = {'PE_mid': sorted(set(pe_tok2))}
# PE no-strip variant (whole entry sign string, numerals removed) = the S-DARK-56.2 PE row
pops_ded['PE_entry_nostrip'] = sorted(set(e['signs'] for e in E if len(e['signs']) >= 2))
for LV in ['seq_raw', 'seq_strong', 'seq_all', 'im77']:
    O = indus_objs(LV)
    pops_tok['Indus_' + LV] = [o['mid'] for o in O if len(o['mid']) >= 1]
    pops_ded['Indus_' + LV] = sorted(set(o['mid'] for o in O if len(o['mid']) >= 2))
pops_tok['UrIII_names'] = ur3_names_tokens()
LB = linb_names_tab()
pops_tok['LinB_names'] = [x['name'] for x in LB]
pops_ded['UrIII_names'] = sorted(set(s for s in ur3_names_dedup() if len(s) >= 2))
pops_ded['LinB_names'] = sorted(set(x['name'] for x in LB if len(x['name']) >= 2))
for k, v in pops_tok.items(): P(f'  tokens {k}: n={len(v)} distinct {len(set(v))}')
for k, v in pops_ded.items(): P(f'  dedup >=2 {k}: n={len(v)} mean len {sum(map(len,v))/len(v):.2f}')

# ---- (a) frequency statistics at common n
NF = 1500
P(f'\n##### (a) identifier-frequency statistics, n = {NF} tokens drawn without replacement x {NDRAW} (PE header: full n = {len(hd_tok1)})')
FK = ['uniq', 'gini_id', 'zipf_id', 'heaps_id', 'top10', 'gt_new']
R['freq'] = {}
for k, v in list(pops_tok.items()) + [(k + '>=2', [t for t in v if len(t) >= 2]) for k, v in pops_tok.items() if k != 'PE_header']:
    runs = []
    for b in range(NDRAW):
        r = random.Random(7300 + b); sub = r.sample(v, min(NF if '>=2' not in k else 1000, len(v))); runs.append(freq_metrics(sub, r))
    R['freq'][k] = {s: (q([x[s] for x in runs], 0.5), q([x[s] for x in runs], 0.025), q([x[s] for x in runs], 0.975)) for s in FK}
    R['freq'][k]['n'] = min(NF if '>=2' not in k else 1000, len(v))
    P(f'  {k:18s} n={R["freq"][k]["n"]:5d} ' + ' '.join(f'{s} {R["freq"][k][s][0]:.3f} [{R["freq"][k][s][1]:.3f},{R["freq"][k][s][2]:.3f}]' for s in FK))

# ---- (b)(c)(d) on deduplicated designations at common n
ND = 1000
P(f'\n##### (b-d) element, order, edge and binding statistics on distinct designations >= 2 elements, n = {ND} x {NDRAW}')
OK = ['el_ttr', 'el_gini', 'el_zipf', 'el_heaps', 'pos_excess', 'pos_bound', 'big_reuse', 'mi_ex', 'pair_pred', 'substr', 'meanL']
EK = ['top10_init', 'top10_fin', 'init_ex', 'fin_ex', 'fin_H', 'fin_H_rand']
BK = ['bind_mi_ex', 'bind_share', 'loyal', 'loyal_null']
R['ded'] = {}
for k, v in pops_ded.items():
    runs = []
    for b in range(NDRAW):
        r = random.Random(7400 + b); sub = r.sample(v, min(ND, len(v)))
        m = D56.metrics(sub, r, nnull=10); m.update(edge_metrics(sub, r, nrand=10)); m.update(bind_metrics(sub, r, nperm=60))
        runs.append(m)
    R['ded'][k] = {s: (q([x[s] for x in runs], 0.5), q([x[s] for x in runs], 0.025), q([x[s] for x in runs], 0.975)) for s in OK + EK + BK}
    P(f'  {k:18s} ' + ' '.join(f'{s} {R["ded"][k][s][0]:.3f} [{R["ded"][k][s][1]:.3f},{R["ded"][k][s][2]:.3f}]' for s in OK))
    P(f'  {"":18s} ' + ' '.join(f'{s} {R["ded"][k][s][0]:.3f} [{R["ded"][k][s][1]:.3f},{R["ded"][k][s][2]:.3f}]' for s in EK + BK))
# what the PE middle edges are
m2 = pops_ded['PE_mid']
P('  PE middle commonest initials', collections.Counter(n[0] for n in m2).most_common(10))
P('  PE middle commonest finals', collections.Counter(n[-1] for n in m2).most_common(10))

# ---- (e) one-element entries (S-DARK-58.3 code)
P('\n##### (e) one- and two-element texts: position class of the sign in long texts (>= 4) of the same corpus; null = frequency draws (1,000x)')
R['minimal'] = {}
pe_texts = [e['signs'] for e in E]
for n in (1, 2):
    R['minimal'][f'PE_entries/{n}'] = C58.run('PE-entries(signs, numerals removed)', pe_texts, n)
one = [e['signs'][0] for e in E if len(e['signs']) == 1]
c = collections.Counter('CLASS' if a in CLASS else 'PREFIX' if a in PREFIX else 'OTHER' for a in one)
nC = len(one); share_cls_tokens = sum(1 for e in E for a in e['signs'] if a in CLASS) / sum(len(e['signs']) for e in E)
P(f'  PE 1-sign entries {nC}: CLASS {c["CLASS"]/nC:.3f} (class share of all entry tokens {share_cls_tokens:.3f}), PREFIX {c["PREFIX"]/nC:.3f}, OTHER (middle-type) {c["OTHER"]/nC:.3f}; commonest {collections.Counter(one).most_common(12)}')
R['minimal']['PE_1sign_frame'] = dict(n=nC, cls=c['CLASS'] / nC, cls_base=share_cls_tokens, pre=c['PREFIX'] / nC, other=c['OTHER'] / nC)
# Indus seals at three levels via the same code (whole-corpus roles, seal minimal texts)
for LV in ['seq_raw', 'seq_strong', 'seq_all']:
    rows, T = C58.indus_wells(LV)
    allT = T['ALL']
    for n in (1, 2):
        sub = [s for s in T['seal'] if len(s) == n]
        R['minimal'][f'Indus_{LV}_seal/{n}'] = C58.run(f'Indus-{LV}-seal', [s for s in allT if len(s) != n] + sub, n)
save('loop73_c1', R)
