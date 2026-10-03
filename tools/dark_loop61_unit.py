"""S-DARK-61 extra: is the (qualifier + head) UNIT chosen independently of the rest of the string, the way an Ur III title is
chosen independently of the name? MI(element before the unit, unit) with finals permuted (1,000x), vs Ur III
MI(second-to-last name element, title) and MI(last name element, title). Usage: python3 tools/dark_loop61_unit.py"""
import sys, json, collections
sys.argv = ['x', '0']
exec(open('tools/dark_loop61.py').read().split('# ================= cycles')[0])
RES = {}
for LV in ['seq_raw', 'seq_strong', 'seq_all', 'im77']:
    names = [n for n in indus_strings(LV, 'wh', 3) if n[-1] in HEADS]
    pairs = [(n[-3], (n[-2], n[-1])) for n in names]
    RES['indus_prev_unit_' + LV] = pf_test(pairs, f'Indus {LV}: element before the unit -> (qualifier, head) UNIT (n={len(pairs)})')
    pairs2 = [(n[-3], n[-1]) for n in names]
    RES['indus_prev_head_' + LV] = pf_test(pairs2, f'Indus {LV}: element before the unit -> head alone')
recs = [json.loads(l) for l in open(C61 + 'ur3_legend_fields.jsonl')]
p = [(r['name_elems'][-2], 'T:' + r['title']) for r in recs if r['title'] and len(r['name_elems']) >= 2]
RES['ur3_2ndlast_title'] = pf_test(p, f'Ur III second-to-last name element -> title (n={len(p)})')
p = [(r['name_elems'][-1], 'T:' + r['title']) for r in recs if r['title'] and len(r['name_elems']) >= 2]
RES['ur3_last_title'] = pf_test(p, f'Ur III last name element -> title (names >= 2 elements, n={len(p)})')
# Linear B / OB: element before the last two -> final two (a name's 'ending' as a unit)
for nm in ['linb_names', 'ob_name_signs', 'ur3_name_signs']:
    nl = [n for n in jl(REFS[nm]) if len(n) >= 3]
    RES[nm + '_prev_unit'] = pf_test([(n[-3], (n[-2], n[-1])) for n in nl], f'{nm}: element before -> final two as a unit (n={len(nl)})')
save('loop61_unit', RES)
