"""Loop 59: summarise cycle 1 / cycle 2 JSONs across the three merge levels (numbers for the STRATEGIES rows).
Usage: python3 tools/dark_loop59_rows.py 1|2
"""
import sys, json
DARK = '/home/user/Indus-/data/derived/dark/'
CY = sys.argv[1]
LV = ['seq_raw', 'seq_strong', 'seq_all']
if CY == '1':
    for lv in LV + ['seq_all_rows']:
        try: J = json.load(open(DARK + f'loop59_c1_{lv}.json'))
        except FileNotFoundError: continue
        m = J['meta']
        print(f'\n== {lv}: fit {m["fit"]}, held {m["held"]}, im {m["im"]}, |V| {m["V"]}, rule pairs {m["rules"]}; MD+H CV ' + ', '.join(f'{k} {v:.3f}' for k, v in J['cv_mdh'].items()))
        for s in ('held', 'im'):
            R = J[s]; models = [k for k in R if k not in ('slots', 'best')]
            print(f' {s}: ' + '; '.join(f'{k} {R[k]["bits"]:.3f} [{R[k]["lo"]:.2f},{R[k]["hi"]:.2f}] gap {100*R[k]["gap"]:.0f}%' for k in models))
            for sl, d in R['slots'].items():
                u = d['unigram']
                print(f'   {sl:7s} n={d["n"]:4d} ' + ' '.join(f'{k}={d[k]:.2f}({100*(1-d[k]/u):.0f}%)' for k in ('KN2', 'frame_only', 'structural', 'combined', 'S366gen', 'loop33_PFA') if k in d))
else:
    for lv in LV:
        try: J = json.load(open(DARK + f'loop59_c2_{lv}.json'))
        except FileNotFoundError: continue
        print(f'\n== {lv}')
        for s in ('pooled', 'held', 'im'):
            R = J[s]
            for ot in ('ALL', 'SEAL', 'TAB', 'OTHER'):
                if ot in R:
                    d = R[ot]; print(f' {s} {ot}: texts {d["texts"]} best {d["best"]:.1f} bits/text (uni {d["uni"]:.1f}; explained {100*d["explained"]:.0f}%); ' + ', '.join(f'{g} {v:.1f}' for g, v in d['by'].items()) + f'; middle share {100*d["middle_share"]:.0f}%')
            for g in ('frame', 'qualifiers', 'counts', 'middle'):
                if g in R: print(f'   {g}: n {R[g]["n"]} uni {R[g]["uni"]:.2f} -> best {R[g]["best"]:.2f} ({100*(1-R[g]["best"]/R[g]["uni"]):.0f}%)')
            I = R['irreducible']; print(f'   irreducible {I["n"]}/{I["of"]} = {I["n"]/I["of"]:.2f}; by slot ' + ', '.join(f'{k} {a}/{b}' for k, (a, b) in I['byslot'].items()))
            print(f'   KN2 better by >0.5 bit on {R["kn2_better"]} tokens, structural on {R["str_better"]}; top KN2-beats contexts: ' + '; '.join(f'{b["zone"]}/{b["prev"]} n={b["n"]} +{b["gain"]:.2f} ({b["top"]})' for b in R['beats'][:8]))
