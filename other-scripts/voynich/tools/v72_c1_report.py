"""v72 cycle 1 summary table (prints; rows are written by hand-checked calls at the bottom)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v72_lib as L

D = {}
for f in os.listdir(os.path.join(L.CK, 'c1')):
    r = json.load(open(os.path.join(L.CK, 'c1', f))); D[(r['name'], r['rule'])] = r


def line(name, rule):
    r = D.get((name, rule))
    if not r: return f'{name:22s} {rule:15s} -'
    rec = r.get('recovery')
    return (f"{name:22s} {rule:15s} n={r['n']:5d} types={r['types']:5d} len={r['mean_len']:.2f} NOISE={r['NOISE']:.3f} "
            f"gen+{r['gain_gen']:.3f} msg+{r['gain_msg']:.3f} (page {r['gain_page']:.3f} big {r['gain_big']:.3f}) "
            f"msgsh+{r['gain_msgsh']:.3f} EXC={r['excess']:.3f} surf={r['surf_bits']:.2f}/{r['surf_bits_ctx']:.2f}"
            + (f" pur={rec['purity']:.3f}/{rec['inv_purity']:.3f}" if rec else ''))


if __name__ == '__main__':
    for rule in L.RULES:
        for c in ['ZL', 'IT'] + ['ZL~' + g for g in L.GENS]: print(line(c, rule))
        print()
    for txt in ['BRU', 'ISI', 'DEU']:
        for mode in ['merge', 'verbose']:
            for rule in ['E2_line', 'E0_identity', 'E1_padding']:
                b = f'{txt}-{mode}'
                for c in [b] + [b + '~' + g for g in L.GENS]:
                    if (c, rule) in D: print(line(c, rule))
            print()
