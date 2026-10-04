"""pe4 cycle 4: power check for cycle 3.  Ur III attribute and name ledgers
cut down to the size of the PE multi-sign middle set (tablets drawn at random
until >= 433 valued records), multi-token records only; 20 replicates, 200
within-tablet permutations each.  Power = share of replicates with p < 0.05.
Also PE multi-sign middles of every class pooled, and by system.
Run: python3 pe4_cycle4.py -> data/pe4_cycle4.json
"""
import json, os, random, statistics as st
from collections import defaultdict
import pe4_cycle3 as c3
from pe4_common import *

c3.NPERM = 200
REPS = 20
TARGET = 433
OUT = os.path.join(PEDATA, 'pe4_cycle4.json')


def cut(rows, rng):
    byt = defaultdict(list)
    for r in rows:
        byt[r[0]].append(r)
    ts = list(byt)
    rng.shuffle(ts)
    out = []
    for t in ts:
        out += byt[t]
        if len(out) >= TARGET:
            break
    return out


def main():
    rng = random.Random(9)
    R = pe_records()
    C = controls()
    herd_c = clean_ledger(C['herd'])
    voc = {w for r in herd_c for w in r['words']}
    tex_c = clean_ledger([dict(r, head=(r['words'][0] if r['words'] else '-')) for r in C['textile']],
                         min_tab=10, min_heads=2)
    m2 = lambda r: tuple(r['words']) if len(r['words']) >= 2 else ()
    full = {
        'herd_attr_m2': c3.prep(herd_c, m2),
        'tex_attr_m2': c3.prep(tex_c, m2),
        'herd_attr_1w': c3.prep(herd_c, lambda r: tuple(r['words']) if len(r['words']) == 1 else ()),
    }
    res = {}
    for k, rows in full.items():
        out = []
        for i in range(REPS):
            r = c3.run('%s#%d' % (k, i), cut(rows, rng), rng)
            if r:
                out.append(r)
        pw = sum(r['p'] < 0.05 for r in out) / len(out)
        res[k] = {'power': pw, 'ratio_med': st.median(r['ratio'] for r in out),
                  'tokens_med': st.median(r['tokens'] for r in out), 'reps': out}
        print('== %s power %.2f median ratio %.2f median tokens %.0f' % (k, pw, res[k]['ratio_med'], res[k]['tokens_med']), flush=True)
    mid = lambda r: tuple(r['attr']) if len(r['attr']) >= 2 else ()
    pe = {'PE_m2_all': c3.prep([r for r in R if r['sys'] in ('SDB', 'C')], mid),
          'PE_m2_C': c3.prep([r for r in R if r['sys'] == 'C'], mid),
          'PE_m2_SDB_anyclass': c3.prep([r for r in R if r['sys'] == 'SDB'], mid)}
    c3.NPERM = 1000
    for k, rows in pe.items():
        res[k] = c3.run(k, rows, rng)
    json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
