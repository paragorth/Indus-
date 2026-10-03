"""pe3 cycle 2: MDL segmentation of middles into recurring sub-units.

Same length-matched samples as cycle 1 (N strings, PE-middle length profile).
Statistic: bits per sign saved by greedy MDL chunking, number of accepted
multi-sign units, share of sign tokens inside them.  Nulls: within-string
shuffle (keeps inventory and lengths) and interpolated Markov-2 (keeps local
transition structure).  Then the full PE middle set is segmented and the
induced unit lexicon is written out.
"""
import json, random, os
from collections import Counter
from pe3_common import *

N = int(os.environ.get('PE3_N', 360))
R = int(os.environ.get('PE3_R', 10))
NN = int(os.environ.get('PE3_NN', 10))
OUT = os.path.join(PEDATA, 'pe3_cycle2.json')


def main():
    rng = random.Random(2)
    pe = pe_middles('A')
    prof = length_profile(pe)
    corpora = {'PE_mid_A': pe, 'PE_full_B': pe_middles('B'), 'PE_mid_C': pe_middles('C')}
    types = {'PE_mid_A': 'PE', 'PE_full_B': 'PE', 'PE_mid_C': 'PE'}
    for k, (p, ty) in CALIB.items():
        corpora[k] = load_calib(k)
        types[k] = ty
    res = {}
    for k, c in corpora.items():
        rows = []
        for r in range(R):
            smp, short = matched_sample(c, prof, N, rng)
            o, lex, _ = mdl_segment(smp)
            sh = [mdl_segment(shuffle_within(smp, rng))[0] for _ in range(NN)]
            mk = [mdl_segment(markov2(smp, rng))[0] for _ in range(NN)]
            row = dict(o)
            for key in ('mdl_gain', 'mdl_units', 'mdl_cov'):
                row['z_sh_' + key], row['sh_' + key] = zscore(o[key], [x[key] for x in sh])
                row['z_mk_' + key], row['mk_' + key] = zscore(o[key], [x[key] for x in mk])
            row['mean_unit_len'] = (sum(len(u) for u in lex) / len(lex)) if lex else 0
            rows.append(row)
        mean = {kk: sum(r[kk] for r in rows if r[kk] == r[kk]) / max(1, sum(1 for r in rows if r[kk] == r[kk]))
                for kk in rows[0]}
        mean['type'] = types[k]
        res[k] = mean
        print(k, types[k], {a: round(b, 3) for a, b in mean.items() if isinstance(b, float)}, flush=True)

    # full PE middle set: induced lexicon
    o, lex, uc = mdl_segment(pe, max_units=2000)
    cnt = Counter(u for s in uc for u in s if len(u) > 1)
    full = {'stats': o, 'n_strings': len(pe),
            'units': [['+'.join(u), v] for u, v in cnt.most_common()]}
    print('FULL PE', o, cnt.most_common(25))
    json.dump({'N': N, 'R': R, 'res': res, 'pe_full_lexicon': full}, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
