"""pe4 cycle 1: MDL of a factorial (slot x value) code vs a free-string (name)
code, PE middles by commodity class against calibrated item-code ledgers
(planted herd code; Ur III Drehem herd lines; Ur III textile lines) and
personal-name lists (Drehem 'ki PN-ta' names; Ur III seal names; Linear B
personnel).  All corpora: N distinct strings of >= 2 tokens, length profile
matched to PE_all, R replicates.  Statistic: bits per string saved by the best
factorial model (k <= 5 slots) over the best Markov name model.

Run: python3 pe4_cycle1.py  -> data/pe4_cycle1.json
"""
import json, os, random, statistics as st, sys
from pe4_common import *

N = int(os.environ.get('PE4_N', 110))
R = int(os.environ.get('PE4_R', 8))
KMAX = 5
OUT = os.path.join(PEDATA, 'pe4_cycle1.json')


def main():
    corp = all_corpora()
    prof = profile(corp['PE_all'][0])
    res = {}
    for name, (types, role) in corp.items():
        rows = []
        for r in range(R):
            rng = random.Random(1000 * r + 7)
            smp = matched(types, prof, N, rng)
            a, F = mdl_compare(smp, kmax=KMAX, seed=r)
            b, G = mdl_compare2(smp, kmax=KMAX, seed=r)
            rows.append({'strict_gain': a['gain_per_str'], 'strict_k': a['k'], 'strict_valid': a['valid_share'],
                         'chunk_gain': b['gain_per_str'], 'chunk_k': b['k'], 'chunk_valid': b['valid_share'],
                         'n': a['n'], 'mean_len': a['mean_len']})
        summ = {k: (st.mean(x[k] for x in rows), st.pstdev(x[k] for x in rows))
                for k in rows[0]}
        res[name] = {'role': role, 'rows': rows, 'summary': summ}
        print('%-10s %-3s n=%3d len=%.2f strict gain %+.2f (sd %.2f) k=%.1f valid %.2f | chunk gain %+.2f (sd %.2f) k=%.1f valid %.2f'
              % (name, role, summ['n'][0], summ['mean_len'][0], summ['strict_gain'][0], summ['strict_gain'][1],
                 summ['strict_k'][0], summ['strict_valid'][0], summ['chunk_gain'][0], summ['chunk_gain'][1],
                 summ['chunk_k'][0], summ['chunk_valid'][0]), flush=True)
    json.dump({'N': N, 'R': R, 'res': res}, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
