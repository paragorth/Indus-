"""v33 cycle 1 report: per-corpus medians over the 32 common rules, and the two built-in controls."""
import glob
from v33_lib import *

COMMON = [r.name for r in rule_set(20)]


def loadall(prefix='c1_'):
    D = {}
    for f in sorted(glob.glob(os.path.join(CK, prefix + '*.json'))):
        n = os.path.basename(f)[len(prefix):-5]
        if n == 'corpora' or not n[0].isupper(): continue
        D[n] = json.load(open(f))
    return D


def corp(n): return n.rsplit('_', 1)[0]


def med(D, c, k, rules=COMMON):
    v = [D[n][r][k] for n in D if corp(n) == c for r in rules if r in D[n]]
    return float(np.median(v)) if v else float('nan')


if __name__ == '__main__':
    D = loadall()
    cs = sorted(set(corp(n) for n in D))
    print('corpus      ' + ' '.join(f'{k:>7}' for k in ['conn', 'NODF', 'zNODF', 'dNODF', 'C', 'zC', 'Q', 'zQ', 'dQ', 'cvrow', 'ginicol']))
    for c in cs:
        print(f'{c:12s}' + ' '.join(f'{med(D, c, k):7.2f}' for k in ['conn', 'NODF', 'zNODF', 'dNODF', 'C', 'zC', 'Q', 'zQ', 'dQ', 'cvrow', 'ginicol']))
    # repair control
    for c in ('V-ZL', 'L-la', 'P-gshuf'):
        zs = {k: [D[n][r]['repair'][k] for n in D if corp(n) == c for r in COMMON if 'repair' in D[n].get(r, {})] for k in ('zNODF', 'zC', 'zQ')}
        print('REPAIR', c, {k: (round(float(np.median(np.abs(v))), 2), round(float(np.mean(np.abs(np.array(v)) < 2)), 2)) for k, v in zs.items() if v})
    # planted vs base, per rule and block
    for pl in ('P-plant30', 'P-plant15'):
        for k in ('dNODF', 'zNODF', 'dQ', 'conn', 'zC'):
            d = [D[f'{pl}_{i}'][r][k] - D[f'V-ZL_{i}'][r][k] for i in (0, 1) for r in COMMON
                 if f'{pl}_{i}' in D and f'V-ZL_{i}' in D]
            bb = [abs(D['V-ZL_0'][r][k] - D['V-ZL_1'][r][k]) for r in COMMON] if 'V-ZL_1' in D else [0]
            if d: print('PLANT', pl, k, 'median delta', round(float(np.median(d)), 3), 'frac>0', round(float(np.mean(np.array(d) > 0)), 2),
                        'median |block diff|', round(float(np.median(bb)), 3))
