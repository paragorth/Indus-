"""v61 cycle 3c: size-matched nulls for the Currier A/B line-edge pull (line-blind pair resampling and within-line shuffle of the subset)
plus size-matched Sanskrit (11k tokens)."""
import json
from multiprocessing import Pool
import v61_lib as L
import v61_edge as E

def job(arg):
    tag, mk, d = arg
    lines = mk()
    if d == 'P':
        lines = L.reverse_text(lines)
    recs, nh, info = L.scan(lines, n_random_classes=4000, seed=1, min_tok=10)
    return {'sub': tag + ' ' + d, 'both': E.index(recs, kind='both', nmin=15), 'n_test3': sum(r['z_test'] > 3 for r in recs)}

def sub(v, n='ZL3b'):
    return lambda: [l for l in L.load_vms(n) if l['lang'] == v]

JOBS = [('nullpairblind A', lambda: L.pair_resample(sub('A')(), seed=3, line_blind_start=True), 'R'),
        ('nullpairblind B', lambda: L.pair_resample(sub('B')(), seed=3, line_blind_start=True), 'R'),
        ('nullpairblind A', lambda: L.pair_resample(sub('A')(), seed=3, line_blind_start=True), 'P'),
        ('nullpairblind A seed4', lambda: L.pair_resample(sub('A')(), seed=4, line_blind_start=True), 'R'),
        ('Sanskrit 11k', lambda: L.opaque(L.load_sanskrit(max_tokens=11000))[0], 'R'),
        ('Sanskrit 11k nullpairblind', lambda: L.pair_resample(L.opaque(L.load_sanskrit(max_tokens=11000))[0], seed=3, line_blind_start=True), 'R')]

def run(i):
    return job(JOBS[i])

if __name__ == '__main__':
    with Pool(2) as p:
        for o in p.imap_unordered(run, range(len(JOBS))):
            print(json.dumps(o), flush=True)
