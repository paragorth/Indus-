"""pe12 cycle 2: THE REAL SEARCH.  Proto-Elamite, five slots (FIRST, LAST = before-number,
FIRSTm / LASTm = multi-sign entries only, AFTERHDR = first sign of the line after the header).
~26,000 hypotheses per slot (300 single features, 20,000 random modular hashes of the
numeral, 6,000 random feature pairs), A/B/C held-out tablets, 2 splits; nulls STRICT /
SIZEM with REPS (LOOSE dropped: cycle 1 showed it is always weaker than STRICT) replicates each (whole search re-run each time).
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe12_common import pe_entries, PEDATA  # noqa
from pe12_search import run_corpus  # noqa

LOG = open(os.path.join(PEDATA, 'pe12_cycle2.log'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


if __name__ == '__main__':
    REPS = int(os.environ.get('REPS', 15))
    E = pe_entries()
    log('PE entries', len(E), 'tablets', len(set(e['tab'] for e in E)))
    res = run_corpus('PE', E, ['LAST', 'FIRST', 'LASTm', 'FIRSTm', 'AFTERHDR'], reps=REPS, log=log,
                     nulls=['STRICT', 'SIZEM'])
    json.dump(res, open(os.path.join(PEDATA, 'pe12_cycle2.json'), 'w'), indent=1)
    log('done')
