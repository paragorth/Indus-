"""pe12 cycle 2b: CHECK DIGITS GIVEN SIZE.  Cycle 1 showed that a planted SIZE-class sign
leaks into the arithmetic family even under the SIZEM null (log2 bands are too coarse).
Here the baseline already knows a fine size cell (half-octave of the value x fraction
flag x highest numeral code) and every null shuffles only inside that cell
(STRICT: same tablet + system + cell; SIZEM: same tablet type + system + cell).
Calibration in the same run: planted size-class sign (must now VANISH) and planted
value-mod-3 sign at 10% and 30% (must SURVIVE).
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe12_common import pe_entries, PEDATA  # noqa
from pe12_search import run_corpus  # noqa
from pe12_cycle1 import plant  # noqa

LOG = open(os.path.join(PEDATA, 'pe12_cycle2b.log'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


if __name__ == '__main__':
    REPS = int(os.environ.get('REPS', 10))
    kw = dict(reps=REPS, nhash=10000, npairs=3000, nulls=['STRICT', 'SIZEM'], log=log, fine=True)
    E = pe_entries()
    res = {}
    res['plantF_size_30'] = run_corpus('plantF_size_30', plant(E, 'size', 0.3), ['LAST'], **kw)
    res['plantF_mod3_30'] = run_corpus('plantF_mod3_30', plant(E, 'mod3', 0.3), ['LAST'], **kw)
    res['plantF_mod3_10'] = run_corpus('plantF_mod3_10', plant(E, 'mod3', 0.1), ['LAST'], **kw)
    res['PEfine'] = run_corpus('PEfine', E, ['LAST', 'FIRST', 'LASTm', 'FIRSTm', 'AFTERHDR'], **kw)
    json.dump(res, open(os.path.join(PEDATA, 'pe12_cycle2b.json'), 'w'), indent=1)
    log('done')
