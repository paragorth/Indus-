"""pe12 cycle 1: CALIBRATION of the check-digit search.
(a) planted PE corpora: hidden check sign CK = f(value mod 3) put in the LAST slot of a
    fraction p of counted entries (p = 0.3, 0.1); a planted SIZE-class sign (log10 band, p 0.3)
(b) proto-cuneiform (all periods): commodity / system dependencies are known to exist
(c) Ur III weight entries, PE-sized sample: unit word (gin2 / ma-na / gu2) is a known
    size-class marker; commodity word is a soft size dependency.
Same engine, same three nulls (STRICT, LOOSE, SIZEM) as the real PE run.
"""
import os, sys, json, copy, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe12_common import pe_entries, PEDATA  # noqa
from pe12_search import run_corpus  # noqa

LOG = open(os.path.join(PEDATA, 'pe12_cycle1.log'), 'a')


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def plant(E, kind, p, seed=5):
    rng = np.random.default_rng(seed)
    out = []
    for e in E:
        e = dict(e); e['signs'] = list(e['signs'])
        if e['sys'] == 'S' and rng.random() < p:
            if kind == 'mod3':
                e['signs'][-1] = 'CK%d' % (int(e['v']) % 3)
            else:
                e['signs'][-1] = 'SZ%d' % min(int(math.log10(max(e['v'], 1))), 2)
        out.append(e)
    return out


if __name__ == '__main__':
    REPS = int(os.environ.get('REPS', 10))
    res = {}
    E = pe_entries()
    for kind, p in [('mod3', 0.3), ('mod3', 0.1), ('size', 0.3)]:
        tag = 'plant_%s_%02d' % (kind, int(p * 100))
        res[tag] = run_corpus(tag, plant(E, kind, p), ['LAST'], reps=REPS, log=log)
    PC = json.load(open(os.path.join(PEDATA, 'pe2_pc_corpus.json')))
    from common import norm_code
    for t in PC:
        for l in t['lines']:
            l['numerals'] = [[n, norm_code(c)] for n, c in l['numerals']]
    EP = pe_entries(PC, corpus='PC')
    log('proto-cuneiform entries', len(EP))
    res['PC'] = run_corpus('PC', EP, ['FIRST', 'LAST'], reps=REPS, log=log)
    from pe12_ur3 import ur3_entries, pe_sized
    EU = pe_sized(ur3_entries())
    log('Ur III weight entries (PE-sized)', len(EU), 'tablets', len(set(e['tab'] for e in EU)))
    res['UR3'] = run_corpus('UR3', EU, ['FIRST', 'LAST'], reps=REPS, log=log)
    json.dump(res, open(os.path.join(PEDATA, 'pe12_cycle1.json'), 'w'), indent=1)
    log('done')
