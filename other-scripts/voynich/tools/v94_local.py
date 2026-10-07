"""v94: local-permutation control (unit-level vs topical-block correspondence) applied to stage-2 rows.
usage: v94_local.py TARGETSET  -> data/v94_ckpt/local_TARGETSET.jsonl"""
import os, sys, json
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ.setdefault('V89_LOWRANK', '2'); os.environ.setdefault('VOY_MODE', 'glyph')
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v94_lib as V
import v94_run as R

def job(row):
    tgt = R.get_tgt(row['target'], 12, 11)
    src = R.get_src(row['src'])
    out = {'target': row['target'], 'src': row['src'], 'B_rc': row['B_rc'], 'zB_rc': row['zB_rc']}
    for w in (3, 6):
        r = V.local_perm_test(src, tgt, row['a'], w=w, seed=w)
        out['zL%d' % w] = r['zL']; out['B'] = r['B']; out['Bperm%d' % w] = r['Bperm_mu']
    return out

if __name__ == '__main__':
    tset = sys.argv[1]
    R.TG = R.targets(tset)
    rows = [json.loads(l) for l in open(os.path.join(V.CK, 'stage2_%s.jsonl' % tset))]
    rows.sort(key=lambda r: r['target'])
    with Pool(2) as p, open(os.path.join(V.CK, 'local_%s.jsonl' % tset), 'w') as f:
        for o in p.imap(job, rows):
            f.write(json.dumps(o) + '\n'); f.flush()
            print('%-28s %-42s B %.3f zB_rc %.1f zL3 %.1f zL6 %.1f' % (o['target'][:28], o['src'][:42], o['B'], o['zB_rc'], o['zL3'], o['zL6']), flush=True)
