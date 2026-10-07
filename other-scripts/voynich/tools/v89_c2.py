"""v89 cycle 2: every Voynich section (ZL3b, VOY_MODE=glyph, whole word types) against every candidate text,
same frozen search as cycle 1. Kill: rotated/reversed unit orders (zB), genre-matched decoys (rank), shuffled-paragraph
null (perm) for the top hits, IT2a replication for the top hits (cycle 3)."""
import os, sys, json, time, zlib
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ.setdefault('V89_LOWRANK', '2'); os.environ.setdefault('VOY_MODE', 'glyph')
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v89_lib as L

TR = os.environ.get('V89_TR', 'ZL3b')
VIEW = os.environ.get('V89_VIEW', 'w')
OUT = os.path.join(L.CK, 'c2_%s_%s.jsonl' % (TR, VIEW))
T = L.all_texts()
SRC_KEYS = [k for k in T if len(T[k]['units']) >= 20]
SEC = L.voynich_sections(TR)
SECS = ['herbalA_pages', 'herbalB_pages', 'pharma_paras', 'bio_paras', 'stars_paras', 'bio_pages', 'stars_pages']
TOBJ = {}

def job(args):
    sec, sk = args
    if sec not in TOBJ:
        TOBJ[sec] = L.Target([L.view(u['tok'], VIEW) for u in SEC[sec]], n_null=12, seed=7)
    tgt = TOBJ[sec]
    src = L.Source(T[sk]['units'])
    t0 = time.time()
    res = L.search(src, tgt, n_rand=400, top=5, refine=30, seed=zlib.crc32((sec + sk).encode()))
    s = L.summarize(res)
    s.update({'target': sec, 'src': sk, 'n': tgt.n, 'sec': time.time() - t0, 'tr': TR, 'view': VIEW})
    return s

if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            d = json.loads(l); done.add((d['target'], d['src']))
    jobs = [(t, s) for t in SECS for s in SRC_KEYS if (t, s) not in done]
    print(len(jobs), 'jobs', flush=True)
    with Pool(2) as p, open(OUT, 'a') as f:
        for s in p.imap_unordered(job, jobs):
            f.write(json.dumps(s) + '\n'); f.flush()
            print('%-15s %-20s A %.3f B %.3f zB %5.1f %4.0fs' % (s['target'], s['src'], s['A'], s['B'], s['zB'], s['sec']), flush=True)
