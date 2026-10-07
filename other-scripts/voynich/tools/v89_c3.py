"""v89 cycle 3 (stranger): is one Voynich section a re-encoding of ANOTHER Voynich section (a translation pair inside
the book, e.g. Currier B herbal re-writing Currier A herbal, pharma re-listing herbal entries)? Same frozen feature,
search and survivor rule as cycles 1-2. Each Voynich section is a 'source'; targets are the other sections.
Positive control: herbal A pages coded through v89's word code (homophones, splits, fillers) must find herbal A.
Kill: block-shuffled target orders (zB) and the 29 real texts as decoys (zDecoy against c2 rows for the same target)."""
import os, sys, json, time, zlib
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ.setdefault('V89_LOWRANK', '2'); os.environ.setdefault('VOY_MODE', 'glyph')
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v89_lib as L

TR = os.environ.get('V89_TR', 'ZL3b')
OUT = os.path.join(L.CK, 'c3_%s.jsonl' % TR)
SEC = L.voynich_sections(TR)
rng = np.random.default_rng(893)
SEC['herbalA_coded'] = [{'t': u['t'], 'tok': x} for u, x in zip(SEC['herbalA_pages'], L.encode_text([u['tok'] for u in SEC['herbalA_pages']], rng))]
PAIRS = [('herbalA_coded', 'herbalA_pages'),
         ('herbalB_pages', 'herbalA_pages'), ('herbalA_pages', 'herbalB_pages'),
         ('pharma_paras', 'herbalA_pages'), ('pharma_paras', 'herbalB_pages'),
         ('bio_paras', 'stars_paras'), ('stars_pages', 'bio_paras'), ('bio_pages', 'stars_paras'),
         ('herbalB_pages', 'stars_paras'), ('pharma_paras', 'stars_paras'), ('herbalB_pages', 'bio_paras'),
         ('bio_paras', 'herbalA_pages'), ('stars_pages', 'herbalA_pages')]
TOBJ = {}

def job(args):
    tg, sk = args
    if tg not in TOBJ:
        TOBJ[tg] = L.Target([u['tok'] for u in SEC[tg]], n_null=12, seed=7)
    src = L.Source(SEC[sk]); t0 = time.time()
    s = L.summarize(L.search(src, TOBJ[tg], n_rand=400, top=5, refine=30, seed=zlib.crc32((tg + sk).encode())))
    s.update({'target': tg, 'src': sk, 'sec': time.time() - t0, 'tr': TR})
    return s

if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT): d = json.loads(l); done.add((d['target'], d['src']))
    jobs = [p for p in PAIRS if p not in done]
    with Pool(2) as p, open(OUT, 'a') as f:
        for s in p.imap_unordered(job, jobs):
            f.write(json.dumps(s) + '\n'); f.flush()
            print('%-15s %-15s A %.3f B %.3f zB %5.1f %4.0fs' % (s['target'], s['src'], s['A'], s['B'], s['zB'], s['sec']), flush=True)
