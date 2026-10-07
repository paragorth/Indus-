"""v89 cycle 1: calibration. Can recurrence-structure alignment find a translation partner among 30 texts?
Targets: translation halves (Celsus, Psalms, Pliny with different chapter cuts, Proverbs), a Voynich-like word code
on Celsus English, and an abridged + coded Celsus English. Partner must top the decoys on held-out pairs (B) and beat
rotated/reversed target orders (zB)."""
import os, sys, json, time, zlib
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ.setdefault('V89_LOWRANK', '2')
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v89_lib as L

OUT = os.path.join(L.CK, 'c1.jsonl')
T = L.all_texts()
SRC_KEYS = [k for k in T if len(T[k]['units']) >= 20]

def targets():
    rng = np.random.default_rng(89)
    ce = [u['tok'] for u in T['celsus_eng']['units'][40:120]]
    tg = {
        'celsus_eng': (ce, 'celsus_lat', ['celsus_eng']),
        'psalms_en': ([u['tok'] for u in T['psalms_en']['units'][20:120]], 'psalms_he', ['psalms_en']),
        'pliny_eng': ([u['tok'] for u in T['pliny_nh_eng']['units'][600:690]], 'pliny_nh_lat', ['pliny_nh_eng']),
        'proverbs_en': ([u['tok'] for u in T['proverbs_en']['units']], 'proverbs_he', ['proverbs_en']),
        'celsus_eng_coded': (L.encode_text(ce, rng), 'celsus_lat', []),   # celsus_eng itself stays in as 2nd partner
    }
    ab = []
    for u in ce:   # abridged: a contiguous excerpt of ~Voynich page size (40-120 tokens)
        k = int(rng.integers(40, 121))
        s = int(rng.integers(0, max(1, len(u) - k)))
        ab.append(u[s:s + k])
    tg['celsus_eng_abr_coded'] = (L.encode_text(ab, rng), 'celsus_lat', [])
    return tg

TG = targets()
TOBJ = {}

def job(args):
    tname, sk = args
    if tname not in TOBJ:
        TOBJ[tname] = L.Target(TG[tname][0], n_null=12, seed=7)
    tgt = TOBJ[tname]
    src = L.Source(T[sk]['units'])
    t0 = time.time()
    res = L.search(src, tgt, n_rand=400, top=5, refine=30, seed=zlib.crc32((tname + sk).encode()))
    s = L.summarize(res)
    s.update({'target': tname, 'src': sk, 'partner': TG[tname][1], 'sec': time.time() - t0})
    return s

if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            d = json.loads(l); done.add((d['target'], d['src']))
    jobs = [(t, s) for t in TG for s in SRC_KEYS if s not in TG[t][2] and (t, s) not in done]
    print(len(jobs), 'jobs', flush=True)
    with Pool(2) as p, open(OUT, 'a') as f:
        for s in p.imap_unordered(job, jobs):
            f.write(json.dumps(s) + '\n'); f.flush()
            print('%-22s %-20s A %.3f B %.3f zB %5.1f %4.0fs' % (s['target'], s['src'], s['A'], s['B'], s['zB'], s['sec']), flush=True)
