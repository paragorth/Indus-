"""v89 cycle 3b: (1) replicate the one near-miss (bio paragraphs <- stars paragraphs, ZL3b zB 3.6) in IT2a, with a new
seed, and in the 'nf' (final glyph dropped) view; reverse direction too. (2) how much recurrence structure is there to
find? Oracle held-out B between a section and its own word-coded copy (identity map), Voynich vs real texts cut to
the same unit count and unit sizes."""
import os, sys, json, time, zlib
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ.setdefault('V89_LOWRANK', '2'); os.environ.setdefault('VOY_MODE', 'glyph')
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v89_lib as L
OUT = os.path.join(L.CK, 'c3b.jsonl')

def rep(args):
    tr, view, tg, sk, seed = args
    S = L.voynich_sections(tr)
    tgt = L.Target([L.view(u['tok'], view) for u in S[tg]], n_null=12, seed=7 + seed)
    src = L.Source([{'tok': L.view(u['tok'], view)} for u in S[sk]])
    s = L.summarize(L.search(src, tgt, n_rand=400, top=5, refine=30, seed=zlib.crc32((tg + sk + tr + view).encode()) + seed))
    s.update({'kind': 'rep', 'tr': tr, 'view': view, 'target': tg, 'src': sk, 'seed': seed})
    return s

def oracle_B(units, seed):
    rng = np.random.default_rng(seed)
    coded = L.encode_text(units, rng)
    n = len(units)
    out = []
    Rs = L.resid_from_sets(L.sets_matrix(units, drop_top=30))
    Rc = L.resid_from_sets(L.sets_matrix(coded, drop_top=30))
    for h in (0, 1):
        P = L.pair_index(n, h)
        out.append(float(L.zvec(Rs, P) @ L.zvec(Rc, P)) / len(P[0]))
    return float(np.mean(out))

def strength(_):
    T = L.all_texts(); S = L.voynich_sections('ZL3b')
    rng = np.random.default_rng(5)
    rows = []
    for sec in ('herbalA_pages', 'bio_paras', 'stars_paras', 'pharma_paras', 'herbalB_pages'):
        V = [u['tok'] for u in S[sec]]; n = len(V); lens = [len(u) for u in V]
        rows.append({'kind': 'strength', 'what': 'voynich:' + sec, 'n': n, 'B': oracle_B(V, 1)})
        for tk in ('celsus_lat', 'celsus_eng', 'culpeper', 'circa_instans_fr', 'leechdoms_v1', 'konrad_plants', 'macer_floridus', 'apicius_eng', 'forme_of_cury', 'antidotarium_nl', 'balneis_synopsis', 'v21_IT', 'v21_LA', 'psalms_en'):
            us = [u['tok'] for u in T[tk]['units'] if len(u['tok']) >= 10]
            if len(us) < n: continue
            vals = []
            for r in range(3):
                o = int(rng.integers(0, len(us) - n + 1))
                ex = []
                for u, k in zip(us[o:o + n], lens):    # same unit count, excerpt of the same size as the Voynich unit
                    s0 = int(rng.integers(0, max(1, len(u) - k))); ex.append(u[s0:s0 + k])
                vals.append(oracle_B(ex, 10 + r))
            rows.append({'kind': 'strength', 'what': '%s@%s' % (tk, sec), 'n': n, 'B': float(np.mean(vals))})
    return rows

if __name__ == '__main__':
    jobs = [('IT2a', 'w', 'bio_paras', 'stars_paras', 0), ('ZL3b', 'w', 'bio_paras', 'stars_paras', 1),
            ('ZL3b', 'nf', 'bio_paras', 'stars_paras', 0), ('ZL3b', 'w', 'stars_paras', 'bio_paras', 0)]
    with Pool(2) as p, open(OUT, 'a') as f:
        a = p.apply_async(strength, (0,))
        for s in p.imap_unordered(rep, jobs):
            f.write(json.dumps(s) + '\n'); f.flush()
            print(s['tr'], s['view'], s['target'], s['src'], s['seed'], 'A %.3f B %.3f zB %.1f' % (s['A'], s['B'], s['zB']), s['a'], flush=True)
        for r in a.get():
            f.write(json.dumps(r) + '\n'); print(r, flush=True)
