"""v94 driver.  python3 v94_run.py screen TARGETSET   |   python3 v94_run.py stage2 TARGETSET K
TARGETSET: 'cal' (re-cut calibration targets hidden in the full pool) or 'voy_ZL3b' / 'voy_IT2a' (Voynich sections).
screen: every target x every pool text, v94 search (grid + random re-segmentations), 6 block-shuffle nulls.
stage2: the top K texts per target by screen B are re-searched with 12 nulls (new seed), then every variant's best
map is re-cut by coordinate ascent on A (recut_greedy); zB from the 12 nulls, held-out B on even pairs."""
import os, sys, json, time, zlib
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ.setdefault('V89_LOWRANK', '2'); os.environ.setdefault('VOY_MODE', 'glyph')
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v94_lib as V
import v89_lib as L

SECS = ['herbalA_pages', 'herbalB_pages', 'pharma_paras', 'bio_paras', 'stars_paras']


def cal_targets():
    T = L.all_texts()
    rng = np.random.default_rng(94)
    ce = [u['tok'] for u in T['celsus_eng']['units'][40:120]]
    out = {}
    out['cal_celsus_recut_coded'] = (L.encode_text(V.recut_units(ce, rng), rng), ['v89_celsus_lat', 'v89_celsus_eng'], ['v89_celsus_eng'])
    ab = []
    for u in ce:
        k = int(rng.integers(40, 121)); s = int(rng.integers(0, max(1, len(u) - k))); ab.append(u[s:s + k])
    out['cal_celsus_abr_recut_coded'] = (L.encode_text(V.recut_units(ab, rng), rng), ['v89_celsus_lat', 'v89_celsus_eng'], ['v89_celsus_eng'])
    out['cal_pliny_en_vs_la'] = ([u['tok'] for u in T['pliny_nh_eng']['units'][600:690]], ['v89_pliny_nh_lat', 'v89_pliny_nh_eng'], ['v89_pliny_nh_eng'])
    ps = [u['tok'] for u in T['psalms_en']['units'][20:120]]
    out['cal_psalms_recut_coded'] = (L.encode_text(V.recut_units(ps, rng), rng), ['v89_psalms_he', 'v89_psalms_en', 'sef_Psalms'], ['v89_psalms_en'])
    return out


def voy_targets(tr):
    S = L.voynich_sections(tr)
    return {'%s:%s' % (tr, s): ([u['tok'] for u in S[s]], [], []) for s in SECS}


def targets(name):
    if name == 'cal': return cal_targets()
    if name.startswith('voy_'): return voy_targets(name[4:])
    raise ValueError(name)


TG = None; TOBJ = {}; SRC = {}


def get_tgt(name, n_null, seed):
    k = (name, n_null, seed)
    if k not in TOBJ:
        TOBJ.clear(); TOBJ[k] = V.Target(TG[name][0], n_null=n_null, seed=seed)
    return TOBJ[k]


def get_src(key):
    if key not in SRC:
        if len(SRC) > 6: SRC.clear()
        SRC[key] = V.Src(V.load_text(key))
    return SRC[key]


def job_screen(args):
    tname, sk = args
    tgt = get_tgt(tname, 6, 7)
    src = get_src(sk)
    t0 = time.time()
    n_rand = 300 if tgt.n < 150 else 120
    res = V.search(src, tgt, n_rand=n_rand, top=3, refine=12 if tgt.n < 150 else 6,
                   seed=zlib.crc32((tname + sk).encode()), max_off=150 if tgt.n < 150 else 60)
    if res is None: return {'target': tname, 'src': sk, 'skip': True}
    s = V.summarize(res)
    s.update({'target': tname, 'src': sk, 'n': tgt.n, 'sec': time.time() - t0, 'nA': res[0]['nA']})
    return s


def job_stage2(args):
    tname, sk = args
    tgt = get_tgt(tname, 12, 11)
    src = get_src(sk)
    t0 = time.time()
    res = V.search(src, tgt, n_rand=400, top=4, refine=20, seed=zlib.crc32((tname + sk + 's2').encode()), max_off=250)
    s1 = V.summarize(res)
    out = []
    for v, r in enumerate(res):
        A, B, cuts = V.recut_greedy(src, tgt, r['a'], v, sweeps=2)
        out.append({'A': A, 'B': B, 'a': r['a']})
    s2 = V.summarize(out)
    s = {'target': tname, 'src': sk, 'n': tgt.n, 'sec': time.time() - t0,
         'B_s2': s1['B'], 'zB_s2': s1['zB'], 'A_s2': s1['A'], 'B_rc': s2['B'], 'zB_rc': s2['zB'], 'A_rc': s2['A'],
         'nullB_rc': [o['B'] for o in out[1:]], 'a': s1['a']}
    if os.environ.get('V94_KEEPCUTS'):
        A, B, cuts = V.recut_greedy(src, tgt, res[0]['a'], 0, sweeps=2)
        s['cuts'] = [int(c) for c in cuts]
    return s


def run(jobs, fn, out):
    done = set()
    if os.path.exists(out):
        for l in open(out):
            d = json.loads(l); done.add((d['target'], d['src']))
    jobs = [j for j in jobs if j not in done]
    print(len(jobs), 'jobs', flush=True)
    with Pool(2) as p, open(out, 'a') as f:
        for s in p.imap_unordered(fn, jobs, chunksize=1):
            f.write(json.dumps(s) + '\n'); f.flush()
            if s.get('skip'): continue
            print('%-28s %-40s %s' % (s['target'][:28], s['src'][:40], ' '.join('%s %.3f' % (k, s[k]) for k in ('B', 'zB', 'B_s2', 'zB_s2', 'B_rc', 'zB_rc') if k in s)), '%.0fs' % s['sec'], flush=True)


def ranked(rows, excl):
    rows = [r for r in rows if not r.get('skip') and r['src'] not in excl]
    rows.sort(key=lambda d: -d['B'])
    return rows


if __name__ == '__main__':
    mode, tset = sys.argv[1], sys.argv[2]
    TG = targets(tset)
    keys = V.pool_keys()
    if mode == 'screen':
        jobs = [(t, k) for k in keys for t in TG if k not in TG[t][2]]
        # order: interleave so one target's object stays cached per worker
        jobs.sort(key=lambda j: (j[0], j[1]))
        run(jobs, job_screen, os.path.join(V.CK, 'screen_%s.jsonl' % tset))
    elif mode == 'stage2':
        K = int(sys.argv[3])
        rows = [json.loads(l) for l in open(os.path.join(V.CK, 'screen_%s.jsonl' % tset))]
        jobs = []
        for t in TG:
            rr = ranked([r for r in rows if r['target'] == t], TG[t][2])
            pick = [r['src'] for r in rr[:K]]
            pick += [r['src'] for r in sorted(rr, key=lambda d: -d['zB'])[:max(2, K // 3)] if r['src'] not in pick]
            pick += [p for p in TG[t][1] if p in keys and p not in pick and p not in TG[t][2]]   # partners always re-tested
            jobs += [(t, p) for p in pick]
        run(jobs, job_stage2, os.path.join(V.CK, 'stage2_%s.jsonl' % tset))
