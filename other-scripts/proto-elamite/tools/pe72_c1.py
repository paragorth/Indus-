"""pe72 cycle 1 (PE) / cycle 2 (PC): build and freeze the v2 reading on TRAINING tablets only, then decode the held-out
half under v1 (pe59-R1) and v2, against twins:
  ALL  : every role set, header-slot sign, weight sign, scoped-allotment condition, sealed flag (within size band)
         and dossier membership replaced by frequency-matched random ones (v2 shuffled-role twin)
  EXT  : the v1 part kept true, only the v2 ADDITIONS replaced (the decisive twin for 'does the update add anything')
  UNIT : random unit ladders with the true v2 roles (random-unit twin)
  TRAN : v2 roles on tablets whose numerals were transplanted from another held-out tablet
  V1ALL: v1 with all roles shuffled (pe59's twin, re-run with the same metrics)
usage: python3 pe72_c1.py PE|PC [n_twins]
"""
import sys, os, json, random, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe72_lib as L
P = L.P

CORPUS = sys.argv[1] if len(sys.argv) > 1 else 'PE'
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 40


def setup():
    if CORPUS == 'PE':
        T = L.pe_tablets(); maps = P.pe_maps()
    else:
        T = L.pc_tablets(); maps = P.pc_maps()
    tr, ho = P.split(T)
    return T, tr, ho, maps


def frozen_roles(T, tr, ho, maps):
    """v1 and v2 role structures; v2 is frozen by hash (training tablets only)."""
    cap, cnt = maps
    cntm = cnt['sex2'] if CORPUS == 'PE' else cnt['S']
    if CORPUS == 'PE':
        w52 = json.load(open(os.path.join(L.DATA, 'pe52_frozen_weights.json')))['signs']
        w58 = json.load(open(os.path.join(L.DATA, 'pe58_frozen_factors.json')))['signs']
        cands = {s: (1 if v['tab'] > 0 else -1) for s, v in w52.items() if v['both']}
        for s, v in w58.items():
            cands.setdefault(s, 1 if v['log'] > 0 else -1)
        wres = L.weight_survival(tr, cands, cap, cntm, nperm=2000)
        R2 = L.build_reading2(P.sha(sorted(t['id'] for t in tr)), wres)
        fz = {'reading': R2, 'heldout_ids_sha': P.sha(sorted(t['id'] for t in ho)),
              'pe59_reading_sha': json.load(open(os.path.join(L.DATA, 'pe59_reading_frozen.json')))['sha256']}
        fz['sha256'] = P.sha(fz)
        fn = os.path.join(L.DATA, 'pe72_reading_frozen.json')
        if os.path.exists(fn):
            old = json.load(open(fn))
            assert old['sha256'] == fz['sha256'], 'v2 reading changed after freezing'
        else:
            json.dump(fz, open(fn, 'w'), indent=1, sort_keys=True)
        return L.roles_v1(), L.roles_v2(R2), fz['sha256'], wres
    wdir, wres = L.derive_weights(tr, cap, cntm)
    doss = L.pc_dossiers(T)
    r2 = L.pc_roles_v2(wdir, doss)
    key = {'PC_KNOWN': {k: sorted(v) for k, v in P.PC_KNOWN.items()}, 'PC_EXT': {k: sorted(v) for k, v in L.PC_EXT.items()},
           'wdir': wdir, 'dossiers': doss}
    return L.pc_roles_v1(), r2, P.sha(key), {'weights': wdir, 'n_dossiers': len(doss),
                                             'dossier_tablets': sum(len(v) for v in doss.values())}


def run(C, ho, numerals=None, gloss=False):
    if numerals is None:
        R = [L.decode(t, C, want_gloss=gloss) for t in ho]
    else:
        R = [L.decode(t, C, numerals_from=n) if n is not None else None for t, n in zip(ho, numerals)]
    return R


def job(args):
    kind, k = args
    T, tr, ho, maps = setup()
    r1, r2, _, _ = CACHE_ROLES
    rng = random.Random(P.seed('pe72%s%s%d' % (CORPUS, kind, k)))
    if kind == 'ALL':
        R = L.twin_roles(r2, T, rng, 'all')
        C = L.Ctx(CORPUS, R, maps, tr, sealed_override=L.shuffle_sealed(T, rng))
    elif kind == 'EXT':
        R = L.twin_roles(r2, T, rng, 'ext', v1roles=r1)
        C = L.Ctx(CORPUS, R, maps, tr, sealed_override=L.shuffle_sealed(T, rng))
    elif kind in ('EXTNW', 'ALLNW'):   # post-hoc: the same twins with the weight component removed
        R = L.twin_roles(r2, T, rng, 'ext' if kind == 'EXTNW' else 'all', v1roles=r1)
        R['wdir'] = {}
        C = L.Ctx(CORPUS, R, maps, tr, sealed_override=L.shuffle_sealed(T, rng))
    elif kind == 'UNIT':
        C = L.Ctx(CORPUS, r2, L.random_units(maps, rng), tr)
    elif kind == 'TRAN':
        C = L.Ctx(CORPUS, r2, maps, tr)
        return L.summarise(run(C, ho, numerals=L.transplant(ho, rng)))
    elif kind == 'V1ALL':
        R = P.shuffle_roles(r1, T, rng); R['version'] = 'v1'
        C = L.Ctx(CORPUS, R, maps, tr)
    return L.summarise(run(C, ho))


CACHE_ROLES = None


def main():
    global CACHE_ROLES
    t0 = time.time()
    T, tr, ho, maps = setup()
    r1, r2, rsha, wres = frozen_roles(T, tr, ho, maps)
    CACHE_ROLES = (r1, r2, rsha, wres)
    out = {'corpus': CORPUS, 'reading_sha': rsha, 'n_train': len(tr), 'n_heldout': len(ho), 'weights': wres}
    C1 = L.Ctx(CORPUS, r1, maps, tr)
    C2 = L.Ctx(CORPUS, r2, maps, tr)
    R1 = run(C1, ho, gloss=True)
    R2 = run(C2, ho, gloss=True)
    out['v1'] = L.summarise(R1); out['v2'] = L.summarise(R2)
    out['v1_train'] = L.summarise(run(C1, tr)); out['v2_train'] = L.summarise(run(C2, tr))
    r2n = dict(r2); r2n['wdir'] = {}
    out['v2_noweights'] = L.summarise(run(L.Ctx(CORPUS, r2n, maps, tr), ho))
    if CORPUS == 'PE':
        # leakage check: weights discovered on the TRAINING half only (same procedure as the PC calibration)
        cap, cnt = maps
        wtr, _ = L.derive_weights(tr, cap, cnt['sex2'])
        r2t = dict(r2); r2t['wdir'] = wtr
        out['v2_trainweights'] = L.summarise(run(L.Ctx(CORPUS, r2t, maps, tr), ho))
        out['trainweights'] = wtr

    # tablet-level transitions v1 -> v2
    tr12 = {'gained_full': [], 'lost_full': [], 'gained_strict': [], 'lost_strict': [], 'gained_close': [], 'lost_close': []}
    for a, b in zip(R1, R2):
        if a is None or b is None:
            continue
        for key, f in (('full', 'full'), ('strict', 'strict')):
            if b[f] and not a[f]:
                tr12['gained_' + key].append(a['id'])
            if a[f] and not b[f]:
                tr12['lost_' + key].append(a['id'])
        if b['C3'] is True and a['C3'] is not True:
            tr12['gained_close'].append(a['id'])
        if a['C3'] is True and b['C3'] is not True:
            tr12['lost_close'].append(a['id'])
    out['transitions'] = tr12
    # why v2 fails / passes: component counts
    out['v2_tab_fail_reasons'] = dict(__import__('collections').Counter(x for r in R2 if r for x in r['tfail']))
    out['v2_tab_pass_reasons'] = dict(__import__('collections').Counter(x for r in R2 if r for x in r['tpass']))
    json.dump({'v1': [r for r in R1 if r], 'v2': [r for r in R2 if r]},
              open(os.path.join(L.CK, 'gloss_%s.json' % CORPUS), 'w'), default=str)
    print('real', json.dumps({'v1': out['v1'], 'v2': out['v2']}), time.time() - t0, flush=True)
    jobs = [(k, i) for k in ('ALL', 'EXT') for i in range(NT)] + [('UNIT', i) for i in range(NT // 2)] + \
           [('TRAN', i) for i in range(NT // 2)] + [('V1ALL', i) for i in range(NT // 2)] + \
           [('EXTNW', i) for i in range(NT // 2)] + [('ALLNW', i) for i in range(NT // 2)]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=4)
    tw = {}
    for (k, i), r in zip(jobs, res):
        tw.setdefault(k, []).append(r)
    out['twins'] = tw
    out['time'] = time.time() - t0
    json.dump(out, open(os.path.join(L.CK, 'c1_%s.json' % CORPUS), 'w'), indent=1, default=str)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
