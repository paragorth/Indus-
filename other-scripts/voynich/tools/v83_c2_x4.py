#!/usr/bin/env python3
"""v83 cycle 2, x4 list-language confound re-run on filtered Voynich text.
x4 (proto-elamite/tools/x4_cycle3.py) retrained the v31 random-classifier vote with list-genre languages and their
generators (REF2): the Voynich generator vote fell from 77% to 49-57% (ZL). Here the Voynich test samples (v31
sampling: 100-word samples, up to 80 per text, one document per section, v31_lib.features) are rebuilt from each
version of the text (v83_harness) and voted on by the same REF2 survivors (one run, NH random classifiers, the
published v31 V_ZL / V_IT samples included as the reference point). Control: label-permuted REF2 (survivor rate).
Out: data/v83_ckpt/c2_x4.json"""
import os, sys, json, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
PE = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'proto-elamite', 'tools'); sys.path.insert(0, PE)
import numpy as np
import v83_harness as H
import v83_parse as P

VERS = os.environ.get('V83_X4_VERS', 'legacy,all,clean,agree,thin-clean-1,thin-clean-2,thin-agree-1,thin-agree-2').split(',')
FEAT = os.path.join(P.CK, 'c2_x4_feats.json')


def feats_for(ver, name):
    import v31_lib as L, v31_feats as F
    H.set_version(ver)
    docs = F.voynich_docs(name)
    S = L.samples(docs, N=100, maxs=80)
    out = []
    for i, s in enumerate(S):
        rng = random.Random(hash(('V_' + name, i)) & 0xffffffff)
        out.append({k: float(v) for k, v in L.features(s, rng).items()})
    return out


def work(a):
    ver, name = a
    return ver, name, feats_for(ver, name)


if __name__ == '__main__':
    FE = json.load(open(FEAT)) if os.path.exists(FEAT) else {}
    todo = [(v, n) for v in VERS for n in ('ZL3b', 'IT2a') if '%s|%s' % (v, n) not in FE]
    if todo:
        from multiprocessing import Pool
        with Pool(2) as Pp:
            for ver, name, F in Pp.imap_unordered(work, todo):
                FE['%s|%s' % (ver, name)] = F; json.dump(FE, open(FEAT, 'w'))
                print('feats', ver, name, len(F), flush=True)
    import x4_cycle3 as X3
    recs, keys, M = X3.load_all()
    extra = []
    for k, F in FE.items():
        for f in F: extra.append(('v83:' + k, None, f))
    recs = recs + extra
    M2 = np.array([[r[2].get(kk, 0.0) for kk in keys] for r in extra], float)
    M = np.vstack([M, np.nan_to_num(M2, nan=0.0, posinf=0.0, neginf=0.0)])
    ob = X3.build

    def build(recs_, M_, ref, leave=None):
        y, grp, tests = ob(recs_, M_, ref, leave)
        corp = np.array([r[0] for r in recs_], object)
        tests = {k: v for k, v in tests.items() if k.startswith('v31:V_') or k.startswith('x4:VOY')}
        for k in sorted(set(corp)):
            if k.startswith('v83:'): tests[k] = corp == k
        return y, grp, tests
    X3.build = build
    out = {}
    nh = int(os.environ.get('NH', 1000))
    out['REF2'] = X3.run_ref(recs, keys, M, 2, nh=nh)
    out['NULL2'] = X3.run_ref(recs, keys, M, 2, permute=True, nh=nh // 2)
    json.dump(out, open(os.path.join(P.CK, 'c2_x4.json'), 'w'), indent=1)
    for tag in ('REF2', 'NULL2'):
        r = out[tag]; print(tag, 'surv', r['nsurv'], '/', r['n'])
        for k, v in sorted(r['votes'].items()):
            print('   %-28s %s' % (k, {c: v[c] for c in X3.CLASSES if v.get(c)}), 'n', v['n'])
