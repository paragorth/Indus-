"""v75 cycle 1: fingerprint table. Two independent replicates (A = discovery, B = held-out): different payload
codes, surface seeds, generator seeds and chunk draws. Output data/v75_ckpt/feats_<rep>.pkl (list of rows)."""
import os, sys, pickle, time, zlib
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_lib as X
import v72_lib as V

K_REAL, K_GEN, K_VOY, K_VGEN = 10, 3, 20, 8


def sseed(s, rep):
    return zlib.crc32(('%s|%s' % (s, rep)).encode()) % 100000


def do_ref(args):
    sid, rep = args
    S = X.systems(); kind = S[sid]['kind']; sd = sseed(sid, rep)
    surf = X.ref_surface(S, sid, sd)
    rows = []
    real = X.extract(surf)
    for ci, ch in enumerate(X.chunks(real, K_REAL, sd + 11)):
        rows.append(dict(rep=rep, sid=sid, base=sid, kind=kind, role='real', feats=X.fingerprint(ch, sd + ci)))
    for g, pages in X.gen_streams(surf, sd + 5).items():
        for ci, ch in enumerate(X.chunks(pages, K_GEN, sd + 13)):
            rows.append(dict(rep=rep, sid=sid + ':' + g, base=sid, kind='GEN', role='gen:' + g, feats=X.fingerprint(ch, sd + ci)))
    return rows


def do_voy(args):
    name, rep = args
    sd = sseed(name, rep)
    surf = X.voy_surface(name); real = X.extract(surf); rows = []
    for half in (0, 1):
        for ci, ch in enumerate(X.chunks(real, K_VOY, sd + 17 + half, page_filter=lambda p: V.leaf_half(p['id']) == half)):
            rows.append(dict(rep=rep, sid='VOY_' + name, base='VOY_' + name, kind='?', role='voy%d' % half,
                             feats=X.fingerprint(ch, sd + ci)))
    for g, pages in X.gen_streams(surf, sd + 5).items():
        for ci, ch in enumerate(X.chunks(pages, K_VGEN, sd + 19)):
            rows.append(dict(rep=rep, sid='VOY_%s:%s' % (name, g), base='VOY_' + name, kind='?', role='vgen:' + g,
                             feats=X.fingerprint(ch, sd + ci)))
    sh = X.gshuffle(real, sd + 23)
    for ci, ch in enumerate(X.chunks(sh, K_VGEN, sd + 29)):
        rows.append(dict(rep=rep, sid='VOY_%s:GSHUF' % name, base='VOY_' + name, kind='?', role='vgen:GSHUF',
                         feats=X.fingerprint(ch, sd + ci)))
    return rows


if __name__ == '__main__':
    reps = sys.argv[1:] or ['A', 'B']
    S = X.systems()
    for rep in reps:
        t0 = time.time()
        jobs = [(s, rep) for s in S]
        rows = []
        with Pool(2) as P:
            for r in P.imap_unordered(do_ref, jobs):
                rows += r; print(rep, r[0]['sid'] if r else '-', len(rows), '%.0fs' % (time.time() - t0), flush=True)
            for r in P.imap_unordered(do_voy, [('ZL3b', rep), ('IT2a', rep)]):
                rows += r; print(rep, 'voy', len(rows), flush=True)
        pickle.dump(dict(feats=X.FEATS, rows=rows), open(os.path.join(X.CK, 'feats_%s.pkl' % rep), 'wb'))
        print('done', rep, len(rows), '%.0fs' % (time.time() - t0), flush=True)
