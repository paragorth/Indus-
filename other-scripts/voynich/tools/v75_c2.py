"""v75 cycle 2: SUBTRACT THE GENERATOR. The signature of a system is not its statistics but what it has that its own
fitted generators lack: delta = mean(real chunks) - mean(generator chunks), per generator (WSHUF, MK2, SELFCIT,
JUNC, SC10), on the 26 non-layout fingerprint features (130 numbers). Generators are fitted to the EXTRACTED stream
(the message layer), so the surface machinery is out of the comparison. Bootstrap draws of chunks give 8 delta
vectors per system. GEN class: a reference system's MK2 and SELFCIT streams treated as if real (their own
generators fitted to them). Voynich-side controls: the Voynich's MK2, SELFCIT, JUNC and global-shuffle streams
treated as if real. Output data/v75_ckpt/delta_<rep>.pkl (same row format as feats_<rep>.pkl)."""
import os, sys, pickle, time, zlib, random
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_lib as X
import v72_lib as V

NL = [i for i, f in enumerate(X.FEATS) if f not in X.LAYOUT]
GN = list(V.GENS)
NB, KR, KG = 8, 16, 6


def sseed(s, rep):
    return zlib.crc32(('c2|%s|%s' % (s, rep)).encode()) % 100000


def deltas(stream, sd):
    """bootstrap delta vectors of a stream against generators fitted to it."""
    R = np.array([X.fingerprint(ch, sd + i) for i, ch in enumerate(X.chunks(stream, KR, sd + 1))])[:, NL]
    Gm = {}
    for g in GN:
        gp = V.GENS[g](stream, seed=sd + 7)
        Gm[g] = np.array([X.fingerprint(ch, sd + 50 + i) for i, ch in enumerate(X.chunks(gp, KG, sd + 3))])[:, NL]
    rng = np.random.default_rng(sd); out = []
    for b in range(NB):
        r = R[rng.choice(len(R), 8, replace=False)].mean(0)
        v = np.concatenate([r - Gm[g][rng.choice(len(Gm[g]), 4, replace=False)].mean(0) for g in GN])
        out.append([float(x) for x in v])
    return out


def do_ref(args):
    sid, rep = args
    S = X.systems(); kind = S[sid]['kind']; sd = sseed(sid, rep)
    surf = X.ref_surface(S, sid, sd); real = X.extract(surf)
    if sum(len(l['w']) for p in real for l in p['lines']) < 1.5 * X.N_TOK: return []
    rows = [dict(rep=rep, sid=sid, base=sid, kind=kind, role='real', feats=v) for v in deltas(real, sd)]
    for g in ('MK2', 'SELFCIT'):
        gs = V.GENS[g](real, seed=sd + 99)
        rows += [dict(rep=rep, sid=sid + ':' + g, base=sid, kind='GEN', role='gen:' + g, feats=v) for v in deltas(gs, sd + 3)]
    return rows


def do_voy(args):
    name, rep = args
    sd = sseed(name, rep); real = X.extract(X.voy_surface(name)); rows = []
    for half in (0, 1):
        sub = [p for p in real if V.leaf_half(p['id']) == half]
        rows += [dict(rep=rep, sid='VOY_' + name, base='VOY_' + name, kind='?', role='voy%d' % half, feats=v)
                 for v in deltas(sub, sd + half)]
    for g in ('MK2', 'SELFCIT', 'JUNC'):
        gs = V.GENS[g](real, seed=sd + 99)
        rows += [dict(rep=rep, sid='VOY_%s:%s' % (name, g), base='VOY_' + name, kind='?', role='vgen:' + g, feats=v)
                 for v in deltas(gs, sd + 5)]
    rows += [dict(rep=rep, sid='VOY_%s:GSHUF' % name, base='VOY_' + name, kind='?', role='vgen:GSHUF', feats=v)
             for v in deltas(X.gshuffle(real, sd + 23), sd + 9)]
    return rows


if __name__ == '__main__':
    reps = sys.argv[1:] or ['A', 'B']
    S = X.systems(); feats = ['%s:%s' % (g, X.FEATS[i]) for g in GN for i in NL]
    for rep in reps:
        t0 = time.time(); rows = []
        with Pool(2) as P:
            for r in P.imap_unordered(do_ref, [(s, rep) for s in S] ):
                rows += r; print(rep, r[0]['sid'] if r else '-', len(rows), '%.0fs' % (time.time() - t0), flush=True)
            for r in P.imap_unordered(do_voy, [('ZL3b', rep), ('IT2a', rep)]):
                rows += r; print(rep, 'voy', len(rows), flush=True)
        pickle.dump(dict(feats=feats, rows=rows), open(os.path.join(X.CK, 'delta_%s.pkl' % rep), 'wb'))
        print('done', rep, len(rows), '%.0fs' % (time.time() - t0), flush=True)
