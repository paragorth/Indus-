"""v85 cycle 1: dictionary sharing ratio (DSR) for Voynich hand pairs vs real multi-scribe controls and planted
private / shared dictionary generators.  Usage: VOY_MODE=glyph python3 v85_c1.py [ZL3b|IT2a]"""
import sys, json, math
from multiprocessing import Pool
import v85_lib as L

NAME = sys.argv[1] if len(sys.argv) > 1 else 'ZL3b'


def voy_jobs():
    P, G = L.voy_groups(NAME)
    def g(sec, hand, lang='B'): return G[(sec, hand, lang)]
    # (label, X pages, Y pages, N, design)
    J = [
        ('H h1(A) x h2(B)', g('H', '1', 'A'), g('H', '2'), 1100, 'same section, diff hand (A/B)'),
        ('H h2 x h3', g('H', '2'), g('H', '3'), 270, 'same section, diff hand'),
        ('H h2 x h5', g('H', '2'), g('H', '5'), 270, 'same section, diff hand'),
        ('H h3 x h5', g('H', '3'), g('H', '5'), 270, 'same section, diff hand'),
        ('h2 H x h2 B', g('H', '2'), g('B', '2'), 1100, 'same hand, diff section'),
        ('h1 H x h1 P', g('H', '1', 'A'), g('P', '1', 'A'), 1000, 'same hand, diff section'),
        ('h3 S(A) x h3 S(B)', g('S', '3', 'A'), g('S', '3'), 360, 'same hand+section, diff language'),
        ('h2 B x h3 S', g('B', '2'), g('S', '3'), 1100, 'diff hand, diff section'),
        ('h2 H x h3 S', g('H', '2'), g('S', '3'), 1100, 'diff hand, diff section'),
        ('h1 H x h3 S', g('H', '1', 'A'), g('S', '3'), 1100, 'diff hand, diff section (A/B)'),
    ]
    return J


def run(job):
    lab, px, py, N, des, lev, k, kind = job
    X, Y = L.level(px, lev), L.level(py, lev)
    if kind == 'real':
        d = L.dsr_pair(X, Y, N, k=k)
    elif kind == 'PRIV':
        gx, _ = L.priv_gen(X, 1, k); gy, _ = L.priv_gen(Y, 2, k)
        d = L.dsr_pair(gx, gy, N, k=k)
    else:   # SHARED: one dictionary made from the pooled hands
        pooled = X + Y
        _, dic = L.priv_gen(pooled, 3, k)
        gx, _ = L.priv_gen(X, 4, k, shared_dict=dic); gy, _ = L.priv_gen(Y, 5, k, shared_dict=dic)
        d = L.dsr_pair(gx, gy, N, k=k)
    d.update(label=lab, design=des, level=lev, k=k, kind=kind)
    return d


if __name__ == '__main__':
    jobs = []
    for lab, px, py, N, des in voy_jobs():
        for lev in ('canon', 'surf'):
            jobs.append((lab, px, py, N, des, lev, 2, 'real'))
        jobs.append((lab, px, py, N, des, 'canon', 3, 'real'))
        if lab in ('H h1(A) x h2(B)', 'H h2 x h3', 'h2 B x h3 S'):
            jobs.append((lab, px, py, N, des, 'canon', 2, 'PRIV'))
            jobs.append((lab, px, py, N, des, 'canon', 2, 'SHARED'))
    if NAME == 'ZL3b':
        for kind, N in [('SAME_CULP', 1100), ('SAME_CULP', 270), ('PROFSAME_CULP', 1100), ('SAME_KONRAD', 1100),
                        ('SAME_MACER', 1100), ('SAME_MACER', 270), ('TWO_CULP_GERARD', 1100), ('TWO_MACER_HILDE', 1100),
                        ('TWO_MACER_HILDE', 270), ('TWO_CULP_CURY', 1100), ('TWO_KONRAD_MACER', 1100)]:
            px, py = L.ctrl_pair(kind)
            des = 'control ' + kind
            for lev in ('canon', 'surf'):
                jobs.append((kind, px, py, N, des, lev, 2, 'real'))
            jobs.append((kind, px, py, N, des, 'canon', 3, 'real'))
            if kind == 'SAME_CULP':
                jobs.append((kind, px, py, N, des, 'canon', 2, 'PRIV'))
    with Pool(2) as pool:
        res = pool.map(run, jobs, chunksize=1)
    L.psave('c1_%s.pkl' % NAME, res)
    for d in res:
        print('%-22s %-6s %-5s k%d %-6s %s' % (d['label'], d['kind'], d['level'], d['k'], '', L.fmt(d)), flush=True)
