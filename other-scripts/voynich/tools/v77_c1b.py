"""v77 cycle 1b: two more stated kill tests.
K7 v61 Currier A line-end lean: stated kill = the same lean from a line-aware generator without sandhi.
   v61 pipeline unchanged (v61_lib.scan, 4,000 random context classes; v61_edge.index kind 'both'); Currier A lines of
   ZL and IT2a against the STACK and JUNC generators (fitted per section|language, no sandhi rule anywhere) and the
   v61 line-blind pair-resampling null.
K8 v57 word-level ink sawtooth (C+): stated kill = an equal stable skew in another single-hand manuscript with
   short words. Six CREMMA Latin manuscripts (4-5 pages each, abbreviation-heavy hands) taken one by one, on
   all words and on short-word increments only (both words <= 4 letters), with page halves for stability; null =
   within-line shuffles (v57_c1.arrow)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v77_lib as L

LOG = open(os.path.join(L.CK, 'c1b.log'), 'a')


def log(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


def k7_job(arg):
    import v61_lib as S, v61_edge as E
    lab, gen, seed, name = arg
    if gen == 'NULLPB':
        lines = S.pair_resample([l for l in S.load_vms(name) if l['lang'] == 'A'], seed=seed, line_blind_start=True)
    else:
        P = L.voy(name) if gen is None else L.generate(name, gen, seed)
        lines = [l for l in L.to_v61(P) if l['lang'] == 'A']
    recs, nh, info = S.scan(lines, n_random_classes=4000, seed=1, min_tok=10)
    out = dict(label=lab, both=E.index(recs, kind='both', nmin=15), finL=E.index(recs, kind='finL', nmin=15),
               n_test3=sum(r['z_test'] > 3 for r in recs), ntok=sum(len(l['words']) for l in lines))
    L.jsave('c1b_k7_%s.json' % lab.replace(':', '_'), out)
    log('K7', json.dumps(out))
    return out


def k8():
    from v57_lib import Corpus
    from v57_c1 import arrow, skew
    import collections
    res = {}
    rng = np.random.default_rng(77)
    for which in ('V', 'L'):
        C = Corpus(which)
        r = C.r
        ms = [C.pages[p].split(':')[0] if which == 'L' else 'VOYNICH' for p in range(len(C.pages))]
        for m in sorted(set(ms)):
            pg = [p for p in range(len(C.pages)) if ms[p] == m]
            for part, sel in (('all', pg), ('h0', pg[0::2]), ('h1', pg[1::2])):
                pl = collections.defaultdict(list)
                for Ln in C.lines:
                    if C.page[Ln[0]] in sel: pl[int(C.page[Ln[0]])].append(Ln)
                a = arrow(C, r, pl, nsur=300)
                # short-word increments: both words <= 4 letters
                short = np.array([len(w) <= 4 for w in C.word])
                inc, incs_null = [], []
                for ls in pl.values():
                    for Ln in ls:
                        if len(Ln) < 3: continue
                        d = np.diff(r[Ln]); ok = short[Ln][1:] & short[Ln][:-1]
                        inc += list(d[ok])
                sk = float(skew(np.array(inc))) if len(inc) > 30 else float('nan')
                nul = []
                for _ in range(200):
                    v = []
                    for ls in pl.values():
                        for Ln in ls:
                            if len(Ln) < 3: continue
                            rr = r[rng.permutation(Ln)]; d = np.diff(rr)
                            ok = short[Ln][1:] & short[Ln][:-1]
                            v += list(d[ok])
                    if len(v) > 30: nul.append(skew(np.array(v)))
                nul = np.array(nul)
                res['%s|%s' % (m, part)] = dict(skew=a['skew'], z=a['z'], n_inc=int(sum(max(0, len(Ln) - 1) for ls in pl.values() for Ln in ls)),
                                               short_skew=sk, short_n=len(inc),
                                               short_z=float((sk - nul.mean()) / (nul.std() + 1e-9)) if len(nul) else float('nan'))
                log('K8', m, part, json.dumps(res['%s|%s' % (m, part)]))
    L.jsave('c1b_k8.json', res)
    return res


if __name__ == '__main__':
    from multiprocessing import Pool
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('k8', 'all'):
        k8()
    if what in ('k7', 'all'):
        jobs = [('ZL3b', None, 0, 'ZL3b'), ('IT2a', None, 0, 'IT2a'), ('ZL:STACK:771', 'STACK', 771, 'ZL3b'),
                ('ZL:STACK:772', 'STACK', 772, 'ZL3b'), ('IT:STACK:773', 'STACK', 773, 'IT2a'),
                ('ZL:JUNC:771', 'JUNC', 771, 'ZL3b'), ('ZL:SELFCIT:771', 'SELFCIT', 771, 'ZL3b'),
                ('ZL:NULLPB:5', 'NULLPB', 5, 'ZL3b')]
        with Pool(2) as pool:
            R = list(pool.imap_unordered(k7_job, jobs))
        L.jsave('c1b_k7_all.json', R)
