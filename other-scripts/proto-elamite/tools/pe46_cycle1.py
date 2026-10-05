"""pe46 cycle 1: calibrate the slot-schema code detector and run it on PE.
For every corpus (PE + planted code + real codes + real names) and two nulls of each (signs shuffled
within strings; strings shuffled across tablets) plus a Markov resynthesis of PE: N random schemas
are scored on half the tablets, the top 20 re-scored on the held-out half.
usage: python3 pe46_cycle1.py [N]
"""
import sys, os, json, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe46_lib as L

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


def build():
    pe = L.pe_tokens(); n = len(pe)
    base = dict(PE=pe, PLANT=L.planted(pe), HTS=L.hts(n), HERD=L.ur3_herd(n), UR3N=L.ur3_names(n),
                LBF=L.linb(True), LBA=L.linb(False))
    out = {}
    for k, t in base.items():
        out[k] = t
        out[k + '~sig'] = L.null_signshuf(t, 11)
        out[k + '~tab'] = L.null_tabshuf(t, 12)
    out['PE~mkv'] = L.null_markov(pe, 13)
    return out


def run(arg):
    k, toks = arg
    f = os.path.join(L.CK, f'c1_{k}.json')
    if os.path.exists(f):
        return k, json.load(open(f))
    C = L.make(toks, k)
    t0 = time.time()
    res = []
    for split in (0, 1):
        out = L.search(C, N // 2, seed=100 + split, split_seed=split)
        s = L.summarize(out)
        s['best'] = dict(sch=L.sch_str(out[0]['sch']), held=out[0]['held']['CODE'], R=out[0]['held']['R'],
                         tgt=out[0]['held']['tgt'], E=out[0]['held']['E'])
        if C.get('true') is not None:
            s['recovery'] = L.recovery(C, out[0]['sch'])
            s['recovery_top5'] = [L.recovery(C, o['sch']) for o in out[:5]]
        res.append(s)
    d = dict(corpus=k, n=C['n'], tabs=int(C['tab'].max() + 1), res=res, secs=time.time() - t0)
    json.dump(d, open(f, 'w'), default=str)
    return k, d


if __name__ == '__main__':
    B = build()
    with Pool(2) as p:
        for k, d in p.imap_unordered(run, list(B.items())):
            r = d['res']
            print(k, d['n'], d['tabs'], 'held_med', [round(x['held_med'], 3) for x in r],
                  'FS', [round(x['held_FS'], 3) for x in r], 'OS', [round(x['held_OS'], 3) for x in r],
                  'rec', [x.get('recovery') for x in r], flush=True)
