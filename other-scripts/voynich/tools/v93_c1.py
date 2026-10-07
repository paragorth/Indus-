"""v93 cycle 1: 400 random interlinear hypotheses x 16 corpora; selection on split 1, test on split 2."""
import sys, os, json, random, time
from multiprocessing import Pool
import v93_lib as L

CORPORA = ['PS_LINE', 'PS_INLINE', 'CE_PARA', 'PS_LINE~shuf', 'CE_PARA~shuf', 'PS_EN', 'PS_HE', 'CE_LA',
           'ZL3b', 'IT2a', 'ZLshuf0', 'ZLshuf1', 'ITshuf0', 'SELFCIT', 'MK2', 'JUNC']
NH = int(os.environ.get('NH', 400))


def hyps():
    rng = random.Random(9301); H, seen = [], set()
    while len(H) < NH:
        h = L.random_hyp(rng); k = json.dumps(h, sort_keys=True)
        if k not in seen: seen.add(k); H.append(h)
    return H


def run(name):
    out = os.path.join(L.CK, 'c1_%s.jsonl' % name.replace('~', '_'))
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['i'] for l in open(out)}
    P = L.corpus(name); v = L.is_voy(name)
    ps = [(p, L.split_of(p['id'], v)) for p in P]
    with open(out, 'a') as f:
        for i, h in enumerate(hyps()):
            if i in done: continue
            r = L.score_hyp(ps, h)
            rec = dict(i=i, h=h, sel=r and r['score'][1], test=r and r['score'][2],
                       gR=r and r['R'][1], n=r and r['n_R'][1])
            f.write(json.dumps(rec) + '\n'); f.flush()
    return name


if __name__ == '__main__':
    t = time.time()
    with Pool(2) as pool:
        for n in pool.imap_unordered(run, CORPORA):
            print(n, 'done', round(time.time() - t), flush=True)
