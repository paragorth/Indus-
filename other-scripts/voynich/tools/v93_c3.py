"""v93 cycle 3: 1,000 random dialogue (of 1,080 possible) (zig-zag) hypotheses x corpora; select on train+selection pages, test on test pages."""
import os, json, random, time
from multiprocessing import Pool
import v93_dial as D
import v93_lib as L
CORPORA = ['PE_DIAL', 'PE_SHUF', 'PE_MONO', 'CE_LA', 'PS_EN', 'ZL3b', 'IT2a', 'ZLshuf0', 'ZLshuf1', 'ITshuf0', 'SELFCIT', 'MK2', 'JUNC']
NH = 1000


def hyps():
    rng = random.Random(9303); H, seen = [], set()
    while len(H) < NH:
        h = D.random_hyp(rng); k = json.dumps(h, sort_keys=True)
        if k not in seen: seen.add(k); H.append(h)
    return H


def run(name):
    out = os.path.join(L.CK, 'c3_%s.jsonl' % name)
    if os.path.exists(out): os.remove(out)
    P = D.corpus(name); isv = L.is_voy(name)
    cache = {}
    with open(out, 'w') as f:
        for i, h in enumerate(hyps()):
            if h['unit'] not in cache: cache[h['unit']] = D.unit_seq(P, h['unit'], isv)
            seq = cache[h['unit']]
            X, voc = D.vectors(seq, h['rep'], tuple(h['band']), h['weight'])
            a = D.zigzag(seq, X, h['scope'], [0, 1]); b = D.zigzag(seq, X, h['scope'], [2])
            f.write(json.dumps(dict(i=i, h=h, sel=a and a['z'], selZ=a and a['Z'], test=b and b['z'], testZ=b and b['Z'])) + '\n')
    return name


if __name__ == '__main__':
    t = time.time()
    with Pool(2) as pool:
        for n in pool.imap_unordered(run, CORPORA): print(n, 'done', round(time.time() - t), flush=True)
