"""v67 cycle 3: learned line compressor (one latent class per line, mixture of multinomials over word
features) on Voynich, generators and plants; latent-class stream scored like the consensus streams.
Out: data/v67_ckpt/c3.jsonl"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v67_lib as X, v67_mix as M
from multiprocessing import Pool

C = None
def init():
    global C
    C = pickle.load(open(os.path.join(X.CK, 'c2_corpora.pkl'), 'rb'))

def work(a):
    n, kinds, K, seed = a
    c, tw, par = C[n]
    out = {'corpus': n, 'kinds': kinds, 'K': K, 'seed': seed}
    for lab, cor in (('real', c), ('twin', tw)):
        z = [str(x) for x in M.fit(M.matrix(cor, kinds), K, seed=seed)]
        s = {}
        for h in (0, 1):
            idx = [i for i in range(len(c)) if par[i] == h]
            r = X.score([z[i] for i in idx], [c[i]['folio'] for i in idx], nshuf=40, seed=seed)
            s['oe'[h]] = {k: round(v, 4) for k, v in r.items()}
        s['all'] = {k: round(v, 4) for k, v in X.score(z, [L['folio'] for L in c], nshuf=60, seed=seed).items()}
        if 'truth' in c[0]:
            s['nmix'] = X.nmi_ex(z, [L['truth'] for L in c])
        s['sec_nmi'] = X.nmi(z, [L['sec'] for L in c])
        out[lab] = s
    return out

if __name__ == '__main__':
    jobs = [(n, k, K, sd) for n in ['ZL', 'IT', 'LRGd', 'CPV', 'LatS', 'LatM', 'LatV', 'ItaM']
            for k in ('p', 's', 'ps', 'b', 'w') for K in (30, 100, 300) for sd in (1,)]
    fn = os.path.join(X.CK, 'c3.jsonl')
    done = set()
    if os.path.exists(fn):
        done = {(d['corpus'], d['kinds'], d['K'], d['seed']) for d in map(json.loads, open(fn))}
    jobs = [j for j in jobs if j not in done]
    with Pool(2, initializer=init) as p, open(fn, 'a') as f:
        for k, o in enumerate(p.imap_unordered(work, jobs)):
            f.write(json.dumps(o) + '\n'); f.flush(); print(k, o['corpus'], o['kinds'], o['K'], o['real']['all']['LANG'], flush=True)
    print('done')
