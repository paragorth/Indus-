"""v67 cycle 2: 1,000 random consensus rules x 9 corpora (+ page-word-shuffled twin of each).
Per rule and corpus: line agreement excess over the twin and LANG, separately on odd and even folios.
Selection on odd folios, test on even folios. Out: data/v67_ckpt/c2.jsonl"""
import sys, os, json, random, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v67_lib as X
from multiprocessing import Pool

C = None; ALPH = None
def build():
    V = X.voynich(); I = X.voynich('IT2a')
    C = {'ZL': V, 'IT': I, 'LRG': X.lrg(V, seed=22, beta=0.3), 'LRGd': X.lrg(V, seed=23, beta=0.3, drift=0.3),
         'CPV': X.cpv(V, seed=24), 'LatS': X.plant(V, 'Latin-Caesar', 'mid', 31, 0.25),
         'LatM': X.plant(V, 'Latin-Caesar', 'mid', 32, 0.6), 'LatV': X.plant(V, 'Latin-Caesar', 'mid', 33, 0.85),
         'ItaM': X.plant(V, 'Italian-Manzoni', 'mid', 35, 0.6)}
    out = {}
    for n, c in C.items():
        fol = []; [fol.append(L['folio']) for L in c if L['folio'] not in fol]
        par = {f: i % 2 for i, f in enumerate(fol)}
        out[n] = (c, X.within_shuffle_words(c, 99), [par[L['folio']] for L in c])
    return out

def init():
    global C, ALPH
    C = pickle.load(open(os.path.join(X.CK, 'c2_corpora.pkl'), 'rb'))
    ALPH = X.alphabet(C['ZL'][0])

def work(a):
    ri, r = a
    res = {}
    for n, (c, tw, par) in C.items():
        tk, ag = X.line_tokens(c, r, ALPH)
        _, ag2 = X.line_tokens(tw, r, ALPH)
        o = {}
        for h in (0, 1):
            idx = [i for i in range(len(c)) if par[i] == h]
            ae = sum(ag[i] - ag2[i] for i in idx) / len(idx)
            s = X.score([tk[i] for i in idx], [c[i]['folio'] for i in idx], nshuf=20, seed=ri)
            s['agx'] = ae
            if 'truth' in c[0]:
                s['nmix'] = X.nmi_ex([tk[i] for i in idx], [c[i]['truth'] for i in idx])
            o['oe'[h]] = {k: round(v, 4) for k, v in s.items()}
        res[n] = o
    return {'i': ri, 'rule': X.rule_name(r), 'r': r, 'res': res}

if __name__ == '__main__':
    if not os.path.exists(os.path.join(X.CK, 'c2_corpora.pkl')):
        pickle.dump(build(), open(os.path.join(X.CK, 'c2_corpora.pkl'), 'wb'))
    C0 = pickle.load(open(os.path.join(X.CK, 'c2_corpora.pkl'), 'rb'))
    rng = random.Random(6702); alph = X.alphabet(C0['ZL'][0])
    rules = [X.random_rule(rng, alph) for _ in range(1000)]
    fn = os.path.join(X.CK, 'c2.jsonl')
    done = set()
    if os.path.exists(fn):
        done = {json.loads(l)['i'] for l in open(fn)}
    todo = [(i, r) for i, r in enumerate(rules) if i not in done]
    with Pool(2, initializer=init) as p, open(fn, 'a') as f:
        for k, o in enumerate(p.imap_unordered(work, todo, chunksize=2)):
            f.write(json.dumps(o) + '\n'); f.flush()
            if k % 50 == 0: print(k, flush=True)
    print('done')
