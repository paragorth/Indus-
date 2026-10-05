"""v67 cycle 1: calibration. Corpora (Voynich, nulls, planted line-per-word encodings, true-word ceiling) x
hand rules + 400 random consensus rules; line-token stream scored by MI / ASYM / REC z vs within-page line shuffle.
Out: data/v67_ckpt/c1.json"""
import sys, os, json, random, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v67_lib as X
from multiprocessing import Pool

def corpora():
    V = X.voynich(); I = X.voynich('IT2a')
    C = {'ZL': V, 'IT': I, 'ZLshufw': X.within_shuffle_words(V, 3),
         'LRG': X.lrg(V, seed=2, beta=0.3), 'LRGd': X.lrg(V, seed=3, beta=0.3, drift=0.3), 'CPV': X.cpv(V, seed=4),
         'LatS': X.plant(V, 'Latin-Caesar', 'mid', 11, 0.25), 'LatM': X.plant(V, 'Latin-Caesar', 'mid', 12, 0.6),
         'LatV': X.plant(V, 'Latin-Caesar', 'mid', 13, 0.85), 'LatH': X.plant(V, 'Latin-Caesar', 'hard', 14),
         'ItaS': X.plant(V, 'Italian-Manzoni', 'mid', 15, 0.25), 'ItaV': X.plant(V, 'Italian-Manzoni', 'mid', 16, 0.85)}
    return C

HAND = [{'red': ('id',), 'ex': 'pre', 'k': k, 'agg': 'mode', 'tie': 'rare'} for k in (1, 2, 3)] + \
       [{'red': ('id',), 'ex': 'bag', 'k': 3, 'agg': 'mode', 'tie': 'rare'},
        {'red': ('id',), 'ex': 'whole', 'k': 0, 'agg': 'mode', 'tie': 'rare'},
        {'red': ('id',), 'ex': 'suf', 'k': 2, 'agg': 'mode', 'tie': 'rare'}]

C = None; ALPH = None
def init():
    global C, ALPH
    C = pickle.load(open(os.path.join(X.CK, 'c1_corpora.pkl'), 'rb'))
    ALPH = X.alphabet(C['ZL'])

def work(a):
    ri, r = a
    out = {}
    for name, cor in C.items():
        toks, agree = X.line_tokens(cor, r, ALPH)
        s = X.score(toks, [L['folio'] for L in cor], nshuf=20, seed=ri)
        s['agree'] = sum(agree) / len(agree)
        if 'truth' in cor[0]:
            tr = [L['truth'] for L in cor]
            s['nmi'] = X.nmi(toks, tr); s['pur'] = X.purity(toks, tr)
        out[name] = s
    return {'i': ri, 'rule': X.rule_name(r), 'r': r, 'res': out}

if __name__ == '__main__':
    C0 = corpora()
    pickle.dump(C0, open(os.path.join(X.CK, 'c1_corpora.pkl'), 'wb'))
    # ceiling: the true word stream itself
    ceil = {}
    for n, cor in C0.items():
        if 'truth' in cor[0]:
            tr = [L['truth'] for L in cor]
            ceil[n] = X.score(tr, [L['folio'] for L in cor], nshuf=100)
    print('ceiling', json.dumps(ceil, indent=0), flush=True)
    rng = random.Random(67)
    alph = X.alphabet(C0['ZL'])
    rules = HAND + [X.random_rule(rng, alph) for _ in range(150)]
    res = []
    with Pool(2, initializer=init) as p:
        for k, o in enumerate(p.imap_unordered(work, list(enumerate(rules)), chunksize=4)):
            res.append(o)
            if k % 50 == 0: print(k, flush=True)
    json.dump({'ceiling': ceil, 'rules': res}, open(os.path.join(X.CK, 'c1.json'), 'w'))
    print('done')
