import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v91_lib as V
# language models from texts NOT used for plants
LAT = V.bigram_model(V.letters('Latin-Caesar', 60000, skip=0.7))
ITA = V.bigram_model(V.letters('Italian-Manzoni', 60000, skip=0.7))
def run(nm, preds, fset, k, o, msg=None):
    L, meta = V.build(nm)
    C = V.Corpus(L, type_salts=V.TYPE_SALTS2)
    seq = V.decode_stream(C, preds, fset, k, o)
    out = {}
    for lang, M in [('lat', LAT), ('ita', ITA)]:
        ll, txt = V.solve_sub(seq, M, iters=5000, restarts=4)
        # null: same stream shuffled (symbol frequencies kept, order destroyed)
        rng = np.random.default_rng(0); sh = seq.copy(); rng.shuffle(sh)
        lln, _ = V.solve_sub(sh, M, iters=5000, restarts=4)
        acc = V.letter_acc(txt, meta['msg']) if meta.get('msg') else None
        out[lang] = (round(ll, 3), round(lln, 3), acc, txt[:70])
    return len(seq), out
if __name__ == '__main__':
    for spec in sys.argv[1:]:
        nm, pr, fs, k, o = spec.split(':')
        print(nm, pr, fs, k, o, run(nm, tuple(pr.split('+')), tuple(fs.split('+')), int(k), int(o)), flush=True)
