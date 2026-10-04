"""v26 cycle 4: CROSS-GENOME control for the margin (AUC on real minus the genome's own re-forge AUC).
A genome that compounds its own effects when refitted has a high re-forge floor, which shrinks the margin by
itself. Apply each corpus's winner (and the planted genome) to the OTHER corpus: if the Latin herbal forged by
the Voynich winner also lands near its floor, the small Voynich margin is a property of the genome, not of the
Voynich. Usage: python3 v26_cross.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v26_lib import *
from v26_evo import PLANT
import v26_eval as E
from multiprocessing import Pool


def main():
    t0 = time.time()
    WV = load('eval_V.json')['winner']; WL = load('eval_LA.json')['winner']
    out = load('cross.json') or {}
    pairs = [('LA', 'Vwin', WV), ('V', 'LAwin', WL), ('LA', 'plant', PLANT), ('V', 'plant', PLANT)]
    for cn, lab, g in pairs:
        key = f'{cn}|{lab}'
        if key in out: continue
        pool = Pool(2, initializer=E.winit, initargs=(cn,))
        rs = pool.map(E.j_train, [(g, s, True) for s in range(5)])
        ne = pool.map(E.j_neg, [(g, s) for s in range(3)])
        pool.close()
        out[key] = dict(genome=gstr(g), ridge=[r[0] for r in rs], gbm=[r[1] for r in rs], neg=ne)
        save('cross.json', out)
        print(key, gstr(g), 'auc', round(np.mean([r[0] for r in rs]), 3), round(np.mean([r[1] for r in rs]), 3),
              'neg', round(np.mean([x[0] for x in ne]), 3), round(np.mean([x[1] for x in ne]), 3), f'[{time.time() - t0:.0f}s]', flush=True)


if __name__ == '__main__':
    main()
