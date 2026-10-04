"""LA-17 prior simulations for ABC (checkpointed in chunks).
usage: python3 la17_sim.py {la,lb} {si,dep} NCHUNKS CHUNK WORKER"""
import sys, os, time
from la17_common import *

ds, mode, nch, ch, wk = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
d = la_data() if ds == 'la' else lb_data()
M = 2000 if ds == 'la' else 10000
K = len(d['E']); tgt = d['inc'].sum()
for c in range(nch):
    fn = os.path.join(OUT, f'sims_{ds}_{mode}_w{wk}_{c:03d}.npz')
    if os.path.exists(fn):
        continue
    rng = np.random.default_rng(1000003 * wk + 7919 * c + (0 if ds == 'la' else 17) + (0 if mode == 'si' else 31))
    TH, SS = [], []
    t0 = time.time()
    for i in range(ch):
        p = draw_prior(K, rng, deposit=(mode == 'dep'))
        o = simulate(p, d['E'], d['D'], M, rng, tgt)
        TH.append(theta_vec(p, K)); SS.append(summaries(o))
    np.savez_compressed(fn + '.tmp.npz', theta=np.array(TH), stats=np.array(SS))
    os.replace(fn + '.tmp.npz', fn)
    print(ds, mode, wk, c, round(time.time() - t0, 1), flush=True)
