import sys, os, json, pickle, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v91_lib as V, v91_excess as E
mode = sys.argv[1]
if mode == 'heldout':
    fr = json.load(open(os.path.join(V.ROOT, 'data', 'v91_frozen.json')))
    h = fr.pop('sha256'); assert V.sha(fr) == h, 'frozen file changed'
    out = {}
    for nm in sys.argv[2].split(','):
        L, _ = V.build(nm); C = V.Corpus(L, type_salts=V.TYPE_SALTS2)
        res = []
        for p, f, k, o in fr['masks']:
            r = V.score_mask(C, tuple(p.split('+')), tuple(f.split('+')), k, o, mode='final')
            res.append(None if r is None else round(r['u'] + r['b'], 2))
        out[nm] = res; print(nm, res, flush=True)
    pickle.dump(out, open(os.path.join(V.CK, 'c3_heldout_%s.pkl' % sys.argv[2].replace(',', '_')), 'wb'))
elif mode == 'read':
    # readability sweep: top-N discovery masks with a 16-64 symbol super-alphabet in corpus `src`, decoded in corpus `nm`
    src, nm, N = sys.argv[2], sys.argv[3], int(sys.argv[4])
    X, _ = E.load(src)
    cand = []
    for (p, f, k, o), s in X.items():
        A = 1
        for ff in f.split('+'): A *= V.FA[ff]
        if 16 <= A ** k <= 64: cand.append((s, p, f, k, o))
    cand.sort(reverse=True)
    seen = set(); sel = []
    for s, p, f, k, o in cand:
        key = ('+'.join(sorted(x for x in p.split('+') if x != 'all')) or 'all', f, k, o)
        if key in seen: continue
        seen.add(key); sel.append((p, f, k, o))
        if len(sel) >= N: break
    L, meta = V.build(nm); C = V.Corpus(L, type_salts=V.TYPE_SALTS2)
    LAT = V.bigram_model(V.letters('Latin-Caesar', 60000, skip=0.7))
    ITA = V.bigram_model(V.letters('Italian-Manzoni', 60000, skip=0.7))
    rng = np.random.default_rng(5)
    res = []
    for p, f, k, o in sel:
        seq = V.decode_stream(C, tuple(p.split('+')), tuple(f.split('+')), k, o)
        if len(seq) < 100: continue
        sh = seq.copy(); rng.shuffle(sh)
        row = [p, f, k, o, len(seq)]
        for M in (LAT, ITA):
            a, txt = V.solve_sub(seq, M, iters=2500, restarts=2)
            b, _ = V.solve_sub(sh, M, iters=2500, restarts=2)
            row += [round(a - b, 3), txt[:40]]
        res.append(row)
    pickle.dump(res, open(os.path.join(V.CK, 'c3_read_%s_%s.pkl' % (src, nm)), 'wb'))
    g = np.array([max(r[5], r[7]) for r in res])
    print(src, nm, len(res), 'gap max %.3f median %.3f n>0.2 %d' % (g.max(), np.median(g), (g > 0.2).sum()), flush=True)
