"""v51 worker: for mapping ids [a, b), render every corpus with the same random
phone mapping and score it. Writes jsonl rows to data/v51_ckpt/<tag>_<a>.jsonl.

usage: python3 v51_run.py TAG A B [corpus,corpus,...] [seconds] [clips]
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v51_lib as V

tag, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
names = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] != 'all' else None
secs = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
nclip = int(sys.argv[6]) if len(sys.argv) > 6 else 1

C = V.corpora()
C.update(V.extra_corpora()) if hasattr(V, 'extra_corpora') else None
if names:
    C = {k: C[k] for k in names}
RI = {k: V.rank_index(w) for k, w in C.items()}
S = V.Scorer(threads=1)
out = os.path.join(V.CKPT, f'{tag}_{a}.jsonl')
done = set()
if os.path.exists(out):
    for line in open(out):
        r = json.loads(line); done.add(r['m'])
f = open(out, 'a')
t0 = time.time()
for m in range(a, b):
    if m in done:
        continue
    M = V.random_mapping(np.random.default_rng(1000 + m))
    xs, meta = [], []
    for ci, (k, w) in enumerate(C.items()):
        for j in range(nclip):
            rng = np.random.default_rng([m, ci, j, 7])
            # synthesis noise seeded by mapping only, so every corpus gets the same voice
            srng = np.random.default_rng([m, j, 99])
            xs.append(V.render(V.clip_words(w, rng), M, RI[k], srng, seconds=secs)); meta.append((k, j))
    res = []
    for i in range(0, len(xs), 8):
        r, _ = S.score(xs[i:i + 8]); res += r
    for (k, j), r in zip(meta, res):
        r.update(m=m, corpus=k, clip=j, pv=M['pv'], stress=M['pros']['stress'])
        f.write(json.dumps(r) + '\n')
    f.flush()
    print(m, round(time.time() - t0, 1), flush=True)
