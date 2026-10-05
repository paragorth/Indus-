"""v51 worker: for mapping ids [a, b), render every corpus with the same random
phone mapping and score it. Writes jsonl rows to data/v51_ckpt/<tag>_<a>.jsonl.

usage: python3 v51_run.py TAG A B [corpus,corpus,...] [seconds] [clips]
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v51_lib as V

tag = sys.argv[1]
if sys.argv[2].startswith('list:'):
    MIDS = [int(x) for x in open(sys.argv[2][5:]).read().split()][int(sys.argv[3].split(':')[0]):int(sys.argv[3].split(':')[1])]
    a = sys.argv[3].split(':')[0]
else:
    a, b = int(sys.argv[2]), int(sys.argv[3]); MIDS = list(range(a, b))
names = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] != 'all' else None
secs = float(sys.argv[5]) if len(sys.argv) > 5 else 6.0
nclip = int(sys.argv[6]) if len(sys.argv) > 6 else 1

C0 = V.corpora()
if hasattr(V, 'extra_corpora'):
    C0.update(V.extra_corpora(C0))
if not names:
    names = list(C0)
# name@d / name@h = first / second half of the corpus (discovery / held-out); ranks from the full corpus
C, RI = {}, {}
for nm in names:
    base, _, half = nm.partition('@')
    w = C0[base]; h = len(w) // 2
    C[nm] = w[:h] if half == 'd' else w[h:] if half == 'h' else w
    RI[nm] = V.rank_index(w)
S = V.Scorer(threads=1)
out = os.path.join(V.CKPT, f'{tag}_{a}.jsonl')
done = set()
if os.path.exists(out):
    for line in open(out):
        r = json.loads(line); done.add(r['m'])
f = open(out, 'a')
t0 = time.time()
for m in MIDS:
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
        r, pl = S.score(xs[i:i + 8])
        if os.environ.get('V51_SAVE_LID'):
            for rr, p in zip(r, pl):
                rr['lid'] = [round(float(v), 5) for v in np.log(p + 1e-9)]
        res += r
    for (k, j), r in zip(meta, res):
        r.update(m=m, corpus=k, clip=j, pv=M['pv'], stress=M['pros']['stress'])
        f.write(json.dumps(r) + '\n')
    f.flush()
    print(m, round(time.time() - t0, 1), flush=True)
