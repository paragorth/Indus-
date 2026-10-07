"""v94 cycle 3: frozen held-out folio test.
For each chosen (target, source): the section's units are split into a TRAIN part (first 60%) and a HELD-OUT part
(last 40%). The search runs on TRAIN only and its best map defines a rate (source segments per unit). The prediction
for the held-out units is the continuation of that map (offset + span, same rate, same scheme/map/drop), plus 4
alternative continuations (rate x 0.8 / 1.25, offset gap +-10% of span) -> 5 frozen maps. All predictions are written
to data/v94_frozen.json and hashed (sha256) BEFORE any held-out scoring.
Test (once): best of the 5 frozen maps scored on ALL held-out pairs (Pearson of residual matrices) vs
  (a) 200 block-shuffled held-out orders under the same 5 maps (best-of-5 each), and
  (b) the same 5-map family placed at 200 random offsets of the same source (best-of-5 each).
Pass: real above 95% of both nulls (p < 0.05 each).
usage: python3 v94_heldout.py freeze   |   python3 v94_heldout.py test"""
import os, sys, json, hashlib, zlib, time
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ.setdefault('V89_LOWRANK', '2'); os.environ.setdefault('VOY_MODE', 'glyph')
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v94_lib as V
import v89_lib as L
import v94_run as R

FROZEN = os.path.join(V.VD, 'data', 'v94_frozen.json')
FROZEN_SHA = os.path.join(V.VD, 'data', 'v94_frozen.sha256')
OUT = os.path.join(V.CK, 'heldout.jsonl')
TRAIN_FRAC = 0.6


def pairs_list():
    return json.load(open(os.path.join(V.CK, 'heldout_pairs.json')))   # [[tset, target, src], ...]


def units_of(tset, tname):
    return R.targets(tset)[tname][0]


def freeze_one(args):
    tset, tname, sk = args
    U = units_of(tset, tname)
    ntr = int(round(len(U) * TRAIN_FRAC))
    tgt = V.Target(U[:ntr], n_null=1, seed=3)
    src = V.Src(V.load_text(sk))
    res = V.search(src, tgt, n_rand=300, top=4, refine=20, seed=zlib.crc32((tname + sk + 'ho').encode()), max_off=200)
    a = res[0]['a']
    nho = len(U) - ntr
    rate = a['span'] / ntr
    M = src.M[a['sch']]
    preds = []
    for f_rate, gap in ((1, 0), (0.8, 0), (1.25, 0), (1, 0.1), (1, -0.1)):
        sp_ = rate * f_rate * nho
        off = a['off'] + a['span'] + gap * a['span']
        if off + sp_ > M:   # running off the end: predict the preceding stretch instead (still unseen)
            off = max(0.0, a['off'] - sp_ - gap * a['span'])
        b = dict(a, off=float(off), span=float(min(sp_, M * 0.98)))
        preds.append(b)
    return {'tset': tset, 'target': tname, 'src': sk, 'n_train': ntr, 'n_ho': nho, 'train_A': res[0]['A'], 'train_B': res[0]['B'], 'train_map': a, 'pred': preds}


def score_full(src, U, a, perm=None):
    """Pearson over all pairs between held-out target residual (in order perm) and source pieces."""
    n = len(U)
    p = np.arange(n) if perm is None else perm
    X = V.sets_matrix([U[i] for i in p])
    Rt = V.resid_from_sets(X)
    cuts = V.make_cuts(src, a, np.array([len(U[i]) for i in p], float))
    Rs = V.resid_from_sets(src.pieces(cuts, a['drop']))
    I, J = np.triu_indices(n, 1)
    x, y = Rt[I, J], Rs[I, J]
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def test_one(fr):
    U = units_of(fr['tset'], fr['target'])[fr['n_train']:]
    src = V.Src(V.load_text(fr['src']))
    real = max(score_full(src, U, a) for a in fr['pred'])
    rng = np.random.default_rng(zlib.crc32((fr['target'] + fr['src']).encode()))
    n = len(U)
    nul_perm = []
    for k in range(200):
        b = max(3, n // 8)
        blocks = [np.arange(i, min(n, i + b)) for i in range(0, n, b)]
        blocks = [bl[::-1] if rng.random() < 0.5 else bl for bl in blocks]
        p = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
        nul_perm.append(max(score_full(src, U, a, p) for a in fr['pred']))
    nul_off = []
    M = src.M[fr['pred'][0]['sch']]
    for k in range(200):
        sh = float(rng.uniform(0, M))
        vals = []
        for a in fr['pred']:
            o = (a['off'] + sh) % max(1.0, M - a['span'])
            vals.append(score_full(src, U, dict(a, off=o)))
        nul_off.append(max(vals))
    nul_perm = np.array(nul_perm); nul_off = np.array(nul_off)
    return {'target': fr['target'], 'src': fr['src'], 'tset': fr['tset'], 'real': real,
            'p_perm': float((1 + (nul_perm >= real).sum()) / 201), 'p_off': float((1 + (nul_off >= real).sum()) / 201),
            'perm_mu': float(nul_perm.mean()), 'off_mu': float(nul_off.mean()), 'n_ho': n}


if __name__ == '__main__':
    if sys.argv[1] == 'freeze':
        jobs = [tuple(x) for x in pairs_list()]
        with Pool(2) as p:
            frs = p.map(freeze_one, jobs, chunksize=1)
        blob = json.dumps(frs, sort_keys=True, indent=1)
        open(FROZEN, 'w').write(blob)
        h = hashlib.sha256(blob.encode()).hexdigest()
        open(FROZEN_SHA, 'w').write(h + '  v94_frozen.json\n')
        print('frozen', len(frs), 'sha256', h)
    else:
        blob = open(FROZEN).read()
        h = hashlib.sha256(blob.encode()).hexdigest()
        assert open(FROZEN_SHA).read().split()[0] == h, 'frozen file changed'
        frs = json.loads(blob)
        with Pool(2) as p, open(OUT, 'w') as f:
            for r in p.imap_unordered(test_one, frs):
                r['sha256'] = h
                f.write(json.dumps(r) + '\n'); f.flush()
                print('%-26s %-40s real %.3f p_perm %.3f p_off %.3f (perm mu %.3f off mu %.3f)' % (r['target'][:26], r['src'][:40], r['real'], r['p_perm'], r['p_off'], r['perm_mu'], r['off_mu']), flush=True)
