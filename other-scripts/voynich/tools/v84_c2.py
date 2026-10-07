"""v84 cycle 2: CAN A MEANINGLESS GENERATOR FAKE THE PAGE VOCABULARY?  The v82 line-mood + harmony generator extended
with page-level mechanisms over word bodies (v84_lib.gen_page: glyph page mood, lexical page mood with AR drift or
recurring discrete moods, per-page seed vocabulary with copy-and-vary, within-page or book-wide self-citation,
section drift via AR persistence). Random settings fitted to ZL3b, then climbed.

Scored on every setting: frozen PB and PB_far (v84_lib, 4 deals in the sweep), the 9 v82 surface statistics
(bands as in v82), and the frozen v81 onset link (3 definitions, 3 re-rolls).
PRE-REGISTERED KILL LINES (written before the sweep):
  size  : PB in [0.037, 0.102] (union of the ZL and IT2a bootstrap bands) AND every surface statistic in band AND
          each onset gain within [0.5x, 2x] of ZL, in 3 of 3 fresh seeds (10 deals);
  shape : size AND PB_far / PB >= 0.80 (Voynich 0.82-0.91; copying generators 0.37-0.71).
"""
import os, sys, json, random, copy, math, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v84_lib as K, v82_lib as K82
from multiprocessing import Pool

Z = I = B = None; MC = {}
VZ = np.array([0.0302, 0.0575, 0.0877])        # ZL frozen onset gains (v82 test, 3 re-rolls, this session)
PB_BAND = (0.037, 0.102)


def init():
    global Z, I, B
    Z = K.voynich('ZL3b'); I = K.voynich('IT2a')
    B = K82.bands(K82.surf_stats(Z), K82.surf_stats(I))


def score(args):
    P, seed, nnull, which = args
    src = Z if which == 'ZL' else I
    try:
        X = K.gen_page(src, P, seed, MC.setdefault(which, {}))
    except Exception as e:
        return dict(P=P, seed=seed, err=repr(e))
    r = K.PB(X, nnull=nnull, variants=('PB', 'PB_far'))
    s = K82.surf_stats(X); ok, bad = K82.match(s, B)
    fz = K82.frozen(K82.Mini(X), nrep=3)
    g = [x[0] for x in fz]
    on_ok = all(0.5 * v <= x <= 2 * v for x, v in zip(g, VZ))
    return dict(P=P, seed=seed, which=which, pb=r['PB'][0], pbz=r['PB'][1], far=r['PB_far'][0], stats=s, ok=ok,
                bad=bad, g=g, on_ok=on_ok)


def flags(r):
    size = r['ok'] and r['on_ok'] and PB_BAND[0] <= r['pb'] <= PB_BAND[1]
    shape = size and r['far'] / max(r['pb'], 1e-9) >= 0.80
    return size, shape


def fit(r):
    if 'err' in r: return -1e9
    d = sum(max(0.0, abs(r['stats'][k] - B[k][0]) / B[k][1] - 1) for k in K82.STAT_KEYS)
    pb = max(r['pb'], 1e-3); far = max(r['far'], 1e-3)
    dpb = abs(math.log(pb / 0.069))
    dfar = max(0.0, 0.85 - far / pb) * 5
    don = float(np.abs(np.log(np.clip(r['g'], 1e-3, None) / VZ)).sum())
    return -(dpb * 2 + dfar + 0.5 * len(r['bad']) + 0.2 * d + 0.3 * don)


def perturb(P, rng):
    Q = copy.deepcopy(P); R = K.sample_params(rng)
    for key in rng.sample([k for k in R if k != 'mech'], rng.randint(1, 3)): Q[key] = R[key]
    for key in ('mood_beta', 'agr_lam', 'p_cite', 'p_mod', 'mood_rate', 'gm_beta', 'lx_beta', 'sd_p', 'sd_mod', 'p_vert'):
        if rng.random() < 0.3 and Q.get(key, 0) > 0: Q[key] = Q[key] * math.exp(rng.gauss(0, 0.4))
    if Q['base'] != 'stack': Q['p_vert'] = 0.0
    if Q['base'] == 'junc': Q['p_cite'] = 0.0
    return Q


def jl(name, recs):
    with open(os.path.join(K.CK, name), 'a') as f:
        for r in recs: f.write(json.dumps(r, default=float) + '\n')


def load(name):
    p = os.path.join(K.CK, name)
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def mechs(P):
    m = []
    if P.get('p_cite', 0) > 0.02 or P.get('p_vert', 0) > 0.02: m.append('copy')
    if P.get('gm_beta', 0) > 0: m.append('gm')
    if P.get('lx_beta', 0) > 0: m.append('lx%s' % ('K' if P.get('lx_K', 0) else ''))
    if P.get('sd_n', 0) > 0: m.append('seed')
    return '+'.join(m) or 'none'


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'sweep':
        N = int(sys.argv[2]); rng = random.Random(8421)
        done = len(load('c2_sweep.jsonl'))
        jobs = [(K.sample_params(rng), 100 + i, 4, 'ZL') for i in range(N)][done:]
        with Pool(2, initializer=init) as Pl:
            buf = []
            for i, r in enumerate(Pl.imap_unordered(score, jobs, chunksize=2)):
                buf.append(r)
                if len(buf) >= 20: jl('c2_sweep.jsonl', buf); buf = []
                if i % 100 == 0: print(i + done, time.strftime('%H:%M:%S'), flush=True)
            jl('c2_sweep.jsonl', buf)
    elif stage == 'climb':
        R = int(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else 'all'
        import zlib; rng = random.Random(8422 + zlib.crc32(tag.encode()) % 1000); init()
        pop = [r for r in load('c2_sweep.jsonl') + load('c2_climb.jsonl') if 'err' not in r]
        if tag != 'all':   # climb inside one mechanism family (no within-page copying unless tag says so)
            pop = [r for r in pop if mechs(r['P']) == tag]
        with Pool(2, initializer=init) as Pl:
            for rd in range(R):
                pop.sort(key=fit, reverse=True); top = pop[:10]
                jobs = []
                for j in range(3):
                    for t in top:
                        Q = perturb(t['P'], rng)
                        if tag != 'all' and mechs(Q) != tag: Q = copy.deepcopy(t['P'])
                        jobs.append((Q, 6000 + rd * 100 + j * 10 + len(jobs), 4, 'ZL'))
                new = [r for r in Pl.imap_unordered(score, jobs) if 'err' not in r]
                for r in new: r['climb'] = tag
                jl('c2_climb.jsonl', new); pop += new
                b = max(pop, key=fit)
                print(tag, 'round', rd, 'best fit %.3f pb %.3f far %.3f bad %s on %s' % (fit(b), b['pb'], b['far'], b['bad'], b['on_ok']), flush=True)
    elif stage == 'final':
        init()
        pop = [r for r in load('c2_sweep.jsonl') + load('c2_climb.jsonl') if 'err' not in r]
        cand = []; seen = set()
        byf = {}
        for r in sorted(pop, key=fit, reverse=True):
            byf.setdefault(mechs(r['P']), []).append(r)
        for m, rs in byf.items():
            for r in rs[:1]:
                key = json.dumps(r['P'], sort_keys=True)
                if key not in seen: seen.add(key); cand.append(r)
        for r in sorted(pop, key=fit, reverse=True)[:6]:
            key = json.dumps(r['P'], sort_keys=True)
            if key not in seen: seen.add(key); cand.append(r)
        jobs = [(r['P'], s, 10, 'ZL') for r in cand for s in (9101, 9102, 9103)]
        with Pool(2, initializer=init) as Pl:
            out = list(Pl.imap(score, jobs))
        for r in out:
            if 'err' in r: continue
            X = K.gen_page(Z, r['P'], r['seed'], {})
            sw = K.PB(X, nnull=10, variants=('PB', 'PB_far'), swap=True)
            r['pb_swap'] = sw['PB'][0]; r['far_swap'] = sw['PB_far'][0]
        jl('c2_final.jsonl', out)
