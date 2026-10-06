"""v72 cycle 1: is the message layer needed? (discovery half only: leaves with even number)

For each corpus x extraction rule: payload stream -> 3-fold page cross-validation inside the discovery half ->
held-out bits per payload token under NOISE / GEN / MSG / MSGSH (see v72_lib.ladder), plus the surface-layer
cost (bits of the written word given its payload, with and without context).
Corpora: Voynich ZL3b and IT2a; four generators fitted to the ZL surface (WSHUF, MK2, SELFCIT, JUNC);
planted controls = Brumati (Italian herbal), Isidore (Latin), German (Alemannic) through a payload code
(verbose 1-2 symbols, or lossy merge to 11 symbols) and the planted surface machinery; the same four
generators fitted to each planted surface (they must lose the message).
Out: data/v72_ckpt/c1/<corpus>__<rule>.json
"""
import os, sys, json, time, random
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v72_lib as L

OUT = os.path.join(L.CK, 'c1'); os.makedirs(OUT, exist_ok=True)
_CACHE = {}


def corpus(name):
    """surface pages of the discovery half (and, for plants, the plain pages)."""
    if name in _CACHE: return _CACHE[name]
    base, gen = name.split('~') if '~' in name else (name, None)
    plain = None
    if base in ('ZL', 'IT'):
        P = L.voynich('ZL3b' if base == 'ZL' else 'IT2a')
        S = [p for p in P if L.leaf_half(p['id']) == 0]
    else:
        txt, mode = base.split('-')
        P = dict(BRU=L.brumati_plain, ISI=L.isidore_plain, DEU=L.german_plain)[txt]()
        P = [p for i, p in enumerate(P) if i % 2 == 0]
        code = L.payload_code([w for p in P for l in p['lines'] for w in l['w']], mode=mode)
        S = L.surface(L.encode_payload(P, code)); plain = P
    if gen:
        S = L.GENS[gen](S, seed=1); plain = None
    _CACHE[name] = (S, plain)
    return S, plain


def run(job):
    name, rule = job
    fn = os.path.join(OUT, f'{name}__{rule}.json')
    if os.path.exists(fn): return job
    t0 = time.time()
    S, plain = corpus(name)
    X = L.extract(S, L.RULES[rule])
    rng = random.Random(3); idx = list(range(len(X))); rng.shuffle(idx)
    agg = {}; folds = 3; per = []
    sb = dict(surf_bits=0.0, surf_bits_ctx=0.0, n=0)
    for f in range(folds):
        te_i = set(idx[f::folds])
        tr = [X[i] for i in range(len(X)) if i not in te_i]; te = [X[i] for i in sorted(te_i)]
        r = L.ladder(tr, te, W=20, seed=f); r.pop('per_tok')
        per.append(r)
        s = L.surface_bits([S[i] for i in range(len(X)) if i not in te_i], tr, [S[i] for i in sorted(te_i)], te)
        for k in ('surf_bits', 'surf_bits_ctx'): sb[k] += s[k] * s['n']
        sb['n'] += s['n']
    n = sum(r['NOISE']['n'] for r in per)
    out = dict(name=name, rule=rule, n=n, sec=0)
    for k in ['NOISE', 'GEN', 'MSG', 'GEN+PAGE', 'GEN+BIG', 'MSGSH']:
        out[k] = sum(r[k]['bits'] * r[k]['n'] for r in per) / n
    for k in ['gain_msg', 'gain_gen', 'gain_page', 'gain_big', 'gain_msgsh']:
        out[k] = sum(r[k] * r['NOISE']['n'] for r in per) / n
    out['gain_msg_z'] = [r['gain_msg_z'] for r in per]
    out['excess'] = out['gain_msg'] - out['gain_msgsh']
    out['surf_bits'] = sb['surf_bits'] / sb['n']; out['surf_bits_ctx'] = sb['surf_bits_ctx'] / sb['n']
    toks = [w for p in X for l in p['lines'] for w in l['w']]
    out['types'] = len(set(toks)); out['ntok'] = len(toks)
    out['mean_len'] = sum(map(len, toks)) / len(toks)
    if plain is not None: out['recovery'] = L.recovery(X, plain)
    out['lam_MSG'] = per[0]['MSG']['lam']
    out['sec'] = time.time() - t0
    json.dump(out, open(fn, 'w'), default=float)
    print(name, rule, round(out['gain_gen'], 3), round(out['gain_msg'], 3), round(out['gain_msgsh'], 3),
          round(out['excess'], 3), round(out['sec']), flush=True)
    return job


def jobs():
    J = []
    vfam = ['ZL', 'IT'] + ['ZL~' + g for g in L.GENS]
    for rule in L.RULES:
        for c in vfam: J.append((c, rule))
    for txt in ['BRU', 'ISI', 'DEU']:
        for mode in ['merge', 'verbose']:
            base = f'{txt}-{mode}'
            for rule in ['E2_line', 'E0_identity', 'E1_padding']:
                J.append((base, rule))
                if mode == 'merge' or rule == 'E2_line':
                    for g in L.GENS: J.append((base + '~' + g, rule))
    return J


if __name__ == '__main__':
    J = jobs(); print(len(J), 'jobs', flush=True)
    with Pool(int(os.environ.get('W', '2'))) as pool:
        for _ in pool.imap_unordered(run, J, chunksize=1): pass
