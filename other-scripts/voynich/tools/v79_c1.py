"""v79 cycle 1: is the line-start chain a COUNTER, a CALENDAR column or a PERIODIC TAG?

(a) Massive random counter search. A hypothesis = a digit set D (3-10 of the chain's top-10 glyphs) in a random cyclic
    order, a step s, a counting unit (body lines, skipping lines whose glyph is not in D = sparse counter; or
    paragraph-first lines = entry numbers), a reset (paragraph, page, never). Score: share of consecutive D-items whose
    digit advances by s, z against the composition expectation. Best of N on discovery leaves -> held-out leaves.
(b) Periodic tag: held-out bits of P(glyph | index mod p) beyond positional drift P(glyph | min(index, 8)),
    index = line in paragraph / in page / in book, p = 2..30.
Nulls: within-paragraph shuffles, first-order Markov surrogates (same table), row-permuted Markov surrogates (same
sharpness, successor identity scrambled), five generators. Controls (real channels, opaque v72 surface, digit written on
70% of lines): Dante verse numbers (units digit), calendar dominical letters, calendar golden numbers (units digit),
Brumati running entry numbers on entry-first lines; negative: Isidore with the v72 surface's own anti-repeat marker.
"""
import sys, os, time, json, random
import numpy as np
import numba
import v79_lib as L

N_HYP = int(os.environ.get('V79_NHYP', '60000'))
FN = 'v79_cycle1.txt'


@numba.njit(cache=True)
def eval_h(g, seg, half, pos, k, s):
    """returns hits[2], pairs[2], f[2,k] (digit counts of the items entering pairs, as predecessors)."""
    hits = np.zeros(2); pairs = np.zeros(2); f = np.zeros((2, k)); f2 = np.zeros((2, k))
    prev = -1; pseg = -1
    for i in range(g.shape[0]):
        x = g[i]
        if x < 0: continue
        d = pos[x]
        if d < 0: continue
        if prev >= 0 and seg[i] == pseg:
            h = half[i]
            pairs[h] += 1; f[h, prev] += 1; f2[h, d] += 1
            if d == (prev + s) % k: hits[h] += 1
        prev = d; pseg = seg[i]
    return hits, pairs, f, f2


def zscore(hits, pairs, f, f2, k, s):
    if pairs < 20: return 0.0, 0.0
    p = 0.0
    a = f / f.sum(); b = f2 / f2.sum()
    for i in range(k): p += a[i] * b[(i + s) % k]
    E = pairs * p
    return (hits - E) / np.sqrt(E * (1 - p) + 1e-9), hits / pairs - p


def make_units(T):
    """sequences for the search: dict name -> (g, seg, half) arrays."""
    body = T['ps'] == 0
    U = {}
    g = T['g'].copy(); g[~body] = -1
    U['line/para'] = (g, T['para'], T['half'])
    U['line/page'] = (g, T['page'], T['half'])
    U['line/book'] = (g, np.zeros_like(T['page']), T['half'])
    # entry unit: paragraph-first lines, own alphabet coded in T['ge']
    ge = T['ge'].copy(); ge[body] = -1
    U['entry/page'] = (ge, T['page'], T['half'])
    U['entry/book'] = (ge, np.zeros_like(T['page']), T['half'])
    return U


def add_entry_alpha(T, k=8):
    from collections import Counter
    c = Counter(T['raw'][i] for i in range(len(T['raw'])) if T['ps'][i])
    al = [x for x, _ in c.most_common(k)]
    ix = {x: i for i, x in enumerate(al)}
    T['ge'] = np.array([ix.get(T['raw'][i], -1) if T['ps'][i] else -1 for i in range(len(T['raw']))])
    T['alpha_e'] = al
    return T


def search(T, nhyp=N_HYP, seed=0, keep=20):
    """random hypothesis search; returns per unit: best by discovery (half 0) and by half 1, with held-out z."""
    rng = np.random.default_rng(seed)
    U = make_units(T)
    A = int(T['g'].max()) + 1; Ae = int(T['ge'].max()) + 1
    res = {}
    for uname, (g, seg, half) in U.items():
        AA = Ae if uname.startswith('entry') else A
        n = nhyp // 4 if uname.startswith('entry') else nhyp
        H = []
        for h in range(n):
            k = int(rng.integers(3, min(10, AA) + 1))
            D = rng.permutation(AA)[:k]
            pos = -np.ones(AA, np.int64); pos[D] = np.arange(k)
            s = int(rng.integers(1, k))
            hits, pairs, f, f2 = eval_h(g, seg, half, pos, k, s)
            z0, d0 = zscore(hits[0], pairs[0], f[0], f2[0], k, s)
            z1, d1 = zscore(hits[1], pairs[1], f[1], f2[1], k, s)
            H.append((z0, z1, d0, d1, k, s, D.tolist()))
        out = {}
        for disc in (0, 1):
            Hs = sorted(H, key=lambda x: -x[disc])
            top = Hs[:keep]
            out['disc%d' % disc] = dict(best_disc_z=top[0][disc], best_held_z=top[0][1 - disc],
                                       best_held_d=top[0][3 - disc], top_mean_held_z=float(np.mean([t[1 - disc] for t in top])),
                                       best=(top[0][4], top[0][5], top[0][6]))
        out['held_z'] = 0.5 * (out['disc0']['best_held_z'] + out['disc1']['best_held_z'])
        out['top_held_z'] = 0.5 * (out['disc0']['top_mean_held_z'] + out['disc1']['top_mean_held_z'])
        out['held_d'] = 0.5 * (out['disc0']['best_held_d'] + out['disc1']['best_held_d'])
        res[uname] = out
    return res


# ------------------------------------------------------------------ periodic tag
def periodic(T, pmax=30):
    """held-out (2-fold by leaf half) bits/line of P(g | idx mod p) minus bits of P(g | min(idx, 8)); best p per index."""
    body = (T['ps'] == 0) & (T['g'] >= 0)
    A = int(T['g'].max()) + 1
    idx_para = np.zeros(len(T['g']), int); idx_page = T['li']; idx_book = np.arange(len(T['g']))
    c = 0
    for i in range(len(T['g'])):
        if i == 0 or T['para'][i] != T['para'][i - 1]: c = 0
        idx_para[i] = c; c += 1

    def ll(feat, nf):
        tot = 0.0; n = 0
        for h in (0, 1):
            tr = body & (T['half'] != h); te = body & (T['half'] == h)
            M = np.full((nf, A), 0.5)
            np.add.at(M, (feat[tr], T['g'][tr]), 1)
            M /= M.sum(1, keepdims=True)
            tot += np.log2(M[feat[te], T['g'][te]]).sum(); n += te.sum()
        return tot / n
    out = {}
    for name, idx in (('para', idx_para), ('page', idx_page), ('book', idx_book)):
        drift = ll(np.minimum(idx, 8), 9)
        base = ll(np.zeros_like(idx), 1)
        best = max(((ll(idx % p, p) - drift, p) for p in range(2, pmax + 1)), key=lambda x: x[0])
        both = max(((ll(np.minimum(idx, 8) * p + idx % p, 9 * p) - drift, p) for p in range(2, pmax + 1)), key=lambda x: x[0])
        out[name] = dict(drift_gain=drift - base, best_mod_minus_drift=best[0], best_p=best[1],
                         drift_plus_mod=both[0], best_p2=both[1])
    return out


# ------------------------------------------------------------------ corpora
def corpora():
    C = {}
    for nm in ('ZL3b', 'IT2a'):
        C[nm] = ('VOY', L.V77.voy(nm), None)
    D = L.dante_plain()[:170]
    C['C_DANTE'] = ('CTL', L.surface_channel(D, lambda l: l['ch'] % 10)[0], 'units digit of verse number, +1, canto reset')
    CA = L.calendar_plain(years=4)
    C['C_DOM'] = ('CTL', L.surface_channel(CA, lambda l: l['dom'])[0], 'dominical letter, +1 mod 7, never resets')
    C['C_GOLD'] = ('CTL', L.surface_channel(CA, lambda l: None if l['gold'] is None else l['gold'] % 10, p_write=0.8)[0],
                   'golden number units digit, +8 mod 19, blanks')
    B = L.brumati_entries_plain()
    C['C_ENTRY'] = ('CTL', L.surface_channel(B, lambda l: l['num'] % 10 if l['ps'] else None, p_write=0.9)[0],
                    'running entry number units digit on entry-first lines')
    I = L.V72.isidore_plain()[:200]
    C['N_ISI'] = ('NEG', L.V72.surface(L.V72.encode_payload(I, L.V72.payload_code([w for p in I for l in p['lines'] for w in l['w']])), seed=7), 'v72 anti-repeat marker, no channel')
    for gname in ('SELFCIT', 'SC10', 'MK2', 'JUNC', 'STACK'):
        C['G_' + gname] = ('GEN', L.V77.generate('ZL3b', gname, 791), None)
    return C


def main():
    t0 = time.time()
    C = corpora()
    R = {}
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    names = list(C)
    if part != 'all': names = names[int(part)::2]
    for name in names:
        kind, pages, note = C[name]
        T, al = L.chain_table(pages); T = add_entry_alpha(T)
        r = dict(kind=kind, note=note, alpha=al, alpha_e=T['alpha_e'], real=search(T, seed=1), per=periodic(T))
        nulls = {}
        rng = np.random.default_rng(5)
        nn = 6 if kind in ('VOY',) else 3
        for nk, fn in (('shuf', lambda T: L.shuffle_within_para(T, np.random.default_rng(int(rng.integers(1e9))))),
                       ('mk1', lambda T: L.markov_surrogate(T, np.random.default_rng(int(rng.integers(1e9))))),
                       ('rowperm', lambda T: L.markov_surrogate(T, np.random.default_rng(int(rng.integers(1e9))), rowperm=True))):
            vals = []
            for j in range(nn):
                Tn = fn(T); Tn['ge'] = T['ge'] if nk != 'shuf' else T['ge']
                if nk == 'shuf':   # also shuffle the entry sequence within page
                    ge = T['ge'].copy()
                    for pg in np.unique(T['page']):
                        ix = np.where((T['page'] == pg) & (T['ps'] == 1))[0]
                        if len(ix) > 1: ge[ix] = ge[rng.permutation(ix)]
                    Tn['ge'] = ge
                vals.append(search(Tn, nhyp=N_HYP // 2, seed=100 + j))
            nulls[nk] = vals
        r['nulls'] = nulls
        R[name] = r
        L.psave('c1_%s.pkl' % name, r)
        print(name, '%.0fs' % (time.time() - t0), {u: round(v['held_z'], 1) for u, v in r['real'].items()},
              {u: (round(v['best_mod_minus_drift'], 4), v['best_p']) for u, v in r['per'].items()}, flush=True)


if __name__ == '__main__':
    main()
