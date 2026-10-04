"""v49 cycle 1: do the off-table words behave like subject terms?

For each corpus the off-table tokens (outside every choice the v45 rule model allows) are split by line role:
OFFpf (paragraph-first line), OFFb (body). Tests, each against a frequency-matched baseline drawn from the same corpus:
  M1 local recurrence: share of the type's other occurrences that fall on the same page (and paragraph), minus the
     mean share for all tokens whose type has the same corpus count (count 2..30). Subject terms recur in their entry.
  M2 label-tied recurrence: for occurrences on other pages, share of page pairs with the same illustration type
     (Voynich: same illus and same Currier language; controls: same section), minus the frequency-matched share;
     also restricted to pairs in different quires (Voynich) to strip neighbourhood.
  M3 position: off-table share at word positions 0-2 vs 3+, paragraph-first vs body lines.
  M4 internal structure: per-glyph surprisal (E3-normalised, glyph trigram with add-0.1, trained on in-table body
     tokens of the other half of pages) of off-table TYPES, regressed on length; OFFpf minus OFFb.
  BR*: share of Latin-binomial tokens off-table vs not; Latin vs Italian off-table structure (ground truth).
Bootstrap over pages (300x) for z. Other occurrences are counted only on pages of the SAME cross-fit fold as the
token (off-table status is defined by the other fold; counting across folds biases M1/M2 against off-table words).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v49_lib import *

FN = 'v49_cycle1.txt'
NB = 300


def quire_of(name, C):
    if name.split('_')[0] in ('V', 'VI'):
        m = meta(); return {p['id']: m.get(p['id'], {}).get('quire') for p in C}
    return {p['id']: p['id'] for p in C}


def trigram(train):
    c3 = Counter(); c2 = Counter(); al = set()
    for w in train:
        s = '^^' + w + '$'
        for i in range(2, len(s)):
            c3[s[i - 2:i + 1]] += 1; c2[s[i - 2:i]] += 1; al.add(s[i])
    V = len(al) + 1
    def surpr(w):
        s = '^^' + w + '$'
        return float(np.mean([-math.log2((c3[s[i - 2:i + 1]] + 0.1) / (c2[s[i - 2:i]] + 0.1 * V)) for i in range(2, len(s))]))
    return surpr


def analyse(name, T, C, rng, pages_subset=None):
    if pages_subset is not None:
        T = [t for t in T if t['page'] in pages_subset]
    lab = illus_of(name, C); qu = quire_of(name, C)
    from v26_lib import split_half
    A, _ = split_half(C, 45); fold = {p['id']: 0 for p in C}; fold.update({p['id']: 1 for p in A})
    cnt = Counter(t['w'] for t in T)
    occ = defaultdict(list)
    for i, t in enumerate(T): occ[t['w']].append(i)
    # per-token local and label features
    loc = np.full(len(T), np.nan); same = np.full(len(T), np.nan); sameq = np.full(len(T), np.nan)
    for w, ix in occ.items():
        c = len(ix)
        if c < 2 or c > 30: continue
        pg = [T[i]['page'] for i in ix]
        for a, i in enumerate(ix):
            others = [pg[b] for b in range(c) if b != a and fold[pg[b]] == fold[pg[a]]]
            if not others: continue
            loc[i] = np.mean([o == pg[a] for o in others])
            far = [o for o in others if o != pg[a]]
            if far:
                same[i] = np.mean([lab[o] == lab[pg[a]] for o in far])
                fq = [o for o in far if qu[o] != qu[pg[a]]]
                if fq: sameq[i] = np.mean([lab[o] == lab[pg[a]] for o in fq])
    cs = np.array([cnt[t['w']] for t in T])
    def base(arr):
        b = {}
        for c in set(cs.tolist()):
            m = (cs == c) & ~np.isnan(arr)
            if m.any(): b[c] = float(arr[m].mean())
        return np.array([b.get(c, np.nan) for c in cs])
    exc = {k: arr - base(arr) for k, arr in (('loc', loc), ('same', same), ('sameq', sameq))}
    groups = {
        'OFFpf': np.array([t['off'] and t['role'] == 'pf' for t in T]),
        'OFFpf1': np.array([t['off'] and t['role'] == 'pf' and not (t['k'] == 0) for t in T]),
        'OFFb': np.array([t['off'] and t['role'] == 'body' for t in T]),
        'INpf': np.array([(not t['off']) and t['role'] == 'pf' for t in T]),
    }
    if T[0]['lat'] is not None:
        groups['LAT'] = np.array([bool(t['lat']) for t in T])
        groups['LAToff'] = np.array([bool(t['lat']) and t['off'] for t in T])
    pages = sorted(set(t['page'] for t in T)); pidx = {p: i for i, p in enumerate(pages)}
    tp = np.array([pidx[t['page']] for t in T])
    res = {}
    pf = np.array([t['role'] == 'pf' for t in T]); off = np.array([t['off'] for t in T])
    res['off_pf'] = float(off[pf].mean()); res['off_body'] = float(off[~pf].mean())
    # bootstrap over pages
    B = [rng.choice(len(pages), len(pages)) for _ in range(NB)]
    bypage = [np.where(tp == i)[0] for i in range(len(pages))]
    for g, m in groups.items():
        r = dict(n=int(m.sum()))
        for k, e in exc.items():
            v = e[m]; ok = ~np.isnan(v)
            r[k] = float(v[ok].mean()) if ok.any() else float('nan'); r[k + '_n'] = int(ok.sum())
        r['single'] = float(np.mean(cs[m] == 1)) if m.any() else float('nan')
        res[g] = r
    # contrasts with bootstrap z: OFFpf - OFFb for loc/same; OFFpf vs 0
    for k, e in exc.items():
        for g in ('OFFpf', 'OFFpf1', 'OFFb'):
            vals = []
            for b in B:
                ix = np.concatenate([bypage[i] for i in b])
                v = e[ix][groups[g][ix]]; v = v[~np.isnan(v)]
                vals.append(v.mean() if len(v) else np.nan)
            vals = np.array(vals); sd = np.nanstd(vals)
            res[g][k + '_z'] = float(res[g][k] / sd) if sd > 0 else float('nan')
    # M3 position
    pos = {}
    for role in ('pf', 'body'):
        for kk in range(5):
            m = np.array([t['role'] == role and (t['k'] == kk if kk < 4 else t['k'] >= 4) for t in T])
            pos[f'{role}{kk}'] = float(off[m].mean()) if m.any() else float('nan')
    res['pos'] = pos
    # M4 internal structure (types), trained on in-table body tokens of the other half of pages
    half = set(pages[::2])
    sc = {}
    for part in (0, 1):
        tr = [norm(t['w']) for t in T if (t['page'] in half) == bool(part) and not t['off'] and t['role'] == 'body']
        f = trigram(tr)
        for t in T:
            if (t['page'] in half) != bool(part) and t['off']:
                sc.setdefault((t['w'], t['role']), f(norm(t['w'])))
    keys = list(sc); y = np.array([sc[k] for k in keys]); ln = np.array([len(norm(k[0])) for k in keys], float)
    X = np.c_[np.ones(len(ln)), ln, ln ** 2]; beta = np.linalg.lstsq(X, y, rcond=None)[0]; resid = y - X @ beta
    rp = resid[[k[1] == 'pf' for k in keys]]; rb = resid[[k[1] == 'body' for k in keys]]
    d = rp.mean() - rb.mean(); se = math.sqrt(rp.var() / len(rp) + rb.var() / len(rb))
    res['M4'] = dict(pf=float(y[[k[1] == 'pf' for k in keys]].mean()), body=float(y[[k[1] == 'body' for k in keys]].mean()),
                     d=float(d), z=float(d / se), npf=len(rp), nb=len(rb))
    if T[0]['lat'] is not None:
        lat = np.array([bool(t['lat']) for t in T])
        res['lat'] = dict(p_off_lat=float(off[lat].mean()), p_off_it=float(off[~lat].mean()),
                          lat_share_OFFpf=float(lat[off & pf].mean()), lat_share_OFFb=float(lat[off & ~pf].mean()),
                          lat_share_all=float(lat.mean()))
        lt = {t['w']: bool(t['lat']) for t in T}
        rl = np.array([r for k, r in zip(keys, resid) if lt.get(k[0])]); ri = np.array([r for k, r in zip(keys, resid) if not lt.get(k[0])])
        res['lat']['M4_lat_minus_it'] = float(rl.mean() - ri.mean())
        res['lat']['M4_z'] = float((rl.mean() - ri.mean()) / math.sqrt(rl.var() / len(rl) + ri.var() / len(ri)))
    return res


if __name__ == '__main__':
    names = sys.argv[1:] or ['V', 'VI', 'GEN0', 'GEN1', 'PL', 'GENN', 'GENT', 'BRe', 'BRf']
    out = jload('c1.json') or {}
    for nm in names:
        rng = np.random.default_rng(491)
        C = get_corpus(nm); T = tokens_table(nm)
        out[nm] = analyse(nm, T, C, rng)
        if nm in ('V', 'VI'):
            H = set(p['id'] for p in C if (illus_of(nm, C)[p['id']][0] == 'H'))
            out[nm + '_H'] = analyse(nm, T, C, rng, H)
        jsave('c1.json', out)
        r = out[nm]
        print(nm, json.dumps({k: r[k] for k in ('off_pf', 'off_body', 'OFFpf', 'OFFb', 'M4')}, default=float), flush=True)
