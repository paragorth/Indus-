"""v76 cycle 2: inside the near-copy groups, align paragraphs and type the slots.
Pairs: cross-page same-section paragraph pairs that share the signature of one of the corpus's top-K templates
(chosen on discovery-half pairs, K=50), evaluated on all pairs and on holdout-half pairs (H1) separately.
Each pair is aligned token by token (global alignment, match = identical extracted token). A SLOT is an aligned
substitution with matching neighbours on both sides (anchored single-token swap); FIXED = matched columns.
Statistics per corpus (pairs pooled):
  FIX   share of aligned columns that match
  SLOTR slots per 100 aligned columns
  POSH  entropy (bits) of slot positions over 6 relative-position bins x {line-initial, medial, line-final},
        minus the same entropy for all aligned columns (negative = slots concentrated in a fixed place)
  PS    page-specificity of slot fillers: filler a (page X) occurs elsewhere on X and not on the partner page Y,
        and b likewise (a swap of page-own items), minus the frequency-matched expectation from random tokens of
        the same two paragraphs (same frequency decile)  -> the IDENTIFIER test
  PSF   the same for FIXED tokens (should be ~0 or negative)
Outputs data/v76_ckpt/c2_<corpus>.json."""
import os, sys, json, gzip, random, math, glob
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v76_lib as V
import v76_c1 as C1
K = 50


def align(a, b):
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), np.int32); D[:, 0] = -np.arange(n + 1); D[0, :] = -np.arange(m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]; row = D[i]; prev = D[i - 1]
        for j in range(1, m + 1):
            s = prev[j - 1] + (2 if ai == b[j - 1] else -1)
            u = prev[j] - 1; l = row[j - 1] - 1
            row[j] = s if s >= u and s >= l else (u if u >= l else l)
    i, j, cols = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (2 if a[i - 1] == b[j - 1] else -1):
            cols.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] - 1:
            cols.append((i - 1, None)); i -= 1
        else:
            cols.append((None, j - 1)); j -= 1
    return cols[::-1]


def posinfo(p):
    out = []; n = len(p['toks'])
    k = 0
    for l in p['lines']:
        for j in range(len(l)):
            lp = 0 if j == 0 else (2 if j == len(l) - 1 else 1)
            out.append(min(5, int(6 * k / n)) * 3 + lp); k += 1
    return out


def ent(c):
    t = sum(c.values()); return -sum(v / t * math.log2(v / t) for v in c.values() if v) if t else 0.0


def analyse(name, P, pairs):
    pages = defaultdict(Counter)
    for p in P: pages[p['page']].update(p['toks'])
    freq = Counter(t for p in P for t in p['toks'])
    ranks = {t: r for r, (t, _) in enumerate(freq.most_common())}
    nt = len(ranks)
    dec = lambda t: min(9, int(10 * ranks[t] / nt))
    pos = [posinfo(p) for p in P]
    fix = slot = cols = 0; posS = Counter(); posA = Counter()
    ps_s, ps_f, ps_r = [], [], []
    fillers = []
    rng = random.Random(1)
    def own(tok, X, Y, nX):          # tok elsewhere on own page X (beyond nX uses) and absent from partner page Y
        return float(pages[X][tok] > nX and pages[Y][tok] == 0)
    for i, j in pairs:
        a, b = P[i], P[j]
        al = align(a['toks'], b['toks'])
        X, Y = a['page'], b['page']
        byd = defaultdict(list)
        for t in a['toks']: byd[('a', dec(t))].append(t)
        for t in b['toks']: byd[('b', dec(t))].append(t)
        for k, (x, y) in enumerate(al):
            if x is None or y is None: continue
            cols += 1; posA[pos[i][x]] += 1
            ta, tb = a['toks'][x], b['toks'][y]
            if ta == tb:
                fix += 1
                if rng.random() < 0.2:
                    ps_f.append(own(ta, X, Y, a['toks'].count(ta)))
                continue
            prevm = k > 0 and al[k - 1][0] is not None and al[k - 1][1] is not None and a['toks'][al[k - 1][0]] == b['toks'][al[k - 1][1]]
            nextm = k + 1 < len(al) and al[k + 1][0] is not None and al[k + 1][1] is not None and a['toks'][al[k + 1][0]] == b['toks'][al[k + 1][1]]
            if not (prevm and nextm): continue
            slot += 1; posS[pos[i][x]] += 1
            v = 0.5 * (own(ta, X, Y, a['toks'].count(ta)) + own(tb, Y, X, b['toks'].count(tb)))
            ra = rng.choice(byd[('a', dec(ta))]); rb = rng.choice(byd[('b', dec(tb))])
            r = 0.5 * (own(ra, X, Y, a['toks'].count(ra)) + own(rb, Y, X, b['toks'].count(rb)))
            ps_s.append(v); ps_r.append(r)
            fillers.append((X, ta, Y, tb, pos[i][x], v))
    res = dict(name=name, npairs=len(pairs), cols=cols, FIX=fix / max(1, cols), SLOTR=100 * slot / max(1, cols),
               nslot=slot, POSH=ent(posS) - ent(posA), PS=float(np.mean(ps_s)) if ps_s else None,
               PSR=float(np.mean(ps_r)) if ps_r else None,
               PSD=float(np.mean(np.array(ps_s) - np.array(ps_r))) if ps_s else None,
               PSD_se=float(np.std(np.array(ps_s) - np.array(ps_r)) / math.sqrt(len(ps_s))) if len(ps_s) > 1 else None,
               PSF=float(np.mean(ps_f)) if ps_f else None,
               posS={str(k): v for k, v in posS.items()}, posA={str(k): v for k, v in posA.items()})
    return res, fillers


def pairs_for(name, P, which):
    d = np.load(os.path.join(V.CK, 'c1_%s.npz' % name))
    sc = d['sc']; npair = d['npair']
    rng = random.Random(7600)
    tmpl = [C1.rand_template(rng) for _ in range(len(sc))]
    ok = npair[:, 1] >= 5
    order = np.argsort(-np.where(ok, sc[:, 1], -1e9))[:K]
    views = [C1.para_view(p) for p in P]
    half = [p['half'] for p in P]
    out = set()
    for k in order:
        lab = C1.signatures(views, tmpl[k], {})
        grp = defaultdict(list)
        for i, g in enumerate(lab): grp[g].append(i)
        for g, ii in grp.items():
            if len(ii) < 2 or len(ii) > 30: continue
            for x in range(len(ii)):
                for y in range(x + 1, len(ii)):
                    i, j = ii[x], ii[y]
                    if P[i]['page'] == P[j]['page'] or P[i]['sec'] != P[j]['sec']: continue
                    if which == 'H1' and not (half[i] == 1 and half[j] == 1): continue
                    out.add((i, j))
    return sorted(out), [int(k) for k in order]


def job(name):
    fn = os.path.join(V.CK, 'c2_%s.json' % name)
    if os.path.exists(fn): return name
    P = json.load(gzip.open(os.path.join(V.CK, 'par_%s.json.gz' % name), 'rt'))
    out = {}
    for which in ('ALL', 'H1'):
        pr, order = pairs_for(name, P, which)
        if len(pr) > 6000: pr = random.Random(5).sample(pr, 6000)
        r, fill = analyse(name, P, pr)
        r['top_templates'] = order[:10]
        out[which] = r
        if which == 'ALL':
            json.dump(fill, gzip.open(os.path.join(V.CK, 'fill_%s.json.gz' % name), 'wt'))
    json.dump(out, open(fn, 'w'))
    return name


if __name__ == '__main__':
    names = sorted(os.path.basename(f)[3:-4] for f in glob.glob(os.path.join(V.CK, 'c1_*.npz')))
    names = [n for n in names if n != 'test']
    with Pool(2) as pool:
        for n in pool.imap_unordered(job, names): print('done', n, flush=True)
