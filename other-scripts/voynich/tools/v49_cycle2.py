"""v49 cycle 2: is the opening-line off-rule lexicon a second language layer, or the known opening-line spelling?

Cycle 1: opening off-table words are internally stranger than body off-table words (+0.21 bits/glyph), and a glyph
comparison shows the difference is dominated by p (and f) gallows. Here:
  A  REFIT after merging all gallows (E3 + p, f -> t; benched cph, cfh -> cth): does the opening-line off-table
     surplus (42% vs 26%) survive, and the M4 strangeness? Controls: BRf (Latin layer must survive the merge),
     GENN (noise must survive too: it is not made of gallows).
  B  SEPARABILITY: a glyph 1-3-gram naive Bayes trained on OFFpf vs OFFb tokens of one page half, AUC on the other
     half, under a normalisation ladder (raw, E3, gallows-merged, gallows-merged without the word's first glyph).
     Positive: BRf (Latin layer), GENT. Negatives: GEN0, GENN.
  C  LEXICON MAP: word-internal typological fingerprint (v48 features minus the running-text ones) of size-matched
     type samples: OFFpf, OFFb, in-table rare types; distance OFFpf-OFFb over the split-half floor, and the nearest
     reference lexicons (rare types of ~60 natural languages from the v48 set). Positive: BRf Latin vs Italian.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v49_lib import *
import v49_cycle1 as C1

FN = 'v49_cycle2.txt'
LEXF = ['h1', 'h2', 'h2r', 'vshare', 'alt', 'bip', 'agree', 'initV', 'finV', 'clus', 'clus2', 'VV', 'sylpw', 'skelH',
        'wl', 'wlcv', 'wlsk', 'w1', 'wlong', 'morph', 'mpw', 'Hp1', 'Hp2', 'Hl1', 'Hl2', 'sufconc', 'sufpre']


def nb_auc(T, f, seed=0):
    """OFFpf vs OFFb tokens; fit on fold of pages, test on the other; AUC averaged over both directions."""
    pages = sorted(set(t['page'] for t in T)); half = set(pages[::2])
    def feats(w):
        s = '^' + f(w) + '$'
        return [s[i:i + n] for n in (1, 2, 3) for i in range(len(s) - n + 1)]
    aucs = []
    for part in (0, 1):
        tr = [t for t in T if t['off'] and ((t['page'] in half) == bool(part))]
        te = [t for t in T if t['off'] and ((t['page'] in half) != bool(part))]
        cnt = {r: Counter() for r in ('pf', 'body')}
        for t in tr: cnt[t['role']].update(feats(t['w']))
        V = len(set(cnt['pf']) | set(cnt['body'])) + 1
        tot = {r: sum(c.values()) for r, c in cnt.items()}
        def score(w):
            return sum(math.log((cnt['pf'][g] + 0.5) / (tot['pf'] + 0.5 * V)) - math.log((cnt['body'][g] + 0.5) / (tot['body'] + 0.5 * V)) for g in feats(w))
        y = np.array([t['role'] == 'pf' for t in te]); s = np.array([score(t['w']) for t in te])
        from scipy.stats import rankdata
        r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
        aucs.append(float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)))
    return float(np.mean(aucs))


def lex_fp(words, seed=0):
    import v48_lib
    ws = [tuple(w) for w in words if w]
    f = v48_lib.fingerprint(ws, seed)
    return {k: f[k] for k in LEXF}


def lexicons(T, f=norm, n=900, rng=None):
    """type samples: OFFpf (not the paragraph's first word), OFFb, INr (in-table rare types)"""
    cnt = Counter(t['w'] for t in T)
    g = {'OFFpf': sorted({f(t['w']) for t in T if t['off'] and t['role'] == 'pf' and t['k'] > 0}),
         'OFFb': sorted({f(t['w']) for t in T if t['off'] and t['role'] == 'body' and t['k'] > 0}),
         'INr': sorted({f(t['w']) for t in T if (not t['off']) and cnt[t['w']] <= 3})}
    if T[0]['lat'] is not None:
        g['LAT'] = sorted({f(t['w']) for t in T if t['lat']}); g['ITr'] = sorted({f(t['w']) for t in T if not t['lat'] and cnt[t['w']] <= 3})
    return g


def sample(ws, n, rng):
    ws = list(ws); rng.shuffle(ws); return ws[:n]


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    out = jload('c2_%s.json' % part) or {}
    if part in ('A', 'all'):
        for nm in ('V_G', 'VI_G', 'BRf_G', 'GENN_G'):
            C = get_corpus(nm); T = tokens_table(nm)
            r = C1.analyse(nm, T, C, np.random.default_rng(492))
            out['A_' + nm] = dict(off_pf=r['off_pf'], off_body=r['off_body'], loc=r['OFFpf']['loc'], loc_z=r['OFFpf']['loc_z'],
                                  M4=r['M4'], pos=r['pos'], lat=r.get('lat'))
            print(nm, json.dumps(out['A_' + nm], default=float), flush=True); jsave('c2_%s.json' % part, out)
    if part in ('B', 'all'):
        ladder = dict(raw=lambda w: w, E3=norm, G=gnorm, G_nofirst=lambda w: gnorm(w)[1:] or '_',
                      G_nop=lambda w: gnorm(w).replace('q', ''))
        for nm in ('V', 'VI', 'GEN0', 'GENN', 'GENT', 'BRe', 'BRf'):
            T = tokens_table(nm); T1 = [t for t in T if t['k'] > 0]
            out['B_' + nm] = {k: nb_auc(T1, f) for k, f in ladder.items()}
            print(nm, out['B_' + nm], flush=True); jsave('c2_%s.json' % part, out)
    if part in ('C', 'all'):
        import v48_lib
        rng = random.Random(493)
        langs = jload('c2_langfp.json')
        if langs is None:
            langs = {}
            for Lc in v48_lib.all_languages():
                toks = v48_lib.tokens(Lc); cnt = Counter(toks)
                rare = [w for w, c in cnt.items() if c <= 3 and len(w) >= 2]
                if len(rare) < 900: continue
                langs[Lc['name']] = dict(fam=Lc['fam'], fp=lex_fp(sample(rare, 900, rng)))
                print('lang', Lc['name'], flush=True)
            jsave('c2_langfp.json', langs)
        names = list(langs); X = np.array([[langs[n]['fp'][k] for k in LEXF] for n in names])
        mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
        D = np.sqrt(((Z[:, None] - Z[None]) ** 2).mean(2)); np.fill_diagonal(D, np.inf); ref = float(np.median(D.min(1)))
        res = {}
        for nm in ('V', 'VI', 'GEN0', 'GENN', 'GENT', 'BRf', 'V_G', 'VI_G', 'BRf_G', 'GENN_G'):
            T = tokens_table(nm)
            f = (lambda w: w) if nm.endswith('_G') else norm
            lx = lexicons(T, f)
            n = min(700, min(len(v) for v in lx.values()))
            zf = {}
            for g, ws in lx.items():
                zs = [(np.array([lex_fp(sample(ws, n, rng), s)[k] for k in LEXF]) - mu) / sd for s in range(3)]
                zf[g] = np.mean(zs, 0)
            dist = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()))
            # split-half floor: two disjoint samples of OFFb
            fl = []
            for s in range(5):
                ws = sample(lx['OFFb'], 2 * n, rng)
                if len(ws) < 2 * n: ws = sample(lx['OFFb'] + lx['INr'], 2 * n, rng)
                a = (np.array([lex_fp(ws[:n], s)[k] for k in LEXF]) - mu) / sd; b = (np.array([lex_fp(ws[n:], s)[k] for k in LEXF]) - mu) / sd
                fl.append(dist(a, b))
            r = dict(n=n, d_pf_b=dist(zf['OFFpf'], zf['OFFb']), d_pf_in=dist(zf['OFFpf'], zf['INr']), d_b_in=dist(zf['OFFb'], zf['INr']),
                     floor=float(np.mean(fl)), floor_sd=float(np.std(fl)), ref_nn=ref)
            if 'LAT' in zf: r['d_lat_it'] = dist(zf['LAT'], zf['ITr']); r['d_lat_b'] = dist(zf['LAT'], zf['OFFb'])
            for g in zf:
                d = np.sqrt(((Z - zf[g]) ** 2).mean(1)); o = np.argsort(d)[:3]
                r['nn_' + g] = [(names[i], langs[names[i]]['fam'], round(float(d[i]), 2)) for i in o]
            # feature-wise largest differences OFFpf - OFFb
            dz = zf['OFFpf'] - zf['OFFb']; o = np.argsort(-np.abs(dz))[:5]
            r['top'] = [(LEXF[i], round(float(dz[i]), 2)) for i in o]
            res[nm] = r; print(nm, json.dumps(r, default=float), flush=True)
            out['C'] = res; jsave('c2_%s.json' % part, out)
