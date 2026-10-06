"""v82 cycle 2: WHAT IS THE ONSET LINK? (if it survives cycle 1)
On ZL3b, IT2a, the best cycle-1 generator, and real texts written through the v72 surface machinery:
  HE  Hebrew Genesis (consonantal; Semitic prefixes w-, h-, b-, l-, m-, k-, $-; article agreement h-N h-ADJ)
  SW  Swahili Wikipedia prose (Bantu noun-class concord written at the FRONT of nouns, adjectives, verbs)
  EO  Esperanto, CS Czech, LA Isidore (agreement written at the END of words)
  SW_REV, HE_REV  the same texts with every word reversed (prefix agreement moved to the end): if the frozen test
                  sees front agreement, the gain must drop
Measures (frozen v81 onset definitions d0 E1c+line k1, d1 q+line stripped k1, d2 E1c+line k2):
  gain   frozen junction-preserving gain (same-section null, 20 re-rolls)
  diag   share of the excess pair table (real minus re-roll mean, all pairs) on the diagonal (same onset twice)
  asym   off-diagonal asymmetry of the excess table, sum|D-D^T| / sum(|D|+|D^T|), and its re-roll noise level
  cross  line-break test: (last word of line L -> first word of line L+1, same page) gain beyond the junction,
         against the within-line pair (second-to-last -> last word) of the same lines
  lag    lag-2 / lag-1 ratio (v81 c3b)
  top    the onset pairs that carry the excess (ZL / IT2a)
"""
import os, sys, re, json, random, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v82_lib as K, v72_lib as V, v78_lib as V78, v81_lib as L81, v81_c3 as C3
from collections import Counter
from multiprocessing import Pool

SCR = sys.argv[2] if len(sys.argv) > 2 else None   # path of the Swahili text (scratch, not committed)


def paras_from(path, keep, rev=False, cap=42000, verse=False):
    txt = open(path, encoding='utf-8').read().lower()
    out = []; n = 0
    blocks = txt.split('\n') if verse else re.split(r'\n\s*\n|\n', txt)
    for b in blocks:
        ws = re.findall('[' + keep + ']+', b)
        if len(ws) < 3: continue
        if rev: ws = [w[::-1] for w in ws]
        out.append(ws); n += len(ws)
        if n >= cap: break
    return out


def build():
    C = {}
    C['VOY_ZL'] = V.voynich('ZL3b'); C['VOY_IT'] = V.voynich('IT2a')
    P = os.path.join(K.ROOT, 'data', 'plain')
    LAT = 'a-z'
    specs = [('HE', os.path.join(P, 'he.txt'), 'א-ת', False, True), ('HE_REV', os.path.join(P, 'he.txt'), 'א-ת', True, True),
             ('EO', os.path.join(P, 'eo.txt'), LAT + 'ĉĝĥĵŝŭ', False, False),
             ('CS', os.path.join(P, 'cs.txt'), LAT + 'áčďéěíňóřšťúůýž', False, False),
             ('LA', os.path.join(P, 'la.txt'), LAT, False, False)]
    if SCR:
        specs += [('SW', SCR, LAT, False, False), ('SW_REV', SCR, LAT, True, False)]
    for i, (name, path, keep, rev, verse) in enumerate(specs):
        pr = paras_from(path, keep, rev, verse=verse)
        pages = V78.plain_pages(pr, name.lower(), cap=40000)
        C[name] = V78.through_surface(pages, 8200 + i)
        C[name + '_PLAIN'] = pages        # the same text without the surface (reference)
    cb = [json.loads(x) for x in open(os.path.join(K.CK, 'c1b.jsonl'))]
    pick = {}
    for r in cb: pick.setdefault(r['cand'], r['P'])
    C['GEN_KILL'] = K.gen_moodagr(C['VOY_ZL'], pick[2], 9101)    # harmony H6 + passage mood, kills 10/10 seeds
    C['GEN_NEAR'] = K.gen_moodagr(C['VOY_ZL'], pick[3], 9101)    # harmony H2 + line mood, Voynich-level gains
    C['GEN_PMI'] = K.gen_moodagr(C['VOY_ZL'], pick[0], 9101)     # fitted harmony table + line mood (ceiling)
    import v77_lib as G
    C['GEN_STACK'] = G.generate('ZL3b', 'STACK', 9102)
    K.psave('c2_corpora.pkl', C)
    return C


def ext_table(c, o, ctx, rng, nrep=20):
    """excess pair table: counts of (ctx_i, o_i+1) over all line-interior pairs, real minus mean re-roll."""
    i1 = c.p1; Kx = int(max(o.max(), ctx.max())) + 1
    def tab(x):
        T = np.zeros((Kx, Kx)); np.add.at(T, (x[i1], o[i1 + 1]), 1); return T
    R = tab(ctx); N = [tab(C3.reroll(c, ctx, rng)) for _ in range(nrep)]
    Nm = np.mean(N, 0)
    return R - Nm, Nm, [n - Nm for n in N[:5]]


def asym(D):
    iu = np.triu_indices(D.shape[0], 1)
    a, b = D[iu], D.T[iu]
    return float(np.abs(a - b).sum() / (np.abs(a).sum() + np.abs(b).sum() + 1e-9))


def gain_pairs(c, o, ctx, ii, jj, tr, te):
    """held-out gain of p(o_j | end_i, ctx_i) over p(o_j | end_i) for explicit pairs (i -> j)."""
    Kx = int(max(o.max(), ctx.max())) + 2
    a, b, e = ctx[ii], o[jj], c.end[ii]; mtr, mte = tr[ii], te[ii]
    uc = np.bincount(o[tr], minlength=Kx).astype(float) + 0.5; pu = uc / uc.sum()
    he, pe = L81._ce_cond(e[mtr], b[mtr], e[mte], b[mte], pu[b[mte]], Kx)
    hea, _ = L81._ce_cond(e[mtr] * Kx + a[mtr], b[mtr], e[mte] * Kx + a[mte], b[mte], pe, Kx)
    return he.mean() - hea.mean()


def reroll_any(c, x, rng):
    """junction-preserving re-roll over ALL positions (same section, same last glyph)."""
    y = x.copy(); key = c.sec * 1000 + c.end
    for kk in np.unique(key):
        ix = np.nonzero(key == kk)[0]; y[ix] = x[ix[rng.permutation(len(ix))]]
    return y


def cross(c, o, rng, nrep=20):
    last = np.nonzero((np.r_[c.ln[1:] != c.ln[:-1], True]) & (c.pos >= 1))[0]
    nxt = last + 1
    ok = (nxt < c.n); last, nxt = last[ok], nxt[ok]
    ok = (c.pg[nxt] == c.pg[last]) & (c.pos[nxt] == 0); last, nxt = last[ok], nxt[ok]
    # same lines, within-line pair second-to-last -> last
    pen = last - 1; ok2 = c.pos[pen] >= 1
    out = {}
    for tag, ii, jj in (('cross', last, nxt), ('cross2', last, nxt + 1), ('inline', pen[ok2], last[ok2])):
        if tag == 'cross2':
            ok3 = (jj < c.n) & (c.ln[np.minimum(jj, c.n - 1)] == c.ln[nxt]); ii, jj = ii[ok3], jj[ok3]
        r = gain_pairs(c, o, o, ii, jj, c.tr, c.te)
        nul = [gain_pairs(c, o, reroll_any(c, o, rng), ii, jj, c.tr, c.te) for _ in range(nrep)]
        out[tag] = (r - np.mean(nul), (r - np.mean(nul)) / (np.std(nul) + 1e-9), int(len(ii)))
    return out


CC = None


def init():
    global CC
    CC = K.pload('c2_corpora.pkl')


def work(name):
    pages = CC[name]; c = K.Mini(pages); rng = np.random.default_rng(822); out = {'name': name}
    out['frozen'] = K.frozen(c, nrep=20)
    out['lag'] = K.lag_ratio(c)
    out['defs'] = []
    for di, d in enumerate(K.DEFS):
        o = L81.onset(c, d)
        D, Nm, noise = ext_table(c, o, o, rng)
        pos = D.clip(min=0)
        dg = float(np.trace(pos) / (pos.sum() + 1e-9))
        same_ex = float(np.trace(D) / (np.trace(Nm) + 1e-9))
        a = asym(D); an = float(np.mean([asym(x) for x in noise]))
        # labels for top cells
        lab = {}
        A = c.G[d['v']][:, :d['k']]
        for t in range(c.n):
            if o[t] not in lab: lab[o[t]] = ''.join(L81.ALPHA[x] for x in A[t] if L81.ALPHA[x] != '$') or '$'
        top = []
        for idx in np.argsort(-D, axis=None)[:12]:
            a_, b_ = np.unravel_index(idx, D.shape)
            top.append((lab.get(a_, '?'), lab.get(b_, '?'), float(D[a_, b_]), float(Nm[a_, b_])))
        bot = []
        for idx in np.argsort(D, axis=None)[:6]:
            a_, b_ = np.unravel_index(idx, D.shape)
            bot.append((lab.get(a_, '?'), lab.get(b_, '?'), float(D[a_, b_]), float(Nm[a_, b_])))
        out['defs'].append(dict(diag=dg, same_ex=same_ex, asym=a, asym_noise=an, top=top, bot=bot,
                                cross=cross(c, o, rng)))
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build()
    else:
        names = list(K.pload('c2_corpora.pkl'))
        res = {}
        with Pool(2, initializer=init) as P:
            for r in P.imap_unordered(work, names):
                res[r['name']] = r
                f = ' '.join('%+.3f(z%.1f)' % x for x in r['frozen'])
                d = ' | '.join('diag %.2f same %+.2f asym %.2f/%.2f cross %+.4f(z%.1f) cross2 %+.4f inline %+.4f(z%.1f)' % (
                    x['diag'], x['same_ex'], x['asym'], x['asym_noise'], x['cross']['cross'][0], x['cross']['cross'][1],
                    x['cross']['cross2'][0], x['cross']['inline'][0], x['cross']['inline'][1]) for x in r['defs'])
                print(r['name'], f, 'lag', ['%.3f/%.3f' % l for l in r['lag']], d, flush=True)
        json.dump(res, open(os.path.join(K.CK, 'c2.json'), 'w'), default=float)
