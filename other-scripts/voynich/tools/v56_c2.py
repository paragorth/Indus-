"""v56 cycle 2: learned pointer TABLES. Instead of an arithmetic address, each word skeleton type (or each word
type) is allowed to point to one arbitrary page: its target is learned on the train pages as the page that the
contexts of that type most resemble (residual affinity R, near pages excluded); on the held-out pages, the
contexts of the same type must again resemble that page (z_hit) AND resemble it more than the pages most like it
(z_spec, the pointer-specificity test: a pointer names ONE page; topical vocabulary only names a neighbourhood).
Token classes: all, labels, paragraph-first words, line-final words, gallows words, frame-only words.
Corpora: ZL, IT2a, controls (Culpeper numerals = opaque pointer table; Culpeper names = real named
cross-references; Brumati herbal = real herbal with no cross-references, topic only), nulls (Markov, self-citation,
relabelled pages, planted positional pointers)."""
import sys, os, json, random, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v56_lib as L
from multiprocessing import Pool

TAG = sys.argv[1] if len(sys.argv) > 1 else 'c2'


def brumati():
    import v54_lib
    P, M = v54_lib.brumati(plant=False)
    pages = []
    for i, p in enumerate(P):
        pages.append(dict(id=p['id'], sec=p['vars']['sec'], quire='Q%02d' % (i // 16), folio=i // 2, lang='-', hand='-',
                          lines=p['lines'], lflags=[(k == 0, 'P') for k in range(len(p['lines']))]))
    return dict(pages=pages, skel=lambda w: w, name='BRUMATI')


def corpus(name):
    if name == 'BRUMATI': return brumati()
    import v56_c1
    return v56_c1.corpus(name)


def table_test(E, TT, C, cls_mask, tok_tr, key='skel', min_tr=2, min_te=1, K=5, rng=None):
    R = E['R']; N = E['N']
    # page-page similarity for the specificity neighbourhood: correlation of R columns over all tokens
    if 'PP' not in E:
        X = R - R.mean(0, keepdims=True); X /= (np.linalg.norm(X, axis=0, keepdims=True) + 1e-9)
        PP = X.T @ X; np.fill_diagonal(PP, -9); E['PP'] = PP
    PP = E['PP']
    types = collections.defaultdict(lambda: ([], []))
    for i, (w, s) in enumerate(TT['words']):
        if not cls_mask[i]: continue
        t = s if key == 'skel' else w
        if not t: continue
        types[t][0 if tok_tr[i] else 1].append(i)
    hit, spec, ntypes = [], [], 0
    learned = []
    for t, (itr, ite) in types.items():
        if len(itr) < min_tr or len(ite) < min_te: continue
        sc = R[itr].sum(0)
        tstar = int(np.argmax(sc))
        if sc[tstar] <= 0: continue
        ntypes += 1
        nb = np.argsort(-PP[tstar])[:K]
        learned.append((t, tstar, ite))
        for i in ite:
            hit.append(R[i, tstar]); spec.append(R[i, tstar] - R[i, nb].mean())
    hit = np.array(hit); spec = np.array(spec)
    def z(a): return float(a.sum() / math.sqrt(len(a)) / (a.std() + 1e-9)) if len(a) > 5 else 0.0
    # pairing null: the learned targets shuffled among the types of the class (keeps class-level hub pages)
    rng = rng or random.Random(0)
    obs = float(hit.sum()) if len(hit) else 0.0
    perm = []
    ts = [x[1] for x in learned]
    for _ in range(200):
        sh = ts[:]; rng.shuffle(sh)
        perm.append(sum(float(R[x[2], s2].sum()) for x, s2 in zip(learned, sh)))
    perm = np.array(perm) if perm else np.zeros(1)
    z_pair = float((obs - perm.mean()) / (perm.std() + 1e-9))
    return dict(ntypes=ntypes, ntest=len(hit), hit_mean=float(hit.mean()) if len(hit) else 0, z_hit=z(hit),
                spec_mean=float(spec.mean()) if len(spec) else 0, z_spec=z(spec), z_pair=z_pair,
                top=[(x[0], int(x[1]), len(x[2])) for x in learned[:0]])


def classes(TT):
    G = TT['glyphs']
    gal = [TT['gi'][g] for g in G if g in L.GALL]
    frame_only = [TT['gi'][g] for g in G if g not in L.FRAME]
    T = len(TT['words'])
    out = dict(all=np.ones(T, bool), label=TT['label'], pfirst=TT['pfirst'], last=TT['last'], first=TT['first'])
    if gal: out['gallows'] = TT['cnt_full'][:, gal].sum(1) > 0
    if frame_only and any(g in L.FRAME for g in G): out['frame_only'] = TT['cnt_full'][:, frame_only].sum(1) == 0
    return out


def run(name):
    C = corpus(name)
    E = L.build_R(C)
    TT = L.token_table(C, E)
    res = dict(name=name)
    for split in range(3):
        trp = L.split_pages(E, seed=split)
        tok_tr = trp[E['tok_page']]
        for cname, m in classes(TT).items():
            if m.sum() < 50: continue
            for key in ('skel', 'word'):
                r = table_test(E, TT, C, m, tok_tr, key=key)
                res['%s|%s|%d' % (cname, key, split)] = r
    # page-relabelled null on the same text: targets are learned and tested on a permuted page space
    json.dump(res, open(os.path.join(L.CK, '%s_%s.json' % (TAG, name)), 'w'))
    return name


if __name__ == '__main__':
    which = sys.argv[2].split(',') if len(sys.argv) > 2 else ['ZL', 'IT2a', 'CULP', 'CULP_names', 'BRUMATI', 'ZL_markov',
                                                              'ZL_selfcit', 'ZL_relab', 'ZL_planted']
    with Pool(2) as P:
        for nm in P.imap_unordered(run, which): print('done', nm, flush=True)
