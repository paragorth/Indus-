"""la76: the archive as a market. Hidden relative values (a money of account) over commodity signs,
massive random search, scored on (S1) equal-value coincidences between entries of different goods on one
document and (S2) equal value of person blocks (word-headed groups) paid in different mixes.
Controls: quantities permuted within commodity x site, planted economy on the real skeleton,
no-economy model (independent quantities), and the uniform-value vector (all goods worth the same)."""
import json, os, collections
import numpy as np

D = os.path.join(os.path.dirname(__file__), '..', 'data')
CK = os.path.join(D, 'la76_ckpt')
TOTAL_WORDS = {('KU', 'RO'), ('PO', 'TO', 'KU', 'RO'), ('KI', 'RO')}
WORD_GOODS = {'NI', '*304', '*308'}       # la8: behave as commodity logograms
SUPPORTS = ('Tablet', 'Lames (short thin tablet)')
TOL = np.log(float(os.environ.get('TOLR', '1.10')))


def entries(read_only=True, frac_mode='int'):
    """One row per number: doc, site, block (person/heading group), commodity, quantity.
    Commodity persists down a document until a new logogram; KU-RO/KI-RO lines skipped.
    frac_mode 'int' = integer part, 'half' = +0.5 if any fraction sign, 'drop' = skip fraction-bearing numbers."""
    d = json.load(open(os.path.join(D, 'corpus_ra.json')))
    rows = []
    for x in d:
        if x['support'] not in SUPPORTS: continue
        com = None; block = 0; is_tot = False; line_com = None
        toks = x['tokens']
        for i, t in enumerate(toks):
            if t['t'] == 'nl':
                is_tot = False; line_com = None; continue
            if t['t'] == 'logo':
                com = t['v'].split('+')[0].lstrip('*') if t['v'] != '*304' else '*304'
                line_com = com; continue
            if t['t'] == 'word':
                w = tuple(t['s'])
                if w in TOTAL_WORDS: is_tot = True; continue
                if len(w) == 1 and w[0] in WORD_GOODS:
                    line_com = w[0]; continue
                block += 1; continue
            if t['t'] == 'num':
                if is_tot: continue
                if read_only and t['st'] != 'read': continue
                q = float(t['v'])
                if t['frac']:
                    if frac_mode == 'drop': continue
                    if frac_mode == 'half': q += 0.5
                if q <= 0: continue
                c = line_com or com
                if c is None: continue
                rows.append(dict(doc=x['id'], site=x['site'], block=block, com=c, q=q))
    return rows


def goods_list(rows, min_docs=4):
    dc = collections.defaultdict(set)
    for r in rows: dc[r['com']].add(r['doc'])
    return sorted([c for c, s in dc.items() if len(s) >= min_docs])


class Market:
    """Precomputed pair and block structure for fast scoring of many value vectors."""
    def __init__(self, rows, goods):
        gi = {g: i for i, g in enumerate(goods)}
        self.goods = goods
        rows = [r for r in rows if r['com'] in gi]
        self.rows = rows
        self.c = np.array([gi[r['com']] for r in rows])
        self.lq = np.log(np.array([r['q'] for r in rows]))
        docs = sorted(set(r['doc'] for r in rows)); di = {d: i for i, d in enumerate(docs)}
        self.docs = docs
        self.g = np.array([di[r['doc']] for r in rows])
        # S1: pairs of entries on one document with different goods; weight 1/npairs(doc)
        A, B, W, PD = [], [], [], []
        byd = collections.defaultdict(list)
        for k, r in enumerate(rows): byd[di[r['doc']]].append(k)
        for dd, ks in byd.items():
            pr = [(a, b) for ii, a in enumerate(ks) for b in ks[ii + 1:] if self.c[a] != self.c[b]]
            for a, b in pr:
                A.append(a); B.append(b); W.append(1.0 / np.sqrt(len(pr))); PD.append(dd)
        self.A = np.array(A, int); self.B = np.array(B, int); self.W = np.array(W); self.PD = np.array(PD, int)
        # S2: person blocks with >=1 good; compare block values within doc where compositions differ
        bk = collections.OrderedDict()
        for k, r in enumerate(rows): bk.setdefault((r['doc'], r['block']), []).append(k)
        self.blocks = list(bk.values()); self.bdoc = np.array([di[d] for d, _ in bk.keys()])
        comp = [frozenset(self.c[k] for k in ks) for ks in self.blocks]
        BA, BB, BD = [], [], []
        byb = collections.defaultdict(list)
        for j, dd in enumerate(self.bdoc): byb[dd].append(j)
        for dd, js in byb.items():
            for ii, a in enumerate(js):
                for b in js[ii + 1:]:
                    if comp[a] != comp[b] and (len(comp[a]) > 1 or len(comp[b]) > 1):
                        BA.append(a); BB.append(b); BD.append(dd)
        self.BA = np.array(BA, int); self.BB = np.array(BB, int); self.BD = np.array(BD, int)
        nb = len(self.blocks)
        self.M = np.zeros((len(rows), nb))
        for j, ks in enumerate(self.blocks):
            for k in ks: self.M[k, j] = 1.0
        # strata for permutation: commodity x site
        st = collections.defaultdict(list)
        for k, r in enumerate(rows): st[(r['com'], r['site'])].append(k)
        self.strata = [np.array(v) for v in st.values() if len(v) > 1]

    def doc_mask(self, docset):
        m = np.array([d in docset for d in self.docs])
        return m[self.PD], m[self.BD] if len(self.BD) else np.zeros(0, bool)

    def s1(self, LV, lq=None, pm=None):
        """LV: (n, K) log values. returns (n,) weighted count of equal-value pairs."""
        lq = self.lq if lq is None else lq
        A, B, W = self.A, self.B, self.W
        if pm is not None: A, B, W = A[pm], B[pm], W[pm]
        base = lq[A] - lq[B]
        d = LV[:, self.c[A]] - LV[:, self.c[B]] + base[None, :]
        return ((np.abs(d) < TOL) * W[None, :]).sum(1)

    def s2(self, LV, lq=None, pm=None):
        lq = self.lq if lq is None else lq
        BA, BB = self.BA, self.BB
        if pm is not None: BA, BB = BA[pm], BB[pm]
        if len(BA) == 0: return np.zeros(len(LV))
        val = np.exp(LV[:, self.c] + lq[None, :]) @ self.M          # (n, nblocks)
        lv = np.log(val + 1e-12)
        d = lv[:, BA] - lv[:, BB]
        return (np.abs(d) < TOL).sum(1).astype(float)

    def permute(self, rng):
        lq = self.lq.copy()
        for ix in self.strata:
            lq[ix] = lq[rng.permutation(ix)]
        return lq


def random_values(rng, n, K, span=np.log(60)):
    LV = rng.uniform(-span, span, size=(n, K)); LV[:, 0] = 0.0
    return LV


def planted_quantities(mk, rng, lv_true, frac=0.5, noise=0.05):
    """Real skeleton; in a random half of documents each block gets one target value V_doc,
    split over its goods, quantity = round(share / value). Other docs keep real quantities."""
    lq = mk.lq.copy()
    docs = rng.permutation(len(mk.docs))[:int(frac * len(mk.docs))]
    sel = set(docs.tolist())
    for j, ks in enumerate(mk.blocks):
        dd = mk.bdoc[j]
        if dd not in sel: continue
        rng2 = np.random.default_rng(int(dd) * 7919 + 1); V = np.exp(rng2.uniform(np.log(5), np.log(200)))
        sh = rng.dirichlet(np.ones(len(ks)))
        for k, s in zip(ks, sh):
            q = max(1, round(V * s / np.exp(lv_true[mk.c[k]]) * np.exp(rng.normal(0, noise))))
            lq[k] = np.log(q)
    return lq
