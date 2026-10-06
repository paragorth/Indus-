#!/usr/bin/env python3
"""LA-69 cycle 3: the CLOZE READER.  Reader-independence measured directly: hide each word of a document and
let a reader who knows only OTHER documents (no sign values) guess it from its neighbours.  An insider reader
who also knows the rest of the SAME document is run alongside; notes to self should be readable by the
insider but not by the outsider (large insider-outsider gap).  Per-token, so length-insensitive.
Also a cross-site outsider (pool from other sites only)."""
import os, sys, math, random, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la69_common as C

K = 5


class Reader:
    def __init__(self, docs):
        self.u = collections.Counter(); self.L = collections.defaultdict(collections.Counter)
        self.R = collections.defaultdict(collections.Counter)
        for d in docs:
            self.add(d)

    @staticmethod
    def seq(d):
        s = ['<L>']
        for x in d['toks']:
            s.append('<L>' if x[0] == 'L' else ('N' if x[0] == 'N' else x[1]))
        s.append('<L>')
        return s

    def add(self, d, sign=1):
        s = self.seq(d)
        for i in range(1, len(s) - 1):
            w = s[i]
            if w in ('<L>', 'N'):
                continue
            self.u[w] += sign; self.L[w][s[i - 1]] += sign; self.R[w][s[i + 1]] += sign

    def guess(self, prev, nxt, k=K):
        tot = sum(self.u.values()) + 1
        best = []
        cands = set()
        for w, c in self.u.most_common(30):
            cands.add(w)
        # candidates seen after prev or before next
        for w in self._near(prev, nxt):
            cands.add(w)
        sc = []
        for w in cands:
            c = self.u[w]
            if c <= 0:
                continue
            s = math.log(c / tot) + math.log((self.L[w][prev] + 0.1) / (c + 1)) + math.log((self.R[w][nxt] + 0.1) / (c + 1))
            sc.append((s, w))
        sc.sort(reverse=True)
        return [w for _, w in sc[:k]]

    def _near(self, prev, nxt):
        return self.idx_prev.get(prev, set()) | self.idx_next.get(nxt, set())

    def index(self):
        self.idx_prev = collections.defaultdict(set); self.idx_next = collections.defaultdict(set)
        for w, c in self.L.items():
            for p, n in c.items():
                if n > 0:
                    self.idx_prev[p].add(w)
        for w, c in self.R.items():
            for p, n in c.items():
                if n > 0:
                    self.idx_next[p].add(w)


def doc_cloze(d, out_reader, in_extra=True):
    s = Reader.seq(d)
    hits_o, hits_i, n = 0, 0, 0
    # insider = outsider + the rest of this document
    ins = None
    if in_extra:
        ins = Reader([])
        ins.u = out_reader.u.copy()
        ins.L = collections.defaultdict(collections.Counter, {w: c.copy() for w, c in out_reader.L.items()})
        ins.R = collections.defaultdict(collections.Counter, {w: c.copy() for w, c in out_reader.R.items()})
        ins.add(d)
        ins.index()
    for i in range(1, len(s) - 1):
        w = s[i]
        if w in ('<L>', 'N'):
            continue
        n += 1
        hits_o += w in out_reader.guess(s[i - 1], s[i + 1])
        if ins is not None:
            # remove this occurrence from the insider's memory
            ins.u[w] -= 1; ins.L[w][s[i - 1]] -= 1; ins.R[w][s[i + 1]] -= 1
            hits_i += w in ins.guess(s[i - 1], s[i + 1])
            ins.u[w] += 1; ins.L[w][s[i - 1]] += 1; ins.R[w][s[i + 1]] += 1
    return n, hits_o, hits_i


def run_set(docs, ref, rng, M=300, npool=4, cross_key=None):
    pools = []
    for _ in range(npool):
        sub = rng.sample(ref, min(M, len(ref) - 1))
        r = Reader(sub); r.index()
        pools.append((r, {x['id'] for x in sub}))
    res = []
    for d in docs:
        src = str(d['id'])[3:] if str(d['id']).startswith('PL:') else d['id']
        cands = [p for p in pools if d['id'] not in p[1] and src not in p[1]]
        if not cands:
            sub = [x for x in rng.sample(ref, min(M + 1, len(ref))) if x['id'] not in (d['id'], src)][:M]
            rr = Reader(sub); rr.index(); cands = [(rr, set())]
        r = cands[rng.randrange(len(cands))][0]
        res.append(doc_cloze(d, r))
    return res


def main():
    rng = random.Random(693)
    S = C.calib_sets()
    out = {}

    def cap(ds, n=300):
        ds = list(ds); rng.shuffle(ds); return ds[:n]
    for cname, st in S.items():
        fin = cap(st['FINAL'])
        for cls, docs in [('NOTE', cap(st['NOTE'])), ('FINAL', fin), ('PLANT', C.plant_notes(fin, rng))]:
            out[(cname, cls)] = (docs, run_set(docs, st['ref'], rng))
        print(cname, flush=True)
    la = C.la_docs()
    out[('LA', 'LA')] = (la, run_set(la, la, rng))
    pl = C.plant_notes(la, rng)
    out[('LA', 'LA_PLANT')] = (pl, run_set(pl, la, rng))
    for name, f in [('OB', C.ob_docs), ('PC', C.pc_docs), ('KH', C.kh_docs)]:
        ds = cap(f())
        out[(name, 'ARCH')] = (ds, run_set(ds, f(), rng))
    # cross-site outsiders: LA by site, LB KN<->PY, UR3 by provenience (matched pool sizes)
    lb = C.lb_docs(); ur = C.ur3_docs()
    for tag, corp in [('LA', la), ('LB', lb), ('UR3', ur)]:
        sites = collections.Counter(d['site'] for d in corp)
        for site, n in sites.most_common(3):
            own = [d for d in corp if d['site'] == site]
            oth = [d for d in corp if d['site'] != site]
            Msz = min(150, len(own) - 1, len(oth))
            docs = cap(own, 150)
            out[('X_' + tag, site + ':same')] = (docs, run_set(docs, own, rng, M=Msz))
            out[('X_' + tag, site + ':other')] = (docs, run_set(docs, oth, rng, M=Msz))
        print('cross', tag, flush=True)
    slim = {k: ([{kk: vv for kk, vv in d.items() if kk != 'toks'} | {'nT': sum(1 for x in d['toks'] if x[0] == 'T')}
                 for d in v[0]], v[1]) for k, v in out.items()}
    pickle.dump(slim, open(os.path.join(C.CK, 'c3_cloze.pkl'), 'wb'))


if __name__ == '__main__':
    main()
