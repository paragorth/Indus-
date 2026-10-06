#!/usr/bin/env python3
"""pe56 cycle 3: DISSECT THE RESIDUES WITH ORACLE FORGERS.
Each cycle-1 residue (ensemble-mean p < 1e-3; Bonferroni flag kept in the report) is re-scored against special forgers,
each of which is GIVEN one kind of real-world knowledge. The kind of knowledge that makes a residue vanish
is its constraint type:
  PERM   the tablet itself with its entry lines (sign string + numeral together) permuted, header in place
         -> a residue that survives is an ORDER constraint (which entry comes where);
  VPERM  the tablet itself with numerals permuted among its entries (strings fixed)
         -> a residue that survives is a STRING-NUMBER binding or arithmetic;
  HDR    a line-bank forger (other folds) whose tablet pools are keyed by the header's first sign and which
         copies a real header -> a residue that vanishes is a HEADER-LINK constraint;
  TOPIC  a forger that takes the entry lines of 1-3 OTHER real tablets sharing >= 1 sign with the target
         tablet's header and deals them in random order -> a residue that vanishes is co-occurrence of
         vocabularies (topic), not a structure of this tablet.
Residue survival: R vs E_x with Poisson p < 1e-3 (E_x from 30 draws per tablet for PERM/VPERM/TOPIC and
the 5-fold HDR forger at 30 reps).
usage: pe56_c3.py NAME [NAME ...]"""
import sys, json, time, random
from pe56_common import *

NR = int(os.environ.get('PE56_C3R', '30'))


def perm_entries(d, rng):
    L = d['lines']
    ei = [i for i, l in enumerate(L) if l['sys'] is not None]
    sh = [L[i] for i in ei]
    rng.shuffle(sh)
    out = list(L)
    for i, l in zip(ei, sh):
        out[i] = l
    return {'id': d['id'], 'site': d['site'], 'lines': out}


def perm_values(d, rng):
    L = d['lines']
    ei = [i for i, l in enumerate(L) if l['sys'] is not None]
    nv = [(L[i]['sys'], L[i]['v']) for i in ei]
    rng.shuffle(nv)
    out = [dict(l) for l in L]
    for i, (s, v) in zip(ei, nv):
        out[i]['sys'], out[i]['v'] = s, v
    return {'id': d['id'], 'site': d['site'], 'lines': out}


def header_of(d):
    L = d['lines']
    fi = next((i for i, l in enumerate(L) if l['sys'] is not None), len(L))
    return L[:fi]


class HdrForger:
    def __init__(self, docs, rng):
        self.rng = rng
        self.by = collections.defaultdict(list)
        self.docs = docs
        for d in docs:
            h = header_of(d)
            k = h[0]['s'][0] if h and h[0]['s'] else '-'
            self.by[k].append(d)

    def forge(self, target):
        h = header_of(target)
        k = h[0]['s'][0] if h and h[0]['s'] else '-'
        pool = self.by.get(k) or self.docs
        if len(pool) < 3:
            pool = self.docs
        n = sum(l['sys'] is not None for l in target['lines'])
        ents = [l for d in pool for l in d['lines'] if l['sys'] is not None]
        return {'id': 'f', 'site': target['site'], 'lines': list(h) + [self.rng.choice(ents) for _ in range(n)]}


class TopicForger:
    def __init__(self, docs, rng):
        self.rng = rng
        self.docs = docs
        self.inv = collections.defaultdict(list)
        for i, d in enumerate(docs):
            for x in set(x for l in d['lines'] for x in l['s']):
                self.inv[x].append(i)

    def forge(self, target):
        hs = set(x for l in header_of(target) for x in l['s'])
        allx = set(x for l in target['lines'] for x in l['s'])
        cand = set()
        for x in (hs or allx):
            cand.update(self.inv.get(x, []))
        cand = sorted(cand) or list(range(len(self.docs)))
        k = self.rng.randint(1, 3)
        src = [self.docs[self.rng.choice(cand)] for _ in range(k)]
        ents = [l for d in src for l in d['lines'] if l['sys'] is not None]
        n = sum(l['sys'] is not None for l in target['lines'])
        self.rng.shuffle(ents)
        while len(ents) < n:
            ents += ents or [{'s': [], 'sys': 'SDB', 'v': 1.0}]
        return {'id': 'f', 'site': target['site'], 'lines': list(header_of(target)) + ents[:n]}


def main():
    for name in sys.argv[1:]:
        fn = os.path.join(CK, 'c3_%s.json' % name)
        if os.path.exists(fn):
            continue
        t0 = time.time()
        c1 = json.load(open(os.path.join(CK, 'c1_%s.json' % name)))
        docs = json.load(open(os.path.join(CK, 'c1_%s_docs.json' % name)))
        freq = freq_set(docs)
        bonf = 0.05 / len(c1['res'])
        keys = set(tuple(x[0]) for x in c1['res'] if x[4] < 1e-3)
        if c1.get('truth'):
            keys |= set(tuple(t) for t in c1['truth'])
        rng = random.Random(seed('pe56c3' + name))
        E = {m: collections.Counter() for m in ['PERM', 'VPERM', 'HDR', 'TOPIC']}
        idx = list(range(len(docs))); rng.shuffle(idx)
        fold = {i: k % 5 for k, i in enumerate(idx)}
        for f in range(5):
            tr = [docs[i] for i in range(len(docs)) if fold[i] != f]
            H = HdrForger(tr, random.Random(rng.randrange(1 << 30)))
            T = TopicForger(tr, random.Random(rng.randrange(1 << 30)))
            for i in range(len(docs)):
                if fold[i] != f:
                    continue
                d = docs[i]
                for _ in range(NR):
                    for m, g in (('PERM', lambda: perm_entries(d, rng)), ('VPERM', lambda: perm_values(d, rng)),
                                 ('HDR', lambda: H.forge(d)), ('TOPIC', lambda: T.forge(d))):
                        for r in relations(g(), freq):
                            if r in keys:
                                E[m][r] += 1
        real = rel_counts(docs, freq)
        c1map = {tuple(x[0]): x for x in c1['res']}
        rows = []
        for k in sorted(keys):
            R = real.get(k, 0)
            row = {'key': list(k), 'R': R, 'E_forger': c1map[k][2] if k in c1map else None,
                   'p_forger': c1map[k][4] if k in c1map else None}
            for m in E:
                e = E[m][k] / NR
                row['E_' + m] = e
                row['p_' + m] = pois_sf(R, e)
            rows.append(row)
        json.dump({'name': name, 'truth': c1.get('truth'), 'rows': rows, 'secs': time.time() - t0},
                  open(fn, 'w'))


if __name__ == '__main__':
    main()
