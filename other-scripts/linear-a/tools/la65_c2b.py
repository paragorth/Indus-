#!/usr/bin/env python3
"""la65 cycle 2b: are the dossier alignments different from alignments of matched random groups?
For every dossier, R groups of the same size and the same site mix are drawn at random and aligned the same
way. Compared: share of FIXED slots (word and logogram constant), STEP slots, whole-list rescales (two members
with >= 3 aligned numbers in one constant ratio != 1), word-logogram coupling, and word typing:
  TPL = word kept constant in an aligned slot of a dossier (template word); VAR = word only in swapped slots.
Linear B: TPL/VAR vs the answer key (TOT, TRA, QUAL vs PLA and unkeyed words, mostly personal names).
Linear A: TPL/VAR vs the la60 frozen roles (any non-entry role) - scored, never used as input.
usage: la65_c2b.py CORPUS  (LAN, LB0, LB1, LB2)"""
import os, sys, json, random
from collections import Counter, defaultdict
import numpy as np
import la65_common as K
from la65_c1 import corpus
from la65_c2 import analyse, slot_type


def profile(A):
    types = Counter(slot_type(r) for a in A for r in a['recs'])
    n = sum(types.values()) or 1
    resc = 0
    for a in A:
        # per member pair, ratios of aligned numbers (both > 0)
        mem = defaultdict(dict)
        for r in a['recs']:
            for k, e in zip(r['mids'], r['ns']):
                mem[k][r['slot']] = e
        ks = sorted(mem)
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                rat = Counter()
                for s in mem[ks[x]]:
                    u, v = mem[ks[x]].get(s), mem[ks[y]].get(s)
                    if u and v and u != v:
                        rat[round(v / u, 3)] += 1
                if rat and rat.most_common(1)[0][1] >= 3:
                    resc += 1
    words = defaultdict(Counter)
    for a in A:
        for r in a['recs']:
            t = slot_type(r)
            for w in set(r['ws']):
                for x in w.split():
                    words[x]['TPL' if t in ('FIX', 'NUM', 'STEP', 'GEN') else 'VAR'] += 1
    return {'fix': (types['FIX'] + types['NUM'] + types['GEN'] + types['STEP']) / n, 'n': n,
            'step': types['STEP'], 'resc': resc, 'types': types, 'words': words}


def wtype(c):
    return 'TPL' if c['TPL'] else 'VAR'


def main():
    name = sys.argv[1]
    R = json.load(open(os.path.join(K.CK, 'c1_dossiers.json')))
    D = R['LANF']['dossiers'] if name == 'LAN' else R[name]['dossiers'] + R[name + 'F']['dossiers']
    seen, DD = set(), []
    for d in D:
        k = tuple(sorted(d['ids']))
        if k not in seen and len(k) >= 2:
            seen.add(k); DD.append(d)
    T, meta = corpus(name)
    pos = {t['id']: t for t in T}
    bysite = defaultdict(list)
    for t in T:
        bysite[t['site']].append(t['id'])
    A = [analyse(T, d['ids']) for d in DD]
    P = profile(A)
    rng = random.Random(K.seed('la65-c2b-' + name))
    reps = 60
    NP = []
    for _ in range(reps):
        AR = []
        for d in DD:
            g = set()
            for i in d['ids']:
                c = [x for x in bysite[pos[i]['site']] if x not in g]
                g.add(rng.choice(c))
            AR.append(analyse(T, sorted(g)))
        NP.append(profile(AR))
    out = {'name': name, 'n_doss': len(DD)}
    for k in ('fix', 'step', 'resc'):
        nv = np.array([p[k] for p in NP])
        out[k] = {'obs': P[k], 'null_mean': float(nv.mean()), 'null_sd': float(nv.std()),
                  'p': float((1 + (nv >= P[k]).sum()) / (1 + reps))}
    # truth enrichment of word types
    if name.startswith('LB'):
        import la63_lib as L63
        key = {w: r for r, ws in L63.LB_KEY_W.items() for w in ws.split()}
        good = lambda w: key.get(w) in ('TOT', 'TRA', 'QUAL')
    else:
        import la60_common as C
        roles = C.prior_reading()['roles']
        good = lambda w: w in roles
    def enrich(Pp):
        W = Pp['words']
        tp = [w for w in W if wtype(W[w]) == 'TPL']
        va = [w for w in W if wtype(W[w]) == 'VAR']
        a = sum(map(good, tp)) / max(1, len(tp))
        b = sum(map(good, va)) / max(1, len(va))
        return a - b, len(tp), len(va), sum(map(good, tp)), sum(map(good, va))
    e = enrich(P)
    ne = np.array([enrich(p)[0] for p in NP])
    out['enrich'] = {'obs': e, 'null_mean': float(ne.mean()), 'p': float((1 + (ne >= e[0]).sum()) / (1 + reps))}
    W = P['words']
    out['tpl_words'] = sorted([w for w in W if wtype(W[w]) == 'TPL'], key=lambda w: -W[w]['TPL'])[:60]
    out['types'] = dict(P['types'])
    json.dump(out, open(os.path.join(K.CK, 'c2b_%s.json' % name), 'w'))
    print(json.dumps({k: v for k, v in out.items() if k != 'tpl_words'}))
    print('TPL words', out['tpl_words'][:40])


if __name__ == '__main__':
    main()
