#!/usr/bin/env python3
"""la83 cycle 2: the la83 battery on 1,000 random perturbations of corpus_ra_v2.json at the measured
corpus error rate (level M) and at twice that rate (level S), plus seed-noise and planted controls.

Measured rates (la83 cycle 1, data only):
  word token reading differs between lineara.xyz and SigLA (variant / nodraw)   97 / 1,775 = 0.0546
  logogram reading differs (same)                                               48 /   627 = 0.0766
  integer differs between the record's 'words' and its 'transcription' field    34 / 1,304 = 0.0261 (upper
      bound: most are notation, e.g. 197 vs 190 + 7)
  fraction letter: no second source; the sign rate 0.0546 is used
  document dropped or counted twice by the parser                               3 / 1,721 = 0.0017
Operator: a perturbed word has one sign replaced by a sign drawn from the corpus sign frequencies; a perturbed
logogram is replaced by one drawn from the logogram frequencies; a perturbed integer gets +-1 or +-10; a perturbed
fraction letter is replaced from the letter frequencies; a dropped document is removed.
Controls: (i) seed noise: the battery on the unperturbed corpus with 100 null seeds; (ii) planted robust: a prefix
X planted on 8 % of word types (la79 plant_affix rule) must keep z >= 2; (iii) marginal M1 (I-, z 1.69) and a
null prefix Y (|z| < 0.4) must flip often.
usage: la83_perturb.py LEVEL N  -> data/la83_ckpt/perturb_LEVEL.jsonl (one line per run)"""
import json, os, sys, random, copy
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la83_battery as B
import la79_claims as K

DATA = os.path.join(HERE, '..', 'data'); CK = os.path.join(DATA, 'la83_ckpt'); os.makedirs(CK, exist_ok=True)
RATES = {'M': dict(w=0.0546, l=0.0766, n=0.0261, f=0.0546, d=0.0017),
         'S': dict(w=0.1092, l=0.1532, n=0.0522, f=0.1092, d=0.0035)}
BASE = json.load(open(os.path.join(DATA, 'corpus_ra_v2.json')))


def unigram(C):
    s, l, f = Counter(), Counter(), Counter()
    for d in C:
        for t in d['tokens']:
            if t['t'] == 'word': s.update(t['s'])
            elif t['t'] == 'logo': l[t['v']] += 1
            elif t['t'] == 'num': f.update(t.get('frac') or [])
    pick = lambda c: (list(c), np.array(list(c.values()), float) / sum(c.values()))
    return pick(s), pick(l), pick(f)


US, UL, UF = unigram(BASE)


def draw(rng, u):
    return u[0][int(np.searchsorted(np.cumsum(u[1]), rng.random()))]


def perturb(C, r, seed):
    rng = random.Random(seed); out = []
    for d in C:
        if d.get('superseded_by'): continue
        if rng.random() < r['d']: continue
        e = dict(d); toks = []
        for t in d['tokens']:
            t = dict(t)
            if t['t'] == 'word' and rng.random() < r['w']:
                s = list(t['s']); s[rng.randrange(len(s))] = draw(rng, US); t['s'] = s
            elif t['t'] == 'logo' and rng.random() < r['l']:
                t['v'] = draw(rng, UL)
            elif t['t'] == 'num':
                if rng.random() < r['n']:
                    dv = rng.choice([1, -1, 10, -10]); t['v'] = t['v'] + dv if t['v'] + dv >= 0 else t['v'] + abs(dv)
                if t.get('frac'):
                    t['frac'] = [draw(rng, UF) if rng.random() < r['f'] else x for x in t['frac']]
            toks.append(t)
        e['tokens'] = toks; out.append(e)
    return out


def pick_null_prefix():
    z = K.affix_z(B.docs79(BASE), 'pre')
    return sorted([k for k, v in z.items() if abs(v) < 0.4])[0]


def plant(C, X, frac=0.08, seed=21):
    rng = random.Random(seed)
    C = copy.deepcopy(C)
    T = sorted({tuple(t['s']) for d in C for t in d['tokens'] if t['t'] == 'word' and len(t['s']) >= 2})
    idx = [i for i, d in enumerate(C) if any(t['t'] == 'word' for t in d['tokens'])]
    for j in rng.sample(range(len(T)), int(frac * len(T))):
        i = rng.choice(idx)
        C[i]['tokens'].append({'t': 'word', 's': [X] + list(T[j]), 'st': 'read', 'fl': []})
    return C


def main():
    level, N = sys.argv[1], int(sys.argv[2])
    fn = os.path.join(CK, 'perturb_%s.jsonl' % level)
    done = sum(1 for _ in open(fn)) if os.path.exists(fn) else 0
    Y = pick_null_prefix(); Xp = 'PLANTX'
    PL = plant(BASE, Xp)
    with open(fn, 'a') as fo:
        for i in range(done, N):
            if level == 'seed':
                r = B.battery(BASE, fast=True, seed=1000 + i, extra=[Y])
            else:
                Cp = perturb(BASE, RATES[level], seed=83000 + i)
                r = B.battery(Cp, fast=True, seed=i, extra=[Y])
                Pp = perturb(PL, RATES[level], seed=83000 + i)
                zp = K.affix_z(B.docs79(Pp), 'pre').get(Xp)
                r['P_planted_prefix'] = dict(z=zp, passed=(zp or 0) >= 2, sign=(zp or 0) > 0)
            r['_i'] = i; r['_Y'] = Y
            fo.write(json.dumps(r, default=float) + '\n'); fo.flush()
            if i % 50 == 0: print(level, i, flush=True)


if __name__ == '__main__':
    main()
