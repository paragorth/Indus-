#!/usr/bin/env python3
"""LA-84 cycle 2: THE SQUEEZE.  Do scribes short of room shorten sign-groups?

Physical lines: la81 layer (data/la81_ckpt/phys.json, lineara.xyz transcription aligned by sign identity).
Occurrence = a sign-group of >= 2 signs on one physical line (groups cut across lines are left out), read status
from corpus_ra (read, no damage flag; matched in order by sign tuple).
Pressure features: last item on its line, last group on its line, line fill (signs on the line / the document's
fullest line), fill relative to the document's median line, last physical line, last two lines.
Short-form rules vs the other documents' vocabulary: P1 a longer attested group starts with it (final signs
dropped), P1x exactly one dropped, S1 a longer one ends with it, I1 one internal sign dropped, ANY.
Random guessing: H random (pressure definition x rule) hypotheses; score = rate(pressure) - rate(rest), z against
within-document permutation of the pressure labels. Fit on documents published <= 1950, test on the rest;
leave-HT-out as a second outside test. Planted: final sign dropped with prob 0.4 at line-final groups on full lines.
Usage: la84_c2.py run [planted]
"""
import sys, os, json, random, collections, re
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la84_common as C
import la78_common as L78

P = json.load(open(os.path.join(C.DATA, 'la81_ckpt', 'phys.json')))
CORP = {d['id']: d for d in json.load(open(os.path.join(C.DATA, 'corpus_ra.json')))}
YEAR = {d['id']: d['year'] for d in L78.load()}


def lex_build():
    lex = collections.defaultdict(set)
    for d in CORP.values():
        for t in d['tokens']:
            if t['t'] == 'word' and t.get('st') in ('read', 'damaged'):
                lex[tuple(C.norm(x) for x in t['s'])].add(d['id'])
    return lex


def occurrences():
    occ = []
    for did, v in P.items():
        if did not in CORP:
            continue
        cw = [(tuple(C.norm(x) for x in t['s']), t) for t in CORP[did]['tokens'] if t['t'] == 'word']
        recs = [r for r in v['recs'] if r['ok'] and r['lines']]
        if not recs:
            continue
        fill = collections.Counter()
        for r in recs:
            for ln in r['lines']:
                fill[ln] += 1
        mx = max(fill.values()); med = float(np.median(list(fill.values())))
        lines = sorted(fill)
        last_item = {}
        last_grp = {}
        for i, r in enumerate(recs):
            ln = r['lines'][-1]
            last_item[ln] = i
            if '-' in r['w']:
                last_grp[ln] = i
        ci = 0
        for i, r in enumerate(recs):
            if '-' not in r['w'] or '+' in r['w']:
                continue
            s = tuple(C.norm(x) for x in r['w'].split('-'))
            # match to corpus word token in order
            j = ci
            while j < len(cw) and cw[j][0] != s:
                j += 1
            if j == len(cw):
                continue
            tok = cw[j][1]; ci = j + 1
            if len(set(r['lines'])) > 1:
                continue
            fl = set(tok.get('fl', [])) - {'cont'}
            if tok.get('st') != 'read' or fl:
                continue
            ln = r['lines'][0]
            occ.append(dict(doc=did, s=s, ln=ln,
                            f=dict(last_item=int(last_item.get(ln) == i), last_grp=int(last_grp.get(ln) == i),
                                   fill=fill[ln] / mx, rel=fill[ln] / max(med, 1),
                                   last_line=int(ln == lines[-1]), last2=int(ln in lines[-2:]),
                                   nlines=len(lines)),
                            ht=CORP[did]['site'] == 'Haghia Triada', year=YEAR.get(did, 2000)))
    return occ


def rules_for(s, doc, lex, byprefix, bysuffix):
    """which short-form rules hold for group s (vs other documents)"""
    def att(v):
        ds = lex.get(v)
        return bool(ds) and (len(ds) - (doc in ds)) > 0
    r = dict(P1=0, P1x=0, S1=0, I1=0)
    for v in byprefix.get(s[0], ()):
        if len(v) <= len(s) or not att(v):
            continue
        if v[:len(s)] == s:
            r['P1'] = 1
            if len(v) == len(s) + 1:
                r['P1x'] = 1
        if v[-len(s):] == s:
            r['S1'] = 1
        if len(v) == len(s) + 1 and any(v[:k] + v[k + 1:] == s for k in range(1, len(v) - 1)):
            r['I1'] = 1
    for v in bysuffix.get(s[-1], ()):     # suffix matches (initial signs dropped)
        if len(v) > len(s) and v[-len(s):] == s and att(v):
            r['S1'] = 1
            break
    r['ANY'] = int(any(r.values()))
    return r


PFEAT = ['last_item', 'last_grp', 'fill', 'rel', 'last_line', 'last2']


def rand_pressure(rng):
    k = rng.choice([1, 1, 2])
    terms = []
    for _ in range(k):
        f = rng.choice(PFEAT)
        if f in ('fill',):
            terms.append((f, rng.choice([0.6, 0.7, 0.8, 0.9, 1.0])))
        elif f == 'rel':
            terms.append((f, rng.choice([1.0, 1.2, 1.5, 2.0])))
        else:
            terms.append((f, 1))
    if rng.random() < 0.5:
        terms.append(('len', rng.choice([3, 4])))
    return terms


def press(o, terms):
    return int(all(o['f'][f] >= t for f, t in terms))


def score(occ, terms, rule, rng, nperm=200):
    x = np.array([press(o, terms) for o in occ]); y = np.array([o['r'][rule] for o in occ])
    if x.sum() < 8 or (1 - x).sum() < 8:
        return None
    obs = y[x == 1].mean() - y[x == 0].mean()
    docs = collections.defaultdict(list)
    for i, o in enumerate(occ):
        docs[o['doc']].append(i)
    blocks = [np.array(v) for v in docs.values() if len(v) > 1]
    nl = []
    xp = x.copy()
    for _ in range(nperm):
        for b in blocks:
            xp[b] = x[b][rng.permutation(len(b))]
        if xp.sum() == 0 or xp.sum() == len(xp):
            continue
        nl.append(y[xp == 1].mean() - y[xp == 0].mean())
    nl = np.array(nl)
    return float(obs), float((obs - nl.mean()) / (nl.std() + 1e-9)), int(x.sum())


def prepare(planted=False, seed=0):
    lex = lex_build()
    occ = occurrences()
    rng = random.Random(seed)
    if planted:
        # plant: final sign dropped with prob 0.4 for groups of >= 3 signs that are last group on a line at fill >= 0.8
        for o in occ:
            if o['f']['last_grp'] and o['f']['fill'] >= 0.8 and len(o['s']) >= 3 and rng.random() < 0.4:
                o['s'] = o['s'][:-1]
                lex[o['s']].add(o['doc'])
    byprefix = collections.defaultdict(set)
    bysuffix = collections.defaultdict(set)
    for v in lex:
        byprefix[v[0]].add(v); bysuffix[v[-1]].add(v)
    for o in occ:
        o['f']['len'] = len(o['s'])
        o['r'] = rules_for(o['s'], o['doc'], lex, byprefix, bysuffix)
    return occ


RULES = ['P1', 'P1x', 'S1', 'I1', 'ANY']


def job(args):
    seed, n, planted = args
    occ = prepare(planted)
    tr = [o for o in occ if o['year'] <= 1950]
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        terms = rand_pressure(rng); rule = rng.choice(RULES)
        s = score(tr, terms, rule, nrng, 100)
        if s:
            out.append(dict(terms=terms, rule=rule, d=s[0], z=s[1], n=s[2]))
    return out


if __name__ == '__main__':
    planted = len(sys.argv) > 2 and sys.argv[2] == 'planted'
    tag = 'planted' if planted else 'real'
    H = int(os.environ.get('LA84_H', '3000'))
    with Pool(2) as pool:
        res = sum(pool.map(job, [(84000 + k, H // 10, planted) for k in range(10)]), [])
    occ = prepare(planted)
    tr = [o for o in occ if o['year'] <= 1950]; te = [o for o in occ if o['year'] > 1950]
    nh = [o for o in occ if not o['ht']]
    # unique hypotheses, survivors = top 25 by z with d > 0 (the squeeze direction: shorter under pressure)
    seen = {}
    for r in res:
        k = (json.dumps(r['terms']), r['rule'])
        seen[k] = r
    U = sorted(seen.values(), key=lambda r: -r['z'])
    surv = [r for r in U if r['d'] > 0][:25]
    rng = np.random.default_rng(7)
    for r in surv:
        a = score(te, r['terms'], r['rule'], rng, 500)
        b = score(nh, r['terms'], r['rule'], rng, 500)
        r['test'] = a; r['nonHT'] = b
    # baseline: random 25 hypotheses from the bank, same tests (what a non-selected hypothesis does out of sample)
    rr = random.Random(5); base = rr.sample(U, min(25, len(U)))
    for r in base:
        r['test'] = score(te, r['terms'], r['rule'], rng, 300)
    out = dict(tag=tag, n_occ=len(occ), n_train=len(tr), n_test=len(te), n_nonHT=len(nh), n_hyp=len(U),
               n_z3=sum(1 for r in U if r['z'] >= 3), n_zm3=sum(1 for r in U if r['z'] <= -3),
               rule_base={k: float(np.mean([o['r'][k] for o in occ])) for k in RULES},
               survivors=surv, base=base)
    json.dump(out, open(os.path.join(C.CK, 'c2_%s.json' % tag), 'w'), indent=1)
    def summ(rs, key):
        v = [r[key] for r in rs if r.get(key)]
        return dict(n=len(v), pos=sum(1 for x in v if x[0] > 0), meanz=float(np.mean([x[1] for x in v])) if v else None)
    print(json.dumps(dict(tag=tag, n_occ=len(occ), n_train=len(tr), n_test=len(te), n_nonHT=len(nh), n_hyp=len(U),
                          n_z3=out['n_z3'], n_zm3=out['n_zm3'], rule_base=out['rule_base'],
                          surv_test=summ(surv, 'test'), surv_nonHT=summ(surv, 'nonHT'), base_test=summ(base, 'test'),
                          top=[(r['terms'], r['rule'], round(r['d'], 3), round(r['z'], 2), r['n'], r['test']) for r in surv[:8]]),
                     indent=1))
