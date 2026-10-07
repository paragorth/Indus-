#!/usr/bin/env python3
"""la83 cycle 1 controls for the corrected parser.
 (a) identity: every record re-parsed by la83_parse (rules on, SigLA flags carried) must give the v1 tokens
     whenever no rule fires -> mismatch count.
 (b) planted defects: 300 random clean records get one defect of a known kind (truncated transliteration list,
     number glyph inside a word, transliteration replaced by its Unicode, ASCII subscript, wrong site).  v1 (la71
     rules) and v2 (la83 rules) parse the damaged record; score = tokens (key + status) equal to the clean parse.
     Shuffled control: the same defects applied, but v2 rules run with a majority map built from shuffled
     sign codes (rebuild rules must then fail)."""
import json, os, sys, random, copy, re
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import build_corpus as bc
import la71_parse as P
import la83_parse as Q

raw = bc.load_raw()
v1 = {d['id']: d for d in json.load(open(os.path.join(HERE, '..', 'data', 'corpus_ra.json')))}
maj = Q.majority(raw)


def keys(toks):
    return [(Q.tkey(t), t.get('st')) for t in toks if t['t'] not in ('nl', 'div')]


def v2_tokens(k, rec, mp, site_only=False):
    pairs, site, log = Q.fix_record(k, rec, mp, RAWV, use_editor=False)
    return Q.finish(Q.parse_pairs(pairs), v1[k]['tokens']), site, log


def v1_tokens(k, rec):
    toks = P.parse_doc(rec)
    for t in toks: t.pop('_codes', None)
    return Q.finish(toks, v1[k]['tokens'])


RAWV = raw
# (a) identity
mis = 0; n = 0
for k, rec in raw.items():
    pairs, site, log = Q.fix_record(k, rec, maj, raw, use_editor=False)
    if log: continue
    n += 1
    if keys(Q.finish(Q.parse_pairs(pairs), v1[k]['tokens'])) != keys(v1[k]['tokens']): mis += 1
print('identity: %d of %d untouched records differ' % (mis, n))

# (b) planted defects
rng = random.Random(83)
clean = [k for k, rec in raw.items() if k in v1 and not Q.fix_record(k, rec, maj, raw, False)[2]
         and any(bc.classify(w, t) and bc.classify(w, t)[0]['t'] == 'word' and '-' in t for w, t in zip(rec['words'], rec['transliteratedWords']))]
NUMS = [chr(0x10107 + i) for i in range(9)]
shuf_codes = list(maj.values()); rng.shuffle(shuf_codes); maj_shuf = dict(zip(maj.keys(), shuf_codes))
res = Counter()
for it in range(300):
    k = rng.choice(clean); rec = copy.deepcopy(raw[k]); kind = ['T1', 'R2', 'R3', 'R5', 'R8'][it % 5]
    W, T = rec['words'], rec['transliteratedWords']
    wi = [i for i, (w, t) in enumerate(zip(W, T)) if '-' in t and '+' not in t and all(Q.ukind(c) in ('sign', 'gap') for c in w)]
    if not wi: continue
    i = rng.choice(wi)
    if kind == 'T1':
        rec['transliteratedWords'] = T[:i]
    elif kind == 'R2':
        ch = rng.choice(NUMS); w = W[i]; t = T[i].split('-'); j = rng.randrange(1, len(t))
        sg = [c for c in w if Q.ukind(c) == 'sign']
        if len(sg) != len(t): continue
        # insert a number between sign j-1 and j (the clean target is word | num | word)
        pos = [p for p, c in enumerate(w) if Q.ukind(c) == 'sign'][j]
        rec['words'][i] = w[:pos] + ch + w[pos:]
        rec['transliteratedWords'][i] = '-'.join(t[:j] + [ch] + t[j:])
        tgt = copy.deepcopy(raw[k]); tgt['words'] = W[:i] + [w[:pos], ch, w[pos:]] + W[i + 1:]
        tgt['transliteratedWords'] = T[:i] + ['-'.join(t[:j]), str(bc.num_value(ch)), '-'.join(t[j:])] + T[i + 1:]
    elif kind == 'R3':
        rec['transliteratedWords'][i] = W[i]
    elif kind == 'R5':
        t = T[i].split('-')
        sub = [j for j, c in enumerate(t) if re.fullmatch(r'[A-Z]{1,2}[₂₃]', c)]
        if not sub: continue
        j = sub[0]; t[j] = t[j][:-1] + {'₂': '2', '₃': '3'}[t[j][-1]]
        rec['transliteratedWords'][i] = '-'.join(t)
    elif kind == 'R8':
        rec['site'] = 'Nowhere'
    target = keys(v1[k]['tokens']) if kind != 'R2' else keys(Q.finish(Q.parse_pairs([(a, b, set()) for a, b in zip(tgt['words'], tgt['transliteratedWords'])]), v1[k]['tokens']))
    RAWV = dict(raw); RAWV[k] = rec
    t2, site2, log = v2_tokens(k, rec, maj)
    if kind == 'T1':  # rebuilt tokens are flagged 'fromuni' (damaged) on purpose: compare readings only
        target = [x[0] for x in target]; kk = lambda t: [x[0] for x in keys(t)]
    else:
        kk = keys
    a = kk(v1_tokens(k, rec)) == target
    b = kk(t2) == target and (kind != 'R8' or site2 == raw[k]['site'])
    t3, site3, _ = v2_tokens(k, rec, maj_shuf)
    c = kk(t3) == target and (kind != 'R8' or site3 == raw[k]['site'])
    if kind == 'R8': a = rec['site'] == raw[k]['site']
    res[(kind, 'n')] += 1; res[(kind, 'v1_ok')] += a; res[(kind, 'v2_ok')] += b; res[(kind, 'shuf_ok')] += c
    res[(kind, 'fired')] += bool(log)
    RAWV = raw
for kind in ['T1', 'R2', 'R3', 'R5', 'R8']:
    print('%s planted %3d  rule fired %3d  v1 recovers %3d  v2 recovers %3d  v2 with shuffled sign map %3d'
          % (kind, res[(kind, 'n')], res[(kind, 'fired')], res[(kind, 'v1_ok')], res[(kind, 'v2_ok')], res[(kind, 'shuf_ok')]))
