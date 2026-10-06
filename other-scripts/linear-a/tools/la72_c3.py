#!/usr/bin/env python3
"""LA-72 cycle 3a: re-check the 37 frozen la70 predictions on the cleaned corpus.

For every prediction whose claim rests on corpus counts, recount it on all (la70's corpus), rd and read,
and list the status of the tokens it rests on (read / damaged / restored / erased).  Order pairs and KU-RO
rates are regenerated with la70_c3.predictions() on each version, with the reading re-gated on all
administrative documents by the clean procedure.  Output: data/la72_ckpt/c3_evidence.json."""
import json, os, re, random
from collections import Counter, defaultdict
import la70_common as L7
from la70_common import sections_v2, com_seq_v2, frac_parts, hash_v2
from la72_common import *
from la60_common import _num_after
from la72_c1 import build
import la70_c3

PRED = os.path.join(HERE, '..', 'loops', 'la70_predictions.txt')


def old_predictions():
    s = open(PRED).read()
    js = s.split('=====JSON=====\n')[1]
    h = hashlib.sha256(js.encode()).hexdigest()
    assert h.startswith('2d7c6c9b'), h
    return json.loads(js), h


def word_hits(A, w, after=None):
    out = []
    for d in A:
        for i, t in enumerate(d['toks']):
            if t[0] == 'W' and t[1] == w:
                nxt = d['toks'][i + 1] if i + 1 < len(d['toks']) else None
                out.append(dict(doc=d['id'], site=d['site'], i=i, st=d['st'][i],
                                next=nxt, next_st=d['st'][i + 1] if nxt else None,
                                first_line=('NL' not in [x[0] for x in d['toks'][:i]])))
    return out


def ki_detail(A):
    out = []
    for d in A:
        toks = d['toks']; nums = [i for i, t in enumerate(toks) if t[0] == 'N']
        for i, t in enumerate(toks):
            if t[0] == 'W' and t[1] == 'KI' and i + 1 < len(toks) and toks[i + 1][0] == 'N':
                prev = [k for k in nums if k < i]
                if prev and toks[prev[-1]][1] > 0:
                    p = prev[-1]
                    out.append(dict(doc=d['id'], ki_amt=toks[i + 1][1], prev_amt=toks[p][1], below=toks[i + 1][1] < toks[p][1],
                                    st_ki=d['st'][i], st_amt=d['st'][i + 1], st_prev=d['st'][p]))
    return out


def p308(A):
    out = []
    for d in A:
        toks = d['toks']
        for i, t in enumerate(toks):
            if t[0] in ('W', 'L') and base_of(t[1]) == '*308':
                j = _num_after(toks, i)
                if j is not None:
                    out.append(dict(doc=d['id'], kind=t[0], amt=toks[j][1], frac=toks[j][2], st=d['st'][i], st_amt=d['st'][j]))
    return out


def db(A):
    out = []
    for d in A:
        for i, t in enumerate(d['toks']):
            if t[0] == 'N' and t[2] and (t[2].count('D') >= 3 or t[2].count('B') >= 2):
                out.append(dict(doc=d['id'], amt=t[1], frac=t[2], st=d['st'][i]))
    return out


def tai(A):
    out = []
    for d in A:
        toks = d['toks']
        for i, t in enumerate(toks):
            if t[0] == 'W' and t[1] == 'TA-I':
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                out.append(dict(doc=d['id'], next=nxt, st=d['st'][i]))
    return out


def sara2(A, R):
    out = []
    for d in A:
        if any(t[0] == 'W' and t[1] == 'SA-RA₂' for t in d['toks']):
            out.append(dict(doc=d['id'], coms=com_seq_v2(d, R['roles']),
                            st=[s for t, s in zip(d['toks'], d['st']) if t[0] == 'L']))
    return out


def sitedef(A, R):
    c = defaultdict(Counter); nocom = Counter()
    for d in A:
        bs = {base_of(t[1]) for t in d['toks'] if t[0] == 'L' or (t[0] == 'W' and R['roles'].get(t[1]) == 'COM')}
        for b in bs:
            c[d['site']][b] += 1
        if not bs:
            nocom[d['site']] += 1
    return {s: dict(c[s].most_common(4), no_commodity_docs=nocom[s]) for s in ('KH', 'ZA', 'ARKH')}


def order_status(A, R):
    """per order pair: agreeing / against counts, and how many lists rest on a damaged logogram."""
    o = R['order']; res = {}
    for d in A:
        seq = []; dam = {}
        for t, s in zip(d['toks'], d['st']):
            if t[0] == 'L' or (t[0] == 'W' and R['roles'].get(t[1]) == 'COM'):
                b = base_of(t[1])
                if b not in seq:
                    seq.append(b); dam[b] = s != 'read'
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                a, b = seq[i], seq[j]
                if a in o and b in o and abs(o[a] - o[b]) > 0.3:
                    early, late = (a, b) if o[a] < o[b] else (b, a)
                    k = '%s<%s' % (early, late)
                    r = res.setdefault(k, Counter())
                    r['agree' if a == early else 'against'] += 1
                    if dam[a] or dam[b]:
                        r['damaged_' + ('agree' if a == early else 'against')] += 1
    return {k: dict(v) for k, v in res.items()}


def main():
    old, h = old_predictions()
    ev = {'old_hash': h}
    for ver in ('all', 'rd', 'read'):
        A = admin(ver)
        R, _ = build(A, A, 'clean', random.Random(seed('la72-c3-%s' % ver)))
        P = la70_c3.predictions(A, R)
        e = dict(regenerated={p['id']: p['claim'] for p in P}, hash=hash_v2(R))
        e['KI'] = ki_detail(A); e['308'] = p308(A); e['DB'] = db(A); e['TAI'] = tai(A)
        e['SARA2'] = sara2(A, R); e['sitedefault'] = sitedef(A, R); e['order'] = order_status(A, R)
        for w in ('*516', '*307', 'A-DU', 'PA-DE', '*28B-NU-MA-RE', 'SI-PI-KI', 'KU-RO'):
            e['hits_' + w] = word_hits(A, w)
        tabs = defaultdict(set); kt = defaultdict(set)
        for d in A:
            tabs[d['site']].add(d['tab'])
            if any(t[0] == 'W' and t[1] == 'KU-RO' for t in d['toks']):
                kt[d['site']].add(d['tab'])
        e['kuro_secs'] = {s: [(d['id'], x[1], sum(y[1] for y in x[2]), x[3]) for d in A if d['site'] == s for x in sections_v2(d, R)]
                          for s in ('HT', 'KH', 'ZA')}
        ev[ver] = e
    json.dump(ev, open(os.path.join(CK, 'c3_evidence.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    # print a comparison of regenerated claims
    for p in old['predictions']:
        c = [ev[v]['regenerated'].get(p['id'], '(not generated)') for v in ('all', 'rd', 'read')]
        if len(set(c)) > 1 or c[0] != p['claim']:
            print(p['id']); print('  old :', p['claim'])
            for v, x in zip(('all', 'rd', 'read'), c):
                print('  %-4s: %s' % (v, x))
    new = set(ev['rd']['regenerated']) - {p['id'] for p in old['predictions']}
    print('new ids on rd:', new, 'read:', set(ev['read']['regenerated']) - {p['id'] for p in old['predictions']})


if __name__ == '__main__':
    main()
