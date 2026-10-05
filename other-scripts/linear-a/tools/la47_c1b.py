#!/usr/bin/env python3
"""la47 cycle 1b: where do the scribes break lines?  For every physical line break, classify the position
in the reading order: (E) entry boundary = the next item is a word and the previous item is a number;
(W) between two words; (N) between a word and its number / inside a number group; (M) inside a word
(the word wraps).  Compare with the re-flow null (same text, a donor page's line-break profile, 200 reps)
and with Linear B (DAMOS).  Output data/la47_ckpt/c1b.json"""
import json, os, random, collections
import numpy as np
import la47_common as C


def classify(pages):
    c = collections.Counter()
    for p in pages:
        it = [i for i in p['items'] if i['k'] != 'd']
        for i in it:
            if i['k'] == 'w' and i['pl2'] > i['pl']: c['M'] += 1
            if i['k'] == 'n' and i['pl2'] > i['pl']: c['Nwrap'] += 1
        for a, b in zip(it, it[1:]):
            if b['pl'] > a['pl2']:
                if b['k'] == 'w' and a['k'] == 'n': c['E'] += 1
                elif b['k'] == 'w' and a['k'] == 'w': c['W'] += 1
                elif b['k'] == 'n': c['N'] += 1
                else: c['O'] += 1
        c['entries'] += sum(1 for a, b in zip(it, it[1:]) if a['k'] == 'n' and b['k'] == 'w')
        c['pages'] += 1
    tot = sum(c[k] for k in ('E', 'W', 'N', 'M', 'Nwrap'))
    return dict(c, tot=tot, fE=c['E'] / max(1, tot), fM=(c['M'] + c['Nwrap']) / max(1, tot),
                cover=c['E'] / max(1, c['entries']))


def main():
    rng = random.Random(7)
    LA = C.la_pages(supports=('Tablet',))
    out = {'LA': classify(LA)}
    for site in ('Haghia Triada', 'Khania', 'Zakros', 'Phaistos'):
        out['LA_' + site] = classify([p for p in LA if p['site'] == site])
    nulls = [classify(C.null_reflow(LA, rng)) for _ in range(200)]
    raw = [classify(C.null_reflow(LA, rng, snap=False)) for _ in range(50)]
    out['rawnull_fE_fM'] = [float(np.mean([x['fE'] for x in raw])), float(np.mean([x['fM'] for x in raw]))]
    for k in ('fE', 'fM', 'cover'):
        v = np.array([n[k] for n in nulls]); out['null_' + k] = [float(v.mean()), float(v.std()), float((v >= out['LA'][k]).mean()), float((v <= out['LA'][k]).mean())]
    for site in ('Haghia Triada', 'Khania'):
        S = [p for p in LA if p['site'] == site]
        v = np.array([classify(C.null_reflow(S, rng))['fE'] for _ in range(100)])
        out['null_fE_' + site] = [float(v.mean()), float(v.std())]
    LB = C.lb_pages(sites={'KN', 'PY'})
    out['LB'] = classify(LB)
    v = [classify(C.null_reflow(LB[:600], rng)) for _ in range(30)]
    out['LB_null_fE'] = [float(np.mean([x['fE'] for x in v])), float(np.std([x['fE'] for x in v]))]
    json.dump(out, open(os.path.join(C.CK, 'c1b.json'), 'w'), indent=1)
    for k, v in out.items(): print(k, v)


if __name__ == '__main__':
    main()
