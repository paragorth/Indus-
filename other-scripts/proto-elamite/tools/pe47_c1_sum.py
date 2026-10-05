"""Summarise pe47 cycle 1: held-out gains vs nulls, role recovery (PLANT, Ur III), PE role read-off."""
import collections, json, re, sys
import numpy as np

R = json.load(open(sys.argv[1]))
ROLE = 'XUCP'
COMM = re.compile(r'^(udu|masz2|gu4|sila4|u8|ud5|kir11|amar|masz2-gal|sze|gurusz|geme2|dumu|ninda|kasz|i3|zi3|tug2|gi|gesz|sa|dug|ab2|anse|anszi|dur3|eme6|munus|niga|sig5|u2|sag|kusz|siki|ku6|mun|gu2|esir2|szum2|zu2-lum|ma-na|sila3|gur|uruda|ku3|masz|ga|ezem|sze-ba|sa2-du11|gu4-niga|udu-niga|gukkal|ur-gir15|dara4|lulim|ganam4|munus)$')


def truth(name, w):
    if name == 'PLANT':
        return {'U': 'U', 'C': 'C', 'P': 'P'}.get(w[0], 'X')
    if name.startswith('UR3'):
        if w.startswith('iti:'):
            return 'P'
        if COMM.match(w.split('{')[0]):
            return 'C'
        return 'other'
    return None


rows = collections.defaultdict(dict)
for d in R:
    rows[d['name']][(d['null'], d['seed'])] = d
for name, D in rows.items():
    real = [D[k]['heldout_best'] for k in D if k[0] == 'real']
    print('==', name)
    for k in sorted(D):
        d = D[k]
        print('  %-5s s%d held %.4f top4 %s abl %s rand_best_inner %.4f' % (k[0], k[1], d['heldout_best'],
              [round(x, 3) for x in d['heldout']], {a: round(b, 4) for a, b in d['ablation'].items()}, d['rand_best_inner']))
    for k in sorted(D):
        if k[0] != 'real':
            continue
        d = D[k]
        cand = d['cand']
        if truth(name, cand[0]) is not None:
            ct = collections.Counter()
            for rho in d['rhos'][:1]:
                for w, r_ in zip(cand, rho):
                    ct[(truth(name, w), ROLE[r_])] += 1
            cls = sorted({a for a, _ in ct})
            print('  role table (truth x assigned)', k, {c: {r_: ct[(c, r_)] for r_ in ROLE} for c in cls})
    if name == 'PE':
        votes = collections.defaultdict(collections.Counter)
        for k in D:
            if k[0] != 'real':
                continue
            d = D[k]
            for rho in d['rhos']:
                for w, r_ in zip(d['cand'], rho):
                    votes[w][ROLE[r_]] += 1
        stab = sorted(((max(v.values()) / sum(v.values()), w, v.most_common(1)[0][0]) for w, v in votes.items()), reverse=True)
        print('  PE most stable non-X roles:', [(w, r_, round(s, 2)) for s, w, r_ in stab if r_ != 'X'][:25])
