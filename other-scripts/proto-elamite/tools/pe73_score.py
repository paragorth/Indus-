#!/usr/bin/env python3
"""pe73: score reader verdicts against the hidden plant keys.
usage: pe73_score.py vnum|knum   (answers: data/pe73_ckpt/<name>_answers.txt, lines 'idx photo copy [note]')
Planted = the shown value is wrong; a correct reader says D on planted, A on unplanted.
On key sheets, D on an UNPLANTED item = the photo/copy disagrees with the CDLI transliteration (candidate correction)."""
import json, os, sys, random
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'pe73_ckpt')
name = sys.argv[1]
Q = json.load(open(os.path.join(CK, name + '_q.json'))); K = json.load(open(os.path.join(CK, name + '_key.json')))
A = {}
for l in open(os.path.join(CK, name + '_answers.txt')):
    if l.startswith('#') or not l.strip(): continue
    p = l.split(None, 3)
    A[int(p[0])] = (p[1], p[2], p[3].strip() if len(p) > 3 else '')
out = {}
for src, j in (('photo', 0), ('copy', 1), ('either', 2)):
    tab = {(pl, v): 0 for pl in (True, False) for v in 'ADU'}
    rows = []
    for i, (q, k) in enumerate(zip(Q, K)):
        if i not in A or A[i][0] == 'X': continue
        if src == 'either':
            ph, cp = A[i][0], A[i][1]
            v = 'D' if 'D' in (ph, cp) else ('A' if 'A' in (ph, cp) else 'U')
        else:
            v = A[i][j]
        tab[(k['plant'], v)] += 1; rows.append((k['plant'], v))
    dec = [(pl, v) for pl, v in rows if v != 'U']
    n_pl = sum(1 for pl, v in dec if pl); n_un = len(dec) - n_pl
    sens = sum(1 for pl, v in dec if pl and v == 'D') / max(1, n_pl)
    spec = sum(1 for pl, v in dec if not pl and v == 'A') / max(1, n_un)
    acc = sum(1 for pl, v in dec if (v == 'D') == pl) / max(1, len(dec))
    # permutation null: shuffle verdicts over decided items
    rng = random.Random(73); vs = [v for _, v in dec]; pls = [pl for pl, _ in dec]; ge = 0
    for _ in range(5000):
        rng.shuffle(vs)
        ge += sum(1 for pl, v in zip(pls, vs) if (v == 'D') == pl) >= acc * len(dec) - 1e-9
    out[src] = dict(n=len(rows), decided=len(dec), unreadable=len(rows) - len(dec), planted_decided=n_pl,
                    sens=round(sens, 3), spec=round(spec, 3), acc=round(acc, 3), p_perm=round((ge + 1) / 5001, 4),
                    table={'%s_%s' % ('plant' if a else 'true', b): c for (a, b), c in tab.items()})
    print(name, src, out[src])
if name == 'knum':
    cand = []
    for i, (q, k) in enumerate(zip(Q, K)):
        if i in A and not k['plant'] and 'D' in A[i][:2]:
            cand.append(dict(i=i, pid=q['pid'], line='%s %s %s' % (q['surface'], q['col'], q['label']), atf=k['true'],
                             photo=A[i][0], copy=A[i][1], note=A[i][2], result=k['result']))
    out['candidates'] = cand
    out['missed_plants'] = [dict(i=i, pid=q['pid'], shown=q['shown'], true=k['true'], v=A.get(i)) for i, (q, k) in enumerate(zip(Q, K)) if k['plant'] and i in A and 'D' not in A[i][:2]]
    for c in cand: print('CANDIDATE', c)
else:
    out['errors'] = [dict(i=i, pid=q['pid'], line='%s %s %s' % (q['surface'], q['col'], q['label']), shown=q['shown'], true=k['true'], plant=k['plant'], v=A[i][:2])
                     for i, (q, k) in enumerate(zip(Q, K)) if i in A and A[i][0] != 'X' and ((k['plant'] and 'D' not in A[i][:2] and 'A' in A[i][:2]) or (not k['plant'] and 'D' in A[i][:2]))]
json.dump(out, open(os.path.join(CK, name + '_score.json'), 'w'), indent=1)
