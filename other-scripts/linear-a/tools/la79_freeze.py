"""LA-79 freeze: outside-corpus predictions for RILA-S1, Anetaki II and any post-2020 Linear A find."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la79_common import load, CK, DATA, sha

D = load()
lot = json.load(open(os.path.join(CK, 'c3_lottery.json')))
R, F = lot['rising'], lot['falling']


def shares(docs):
    n = r = f = 0
    for d in docs:
        for w in d['words']:
            for s in w['s']:
                n += 1; r += s in R; f += s in F
    return dict(n=n, rising=round(r / max(n, 1), 4), falling=round(f / max(n, 1), 4))


cur = shares(D)
tab = {k: shares([d for d in D if d['sup'] == 'tablet' and (d['year'] <= 1950) == (k == 'past')]) for k in ('past', 'later')}
nonht = shares([d for d in D if d['g'] != 'HT'])
fb = json.load(open(os.path.join(CK, 'c3_fb.json')))
obj = dict(
    loop='la79', date='2026-10-07',
    P1=dict(what='sign drift: share of the 15 rising vs 15 falling signs (lottery consensus) in a new find set',
            rising=R, falling=F, current_corpus=cur, current_non_HT=nonht, tablets_past_vs_later=tab,
            rule='on >= 200 new sign tokens from outside Hagia Triada: rising share > %.3f and falling share < %.3f '
                 '(current corpus); support if both hold, kill if both fail' % (cur['rising'], cur['falling'])),
    P2=dict(what='survival of random role readings', rule='la79_c2 generator, 20,000 hypotheses fitted on all current documents, '
            'scored on >= 50 new clean sign-groups (prior mode): share with gain > 0 <= 0.20 supports drift-like behaviour; >= 0.30 kills it'),
    P3=dict(what='backward > forward transfer', rule='random readings (la79_c2 generator, 1,500) taught by the non-HT documents gain on a new '
            'non-HT archive more often than those taught by the HT documents; support if share(nonHT-taught) - share(HT-taught) >= 0.05, '
            'kill if <= 0', real_fb=fb['real']),
    P4=dict(what='claims that passed the counterfactual-history test, with kill lines on new finds',
            items=['aggregate prefixing and suffixing: z >= 2 on >= 150 new types of >= 3 signs (kill: z < 0.5)',
                   '-ME suffix: z >= 1.5 on >= 150 new types (kill: z < 0)',
                   'Linear B shared 3+ sign words: >= 2x a position-preserving shuffle on >= 100 new types (kill: <= 1x)',
                   'commodity order (la20): >= 15 of the next 20 new pairs (kill: <= 10 of 20)']))
h = sha(obj)
obj_path = os.path.join(DATA, 'la79_frozen_predictions.json')
json.dump(obj, open(obj_path, 'w'), sort_keys=True, indent=1)
open(os.path.join(DATA, 'la79_frozen_predictions.sha256'), 'w').write(h + '  (sha256 of json.dumps(obj, sort_keys=True))\n')
print(h); print(json.dumps(dict(cur=cur, tab=tab, nonht=nonht)))
