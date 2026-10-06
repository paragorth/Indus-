#!/usr/bin/env python3
"""LA-62 cycle 2: (a) how well can we transcribe Linear A from images? Blind test on 10 documents we already have
(random draw, seed 62, from administrative documents with a lineara.xyz facsimile and >= 4 tokens). The readings in
data/la62_ckpt/blind_readings.json were written from the facsimile drawings BEFORE the corpus transcription was
looked at. Scored against the corpus: number recall/precision (integer values, multiset), token-kind counts, and sign
accuracy for every sign that was named (not '?'). Control: the same readings scored against a random other
document from the same pool (200 draws per document) = chance agreement.
(b) the new documents (cycle 1) as token skeletons with a confidence per token, built ONLY from the open prose of
Ariadne Suppl. 5 (no legible image exists). Written to data/la62_ckpt/new_docs.json.
"""
import json, os, random, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data'); CK = os.path.join(D, 'la62_ckpt')
sys.path.insert(0, HERE)
from la60_common import load_la, admin_docs


def flat(reading):
    toks = []
    for line in reading:
        for tok, c in line:
            if isinstance(tok, int):
                toks.append(('N', tok, c))
            elif tok.startswith('L:'):
                toks.append(('L', tok[2:], c))
            else:
                toks.append(('W', tok, c))
    return toks


def truth_toks(d):
    return [(t[0], t[1]) for t in d['toks'] if t[0] != 'NL']


def signs_of(toks):
    out = []
    for k, v, *_ in toks:
        if k == 'W':
            out += v.split('-')
        elif k == 'L':
            out += [re.sub(r"[\[\]'?]", '', v.split('+')[0])]
    return out


def score(blind, truth):
    bn = Counter(v for k, v, c in blind if k == 'N'); tn = Counter(v for k, v in truth if k == 'N')
    hit = sum((bn & tn).values())
    named = [s for s in signs_of(blind) if s != '?']
    ts = Counter(signs_of([(k, v) for k, v in truth]))
    nc = sum((Counter(named) & ts).values())
    kinds_b = Counter(k for k, *_ in blind); kinds_t = Counter(k for k, _ in truth)
    return dict(num_hit=hit, num_blind=sum(bn.values()), num_true=sum(tn.values()), signs_named=len(named),
                signs_named_ok=nc, signs_true=sum(ts.values()), kinds_blind=dict(kinds_b), kinds_true=dict(kinds_t),
                kind_abs_err=sum(abs(kinds_b[k] - kinds_t[k]) for k in 'WLN'))


def main():
    B = json.load(open(os.path.join(CK, 'blind_readings.json')))
    A = {d['id']: d for d in admin_docs(load_la())}
    pool = [d for d in A.values() if sum(t[0] != 'NL' for t in d['toks']) >= 4]
    rng = random.Random(62)
    rows = []; tot = Counter(); ctl = Counter()
    for i, rd in B.items():
        if i.startswith('_'):
            continue
        bl = flat(rd); tr = truth_toks(A[i])
        s = score(bl, tr)
        for k in ('num_hit', 'num_blind', 'num_true', 'signs_named', 'signs_named_ok', 'signs_true', 'kind_abs_err'):
            tot[k] += s[k]
        c = Counter()
        for _ in range(200):
            o = rng.choice(pool)
            while o['id'] == i:
                o = rng.choice(pool)
            so = score(bl, truth_toks(o))
            for k in ('num_hit', 'num_true', 'signs_named_ok', 'kind_abs_err'):
                c[k] += so[k] / 200
        for k in c:
            ctl[k] += c[k]
        rows.append((i, s, dict(c), ' '.join('%s' % v for k, v in tr)))
    out = dict(rows=rows, total=dict(tot), control=dict(ctl))
    for i, s, c, t in rows:
        print('%-9s numbers %d/%d true (%d read; control %.1f)  named signs %d ok of %d named (control %.1f), true signs %d  kind err %d (control %.1f)'
              % (i, s['num_hit'], s['num_true'], s['num_blind'], c['num_hit'], s['signs_named_ok'], s['signs_named'],
                 c['signs_named_ok'], s['signs_true'], s['kind_abs_err'], c['kind_abs_err']))
        print('          truth:', t[:160])
    print('TOTAL', dict(tot), 'CONTROL', {k: round(v, 1) for k, v in ctl.items()})

    # (b) new documents from prose. conf = probability the token is as written (kind and identity).
    new = {
        'KNZg58': dict(site='KN', support='ivory object', source='Ariadne Suppl. 5 (2024) 40-41, prose only',
                       faces={
                           'alpha': [['L', '*4xxVAS+S1', None, 0.6], ['L', '*4xxVAS+S2', None, 0.6], ['N', 10, '', 0.6],
                                     ['W', '*NEW', None, 0.5], ['W', '*180', None, 0.6], ['W', '*181', None, 0.6]],
                           'beta': [['W', '?-?-?-?', None, 0.7], ['L', 'HIDE', None, 0.5]],
                           'gamma': [['L', 'SUS+boar', None, 0.5], ['L', 'SUS', None, 0.4]],
                           'delta': [['L', '*4xxVAS+S3', None, 0.6], ['N', 1, 'F1', 0.4], ['L', '*4xxVAS+S4', None, 0.6],
                                     ['N', 1, 'F2F3F4F5F6', 0.4]]},
                       notes='Numbers in delta: units (value not given) and a run of six different fraction signs; '
                             'integer values here are placeholders (only presence is used). Order of faces as published.'),
        'KNZg57B': dict(site='KN', support='ivory object', source='Ariadne Suppl. 5 (2024) 39',
                        faces={'B': [['W', 'g1', None, 0.8], ['W', 'g2', None, 0.8], ['W', 'g3', None, 0.8], ['W', 'g4', None, 0.8],
                                     ['W', 'g5', None, 0.8], ['W', 'g6', None, 0.8], ['L', 'GRA', None, 0.8], ['L', 'FAR?', None, 0.4],
                                     ['L', 'OLIV', None, 0.6]]}, notes='"six groups ... followed by three logograms: gra, probably far and oliv", read left to right'),
        'KNZg57C': dict(site='KN', support='ivory object', source='Ariadne Suppl. 5 (2024) 39-40',
                        faces={'C': [['W', 'g%d' % k, None, 0.6] for k in range(1, 10)] + [['L', 'TELA', None, 0.6]] * 4 +
                               [['L', 'TELA+KA', None, 0.6]]}, notes='order of groups vs TELA unknown'),
        'KNZg57D': dict(site='KN', support='ivory object', source='Ariadne Suppl. 5 (2024) 40',
                        faces={'D': [['L', 'HIDE', None, 0.5]] * 8 + [['L', 'HIDE+KA/QE/KO', None, 0.3]]}),
        'KNZg57A': dict(site='KN', support='ivory object', source='Ariadne Suppl. 5 (2024) 36-38',
                        faces={'A': [['L', 'ANIMAL', None, 0.7]] * 12 + [['L', 'VAS-amphora', None, 0.7]] * 6 +
                               [['L', 'VAS-rhyton(A664)', None, 0.7], ['L', 'VAS-tripod', None, 0.7], ['L', 'VAS+PA', None, 0.6],
                                ['L', 'VAS+RU', None, 0.6]]}),
    }
    json.dump(dict(blind=out, new_docs=new), open(os.path.join(CK, 'c2.json'), 'w'), indent=1, ensure_ascii=False)
    json.dump(new, open(os.path.join(CK, 'new_docs.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
