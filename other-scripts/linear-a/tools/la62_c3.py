#!/usr/bin/env python3
"""LA-62 cycle 3: score the frozen predictions (la60 sha256 538ad6fa..., plus la46/la51/la53/la54 frozen items) on
the documents outside the corpus (cycle 1), using the cycle-2 skeletons, and decode them with the frozen reading.

Decoding: la60 decoder with the frozen PRIOR reading (hash c47f937b...), acceptor trained on all 387 administrative
documents. Controls: EMPTY reading, 200 role-shuffled readings, 100 random readings (la60_c2 functions).
'Explained' = share of tokens that get a lexicon role (TOT, RES, TRX, HDR, COM, CENT) or an entry slot (ENT1);
'UNK' tokens are unexplained.
Commodity profile test: does the frozen Knossos commodity profile (top classes) cover the new Knossos documents'
commodity classes better than the frozen profiles of the other sites (control)?
"""
import hashlib, json, math, os, random, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data'); CK = os.path.join(D, 'la62_ckpt')
sys.path.insert(0, HERE)
from la60_common import load_la, admin_docs, prior_reading, reading_hash, shuffle_reading, base_of, LIB_SUPPORTS
from la60_decode import decode, acceptor, roles_doc
import la60_c2 as C2

FROZEN = os.path.join(D, 'la60_ckpt', 'c3_frozen_predictions.json')
SHA = '538ad6fa64360eb8178f65a54ea78aa15d9dcee3532eb5181a704ec3063c50fb'


def vclass(x):
    """Commodity class of a logogram name: vessel logograms (*4xx, VAS) -> VAS, else its base."""
    b = base_of(x)
    if b.startswith('*4') or 'VAS' in b:
        return 'VAS'
    if b.startswith('SUS'):
        return 'SUS'
    if b.startswith('HIDE'):
        return 'HIDE'
    return b.rstrip('?')


def to_doc(i, nd):
    toks = []
    for face, ts in nd['faces'].items():
        if toks:
            toks.append(['NL', None, None])
        for k, v, fr, c in ts:
            toks.append([k, v, fr if k == 'N' else None])
    return dict(id=i, tab=i, site=nd['site'], support=nd['support'], pub='new', scribe='', lib=nd['support'] in LIB_SUPPORTS,
                toks=toks)


def explained(d, R):
    rs = [r for r in roles_doc(d, R) if r not in ('N', 'NL')]
    ok = sum(1 for r in rs if r in ('TOT', 'RES', 'TRX', 'HDR', 'COM', 'CENT', 'ENT1', 'ENT'))
    lex = sum(1 for r in rs if r in ('TOT', 'RES', 'TRX', 'HDR', 'CENT') or (r == 'COM'))
    return ok, lex, len(rs), rs


def main():
    h = hashlib.sha256(open(FROZEN, 'rb').read()).hexdigest()
    assert h == SHA, h
    F = json.load(open(FROZEN))
    new = json.load(open(os.path.join(CK, 'new_docs.json')))
    docs = [to_doc(i, nd) for i, nd in new.items()]
    A = admin_docs(load_la())
    RP = prior_reading(); assert reading_hash(RP).startswith('c47f937b')
    EMPTY = dict(name='EMPTY', roles={}, order={}, lib=set(), site_default={}, ent_slot_unknown=True)
    rng = random.Random(6200)
    out = dict(frozen_sha=h, n_train=len(A))
    print('frozen sha ok; train docs', len(A))
    print('frozen rule check: ivory support counted as non-administrative by la60 LIB_SUPPORTS:',
          {d['id']: d['lib'] for d in docs})
    # ---------------- decoding
    dec = {}
    for R in (RP, EMPTY):
        acc = acceptor(A, R)
        for d in docs:
            ch, g = decode(d, R, acc)
            ok, lex, n, rs = explained(d, R)
            dec[(R['name'], d['id'])] = dict(full=ch['full'], cover=ch['cover'], roles=ch['roles'], n_tot=ch['n_tot'],
                                             agree=ch['n_order_agree'], viol=ch['n_order_viol'], explained=ok, lexicon=lex,
                                             n=n, roles_seq=rs, gloss=g, bad=ch['bad_steps'])
            print(R['name'], d['id'], 'FULL' if ch['full'] else '-', 'explained %d/%d lexicon %d' % (ok, n, lex),
                  'order +%d -%d' % (ch['n_order_agree'], ch['n_order_viol']), 'bad steps', ch['bad_steps'])
            print('   ', g)
    out['decode'] = {'%s|%s' % k: v for k, v in dec.items()}
    # shuffled / random controls for the explained share and FULL
    ctl = Counter(); NS = 200
    for s in range(NS):
        for kind in ('shuf', 'rand'):
            if kind == 'rand' and s >= NS // 2:
                continue
            R = shuffle_reading(RP, rng) if kind == 'shuf' else C2.random_reading(A, RP, rng)
            acc = acceptor(A, R)
            for d in docs:
                ch, _ = decode(d, R, acc)
                ok, lex, n, _ = explained(d, R)
                ctl[(kind, d['id'], 'ok')] += ok; ctl[(kind, d['id'], 'lex')] += lex; ctl[(kind, d['id'], 'full')] += ch['full']
                ctl[(kind, d['id'], 'ge')] += ok >= dec[('PRIOR', d['id'])]['explained']
    for d in docs:
        for kind, m in (('shuf', NS), ('rand', NS // 2)):
            print('control %s %s: explained mean %.2f, lexicon mean %.2f, FULL %.2f, P(explained >= PRIOR) %.2f'
                  % (kind, d['id'], ctl[(kind, d['id'], 'ok')] / m, ctl[(kind, d['id'], 'lex')] / m,
                     ctl[(kind, d['id'], 'full')] / m, ctl[(kind, d['id'], 'ge')] / m))
    out['controls'] = {'|'.join(k): v for k, v in ctl.items()}
    # ---------------- commodity-profile test (frozen per-site top classes)
    obs = Counter()
    for d in docs:
        for t in d['toks']:
            if t[0] == 'L':
                c = vclass(t[1])
                if c not in ('ANIMAL', 'FAR'):
                    obs[c] = 1
    prof = {}
    for s, v in F['per_site'].items():
        prof[s] = {vclass(k) for k in v['commodity_share']}
    cov = {s: sorted(obs.keys() & p) for s, p in prof.items()}
    print('observed commodity classes on new KN documents', sorted(obs))
    for s in prof:
        print('  frozen %s profile %s covers %d: %s' % (s, sorted(prof[s]), len(cov[s]), cov[s]))
    out['profile'] = dict(observed=sorted(obs), coverage={s: cov[s] for s in cov})
    # ---------------- order pairs
    pairs = F['commodity_pairs']
    o57 = ['GRA', 'OLIV']
    k = 'GRA before OLIV'
    print('order: KN Zg 57 face B writes GRA ... OLIV; frozen', k, pairs[k], '-> agree; log2 LR vs coin',
          round(math.log2(pairs[k]['p'] / 0.5), 2))
    out['order'] = dict(pair=k, frozen=pairs[k], observed='agree')
    json.dump(out, open(os.path.join(CK, 'c3.json'), 'w'), indent=1, ensure_ascii=False, default=str)


if __name__ == '__main__':
    main()
