#!/usr/bin/env python3
"""LA-72 cycle 1: rebuild the la70 v2 reading on the restoration-aware corpus and freeze it by hash.

For each version (all = la70's corpus, rd = default, read = check), on the TRAINING half of the la60
primary split ('la60-main'):
  (a) la70 procedure (control): KI test on all documents (as la70 c1a did), D/B and site defaults by hand,
      then la70's train-only gate.  On 'all' this must reproduce la70's frozen hash 182fd0d6.
  (b) clean procedure: every rule re-derived on training documents only - KI survival test (train only),
      D/B values (no training amount with D >= 5 or B >= 3), site defaults (training modal commodity),
      the la70 gates for every item, staple-order positions and the *308 rule.
Gate stability over the 10 la60 splits per version.  Item-level audit: for every candidate item, its
token counts by status (read / damaged / restored / erased) and how many tokens 'rd' removes.
Freeze: V2c = procedure (b) on rd -> data/la72_ckpt/frozen_V2c.json (sha256)."""
import json, os, random, copy
from collections import Counter, defaultdict
import la70_common as L7
from la70_common import gate_v2, hash_v2, CAND, SITE_DEF, FRAC_VAL
from la72_common import *

CAND0 = copy.deepcopy(L7.CAND)


def set_ki(red):
    L7.CAND.clear(); L7.CAND.update(CAND0)
    L7.CAND['KI'] = ('RED', 'C', 'reduced amount (smaller than the entry above)') if red else ('COM', 'B', 'commodity sign')


def build(tr, A, procedure, rng):
    info = {}
    if procedure == 'la70':
        k = ki_gate(A, rng)          # la70 c1a ran on all admin documents
        info['ki'] = k
        set_ki(k['survives'])
        R = gate_v2(tr, 'V2')
    else:
        k = ki_gate(tr, rng)
        info['ki'] = k
        set_ki(k['survives'])
        R = gate_v2(tr, 'V2c')
        db = db_gate(tr); info['db'] = db
        if not db['keep']:
            R['rules'] = dict(R['rules'], DB_values=False); R['v1close'] = True
        sd, si = site_default_gate(tr, R['roles'], SITE_DEF); info['site_default'] = si
        R['site_default'] = sd
    return R, info


def summary(R):
    return dict(roles=R['roles'], order=R['order'], rules=R['rules'], site_default=R['site_default'],
                dropped=R['dropped'], ungated=R['ungated'], sha256=hash_v2(R))


def item_audit():
    """per candidate item: tokens by status in the full corpus (admin docs), and what rd/read remove."""
    raw = json.load(open(os.path.join(D, 'corpus_ra.json')))
    cnt = defaultdict(Counter)
    items = set(CAND0) | {'KI'}
    for d in raw:
        for t in d['tokens']:
            if t['t'] == 'word':
                w = '-'.join(t['s'])
                if w in items:
                    cnt[w][t['st']] += 1
            elif t['t'] == 'logo' and t['v'].split('+')[0] in ('*308', 'CYP', 'GRA', 'OLE', 'OLIV', 'VIR', 'NI', 'VIN', 'QA2'):
                cnt['L:' + t['v'].split('+')[0]][t['st']] += 1
    return {k: dict(v) for k, v in sorted(cnt.items())}


def subscripts():
    out = {}
    for ver in ('all', 'rd', 'read'):
        A = load(ver)
        out[ver] = sum(1 for d in A for t in d['toks'] if t[0] == 'W' and any(c in t[1] for c in '₂₃'))
    return out


def main():
    res = {'item_audit': item_audit(), 'subscript_words': subscripts()}
    for ver in ('all', 'rd', 'read'):
        A = admin(ver)
        tr, te = split(A, 'la60-main')
        for proc in ('la70', 'clean'):
            rng = random.Random(seed('la72-c1-%s-%s' % (ver, proc)))
            R, info = build(tr, A, proc, rng)
            s = summary(R); s.update(info=info, n_admin=len(A), n_train=len(tr), n_test=len(te))
            res['%s/%s' % (ver, proc)] = s
            print(ver, proc, 'admin', len(A), 'train', len(tr), 'hash', s['sha256'][:12], 'KI', info['ki'].get('n'),
                  round(info['ki'].get('share') or 0, 3), info['ki'].get('p'), info['ki']['survives'])
            print('   dropped', R['dropped'], 'ungated', R['ungated'], 'rules', R['rules'], 'site', R['site_default'])
            if ver == 'rd' and proc == 'clean':
                fz = dict(name='V2c', version='rd', roles=R['roles'], order=R['order'], lib=sorted(R['lib']),
                          site_default=R['site_default'], rules=R['rules'], v1close=bool(R.get('v1close')),
                          frac_values=FRAC_VAL, ungated=R['ungated'], dropped_by_train_gate=R['dropped'],
                          grades={w: L7.CAND[w][1] for w in R['roles']}, glosses={w: L7.CAND[w][2] for w in R['roles']},
                          sha256=s['sha256'], split='la60-main', train_docs=len(tr), test_docs=len(te),
                          test_ids=sorted(d['id'] for d in te), train_gates=info)
                json.dump(fz, open(os.path.join(CK, 'frozen_V2c.json'), 'w'), ensure_ascii=False, indent=1, sort_keys=True)
                print('FROZEN V2c', s['sha256'])
        # gate stability over 10 splits (clean procedure)
        stab = Counter(); kis = 0; dbk = 0; sdk = Counter()
        for k in range(10):
            tr2, _ = split(A, 'la60-c2-split%d' % k)
            R2, inf2 = build(tr2, A, 'clean', random.Random(seed('la72-c1-stab%d-%s' % (k, ver))))
            for w in R2['dropped']:
                stab[w] += 1
            kis += inf2['ki']['survives']; dbk += inf2['db']['keep']
            for s_ in R2['site_default']:
                sdk[s_] += 1
        res['%s/stability' % ver] = dict(dropped_per_split=dict(stab), ki_survives=kis, db_keep=dbk, site_default_kept=dict(sdk))
        print(ver, 'stability (of 10): dropped', dict(stab), 'KI survives', kis, 'DB kept', dbk, 'site defaults kept', dict(sdk))
    json.dump(res, open(os.path.join(CK, 'c1.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    print('subscript words', res['subscript_words'])


if __name__ == '__main__':
    main()
