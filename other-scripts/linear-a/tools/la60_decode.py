#!/usr/bin/env python3
"""LA-60 decoder: a frozen reading R + a role acceptor learned on training documents turn a
held-out document into a structured gloss and a set of checks.

Checks per document
  cover   every word/logogram gets a role: lexicon role, or an unknown word in an entry slot
          (directly before a number, or before a logogram/commodity sign + number)
  roles   every role-to-role step (line breaks skipped) was seen in training under the same reading
  totals  every TOT with >= 2 entries in its section closes (fraction slack, la60_common.close_test)
  order   no two commodities written against R's order (margin 0.3 on the Bradley-Terry scale)
  substantive  at least one total checked, or a commodity entry with a number
FULL = cover & roles & totals & order & substantive.
"""
from collections import Counter
from la60_common import role_of, base_of, close_test, sections, _num_after

MARGIN = 0.3


def roles_doc(d, R):
    toks = d['toks']; out = []
    for i, t in enumerate(toks):
        if t[0] in ('N', 'NL'):
            out.append(t[0]); continue
        r = role_of(t, R)
        if t[0] == 'W' and t[1] not in R['roles']:
            # unknown word: entry name only in an entry slot
            j = i + 1
            while j < len(toks) and toks[j][0] == 'NL':
                j += 1
            nxt = toks[j] if j < len(toks) else None
            nxt2 = toks[j + 1] if j + 1 < len(toks) else None
            slot = nxt is not None and (nxt[0] == 'N' or (nxt[0] in ('L', 'W') and role_of(nxt, R) in ('COM', 'CENT')
                                                          and nxt2 is not None and nxt2[0] == 'N'))
            if t[1] in R.get('lib', ()):
                r = 'LIB'
            elif slot:
                r = 'ENT' if r == 'ENT' else 'ENT1'
            else:
                r = 'UNK'
        out.append(r)
    return out


def acceptor(train, R):
    c = Counter()
    for d in train:
        rs = [r for r in roles_doc(d, R) if r != 'NL']
        rs = ['<s>'] + rs + ['</s>']
        for a, b in zip(rs, rs[1:]):
            c[(a, b)] += 1
    return c


def decode(d, R, acc, lb=False):
    toks = d['toks']; rs = roles_doc(d, R)
    ch = {}
    ch['cover'] = 'UNK' not in rs and 'LIB' not in rs
    seq = ['<s>'] + [r for r in rs if r != 'NL'] + ['</s>']
    bad = [(a, b) for a, b in zip(seq, seq[1:]) if acc[(a, b)] == 0]
    ch['roles'] = not bad
    secs = sections(d, R['roles'], lb=lb)
    tested = [s for s in secs if s[3] is not None]
    ch['n_tot'] = len(tested); ch['n_close'] = sum(1 for s in tested if s[3])
    ch['totals'] = all(s[3] for s in tested)
    coms = []
    for t, r in zip(toks, rs):
        if r == 'COM' and t[0] in ('L', 'W'):
            b = base_of(t[1])
            if b not in coms:
                coms.append(b)
    o = R['order']; viol = 0; agree = 0
    for i in range(len(coms)):
        for j in range(i + 1, len(coms)):
            a, b = coms[i], coms[j]
            if a in o and b in o:
                if o[a] > o[b] + MARGIN:
                    viol += 1
                elif o[a] < o[b] - MARGIN:
                    agree += 1
    ch['order'] = viol == 0; ch['n_order_agree'] = agree; ch['n_order_viol'] = viol
    com_entry = any(r == 'COM' and k + 1 < len(toks) and toks[k + 1][0] == 'N' for k, r in enumerate(rs))
    ch['substantive'] = bool(tested) or com_entry or agree > 0
    ch['full'] = ch['cover'] and ch['roles'] and ch['totals'] and ch['order'] and ch['substantive']
    ch['bad_steps'] = bad[:3]
    return ch, gloss(d, R, rs, secs, coms)


def fmt_num(t):
    return str(t[1]) + (' ' + t[2] if t[2] else '')


ROLE_WORD = {'HDR': 'heading sign', 'TRX': 'heading/transaction', 'TOT': 'TOTAL', 'RES': 'residue/sub-heading',
             'COM': 'commodity', 'CENT': 'compound entry', 'ENT': 'entry', 'ENT1': 'entry', 'SGL': 'single sign',
             'UNK': '??', 'LIB': 'libation-register word'}


def gloss(d, R, rs, secs, coms):
    toks = d['toks']; parts = []; i = 0
    head = []
    while i < len(toks) and rs[i] in ('HDR', 'TRX', 'UNK', 'NL', 'RES') and not (i + 1 < len(toks) and toks[i + 1][0] == 'N'):
        if toks[i][0] != 'NL':
            head.append('%s [%s]' % (toks[i][1], ROLE_WORD[rs[i]]))
        i += 1
    out = d['id'] + ': '
    if head:
        out += 'header ' + ', '.join(head) + '; '
    cur = []
    tot_at = {s[0]: s for s in secs}
    pending = None
    while i < len(toks):
        t = toks[i]; r = rs[i]
        if t[0] == 'NL':
            i += 1; continue
        if t[0] == 'N':
            cur.append(fmt_num(t)); i += 1; continue
        if r == 'TOT':
            j = _num_after(toks, i)
            s = tot_at.get(i)
            if j is not None:
                tag = ''
                if s is not None and s[3] is not None:
                    tag = ' (sum of entries %d%s: %s)' % (sum(e[1] for e in s[2]),
                                                          '+fr' if any(e[2] for e in s[2]) else '', 'closes' if s[3] else 'does NOT close')
                cur.append('| %s=TOTAL %s%s |' % (t[1], fmt_num(toks[j]), tag))
                i = j + 1; continue
        lab = t[1] if r in ('ENT', 'ENT1', 'SGL') else '%s[%s]' % (t[1], ROLE_WORD.get(r, r))
        cur.append(lab); i += 1
    out += ' '.join(cur)
    if coms:
        out += ' || commodities in order: ' + ' > '.join(coms)
    elif R['site_default'].get(d['site']):
        out += ' || no commodity written; site default %s' % R['site_default'][d['site']]
    return out
