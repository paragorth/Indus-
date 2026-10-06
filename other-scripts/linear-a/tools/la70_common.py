#!/usr/bin/env python3
"""LA-70 CRACK ATTEMPT v2: the la60 reading updated with what survived la61-la69 (incl. the la67 kill
sweep) and nothing that did not; frozen on training documents only (train-gated, sha256); decoded on
held-out documents against shuffled-role, random and empty readings and against la60 v1; calibrated
with a Linear B partial reading of matching kind and size.

V2 candidate items (hand list, chosen BEFORE this loop from FINDINGS.md grades; the list itself was
selected by loops that saw all documents, so the candidate list is contaminated; every fitted part and
every keep/drop gate below uses training documents only):
  TOT  KU-RO, PO-TO-KU-RO                                   A   (FINDINGS 2a)
  RES  KI-RO                                                B   (FINDINGS 2a)
  COM  logograms (default); NI KI DI RE TI MA               A / B (la45)
       *304 *308                                            B   (la8)
       OLE+U / OLE+MI / OLE+DI -> ordinary logograms (la45 'entries' KILLED by la67)
  HDR  *301 RO ZE TE                                        B   (la45/la49; KA KU SI I dropped: la28 KILLED,
                                                                 A U DA KA-NA KU-RE dropped: la40 KILLED)
       *516 *307 A-DU (heading words)                       C   (la67 1j: only words passing the decoy heading test)
  TRX  SA-RA2 KA-PA A-PA-RA-NE MA-KA-RI-TE KU-NI-SU        B   (la8, la46, la53)
  MRK  PA-DE  marked entry (own line, small amount)        C   (la53, la67 3h)
  TPL  *28B-NU-MA-RE SI-PI-KI  Zakros wine-template words  C   (la65, la67 1f)
  RED  KI  reduced amount (only if it survives la70 cycle 1) C (la66)
  rules  *308 amounts carry a fraction                     C   (la63, la67 1c)
         D = 1/5, B = 1/3 (other fraction signs unvalued)   C   (la27, la67 1h)
  order  staple order GRA < olive goods (OLE, OLIV) < NI < VIN (la50 C+, la67 2k) with CYP first and
         VIR beside OLE, QA2 with VIN (la20 A existence / B order); *307 removed (la67 KILLED)
  site defaults  KH CYP, ZA GRA, ARKH GRA                  B   (la54)
  annotations (gloss only, no decoding role): QA2, TE Villa magazines; NI, A-DU houses (la51 C+, la67 3c);
         TA-I with AROM (C, la67 1e); first-sign runs outside HT (C+, la67 2n)
Dropped as not tested or killed: *306 E SU (C, never tested), TA O JA DA *164 *411-VS (C), KU-PA (no
surviving source), KU-RE KA-NA (KILLED), *318 (KILLED), SI/NI/TA2 commodity links (KILLED), CENT role.
No Linear B sound value enters any Linear A item.
"""
import json, os, re, sys, math, random, hashlib
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from la60_common import (load_la, load_lb, admin_docs, split, base_of, bt_order, seed, prior_reading,
                         induce_reading, _num_after, close_test, lb_prior_reading, lb_draw, reading_hash)
import la60_decode

D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la70_ckpt')
os.makedirs(CK, exist_ok=True)

KI_SURVIVES = os.path.exists(os.path.join(CK, 'ki_survives.flag'))

CAND = {
    'KU-RO': ('TOT', 'A', 'total (running sum closes)'),
    'PO-TO-KU-RO': ('TOT', 'A', 'grand total'),
    'KI-RO': ('RES', 'B', 'residue / sub-heading'),
    'NI': ('COM', 'B', 'commodity sign'), 'KI': ('COM', 'B', 'commodity sign'), 'DI': ('COM', 'B', 'commodity sign'),
    'RE': ('COM', 'B', 'commodity sign'), 'TI': ('COM', 'B', 'commodity sign'), 'MA': ('COM', 'B', 'commodity sign'),
    '*304': ('COM', 'B', 'commodity sign'), '*308': ('COM', 'B', 'commodity sign (fraction-bound, C)'),
    '*301': ('HDR', 'B', 'heading sign'), 'RO': ('HDR', 'B', 'heading sign'), 'ZE': ('HDR', 'B', 'heading sign'),
    'TE': ('HDR', 'B', 'heading sign (Villa-magazine word, C+)'),
    '*516': ('HDR', 'C', 'heading word'), '*307': ('HDR', 'C', 'heading word'),
    'A-DU': ('HDR', 'C', 'heading word (house word, C+)'),
    'SA-RA₂': ('TRX', 'B', 'heading/transaction word (opens GRA/CYP > NI > VIN basket)'),
    'KA-PA': ('TRX', 'B', 'heading/transaction word'), 'A-PA-RA-NE': ('TRX', 'B', 'heading/transaction word'),
    'MA-KA-RI-TE': ('TRX', 'B', 'heading/transaction word'), 'KU-NI-SU': ('TRX', 'B', 'heading/transaction word'),
    'PA-DE': ('MRK', 'C', 'marked entry (own line, small amount)'),
    '*28B-NU-MA-RE': ('TPL', 'C', 'fixed line word of the Zakros wine template'),
    'SI-PI-KI': ('TPL', 'C', 'fixed line word of the Zakros wine template'),
}
if KI_SURVIVES:
    CAND['KI'] = ('RED', 'C', 'reduced amount (smaller than the entry above)')

ORDER_V2 = {'CYP': 0.0, 'GRA': 0.0, 'OLE': 1.0, 'OLIV': 1.0, 'VIR': 1.0, 'NI': 2.0, 'VIN': 3.0, 'QA2': 3.0}
SITE_DEF = {'KH': 'CYP', 'ZA': 'GRA', 'ARKH': 'GRA'}
LIB = prior_reading()['lib']
ANNOT = {'QA₂': 'Villa-magazine word C+', 'QA2': 'Villa-magazine word C+', 'TE': 'Villa-magazine word C+',
         'NI': 'house word C+', 'A-DU': 'house word C+', 'TA-I': 'goes with AROM, C'}
FRAC_VAL = {'D': 0.2, 'B': 1.0 / 3}


# ------------------------------------------------------------------ fractions
def frac_parts(s):
    return re.findall(r'JE|L\d|[A-Z]', s or '')


def frac_known(s):
    """(known value, number of unvalued fraction signs)"""
    k = 0.0; u = 0
    for p in frac_parts(s):
        if p in FRAC_VAL:
            k += FRAC_VAL[p]
        else:
            u += 1
    return k, u


def close_v2(total_int, total_fr, ents):
    """Value-aware closure: D and B valued; other fraction signs unvalued (la60 slack for them)."""
    tk, tu = frac_known(total_fr)
    s_int = sum(e[1] for e in ents)
    sk = 0.0; nu = 0
    for e in ents:
        k, u = frac_known(e[2]); sk += k; nu += (u > 0)
    T_lo, T_hi = total_int + tk, total_int + tk + (0.999 if tu else 0.0)
    S_lo = s_int + sk
    S_hi = S_lo + (nu // 2 + 1 if nu else 0.0)
    return not (T_hi < S_lo - 0.02 or T_lo > S_hi + 0.02)


# ------------------------------------------------------------------ the v2 reading
def gate_v2(train, name='V2', strict=False):
    """Train-only gates: each candidate item is kept only if it meets its own criterion on training
    documents; order positions kept only where training pairs do not contradict them."""
    roles = {}; dropped = {}
    occ = Counter(); fn = Counter(); head = Counter(); site_occ = defaultdict(Counter)
    tot_close = Counter(); tot_try = Counter(); ki_down = [0, 0]; f308 = [0, 0]
    for d in train:
        toks = d['toks']
        first_line = []
        for t in toks:
            if t[0] == 'NL':
                break
            first_line.append(t)
        for i, t in enumerate(toks):
            if t[0] != 'W':
                continue
            w = t[1]; occ[w] += 1; site_occ[w][d['site']] += 1
            j = _num_after(toks, i)
            if j is not None and j == i + 1:
                fn[w] += 1
            if t in first_line and j is None:
                head[w] += 1
            if w == 'KI' and i + 1 < len(toks) and toks[i + 1][0] == 'N':
                prev = [x for x in toks[:i] if x[0] == 'N']
                if prev and prev[-1][1] > 0:
                    ki_down[1] += 1; ki_down[0] += toks[i + 1][1] < prev[-1][1]
            if w == '*308' and j is not None:
                f308[1] += 1; f308[0] += bool(toks[j][2])
        for s in _sections_raw(d, {'KU-RO': 'TOT', 'PO-TO-KU-RO': 'TOT', 'KI-RO': 'RES'}, set()):
            if s[3] is not None:
                tot_try[s[4]] += 1; tot_close[s[4]] += s[3]
    ungated = []
    for w, (r, g, gl) in CAND.items():
        if occ[w] == 0 and not strict:
            roles[w] = r; ungated.append(w)       # absent from training: kept as prior, not gated
            continue
        ok = occ[w] >= 1
        if r == 'TOT':
            ok = tot_close[w] >= 1 or (occ[w] >= 1 and w == 'PO-TO-KU-RO' and 'KU-RO' in roles)
        elif r == 'COM':
            ok = occ[w] >= 2 and fn[w] / occ[w] >= 0.5
        elif r == 'HDR':
            ok = occ[w] >= 1 and head[w] / occ[w] >= 0.4
        elif r == 'MRK':
            ok = occ[w] >= 1 and fn[w] >= 1
        elif r == 'TPL':
            ok = occ[w] >= 1 and set(site_occ[w]) == {'ZA'}
        elif r == 'RED':
            ok = ki_down[1] >= 2 and ki_down[0] / ki_down[1] >= 0.6
        if ok:
            roles[w] = r
        else:
            dropped[w] = r
    # order: keep a commodity's hand position unless training pairs contradict it (> half against)
    wins = Counter()
    for d in train:
        s = com_seq_v2(d, roles)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                wins[(s[i], s[j])] += 1
    order = {}
    for a in ORDER_V2:
        agree = against = 0
        for b in ORDER_V2:
            if a == b or abs(ORDER_V2[a] - ORDER_V2[b]) < 0.3:
                continue
            early, late = (a, b) if ORDER_V2[a] < ORDER_V2[b] else (b, a)
            agree += wins[(early, late)]; against += wins[(late, early)]
        if agree + against == 0 or agree >= against:
            order[a] = ORDER_V2[a]
        else:
            dropped['order:' + a] = ORDER_V2[a]
    rules = {'frac308': f308[0] == f308[1] and '*308' in roles and (f308[1] >= 1 or not strict),
             'DB_values': True, 'ki_red': 'KI' in roles and roles['KI'] == 'RED'}
    return dict(name=name, roles=roles, order=order, lib=set(LIB), site_default=dict(SITE_DEF),
                ent_slot_unknown=True, rules=rules, dropped=dropped, ungated=ungated)


def hash_v2(R):
    s = json.dumps(dict(roles=sorted(R['roles'].items()), order=sorted((k, round(v, 6)) for k, v in R['order'].items()),
                        lib=sorted(R['lib']), site_default=sorted(R['site_default'].items()),
                        rules=sorted(R.get('rules', {}).items()), fv=sorted(FRAC_VAL.items())), ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()


def com_seq_v2(doc, roles):
    out = []
    for t in doc['toks']:
        if t[0] == 'L' or (t[0] == 'W' and roles.get(t[1]) == 'COM'):
            b = base_of(t[1])
            if b == 'OLIV':
                b = 'OLIV'
            if b not in out:
                out.append(b)
    return out


# ------------------------------------------------------------------ decoding
def _sections_raw(doc, roles, red_words, v2close=True):
    """(tot_index, total, entries, closes, word); numbers after a RED word are kept out of the running sum."""
    toks = doc['toks']; out = []; ent = []; skip = set(); xin = False
    for i, t in enumerate(toks):
        if t[0] == 'X':
            # la71: a removed number or section word in this section -> its closure is untestable
            if t[1] == 'num' or t[2]: xin = True
            continue
        if t[0] == 'W' and roles.get(t[1]) in ('TOT', 'RES'):
            j = _num_after(toks, i)
            if j is not None:
                skip.add(j)
                if roles.get(t[1]) == 'TOT':
                    if len(ent) >= 2 and not xin:
                        ok = close_v2(toks[j][1], toks[j][2], ent) if v2close else \
                            close_test(toks[j][1], sum(e[1] for e in ent), 0, sum(1 for e in ent if e[2]))
                    else:
                        ok = None
                    out.append((i, toks[j][1], list(ent), ok, t[1]))
            ent = []; xin = False
            continue
        if t[0] == 'W' and t[1] in red_words and i + 1 < len(toks) and toks[i + 1][0] == 'N':
            skip.add(i + 1)
        if t[0] == 'N' and i not in skip:
            ent.append((i, t[1], t[2]))
    return out


def sections_v2(doc, R):
    red = {w for w, r in R['roles'].items() if r == 'RED'}
    return _sections_raw(doc, R['roles'], red, v2close=not R.get('v1close'))


def roles_doc_v2(d, R):
    rs = la60_decode.roles_doc(d, R)
    # an unknown word directly before a RED/MRK word + number also sits in an entry slot
    return rs


def acceptor_v2(train, R):
    c = Counter()
    for d in train:
        rs = [r for r in roles_doc_v2(d, R) if r != 'NL']
        rs = ['<s>'] + rs + ['</s>']
        for a, b in zip(rs, rs[1:]):
            c[(a, b)] += 1
    return c


def decode_v2(d, R, acc, margin=0.3):
    toks = d['toks']; rs = roles_doc_v2(d, R); ch = {}
    ch['cover'] = 'UNK' not in rs and 'LIB' not in rs
    seq = ['<s>'] + [r for r in rs if r != 'NL'] + ['</s>']
    bad = [(a, b) for a, b in zip(seq, seq[1:]) if acc[(a, b)] == 0]
    ch['roles'] = not bad
    secs = sections_v2(d, R)
    tested = [s for s in secs if s[3] is not None]
    ch['n_tot'] = len(tested); ch['n_close'] = sum(1 for s in tested if s[3])
    ch['totals'] = all(s[3] for s in tested)
    coms = []
    for t, r in zip(toks, rs):
        if r == 'COM' and t[0] in ('L', 'W'):
            b = base_of(t[1])
            if b not in coms:
                coms.append(b)
    o = R['order']; viol = agree = 0
    for i in range(len(coms)):
        for j in range(i + 1, len(coms)):
            a, b = coms[i], coms[j]
            if a in o and b in o:
                if o[a] > o[b] + margin:
                    viol += 1
                elif o[a] < o[b] - margin:
                    agree += 1
    ch['order'] = viol == 0; ch['n_order_agree'] = agree; ch['n_order_viol'] = viol
    # rule checks
    rule_ok = True; n_rule = 0
    rules = R.get('rules', {})
    for i, t in enumerate(toks):
        if t[0] == 'W' and R['roles'].get(t[1]) == 'COM' and t[1] == '*308' and rules.get('frac308'):
            j = _num_after(toks, i)
            if j is not None:
                n_rule += 1
                rule_ok &= bool(toks[j][2])
        if t[0] == 'W' and R['roles'].get(t[1]) == 'RED' and i + 1 < len(toks) and toks[i + 1][0] == 'N':
            prev = [x for x in toks[:i] if x[0] == 'N']
            if prev:
                n_rule += 1
                rule_ok &= toks[i + 1][1] <= prev[-1][1]
    ch['rules'] = rule_ok; ch['n_rule'] = n_rule
    com_entry = any(r == 'COM' and k + 1 < len(toks) and toks[k + 1][0] == 'N' for k, r in enumerate(rs))
    ch['substantive'] = bool(tested) or com_entry or agree > 0 or n_rule > 0
    ch['full'] = ch['cover'] and ch['roles'] and ch['totals'] and ch['order'] and ch['substantive'] and ch['rules']
    ch['strict'] = ch['full'] and (ch['n_close'] > 0 or agree > 0 or n_rule > 0)
    ch['bad_steps'] = bad[:3]
    return ch, rs, secs, coms


def shuffle_v2(R, rng):
    items = sorted(R['roles'].items())
    types = [w for w, r in items]; labs = [r for w, r in items]
    rng.shuffle(labs)
    ok = sorted(R['order']); ov = [R['order'][k] for k in ok]; rng.shuffle(ov)
    return dict(name=R['name'] + '-shuf', roles=dict(zip(types, labs)), order=dict(zip(ok, ov)), lib=R['lib'],
                site_default=R['site_default'], ent_slot_unknown=True, rules=R.get('rules', {}))


def random_v2(tr, R, rng):
    vocab = sorted({t[1] for d in tr for t in d['toks'] if t[0] == 'W'})
    labs = list(R['roles'].values())
    types = rng.sample(vocab, min(len(labs), len(vocab)))
    roles = dict(zip(types, labs))
    bases = sorted({base_of(t[1]) for d in tr for t in d['toks'] if t[0] == 'L'})
    ov = list(R['order'].values()); rng.shuffle(ov)
    ob = rng.sample(bases, min(len(ov), len(bases)))
    return dict(name='RANDOM', roles=roles, order=dict(zip(ob, ov)), lib=set(), site_default={},
                ent_slot_unknown=True, rules={})


EMPTY = dict(name='EMPTY', roles={}, order={}, lib=set(), site_default={}, ent_slot_unknown=True, rules={})


def score_v2(tr, te, R):
    acc = acceptor_v2(tr, R); c = Counter()
    for d in te:
        ch = decode_v2(d, R, acc)[0]
        c['full'] += ch['full']; c['strict'] += ch['strict']; c['close'] += ch['n_close']
        c['tot_tested'] += ch['n_tot']; c['agree'] += ch['n_order_agree']; c['viol'] += ch['n_order_viol']
        c['cover'] += ch['cover']; c['rule_n'] += ch['n_rule']
    return dict(c)


# ------------------------------------------------------------------ LB calibration reading
def lb_matched_reading(train, R_la):
    """CALIBRATION ONLY: a true partial Linear B reading matched to V2 in kind and size: TOT (to-so),
    RES (o-pe-ro), heading/transaction words (true LB heading words, as many as V2 has HDR+TRX+MRK+TPL
    multi-sign items), order fitted on LB training documents only. Gated on training like V2."""
    n_head = sum(1 for w, r in R_la['roles'].items() if r in ('HDR', 'TRX', 'MRK', 'TPL'))
    heads = ['A-PU-DO-SI', 'O-U-DI-DO-SI', 'PA-RO', 'E-KE', 'O-NA-TO', 'KE-KE-ME-NA', 'KI-TI-ME-NA', 'DA-MO',
             'PE-MO', 'PE-MA', 'O-DA-A2', 'KO-TO-NA', 'DO-SO-MO', 'O-PI', 'DE-DE-ME-NO', 'DO-SE', 'TE-RE-TA',
             'WO-ZE-E', 'E-RO2', 'ME-TA-QE', 'E-KE-QE']
    occ = Counter(t[1] for d in train for t in d['toks'] if t[0] == 'W')
    roles = {}
    for w in ['TO-SO', 'TO-SA', 'TO-SO-DE', 'TO-SA-DE']:
        if occ[w]:
            roles[w] = 'TOT'
    if occ['O-PE-RO']:
        roles['O-PE-RO'] = 'RES'
    k = 0
    for w in heads:
        if occ[w] and k < n_head:
            roles[w] = 'TRX'; k += 1
    return dict(name='LBMATCH', roles=roles, order=bt_order(train, roles), lib=set(), site_default={},
                ent_slot_unknown=True, rules={})


def score_lb(tr, te, R):
    """Linear B through the la60 decoder (values known: lb=True)."""
    acc = la60_decode.acceptor(tr, R); c = Counter()
    for d in te:
        ch, g = la60_decode.decode(d, R, acc, lb=True)
        strict = ch['full'] and (ch['n_close'] > 0 or ch['n_order_agree'] > 0)
        c['full'] += ch['full']; c['strict'] += strict; c['close'] += ch['n_close']
        c['agree'] += ch['n_order_agree']; c['viol'] += ch['n_order_viol']
    return dict(c)


def score_v1(tr, te, R):
    acc = la60_decode.acceptor(tr, R); c = Counter()
    for d in te:
        ch, g = la60_decode.decode(d, R, acc)
        strict = ch['full'] and (ch['n_close'] > 0 or ch['n_order_agree'] > 0)
        c['full'] += ch['full']; c['strict'] += strict; c['close'] += ch['n_close']
        c['tot_tested'] += ch['n_tot']; c['agree'] += ch['n_order_agree']; c['viol'] += ch['n_order_viol']
    return dict(c)
