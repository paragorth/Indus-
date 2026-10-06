#!/usr/bin/env python3
"""Restoration-aware Proto-Elamite corpus parser (pe74).

The original parser (tools/build_corpus.py -> data/pe_corpus.json) strips the ATF editorial marks and
keeps every sign and numeral as if it were read on the clay.  This module re-reads each line's raw ATF
body (the 'raw' field of pe_corpus.json, which is the verbatim CDLI line) and gives every token a status:

  read       no mark, or only '!' (editor's correction of a sign that is on the clay)
  damaged    '#'  (sign visible but damaged)
  uncertain  '?'  (reading uncertain; '#?' counts as uncertain)
  restored   inside [...] (editor's restoration of text that is lost), wholly or in part
  supplied   inside <...> (editor's addition of something the scribe omitted; not on the clay)

The worst mark wins (supplied = restored > uncertain > damaged > read).  Token classification (sign vs
numeral, lacunae '...' skipped) is identical to build_corpus.py, so mode 'all' reproduces
pe_corpus.json token for token (checked by `python3 pe74_parse.py --check`).

Modes for analysis:
  'all'  every token (the old behaviour)
  'rd'   read + damaged only (drops uncertain, restored, supplied)
  'r'    read only (also drops damaged)

load(mode) returns the pe_corpus.json structure with dropped tokens removed.  Each line also gets
  sign_status / num_status : statuses of the kept tokens
  n_dropped_signs / n_dropped_nums : how many tokens the mode dropped (a dropped damaged / uncertain sign
                  stays as 'x' with status 'dropped'; a dropped restored sign is removed)
  nums_complete : False if any numeral token of the line was removed (use it to exclude such lines
                  or tablets from arithmetic: a part-numeral is not the numeral)
Use load_marked() for the full corpus with statuses and no filtering.
"""
import json, os, re, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RANK = {'read': 0, 'damaged': 1, 'uncertain': 2, 'restored': 3, 'supplied': 3}
KEEP = {'all': {'read', 'damaged', 'uncertain', 'restored', 'supplied'},
        'rd': {'read', 'damaged'},
        'r': {'read'}}
NUMTOK = re.compile(r'^(\d+|n)\(((?:N[0-9A-Z]+|n)(?:@[a-z])?)\)$')


def norm_code(c):
    c = c.replace('N1@', 'N01@')
    return 'N01' if c == 'N1' else c


def _char_states(body):
    """For each char: (in_square, in_angle).  Brackets may span tokens and commas."""
    sq = ang = False
    out = []
    for ch in body:
        if ch == '[':
            sq = True
            out.append((None, None))
            continue
        if ch == ']':
            sq = False
            out.append((None, None))
            continue
        if ch == '<':
            ang = True
            out.append((None, None))
            continue
        if ch == '>':
            ang = False
            out.append((None, None))
            continue
        out.append((sq, ang))
    return out


def _tokens(body):
    """Yield (token_text_with_marks, side, restored_frac, supplied_any) in reading order.
    side 0 = before the first comma, 1 = after."""
    st = _char_states(body)
    comma = body.find(',')
    toks = []
    i, n = 0, len(body)
    while i < n:
        if body[i].isspace() or (i == comma):
            i += 1
            continue
        j = i
        while j < n and not body[j].isspace() and j != comma:
            j += 1
        piece = body[i:j]
        sts = [s for s in st[i:j] if s[0] is not None]
        content = [s for k, s in zip(piece, st[i:j]) if s[0] is not None and k not in '#?!*']
        rfrac = (sum(1 for s in content if s[0]) / len(content)) if content else 0.0
        sup = any(s[1] for s in sts)
        side = 0 if (comma < 0 or i < comma) else 1
        toks.append((piece, side, rfrac, sup))
        i = j
    return toks


def status_of(piece, rfrac, sup):
    s = 'read'
    if '#' in piece:
        s = 'damaged'
    if '?' in piece:
        s = 'uncertain'
    if rfrac > 0:
        s = 'restored'
    if sup:
        s = 'supplied'
    return s


def parse_body(body):
    """Return signs, sign_status, numerals, num_status, partial_restored_flags (same classification
    as build_corpus.parse_side, applied to each comma side)."""
    out = {0: ([], [], [], [], []), 1: ([], [], [], [], [])}
    for piece, side, rfrac, sup in _tokens(body):
        signs, sst, nums, nst, part = out[side]
        if '...' in piece:
            continue
        stt = status_of(piece, rfrac, sup)
        if piece.startswith('|') or piece.startswith('[|') or piece.startswith('<|'):
            pass
        t = re.sub(r'[\[\]#?!*<>]', '', piece)
        if t.startswith('|'):
            signs.append(t); sst.append(stt); part.append(0 < rfrac < 1)
            continue
        m = NUMTOK.match(t)
        if m:
            c = m.group(1)
            nums.append([int(c) if c != 'n' else None, m.group(2)]); nst.append(stt)
            continue
        if t in ('', '+'):
            continue
        signs.append(t); sst.append(stt); part.append(0 < rfrac < 1)
    s0, ss0, n0, ns0, p0 = out[0]
    s1, ss1, n1, ns1, p1 = out[1]
    return s0 + s1, ss0 + ss1, n0 + n1, ns0 + ns1, p0 + p1


def site_of(prov):
    m = re.search(r'mod\. ([^)]+)\)', prov or '')
    s = m.group(1) if m else (prov or 'unknown')
    return s.replace('Shush', 'Susa')


def volume_of(desig):
    d = desig.split(',')[0].strip()
    if d.startswith('Fouilles de Sialk'):
        return 'Sialk I'
    return d


def load_marked(fix_n=True):
    """fix_n: a bare 'n' (e.g. '[n]', a numeral of unknown count with no unit) is not a sign; drop it
    from signs and set line['unknown_numeral'].  build_corpus.py kept it as a sign (52 tokens)."""
    T = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
    for t in T:
        t['site'] = site_of(t['provenience'])
        t['volume'] = volume_of(t['designation'])
        for l in t['lines']:
            s, ss, nu, ns, part = parse_body(l['raw'])
            l['signs'] = s
            l['sign_status'] = ss
            l['numerals'] = [[n, norm_code(c)] for n, c in nu]
            l['num_status'] = ns
            l['sign_partial'] = part
            l['unknown_numeral'] = 'n' in s
            if fix_n and 'n' in s:
                k = [i for i, x in enumerate(s) if x != 'n']
                l['signs'] = [s[i] for i in k]
                l['sign_status'] = [ss[i] for i in k]
                l['sign_partial'] = [part[i] for i in k]
    return T


def load(mode='rd', fix_n=True, mark_raw=True):
    """Corpus with tokens outside KEEP[mode] filtered (see module doc).
    Signs: a dropped damaged or uncertain sign becomes 'x' (a sign is on the clay but its reading is not
    accepted; 'x' is the corpus's own 'unreadable' token, so positions and line lengths stay); a dropped
    restored or supplied sign is removed (it is not on the clay; the raw ATF already shows '[' or '<').
    Numerals: a dropped numeral is removed; with mark_raw the line gets lacuna = True and ' [...]' appended
    to raw, so older code's numeral-cleanliness tests (lacuna, '[' or '...' after the comma) exclude it.
    Per line: n_dropped_signs, n_dropped_nums, nums_complete; per tablet: nums_complete."""
    return apply_mode(load_marked(fix_n), mode, mark_raw)


def apply_mode(T, mode, mark_raw=True):
    """Filter a load_marked() corpus in place (see load())."""
    keep = KEEP[mode]
    for t in T:
        for l in t['lines']:
            sg, st = [], []
            nd = 0
            for s, x in zip(l['signs'], l['sign_status']):
                if x in keep:
                    sg.append(s); st.append(x)
                else:
                    nd += 1
                    if x in ('damaged', 'uncertain'):
                        sg.append('x'); st.append('dropped')
            kn = [i for i, x in enumerate(l['num_status']) if x in keep]
            l['n_dropped_signs'] = nd
            l['n_dropped_nums'] = len(l['numerals']) - len(kn)
            l['nums_complete'] = l['n_dropped_nums'] == 0
            l['signs'], l['sign_status'] = sg, st
            l['numerals'] = [l['numerals'][i] for i in kn]
            l['num_status'] = [l['num_status'][i] for i in kn]
            if mark_raw and mode != 'all' and l['n_dropped_nums']:
                l['raw'] = l['raw'] + ' [...]'
                l['lacuna'] = True
        t['nums_complete'] = all(l['nums_complete'] for l in t['lines'])
    return T


def corpus_file(mode):
    """Write (once) and return the path of the filtered corpus in pe_corpus.json format."""
    if mode == 'all':
        return os.path.join(DATA, 'pe_corpus.json')
    d = os.path.join(DATA, 'pe74_ckpt')
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, 'corpus_%s.json' % mode)
    src = os.path.join(DATA, 'pe_corpus.json')
    if not os.path.exists(fn) or os.path.getmtime(fn) < os.path.getmtime(src) \
            or os.path.getmtime(fn) < os.path.getmtime(os.path.abspath(__file__)):
        T = load(mode)
        for t in T:
            for l in t['lines']:
                l['numerals'] = [list(x) for x in l['numerals']]
        json.dump(T, open(fn, 'w'))
    return fn


def check_against_old():
    """Mode 'all' must equal pe_corpus.json token for token."""
    old = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
    new = load('all', fix_n=False)
    bad = 0
    for a, b in zip(old, new):
        for la, lb in zip(a['lines'], b['lines']):
            na = [[n, norm_code(c)] for n, c in la['numerals']]
            if la['signs'] != lb['signs'] or na != lb['numerals']:
                bad += 1
                if bad <= 5:
                    print('MISMATCH', a['id'], la['raw'], la['signs'], lb['signs'], na, lb['numerals'])
    print('lines differing from pe_corpus.json:', bad)
    return bad


if __name__ == '__main__':
    if '--check' in sys.argv:
        sys.exit(1 if check_against_old() else 0)
    T = load_marked()
    c = collections.Counter()
    for t in T:
        for l in t['lines']:
            for s in l['sign_status']:
                c[('sign', s)] += 1
            for s in l['num_status']:
                c[('num', s)] += 1
    for k in sorted(c):
        print(k, c[k])
