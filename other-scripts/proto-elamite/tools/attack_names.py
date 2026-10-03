#!/usr/bin/env python3
"""Attack 3: do PE 'name-like' strings read as Elamite names under the
Linear Elamite (LE) values of Desset et al. 2026 (data/le_pe_graphic_matches.json)?

Uses the FROZEN lists data/elamite_name_list_frozen.json and
data/control_name_list_frozen.json (built by build_elamite_list.py, before this test).

Declared, fixed spelling rules (applied identically to PE readings and to names):
  N1 (Elamite phonology after Paper/Reiner, as stated by Zadok 1984 intro):
     e=i, o=u, b=p, d=t, g=k, z=s=ṣ, ṭ=t, q=k, y=i; syllable boundaries removed;
     runs of the same letter collapsed (so CV-VC 'ba-an' = 'ban', geminates single).
  N2 = N1 plus š=s (Zadok notes s/š interchange; the OCR also loses š).
  A trailing vowel after a match is free (dead final vowel of a CV sign) because
  matching is by substring.
PE units: distinct sign strings (entry lines and header lines, no 'x'), >=2 signs,
>=70% of signs with an LE value, >=2 valued signs. Signs without a value break
the reading ('#'); a match may not cross a break.
Sign -> value: 'lenient' = base sign (variants merged; every LE value listed for
that base); 'strict' = exact variant form only.
Element hit: an element with >=2 vowels occurs as a substring of a reading.
Exact hit: the whole reading equals a whole name (or does so minus a final vowel).
"""
import collections, hashlib, json, os, random, re, sys
from common import load, entries, base, header, DATA, is_sign

random.seed(2026)
NSHUF = int(os.environ.get('NSHUF', 1000))
NLIST = int(os.environ.get('NLIST', 200))

EL = json.load(open(os.path.join(DATA, 'elamite_name_list_frozen.json')))
CT = json.load(open(os.path.join(DATA, 'control_name_list_frozen.json')))
SHA = {f: hashlib.sha256(open(os.path.join(DATA, f), 'rb').read()).hexdigest()
       for f in ('elamite_name_list_frozen.json', 'control_name_list_frozen.json')}
LE = json.load(open(os.path.join(DATA, 'le_pe_graphic_matches.json')))['matches']

VOW = set('aiu')


def norm(s, lvl):
    s = s.lower().replace('-', '').replace('ḫ', 'h')
    for a, b in (('e', 'i'), ('o', 'u'), ('b', 'p'), ('d', 't'), ('g', 'k'), ('z', 's'), ('ṣ', 's'),
                 ('ṭ', 't'), ('q', 'k'), ('y', 'i')):
        s = s.replace(a, b)
    if lvl == 2:
        s = s.replace('š', 's')
    s = re.sub(r'[^a-zš#]', '', s)
    return re.sub(r'(.)\1+', r'\1', s)


def nvow(s):
    return len(re.findall(r'[aiu]+', s))


# ------------------------------------------------------------ name sets
NONEL = {'Achaemenid', 'RAE', 'N/LB', 'NA', 'SB'}
el_names = [n['form'] for n in EL['names'] if n['period'] not in NONEL]
el_elems = [e['form'] for e in EL['elements']]


def sets(lvl):
    nm = {norm(x, lvl) for x in el_names}
    nm = {x for x in nm if nvow(x) >= 2}
    el = {norm(x, lvl) for x in el_elems}
    el = {x for x in el if nvow(x) >= 2}
    return nm, el


# ------------------------------------------------------------ LE values
def raw_val(k):
    if k.endswith('_sh'):
        return 'š' + re.sub(r'\d', '', k.split('_')[0])[1:]
    if k == 'u2_w':
        return 'u'
    return re.sub(r'\d', '', k)


def corpus_form(m):
    """LE-list sign id (e.g. M263a, M218a+M101) -> corpus strict form and base form."""
    parts = m.split('+')
    sf, bf = [], []
    for p in parts:
        mm = re.match(r'M(\d+)([a-z0-9]*)$', p)
        num = 'M%03d' % int(mm.group(1))
        sf.append(num + ('~' + mm.group(2) if mm.group(2) else ''))
        bf.append(num)
    if len(parts) > 1:
        return '|' + '+'.join(sf) + '|', '|' + '+'.join(bf) + '|'
    return sf[0], bf[0]


STRICT, LEN = collections.defaultdict(set), collections.defaultdict(set)
for k, v in LE.items():
    if k.startswith('US'):
        continue
    for m in v:
        sf, bf = corpus_form(m)
        STRICT[sf].add(raw_val(k))
        LEN[bf].add(raw_val(k))

# ------------------------------------------------------------ PE units
T = load()
units = collections.defaultdict(set)   # tuple(raw signs) -> tablets
kind = {}
for e in entries(T, base_signs=False):
    if 'x' in e['signs']:
        continue
    units[tuple(e['signs'])].add(e['tablet'])
    kind.setdefault(tuple(e['signs']), 'entry')
for t in T:
    if not t['lines']:
        continue
    l = t['lines'][0]
    if l['numerals']:
        continue
    sg = [s for s in l['signs'] if is_sign(s) or s == 'x']
    if sg and 'x' not in sg:
        units[tuple(sg)].add(t['id'])
        kind.setdefault(tuple(sg), 'header')

FRAME_I = {'M387', 'M157', 'M370', 'M124', 'M305', 'M038', 'M111'}
FRAME_F = {'M288', 'M297', 'M263', 'M346', 'M264', 'M072', 'M003', 'M354', 'M371', 'M096', 'M376', 'M036'}


def key_of(s, mode):
    if mode == 'strict':
        return s if s in STRICT else None
    b = base(s)
    return b if b in LEN else None


def select(mode, middle=False):
    out = []
    for u, tabs in units.items():
        sg = list(u)
        if middle:
            if sg and base(sg[0]) in FRAME_I:
                sg = sg[1:]
            if sg and base(sg[-1]) in FRAME_F:
                sg = sg[:-1]
        if len(sg) < 2:
            continue
        ks = [key_of(s, mode) for s in sg]
        nv = sum(k is not None for k in ks)
        if nv >= 2 and nv / len(ks) >= 0.7:
            out.append((u, ks, tabs))
    # merge identical key strings (variants collapse in lenient mode)
    m = collections.OrderedDict()
    for u, ks, tabs in out:
        kk = tuple(ks)
        if kk not in m:
            m[kk] = {'units': [], 'tabs': set()}
        m[kk]['units'].append(u)
        m[kk]['tabs'] |= tabs
    return m


def readings(ks, VM, lvl, cap=64):
    rs = ['']
    for k in ks:
        opts = sorted(VM[k]) if k is not None else ['#']
        rs = [r + ('#' if o == '#' else o) for r in rs for o in opts][:cap]
    return {norm(r, lvl) for r in rs}


def score(S, VM, lvl, nm, el, detail=False):
    ehit = xhit = 0
    det = []
    for ks, info in S.items():
        rs = readings(ks, VM, lvl)
        found = set()
        exact = False
        for r in rs:
            for seg in r.split('#'):
                if not seg:
                    continue
                for e in el:
                    if e in seg:
                        found.add(e)
            if '#' not in r and (r in nm or (r[-1:] in VOW and r[:-1] in nm)):
                exact = True
        if found:
            ehit += 1
        xhit += exact
        if detail and (found or exact):
            det.append({'signs': ' '.join(info['units'][0]), 'reading': sorted(rs)[:4],
                        'elements': sorted(found, key=len, reverse=True)[:6], 'exact_name': exact,
                        'tablets': sorted(info['tabs'])[:8], 'n_tablets': len(info['tabs']),
                        'longest': max((len(x) for x in found), default=0)})
    return ehit, xhit, det


def score_long(S, VM, lvl, el, minlen=5):
    """strings with an element hit of >= minlen phonemes"""
    el2 = {e for e in el if len(e) >= minlen}
    n = 0
    for ks in S:
        if any(e in seg for r in readings(ks, VM, lvl) for seg in r.split('#') for e in el2):
            n += 1
    return n


def shuffled(VM, keys):
    vals = [VM[k] for k in keys]
    random.shuffle(vals)
    return dict(zip(keys, vals))


# control element lists: random whole-syllable chunks of control names, matched in length to Elamite elements
ctrl_names = [n['form'] for n in CT['names']]


def control_elements(el, lvl):
    out = set()
    lens = [len(e) for e in el]
    for L in lens:
        for _ in range(200):
            n = random.choice(ctrl_names).split('-')
            i = random.randrange(len(n))
            for j in range(i + 1, len(n) + 1):
                c = norm('-'.join(n[i:j]), lvl)
                if len(c) >= L:
                    break
            if len(c) == L and nvow(c) >= 2:
                out.add(c)
                break
    return out


def control_names(nm, lvl):
    pool = collections.defaultdict(list)
    for n in ctrl_names:
        c = norm(n, lvl)
        if nvow(c) >= 2:
            pool[len(c)].append(c)
    out = set()
    for x in nm:
        L = len(x)
        while not pool.get(L):
            L -= 1
        out.add(random.choice(pool[L]))
    return out


def freq_shuffled(VM, keys, S, binsize=5):
    """permute values only among signs of similar token frequency in the tested strings"""
    f = collections.Counter(k for ks in S for k in ks if k is not None)
    ks = sorted(keys, key=lambda k: -f[k])
    out = {}
    for i in range(0, len(ks), binsize):
        b = ks[i:i + binsize]
        v = [VM[k] for k in b]
        random.shuffle(v)
        out.update(zip(b, v))
    return out


def order_shuffled(S):
    m = collections.OrderedDict()
    for ks, info in S.items():
        k2 = list(ks)
        random.shuffle(k2)
        k2 = tuple(k2)
        if k2 not in m:
            m[k2] = {'units': [], 'tabs': set()}
        m[k2]['units'] += info['units']
        m[k2]['tabs'] |= info['tabs']
    return m


def summ(obs, xs):
    n = len(xs)
    return {'mean': round(sum(xs) / n, 2), 'p_ge': sum(x >= obs for x in xs) / n}


res = {'frozen_sha256': SHA, 'n_shuffles': NSHUF, 'n_control_lists': NLIST, 'runs': {}}
for mode in ('lenient', 'strict'):
    VM = LEN if mode == 'lenient' else STRICT
    for middle in (False, True):
        S = select(mode, middle)
        used = sorted({k for ks in S for k in ks if k is not None})
        for lvl in (1, 2):
            nm, el = sets(lvl)
            tag = '%s_%s_N%d' % (mode, 'middle' if middle else 'full', lvl)
            eh, xh, det = score(S, VM, lvl, nm, el, detail=True)
            lh = score_long(S, VM, lvl, el)
            r = {'n_strings': len(S), 'n_strings_multi_tablet': sum(len(i['tabs']) > 1 for i in S.values()),
                 'n_signs_used': len(used), 'n_elements': len(el), 'n_names': len(nm),
                 'obs': {'element_hit_strings': eh, 'exact_name_strings': xh, 'long_element_hit_strings': lh}}
            for cname, fn in (('value_shuffle', lambda: shuffled(VM, used)),
                              ('value_shuffle_freq_matched', lambda: freq_shuffled(VM, used, S))):
                se, sx, sl = [], [], []
                for _ in range(NSHUF):
                    V2 = dict(VM); V2.update(fn())
                    a, b, _d = score(S, V2, lvl, nm, el)
                    se.append(a); sx.append(b); sl.append(score_long(S, V2, lvl, el))
                r[cname] = {'element': summ(eh, se), 'exact': summ(xh, sx), 'long5': summ(lh, sl)}
            se, sx = [], []
            for _ in range(min(NSHUF, 300)):
                a, b, _d = score(order_shuffled(S), VM, lvl, nm, el)
                se.append(a); sx.append(b)
            r['order_shuffle'] = {'element': summ(eh, se), 'exact': summ(xh, sx)}
            if mode == 'lenient':
                # (b) list control: same PE readings, Elamite list vs random non-Elamite onomastica
                ce, cx, cl, dreal = [], [], [], []
                for _ in range(NLIST):
                    cel = control_elements(el, lvl)
                    cnm = control_names(nm, lvl)
                    a, b, _d = score(S, VM, lvl, cnm, cel)
                    ce.append(a); cx.append(b); cl.append(score_long(S, VM, lvl, cel))
                    dreal.append(eh - a)
                r['list_control'] = {'element': summ(eh, ce), 'exact': summ(xh, cx), 'long5': summ(lh, cl)}
                # interaction: is the Elamite-over-control margin larger with the real LE values
                # than with shuffled values? (removes pure value-inventory phonotactics)
                dsh = []
                for _ in range(NLIST):
                    V2 = dict(VM); V2.update(shuffled(VM, used))
                    a, _b, _d = score(S, V2, lvl, nm, el)
                    c, _b, _d = score(S, V2, lvl, control_names(nm, lvl), control_elements(el, lvl))
                    dsh.append(a - c)
                md = sum(dreal) / len(dreal)
                r['list_x_value_interaction'] = {'margin_real_values': round(md, 2),
                                                 'margin_shuffled_values': summ(md, dsh)}
            det.sort(key=lambda d: (-d['longest'], -d['n_tablets']))
            r['top'] = det[:25]
            res['runs'][tag] = r
            print(tag, json.dumps({k: v for k, v in r.items() if k != 'top'}), flush=True)


# (c) famous claim: Inšušinak / šuši
VM = LEN
allS = select('lenient', False)
hits = []
for ks, info in allS.items():
    for lvl in (1, 2):
        for rd in readings(ks, VM, lvl):
            if ('šuši' if lvl == 1 else 'susi') in rd:
                hits.append({'signs': ' '.join(info['units'][0]), 'reading': rd, 'N': lvl, 'tablets': sorted(info['tabs'])})
# raw sign bigram search independent of the 70% filter: šu (M226~c/ca or base M226) followed by še/ši (M032)
raw = []
for u, tabs in units.items():
    b = [base(s) for s in u]
    for i in range(len(b) - 1):
        if b[i] == 'M226' and b[i + 1] == 'M032':
            raw.append({'signs': ' '.join(u), 'tablets': sorted(tabs)})
cnt = collections.Counter(base(s) for u in units for s in u)
res['famous'] = {'reading_hits': hits, 'raw_M226_M032_bigrams': raw,
                 'token_counts': {'M226 (šu)': cnt['M226'], 'M032 (še/ši)': cnt['M032'],
                                  'M131 (i)': cnt['M131'], 'M066 (i)': cnt['M066']},
                 'note': 'LE has no plain n sign among the matches, so in-šu-ši-na-ak would have to be spelled i-šu-ši-na-ka/ki(-k) in this value set.'}
print('famous', res['famous'])
json.dump(res, open(os.path.join(DATA, 'attack_names.json'), 'w'), ensure_ascii=False, indent=1)
