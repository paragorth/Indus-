#!/usr/bin/env python3
"""v83 cycle 1: validate the uncertainty-aware parser and count uncertainty by section and transcription.
(a) round trip against the legacy parser (data/derived/<name>_lines.json)
(b) hand-checked gold lines (data/derived/v83_gold_lines.json)
(c) planted control: markers planted at random words of clean ZL lines must come back on exactly those words
(d) shares of words with any flag, by transcription x section / language / line type / flag
(e) do flags mark real reading problems? P(three transcriptions disagree | flag) vs no flag
Out: data/v83_ckpt/c1.json; rows printed for loops/v83_cycle1.txt."""
import os, sys, json, random, re, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v83_parse as P

OUT = {}
D = {nm: P.records(nm) for nm in ('ZL3b', 'IT2a', 'GC2a')}

# (a) round trip
rt = {}
for nm in ('ZL3b', 'IT2a'):
    L = P.legacy_lines(nm)
    li = {(r['folio'], r['n']): r['words'] for r in L}
    ni = {(r['folio'], r['n']): r['words'] for r in D[nm]}
    diff = [k for k in set(li) | set(ni) if li.get(k) != ni.get(k)]
    det = []
    for k in sorted(diff):
        a, b = li.get(k), ni.get(k)
        det.append((k, a if a is None else len(a), b if b is None else len(b),
                    [w for w in (a or []) if w not in (b or [])][:3]))
    rt[nm] = dict(lines_legacy=len(li), lines_new=len(ni), lines_differ=len(diff), detail=det[:20],
                  words_legacy=sum(len(v) for v in li.values()), words_new=sum(len(v) for v in ni.values()))
OUT['roundtrip'] = rt

# (b) gold
G = json.load(open(os.path.join(P.DATA, 'derived', 'v83_gold_lines.json')))
gold = {}
for nm in ('ZL3b', 'IT2a', 'GC2a'):
    idx = {'%s.%d' % (r['folio'], r['n']): r for r in D[nm]}
    tot = ok = 0; bad = []
    for key, exp in G[nm].items():
        r = idx[key]
        if len(r['words']) != len(exp):
            bad.append((key, 'count', len(r['words']), len(exp))); tot += len(exp); continue
        for j, (w, fl) in enumerate(exp):
            tot += 1
            want = set(x for x in fl.split(',') if x)
            got = set(r['flags'][j])
            wok = (nm == 'GC2a') or (r['words'][j] == w)
            if want == got and wok: ok += 1
            else: bad.append((key, j, w, r['words'][j], sorted(want), sorted(got)))
    gold[nm] = dict(tokens=tot, correct=ok, lines=len(G[nm]), errors=bad)
OUT['gold'] = gold

# (c) planted control on clean ZL lines (re-parsed from synthetic IVTFF text)
rng = random.Random(83)
clean = [r for r in D['ZL3b'] if r['ltype'] == 'P' and len(r['words']) >= 3 and not any(r['flags'])
         and all(re.fullmatch(r'[a-z]+', w) for w in r['words'])]
KINDS = ['alt', 'ill', 'rare', 'lig', 'mark', 'usp', 'cmt', 'dmg']
tot = okc = 0; errs = collections.Counter(); per = collections.Counter(); perok = collections.Counter()
for t in range(3000):
    r = rng.choice(clean)
    ws = list(r['words']); seps = ['.'] * (len(ws) - 1)
    j = rng.randrange(len(ws)); kind = rng.choice(KINDS)
    exp = [set() for _ in ws]
    w = ws[j]; i = rng.randrange(len(w))
    if kind == 'alt':
        ws[j] = w[:i] + '[' + w[i] + ':' + rng.choice('aoyedk') + ']' + w[i + 1:]; exp[j].add('alt')
    elif kind == 'ill':
        ws[j] = w[:i] + '?' + w[i + 1:]; exp[j].add('ill')
    elif kind == 'rare':
        ws[j] = w[:i] + '@%d;' % rng.randrange(128, 256) + w[i + 1:]; exp[j].add('rare')
    elif kind == 'lig':
        k = min(len(w), i + 2); ws[j] = w[:i] + '{' + w[i:k] + '}' + w[k:]; exp[j].add('lig')
    elif kind == 'mark':
        ws[j] = w[:i + 1] + "'" + w[i + 1:]; exp[j].add('mark')
    elif kind == 'usp':
        if j == len(ws) - 1: j -= 1
        seps[j] = ','; exp = [set() for _ in ws]; exp[j].add('usp'); exp[j + 1].add('usp')
    elif kind == 'cmt':
        ws[j] = w + rng.choice(['<!corr?>', '<!unclear>', '<!funny d>', '<!faint>']); exp[j].add('cmt')
    elif kind == 'dmg':
        ws[j] = w + rng.choice(['<!tear>', '<!hole>', '<!gap>']); exp[j].add('dmg')
    # plus a positional comment that must NOT flag anything
    if rng.random() < 0.3:
        ws[0] = '<!%02d:%02d>' % (rng.randrange(12), rng.choice([0, 30])) + ws[0]
    if rng.random() < 0.3:
        k = rng.randrange(len(seps)) if seps else None
        if k is not None and seps[k] == '.': seps[k] = rng.choice(['<->', '<~>'])
    text = ws[0] + ''.join(s + x for s, x in zip(seps, ws[1:]))
    words, sp, _, _ = P.parse_line_text(text)
    per[kind] += 1
    good = len(words) == len(exp) and all(set(d['flags']) == e for d, e in zip(words, exp))
    # word identity: the first reading with markers removed must equal the original word, except '?' and rare codes
    if good:
        for d, orig in zip(words, r['words']):
            ww = d['w']
            if kind in ('ill', 'rare'):
                if len(ww) != len(orig): good = False
            elif ww != orig: good = False
    tot += 1; okc += good; perok[kind] += good
    if not good: errs[kind] += 1
OUT['planted'] = dict(trials=tot, correct=okc, by_kind={k: [perok[k], per[k]] for k in KINDS}, errors=dict(errs))

# (d) shares
SEC = dict(H='herbal', S='stars', B='bio', P='pharma', T='text', C='cosmo', Z='zodiac', A='astro')
shares = {}
for nm, R in D.items():
    tab = collections.defaultdict(lambda: collections.Counter())
    for r in R:
        for j, f in enumerate(r['flags']):
            for key in ('ALL', 'sec:' + SEC.get(r['illus'], str(r['illus'])), 'lang:' + str(r['lang']),
                        'ltype:' + r['ltype'], 'hand:' + str(r['hand'])):
                c = tab[key]; c['n'] += 1; c['any'] += bool(f); c['glyph'] += bool(set(f) & P.GLYPH_FLAGS)
                c['agree'] += r['agree'][j]; c['agreeclean'] += r['agree'][j] and not f
                for x in f: c[x] += 1
    shares[nm] = {k: dict(v) for k, v in tab.items()}
OUT['shares'] = shares

# (e) are flagged words read differently by the other transcribers?
ev = {}
for nm in ('ZL3b', 'GC2a', 'IT2a'):
    c = collections.Counter()
    for r in D[nm]:
        if r['ltype'] != 'P': continue
        for j, f in enumerate(r['flags']):
            g = 'none' if not f else ('usp_only' if set(f) == {'usp'} else 'glyph')
            c[(g, 'n')] += 1; c[(g, 'agree')] += r['agree'][j]
            for x in f:
                c[('f:' + x, 'n')] += 1; c[('f:' + x, 'agree')] += r['agree'][j]
    ev[nm] = {g: (c[(g, 'agree')], c[(g, 'n')], round(c[(g, 'agree')] / max(1, c[(g, 'n')]), 3))
              for g in sorted({k[0] for k in c})}
OUT['flag_vs_agree'] = ev

# GC: glyphs that the v74 converter deleted
gcraw = open(P.SRC['GC2a'], encoding='latin-1').read().split('\n')
nbang = sum(len(re.findall(r'!', re.sub(r'<![^>]*>', '', l.split('>', 1)[1] if '>' in l else ''))) for l in gcraw if l.startswith('<f') and '.' in l.split('>')[0])
npct = sum(len(re.findall(r'%', (l.split('>', 1)[1] if '>' in l else '').replace('<%>', ''))) for l in gcraw if l.startswith('<f') and '.' in l.split('>')[0])
OUT['gc_v74_deleted'] = dict(bang=nbang, pct=npct)
json.dump(OUT, open(os.path.join(P.CK, 'c1.json'), 'w'), indent=1, default=str)

print('ROUNDTRIP', json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'detail'} for k, v in rt.items()}))
for k, v in rt.items(): print(k, v['detail'][:8])
print('GOLD', {k: (v['correct'], v['tokens'], v['lines']) for k, v in gold.items()})
for k, v in gold.items():
    for e in v['errors']: print('  gold err', k, e)
print('PLANTED', OUT['planted'])
for nm in shares:
    s = shares[nm]
    print('SHARES', nm)
    for key in sorted(s):
        c = s[key]
        if c['n'] < 100: continue
        print('   %-12s n=%6d any=%.3f glyph=%.3f usp=%.3f alt=%.3f ill=%.3f rare=%.3f lig=%.3f ldmg=%.3f agree=%.3f agreeclean=%.3f' % (
            key, c['n'], c['any'] / c['n'], c['glyph'] / c['n'], c.get('usp', 0) / c['n'], c.get('alt', 0) / c['n'],
            c.get('ill', 0) / c['n'], c.get('rare', 0) / c['n'], c.get('lig', 0) / c['n'], c.get('ldmg', 0) / c['n'],
            c['agree'] / c['n'], c['agreeclean'] / c['n']))
print('FLAG vs AGREE', json.dumps(ev))
print('GC deleted by v74', OUT['gc_v74_deleted'])
