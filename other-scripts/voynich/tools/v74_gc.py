"""v74 cycle 2a: Glen Claston's transcription (GC2a-n.txt, v101 alphabet, voynich.nu IVTFF) converted to EVA.
Mapping = composition of the two published bitrans tables (voynich.nu/software/bitrans): STA-v101_def.bit
(v101 <-> STA) and STA-Eva_def.bit (STA <-> Eva-). STA codes present in v101 but absent from the Eva table fall
back to the Eva form of their STA family's first member (letter + '1'); braces and apostrophes are dropped;
codes with no Eva form become '?'. Longest-match tokenisation of v101 text.
Validation (independent of the P2 statistics): line-level agreement with ZL3b on the same loci (word count,
glyph error rate by edit distance), per-glyph confusion.
Out: data/v74_ckpt/GC_lines.json (v72 corpus line records), data/v74_ckpt/gc_validation.json."""
import os, sys, re, json, collections, difflib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v74_lib as X
GCD = os.path.join(X.SCR, 'v74', 'gc')


def table(fn):
    d = {}
    for line in open(fn, encoding='latin-1'):
        line = line.rstrip('\n')
        if not line or line.startswith('#') or line.startswith('<'): continue
        p = line.split(None, 1)
        if len(p) == 2: d[p[0]] = p[1].strip()
    return d


def mapping():
    v = table(os.path.join(GCD, 'STA-v101_def.bit')); e = table(os.path.join(GCD, 'STA-Eva_def.bit'))
    m = {}; how = {}
    for sta, sym in v.items():
        ev = e.get(sta)
        src = 'direct'
        if ev is None:
            # compound STA codes (e.g. 'E3B8') -> concatenate parts; else family first member
            parts = re.findall(r'[A-Z][0-9a-z]', sta)
            if len(parts) > 1 and all(pp in e or (pp[0] + '1') in e for pp in parts):
                ev = ''.join(e.get(pp, e.get(pp[0] + '1')) for pp in parts); src = 'compound'
            elif sta[0] + '1' in e:
                ev = e[sta[0] + '1']; src = 'family'
            else:
                ev = '?'; src = 'none'
        if '@' in ev and sta[0] + '1' in e and '@' not in e[sta[0] + '1']:
            ev = e[sta[0] + '1']; src += '-family'     # rare Eva extended glyph -> its STA family's first member
        ev = ev.replace('{', '').replace('}', '').replace("'", '')
        if '@' in ev: ev = '?'; src += '-unknown'
        m[sym] = ev; how[sym] = (sta, src)
    return m, how


def convert(word, m, keys):
    out, i = [], 0
    while i < len(word):
        for k in keys:
            if word.startswith(k, i):
                out.append(m[k]); i += len(k); break
        else:
            out.append('?'); i += 1
    return ''.join(out)


def records(path, m):
    keys = sorted(m, key=len, reverse=True)
    pages = {}; recs = []
    for line in open(path, encoding='latin-1'):
        line = line.rstrip('\n')
        mh = re.match(r'<(f[0-9a-z]+)>\s+<!(.*)>', line)
        if mh:
            pages[mh.group(1)] = dict(re.findall(r'\$(\w)=(\w+)', mh.group(2))); continue
        mm = re.match(r'<(f[0-9a-z]+)\.(\d+)[,;]?([^>]*)>\s*(.*)', line)
        if not mm: continue
        folio, n, loc, txt = mm.group(1), int(mm.group(2)), mm.group(3), mm.group(4)
        t = txt.replace('<->', '-')
        t = re.sub(r'<[^>]*>', '', t)
        t = re.sub(r'\[([^:\]]*):[^\]]*\]', r'\1', t)
        t = t.replace('{', '').replace('}', '').replace('!', '').replace('%', '')
        # keep @nnn; codes intact for the converter
        words = [w for w in re.split(r'[.,\-]', t) if w]
        ev = [convert(w, m, keys) for w in words]
        h = pages.get(folio, {})
        ltc = loc
        recs.append(dict(folio=folio, quire=h.get('Q'), panel=h.get('P'), illus=h.get('I'), lang=h.get('L'), hand=h.get('H'),
                         n=n, ltype=(re.sub(r'[^A-Z]', '', ltc)[:1] or '?'), ltcode=ltc,
                         para_start=('<%>' in txt) or ltc.startswith('@'), para_end='<$>' in txt, words=ev, raw=words,
                         uncertain=['?' in w for w in ev]))
    return recs


def lev(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != 'equal')


SENS = {'(': 'y', 'Q': 'm'}   # sensitivity variant GCb: ZL-majority glyph for the family fallbacks that disagree

if __name__ == '__main__':
    m, how = mapping()
    mb = dict(m); mb.update(SENS)
    Rb = records(os.path.join(GCD, 'GC2a-n.txt'), mb)
    for r in Rb: r.pop('raw')
    X.jsave('GCb_lines.json', Rb)
    R = records(os.path.join(GCD, 'GC2a-n.txt'), m)
    Z = X.ivtff(os.path.join(X.ROOT, 'data', 'ZL3b-n.txt'))
    from v18_lib import glyphs
    # validation against ZL on the same loci
    tot = err = nl = samewc = 0; wtot = wsame = 0
    conf = collections.Counter(); zlg = collections.Counter()
    for r in R:
        z = Z.get((r['folio'], r['n']))
        if not z or r['ltype'] != 'P': continue
        a = [g for w in z['words'] for g in glyphs(w)]; b = [g for w in r['words'] for g in glyphs(w)]
        nl += 1; tot += len(a); err += lev(a, b); samewc += len(z['words']) == len(r['words'])
        if len(z['words']) == len(r['words']):
            for x, y in zip(z['words'], r['words']):
                wtot += 1; wsame += x == y
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == 'replace' and i2 - i1 == j2 - j1:
                for x, y in zip(a[i1:i2], b[j1:j2]): conf[(x, y)] += 1
            zlg.update(a[i1:i2])
    gcg = collections.Counter(g for r in R if r['ltype'] == 'P' for w in r['words'] for g in glyphs(w))
    val = dict(lines=nl, glyph_error_rate=err / max(tot, 1), same_word_count=samewc / max(nl, 1),
               word_identity_rate=wsame / max(wtot, 1), top_confusions=[(a, b, n) for (a, b), n in conf.most_common(15)],
               gc_glyph_freq=gcg.most_common(30), mapping={k: (v, how[k]) for k, v in m.items()})
    X.jsave('gc_validation.json', val)
    for r in R: r.pop('raw')
    X.jsave('GC_lines.json', R)
    print(json.dumps({k: v for k, v in val.items() if k != 'mapping'})[:3000])
    print('P lines', sum(r['ltype'] == 'P' for r in R), 'tokens', sum(len(r['words']) for r in R if r['ltype'] == 'P'))
