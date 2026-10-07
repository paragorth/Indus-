"""la81: where the scribe cut a sign-group at the end of a physical line.
Physical lines: lineara.xyz 'transcription' (GORILA layout); logical words: 'transliteratedWords'.
A document is used only if the Unicode character class string of its words matches the
transcription exactly (no fuzzy alignment), so every cut position is read off, not inferred.
Damage flags come from corpus_ra.json (word tokens matched in order by sign sequence)."""
import json, os, re, sys, collections, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la81_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)


def cls(ch):
    o = ord(ch)
    if o == 0x10101: return 'D'
    if 0x10107 <= o <= 0x1013F: return 'N'
    if 0x10740 <= o <= 0x10755: return 'F'
    if (0x10600 <= o <= 0x1073F) or (0x10760 <= o <= 0x10767) or (0x10000 <= o <= 0x100FF): return 'S'
    if ch.isspace(): return None
    return 'L'


def wcls(w):
    """expected class string of a transliterated word entry"""
    if w == '\n': return None
    if re.fullmatch(r'[\d]+', w):
        return 'N' * sum(1 for c in w if c != '0')
    if '⁄' in w: return 'F' * max(1, len(re.findall('⁄', w)))
    if all(cls(c) in ('D', 'L', 'S', 'N', 'F') for c in w) and any(ord(c) > 0xFFFF for c in w):
        return ''.join(cls(c) for c in w)
    if '-' in w: return 'S' * len(w.split('-'))
    return None  # logogram / single sign: length uncertain (ligatures)


def _entry_opts(w):
    """possible class strings an entry of transliteratedWords consumes in parsedInscription"""
    if w in ('—', '[?]', '?'): return ['*']
    if re.fullmatch(r'\d+', w): return ['N' * sum(1 for c in w if c != '0')]
    if '⁄' in w: return ['F' * len(re.findall('⁄', w))]
    if any(ord(c) > 0xFFFF for c in w):
        e = ''.join(cls(c) or '' for c in w); return [e, ''] if set(e) <= {'L'} else [e, '*']
    if '-' in w and '+' not in w: return ['S' * len(w.split('-'))]
    k = w.count('+') + 1
    return ['S' * m for m in range(1, k + 2)] + ['*']


def build():
    import difflib
    J = json.load(open(os.path.join(CK, 'lineara_js.json')))
    C = {d['id']: d for d in json.load(open(os.path.join(DATA, 'corpus_ra.json')))}
    out = {}; st = collections.Counter()
    for did, j in J.items():
        t = j['t'] or ''; p = j['p'] or ''; W = j['w'] or []
        tch = []; line = 0
        for ch in t:
            if ch == '\n': line += 1; continue
            if cls(ch): tch.append((ch, line))
        pch = [ch for ch in p if ch != '\n' and cls(ch)]
        # 1. segment p into entries: DP with free skipping of L chars between entries
        act = ''.join(cls(c) for c in pch)
        ents = [w for w in W if w != '\n']
        opts = [_entry_opts(w) for w in ents]
        best = {0: []}
        def skipL(pos):
            while pos < len(act) and act[pos] == 'L': pos += 1
            return pos
        best = {skipL(0): []}
        for op in opts:
            nb = {}
            for pos, path in best.items():
                for e in op:
                    if e == '*':
                        for m in range(0, 4):
                            q = skipL(pos + m)
                            if q <= len(act) and q not in nb: nb[q] = path + [(pos, m)]
                        continue
                    if act[pos:pos + len(e)] == e:
                        q = skipL(pos + len(e))
                        if q not in nb: nb[q] = path + [(pos, len(e))]
            best = nb
            if not best: break
        if len(act) not in best: st['noseg'] += 1; continue
        spans = best[len(act)]
        # 2. map p chars to physical lines by identity alignment with t
        sm = difflib.SequenceMatcher(None, ''.join(pch), ''.join(c for c, _ in tch), autojunk=False)
        pl = [None] * len(pch); m = 0
        for a, b, n in sm.get_matching_blocks():
            for k in range(n): pl[a + k] = tch[b + k][1]
            m += n
        q = m / max(1, len(pch))
        if q < 0.8: st['lowmatch'] += 1; continue
        recs = []
        for w, (pos, ln) in zip(ents, spans):
            lines = pl[pos:pos + ln]
            recs.append(dict(w=w, lines=lines, cls=act[pos:pos + ln], ok=all(x is not None for x in lines)))
        out[did] = dict(site=C.get(did, {}).get('site', ''), support=C.get(did, {}).get('support', ''),
                        scribe=C.get(did, {}).get('scribe', ''), nphys=line + 1, q=round(q, 3), recs=recs)
        st['ok'] += 1
    json.dump(out, open(os.path.join(CK, 'phys.json'), 'w'))
    print(dict(st))
    return out


def load_phys():
    f = os.path.join(CK, 'phys.json')
    return json.load(open(f)) if os.path.exists(f) else build()


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


if __name__ == '__main__':
    o = build()
    sp = collections.Counter(); ex = []
    for did, d in o.items():
        for r in d['recs']:
            if r.get('ok') and len(set(r['lines'])) > 1:
                sp[r['cls'][0]] += 1
                if r['cls'][0] == 'S' and len(ex) < 40: ex.append((did, r['w'], r['lines']))
    print(sp); [print(e) for e in ex]
