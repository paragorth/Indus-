"""v74 TESTING THE FROZEN v72 PREDICTIONS OUTSIDE THE TRANSCRIPTION: shared helpers.
 - ivtff(path): per-line words and separators ('.' certain space, ',' uncertain space) from an IVTFF file
 - glyph-unit splitting, the v72 target and control junction classes
Images stay in the scratch cache; only numbers go to data/v74_ckpt/ (git-ignored)."""
import os, re, json, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v74_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
EVA_MULTI = ['cth', 'ckh', 'cph', 'cfh', 'ch', 'sh']
# frozen P1 junctions (v72_final.txt)
TARGETS = [('or', 'aiin'), ('s', 'aiin'), ('r', 'aiin'), ('r', 'ain'), ('ar', 'al'), ('o', 'l'), ('ol', 'chedy')]
TSET = set(TARGETS)


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def jsave(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=float)


def jload(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


def vglyphs(w):
    out, i = [], 0
    while i < len(w):
        for m in EVA_MULTI:
            if w.startswith(m, i):
                out.append(m); i += len(m); break
        else:
            out.append(w[i]); i += 1
    return out


def clean(t):
    t = t.replace('<->', '-')                        # drawing intrusion = a third separator type
    t = re.sub(r'<[^>]*>', '', t)                    # inline comments, paragraph and line markers
    t = re.sub(r'\[([^:\]]*):[^\]]*\]', r'\1', t)    # [a:b] alternatives -> first reading
    t = re.sub(r'@\d+;', '?', t)
    t = t.replace('{', '').replace('}', '').replace("'", '').replace('!', '').replace('%', '')
    return t


def ivtff(path):
    """{(folio, n): dict(loc, words, seps)} for every transcribed line."""
    out = {}
    for line in open(path, encoding='utf-8', errors='replace'):
        if not line.startswith('<f'): continue
        m = re.match(r'<(f[0-9a-z]+)\.(\d+)[,;]?([^>]*)>\s*(.*)', line.rstrip('\n'))
        if not m: continue
        folio, n, loc, txt = m.group(1), int(m.group(2)), m.group(3), clean(m.group(4))
        toks = re.split(r'([.,\-])', txt)
        words, seps = [], []
        for i, tk in enumerate(toks):
            if i % 2 == 0:
                if tk.strip():
                    words.append(tk.strip())
                elif words and i > 0:
                    pass
            else:
                if words and len(seps) < len(words): seps.append(tk)
        seps = seps[:max(0, len(words) - 1)]
        out[(folio, n)] = dict(loc=loc, words=words, seps=seps)
    return out
