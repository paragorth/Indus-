"""pe70 'a seal means someone signed for it': shared data.

Three corpora reduced to the same shape: tablet -> opaque token set, structure numbers, physical size,
sealed flag, seal ids.  Seal data are presence and identity only (CDLI catalogue seal_id, ATF seal notes,
the pe33/pe35 figure and catalogue numbers); no picture and no reading of any sign is used.

PE  : data/pe_corpus.json + pe_raw.atf seal notes + pe8_meta / pe10_dims (CDLI catalogue) + pe33/pe35 seal ids.
UR3 : CDLI bulk ATF + catalogue (scratchpad); words as opaque integer ids; @seal / @envelope text dropped
      from content and kept apart (legend) for validation only.
PC  : data/pe2_pc_corpus.json (Uruk V-III) + CDLI catalogue seal_id / ATF @seal.
"""
import json, os, re, sys, csv, collections, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe70_ckpt')
LOOPS = os.path.join(HERE, '..', 'loops')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
os.makedirs(CK, exist_ok=True)


def _cat(periods):
    csv.field_size_limit(10 ** 9)
    out = {}
    for row in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8')):
        if not any(row['period'].startswith(p) for p in periods):
            continue
        out['P%06d' % int(row['id_text'])] = row
    return out


def _f(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except Exception:
        return None


def size_band(area, nlines, qs):
    if area is None:
        return 'L%d' % (0 if nlines <= 3 else 1 if nlines <= 7 else 2 if nlines <= 15 else 3)
    return 'A%d' % int(np.searchsorted(qs, np.log(area)))


# ---------------------------------------------------------------- PE
def build_pe():
    from pe18_common import tablets
    T, _ = tablets()
    dims = json.load(open(os.path.join(DATA, 'pe10_dims.json')))
    meta = json.load(open(os.path.join(DATA, 'pe8_meta.json')))
    m33 = {m['id']: m for m in json.load(open(os.path.join(DATA, 'pe33_seal_motifs.json')))['tablets']}
    m35 = {m['id']: m for m in json.load(open(os.path.join(DATA, 'pe35_seal_motifs.json')))['tablets']}
    recs = []
    for t in T:
        seals = set('PES%04d' % int(s[3:]) for s in t['seals'])
        if t['id'] in m33 and m33[t['id']]['primary']:
            seals |= {'PES%04d' % f for f in m33[t['id']]['figs']}
        if t['id'] in m33:
            seals |= {'PES%04d' % int(str(s).replace('PES', '')) for s in m33[t['id']].get('pes_all', [])}
        if t['id'] in m35:
            seals |= {'PES%04d' % a for a in m35[t['id']]['amiet']}
        mt = meta.get(t['id'], {})
        sealed = bool(t['sealed'] or seals or mt.get('seal_note') or mt.get('seal_id'))
        d = dims.get(t['id'], {})
        h, w, th = _f(d.get('h')), _f(d.get('w')), _f(d.get('t'))
        recs.append(dict(id=t['id'], vol=t['vol'], site=t['site'], toks=sorted(set(t['toks'])), hdr=t['hdr'],
                         sys=t['sys'], fmt=list(map(float, t['fmt'])), n_lines=t['n_lines'], n_ent=t['n_ent'],
                         finals=t['finals'], area=(h * w if h and w else None), thick=th,
                         hw=(h / w if h and w else None), pres=d.get('pres', ''),
                         sealed=sealed, seals=sorted(seals)))
    return recs


# ---------------------------------------------------------------- generic ATF reader (UR3, PC)
NUMRE = re.compile(r'^(\d|n\(|\d+/\d|N\d)')


def read_atf(want, keep_numeral_codes=False):
    """yield (pid, content_lines(list of token lists), legend_tokens, has_seal_surface)"""
    cur = None
    for raw in open(os.path.join(SCR, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            if cur and cur[0] in want:
                yield cur
            pid = raw[1:8]
            cur = [pid, [], [], False, 'tablet']
            continue
        if cur is None or cur[0] not in want:
            continue
        s = raw.strip()
        if not s:
            continue
        if s.startswith('@'):
            k = s[1:].split()[0].lower()
            if k == 'seal':
                cur[3] = True; cur[4] = 'seal'
            elif k in ('envelope',):
                cur[4] = 'envelope'
            elif k in ('tablet', 'bulla', 'object'):
                cur[4] = 'tablet'
            continue
        if s.startswith('$'):
            if re.search(r'seal', s, re.I) and cur[4] != 'seal':
                cur[3] = True
            continue
        if s.startswith('#') or s.startswith('>>') or s.startswith('=:'):
            continue
        m = re.match(r'^[0-9a-z\'.]+\.\s+(.*)$', s)
        if not m:
            continue
        body = m.group(1)
        toks = []
        for w in re.split(r'[\s,]+', body):
            w = re.sub(r'[\[\]#?!<>{}*]', '', w)
            if not w or w in ('x', '...', '|', ':', '$'):
                continue
            if NUMRE.match(w):
                if keep_numeral_codes:
                    toks.append('#' + re.sub(r'^[\dn]+\(', '', w).rstrip(')'))
                continue
            toks.append(w)
        if cur[4] == 'seal':
            cur[2].extend(toks)
        elif cur[4] == 'tablet' and toks:
            cur[1].append(toks)
    if cur and cur[0] in want:
        yield cur


def build_ur3(sites=('Umma', 'Puzrish-Dagan', 'Girsu'), maxn=10 ** 6, seed=70):
    cat = _cat(['Ur III'])
    rng = np.random.default_rng(seed)
    ids = [p for p, r in cat.items() if r['object_type'] == 'tablet' and any(s in r['provenience'] for s in sites)]
    ids = set(rng.choice(sorted(ids), min(maxn, len(ids)), replace=False))
    recs = []
    for pid, lines, leg, hs, _ in read_atf(ids):
        r = cat[pid]
        sid = [x.strip() for x in r['seal_id'].split(';') if x.strip() and x.strip() != 'Sx']
        sealed = bool(r['seal_id'].strip() or hs)
        if not lines:
            continue
        h, w, th = _f(r['height']), _f(r['width']), _f(r['thickness'])
        toks = [x for l in lines for x in l]
        recs.append(dict(id=pid, vol=next(s for s in sites if s in r['provenience']), toks=sorted(set(toks)),
                         lines=lines, n_lines=len(lines), area=(h * w if h and w else None), thick=th,
                         hw=(h / w if h and w else None), sealed=sealed, seals=sorted(set(sid)), legend=leg,
                         date=r['date_of_origin']))
    return recs


def build_pc():
    cat = _cat(['Uruk', 'Jemdet'])
    C = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    want = {c['id'] for c in C}
    sealsurf = {pid: hs for pid, _, _, hs, _ in read_atf(want)}
    recs = []
    for c in C:
        r = cat.get(c['id'])
        if r is None:
            continue
        lines = [l for l in c['lines'] if l['signs'] or l['numerals']]
        if not lines:
            continue
        sid = [x.strip() for x in r['seal_id'].split(';') if x.strip() and x.strip() != 'Sx']
        sealed = bool(r['seal_id'].strip() or sealsurf.get(c['id']))
        toks = sorted({re.sub(r'~[a-z0-9]+', '', s) for l in lines for s in l['signs'] if s not in ('x', 'X')})
        codes = sorted({n[1] for l in lines for n in l['numerals']})
        h, w, th = _f(r['height']), _f(r['width']), _f(r['thickness'])
        recs.append(dict(id=c['id'], vol=c['period'] + '|' + c['provenience'][:12], toks=toks, codes=codes,
                         n_lines=len(lines), n_num=sum(bool(l['numerals']) for l in lines),
                         area=(h * w if h and w else None), thick=th, hw=(h / w if h and w else None),
                         sealed=sealed, seals=sorted(set(sid)), period=c['period']))
    return recs


def get(name):
    p = os.path.join(CK, name + '.pkl')
    if os.path.exists(p):
        return pickle.load(open(p, 'rb'))
    R = {'pe': build_pe, 'ur3': build_ur3, 'pc': build_pc}[name]()
    pickle.dump(R, open(p, 'wb'))
    return R


def bands(R, k=5):
    la = np.array([np.log(r['area']) for r in R if r['area']])
    qs = np.quantile(la, np.linspace(0, 1, k + 1)[1:-1])
    return np.array([size_band(r['area'], r['n_lines'], qs) for r in R])


if __name__ == '__main__':
    for n in sys.argv[1:]:
        R = get(n)
        print(n, len(R), 'sealed', sum(r['sealed'] for r in R), 'with seal id', sum(bool(r['seals']) for r in R),
              'seals >=2 tablets', sum(1 for v in collections.Counter(s for r in R for s in r['seals']).values() if v >= 2))
