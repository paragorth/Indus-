#!/usr/bin/env python3
"""LA-69 NOTES TO SELF: a reader-independence fingerprint.

Hypothesis (arrow in the dark): Linear A documents are private working notes, readable only by the writer
and people in the room, not records for a later reader.  Each document is scored on how far it can be
understood from the REST of its corpus (vocabulary, word pairs, line shapes and opening words attested in
other documents), how explicit it is (labelled numbers, header line, explicit total, words per number,
formulaic repetition) and how private its abbreviations are.  Calibrated on administrations of known purpose:
  LB  Linear B KN+PY (DAMOS): leaf-shaped (unnumbered or lettered lines) vs page-shaped (numbered lines);
      PY Eb+Eo (known drafts) vs PY Ep+En (their fair copies).
  UR3 Ur III admin (CDLI): undated AND unsealed (internal) vs dated AND sealed (record for others).
      Date lines (mu / iti / u4 ...) and seal surfaces are removed from every document before scoring.
  OA  Old Assyrian (CDLI): merchants' memoranda vs legal documents (debt notes, contracts).
  PLANT  final records with context deleted (headers, totals, labels, private abbreviations, merged lines).
No sign values are used anywhere; tokens are opaque ids.
"""
import os, re, sys, json, math, random, csv, collections, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la69_ckpt')
os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(HERE, '..', 'loops')
import la57_common as L57
L57.CK = CK                       # every cache goes to la69_ckpt
SCRATCH = L57.SCRATCH


def seed(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def cache(name, fn):
    p = os.path.join(CK, name + '.json')
    if os.path.exists(p):
        return [dict(d, toks=[tuple(x) for x in d['toks']]) for d in json.load(open(p))]
    out = fn()
    json.dump(out, open(p, 'w'))
    return out


def ok(d, minT=2, minN=1):
    return sum(1 for x in d['toks'] if x[0] == 'T') >= minT and sum(1 for x in d['toks'] if x[0] == 'N') >= minN


# ------------------------------------------------------------------ Linear A
def _la():
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        toks = []
        for t in ins['tokens']:
            if t['t'] == 'word':
                toks.append(('T', '-'.join(t['s'])))
            elif t['t'] == 'logo':
                toks.append(('T', 'L:' + t['v']))
            elif t['t'] == 'num':
                toks.append(('N', float(t['v']), bool(t['frac'])))
            elif t['t'] == 'nl' and toks and toks[-1][0] != 'L':
                toks.append(('L',))
        while toks and toks[-1][0] == 'L':
            toks.pop()
        out.append({'id': ins['id'], 'sys': 'LA', 'site': ins['site'], 'support': ins.get('support', ''),
                    'scribe': ins.get('scribe') or '', 'toks': toks})
    return out


def la_docs():
    return [d for d in cache('la_docs', _la) if ok(d)]


# ------------------------------------------------------------------ Linear B with shape class
def _lb():
    base = L57._lb()
    shape = {}
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        labs = re.findall(r'^\s*\.(\w+)', d.get('content') or '', re.M)
        if labs and all(x[:1].isdigit() for x in labs):
            shape[h] = 'page'
        elif not labs or all(x.isalpha() for x in labs):
            shape[h] = 'leaf'
        else:
            shape[h] = 'mix'
    for d in base:
        d['shape'] = shape.get(d['id'], 'mix')
    return base


def lb_docs():
    return [d for d in cache('lb_docs', _lb) if ok(d)]


# ------------------------------------------------------------------ CDLI with date / seal flags
DATE = re.compile(r"^(mu|iti|u4|itu|li-mu-um|li-mu|limum|iti-kam|{iti})(?=\s|$)|^u4-\d|^iti-")


def _cdli_flags(period_pred, tag, genre_pred, nmax=None):
    csv.field_size_limit(10 ** 9)
    keep = {}
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if period_pred(row['period']) and genre_pred(row.get('genre', ''), row.get('subgenre', '')):
            keep['P%06d' % int(row['id_text'])] = (row['provenience'].split(' (')[0],
                                                   row.get('genre', ''), row.get('subgenre', ''))
    docs = []
    st = {}

    def flush():
        cur = st.get('cur')
        if not cur:
            return
        t = list(st['toks'])
        while t and t[-1][0] == 'L':
            t.pop()
        prov, g, sg = keep[cur]
        docs.append({'id': cur, 'sys': tag, 'site': prov, 'genre': g, 'subgenre': sg,
                     'dated': st['dated'], 'sealed': st['sealed'], 'toks': t})
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            flush()
            cur = raw[1:8]
            st = {'cur': cur if cur in keep else None, 'toks': [], 'surf': 'obverse', 'dated': False,
                  'sealed': False}
            continue
        if not st.get('cur'):
            continue
        if raw.startswith('@'):
            w = raw[1:].split()[0] if raw[1:].split() else ''
            if w in ('seal', 'envelope'):
                st['surf'] = 'seal'; st['sealed'] = True
            elif w in ('obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge', 'column', 'tablet'):
                if st['surf'] == 'seal' and w != 'column':
                    st['surf'] = w
            continue
        if raw.startswith('$') and 'seal' in raw.lower():
            st['sealed'] = True
            continue
        if st['surf'] == 'seal' or not raw[:1].isdigit():
            continue
        m = re.match(r"^\d+'?\.\s+(.*)$", raw.rstrip())
        if not m:
            continue
        body = re.sub(r'[\[\]#?!<>*]|\(\$.*?\$\)', '', m.group(1)).replace('_', ' ').strip()
        if DATE.match(body.lower()):
            st['dated'] = True               # date line: flag it, but do NOT keep its tokens
            continue
        line, nums = [], []
        for x in body.split():
            mm = L57.NUMRE.match(x)
            if mm:
                nums.append((mm.group(1), mm.group(2)))
                continue
            if nums:
                line.append(L57._numtok(nums, line)); nums = []
            if x in ('...', 'x') or x.startswith('x-') or x == 'n' or x.startswith('n('):
                continue
            line.append(('T', x.lower()))
        if nums:
            line.append(L57._numtok(nums, line))
        if line:
            st['toks'].extend(line); st['toks'].append(('L',))
    flush()
    docs = [d for d in docs if ok(d)]
    rng = random.Random(seed('la69-' + tag))
    rng.shuffle(docs)
    return docs[:nmax] if nmax else docs


def ur3_docs():
    def build():
        ds = _cdli_flags(lambda p: p.startswith('Ur III'), 'UR3', lambda g, s: g == 'Administrative')
        notes = [d for d in ds if not d['dated'] and not d['sealed']]
        finals = [d for d in ds if d['dated'] and d['sealed']]
        rest = [d for d in ds if (d['dated'] != d['sealed'])]
        rng = random.Random(7)
        rng.shuffle(notes); rng.shuffle(finals); rng.shuffle(rest)
        return notes[:2500] + finals[:2500] + rest[:2500]
    return cache('ur3_docs', build)


def oa_docs():
    return cache('oa_docs', lambda: _cdli_flags(lambda p: p.startswith('Old Assyrian'), 'OA',
                                                lambda g, s: g.startswith('Administrative') or g.startswith('Legal')))


def ob_docs():
    return cache('ob_docs', lambda: _cdli_flags(lambda p: p.startswith('Old Babylonian'), 'OB',
                                                lambda g, s: g == 'Administrative', nmax=4000))


def pc_docs():
    return [d for d in cache('pc_docs', L57._pc) if ok(d)]


def kh_docs():
    return [d for d in cache('kh_docs', L57._kh) if ok(d)]


# ------------------------------------------------------------------ calibration sets
def calib_sets():
    """{corpus: {'ref': docs, 'NOTE': docs, 'FINAL': docs, 'group': key}} (each corpus may hold 2 contrasts)."""
    lb = lb_docs()
    E = lambda d: d['site'] == 'PY' and d.get('series', '')[:2] in ('Eb', 'Eo', 'Ep', 'En', 'Ea', 'Ed', 'Eq', 'Er', 'Es')
    ur = ur3_docs()
    oa = oa_docs()
    S = {
        'LB_shape': {'ref': lb, 'NOTE': [d for d in lb if d['shape'] == 'leaf' and not E(d)],
                     'FINAL': [d for d in lb if d['shape'] == 'page' and not E(d)]},
        'LB_draft': {'ref': [d for d in lb if not E(d)],
                     'NOTE': [d for d in lb if d['site'] == 'PY' and d.get('series', '')[:2] in ('Eb', 'Eo')],
                     'FINAL': [d for d in lb if d['site'] == 'PY' and d.get('series', '')[:2] in ('Ep', 'En')]},
        'UR3': {'ref': ur, 'NOTE': [d for d in ur if not d['dated'] and not d['sealed']],
                'FINAL': [d for d in ur if d['dated'] and d['sealed']]},
        'OA': {'ref': oa, 'NOTE': [d for d in oa if 'memo' in d['subgenre'].lower()],
               'FINAL': [d for d in oa if d['genre'].startswith('Legal')]},
    }
    return S


# ------------------------------------------------------------------ planted notes (context deletion)
def plant_notes(docs, rng, p_abbr=0.5, p_drop_label=0.3, p_merge=0.4):
    out = []
    for d in docs:
        lines, cur = [], []
        for x in d['toks']:
            if x[0] == 'L':
                if cur:
                    lines.append(cur)
                cur = []
            else:
                cur.append(x)
        if cur:
            lines.append(cur)
        # 1 drop header lines (no number) at the top
        while len(lines) > 1 and not any(x[0] == 'N' for x in lines[0]):
            lines.pop(0)
        # 2 drop explicit totals: lines whose number = sum of earlier numbers
        seen, keep = [], []
        for ln in lines:
            ns = [x[1] for x in ln if x[0] == 'N' and x[1] is not None]
            if ns and len(seen) >= 2 and abs(ns[-1] - sum(seen)) < 1e-6 and len(keep) >= 1:
                continue
            seen += ns
            keep.append(ln)
        lines = keep or lines
        # 3 private abbreviations + 4 drop labels the writer knows
        abbr = {}
        new = []
        for ln in lines:
            nl = []
            for i, x in enumerate(ln):
                if x[0] == 'T':
                    nxtN = i + 1 < len(ln) and ln[i + 1][0] == 'N'
                    if nxtN and rng.random() < p_drop_label and sum(1 for y in ln if y[0] == 'T') > 1:
                        continue
                    if rng.random() < p_abbr:
                        abbr.setdefault(x[1], 'ab:%s:%d' % (d['id'], len(abbr)))
                        nl.append(('T', abbr[x[1]]))
                        continue
                nl.append(x)
            if nl:
                new.append(nl)
        # 5 merge lines (format varies)
        toks = []
        for i, ln in enumerate(new):
            if i and rng.random() >= p_merge:
                toks.append(('L',))
            toks.extend(ln)
        nd = dict(d, toks=toks, id='PL:' + str(d['id']))
        if ok(nd, 1, 1):
            out.append(nd)
    return out


# ------------------------------------------------------------------ features
FEATS = ['VOC', 'VOC3', 'BIG', 'LSH', 'FIRST', 'LAB', 'HDR', 'TOT', 'ENT', 'PRIV', 'REP', 'BARE', 'XGRP']
SIGN = {'VOC': 1, 'VOC3': 1, 'BIG': 1, 'LSH': 1, 'FIRST': 1, 'LAB': 1, 'HDR': 1, 'TOT': 1, 'ENT': 1,
        'PRIV': -1, 'REP': 1, 'BARE': -1, 'XGRP': 1}
NULLADJ = ['VOC', 'VOC3', 'BIG', 'FIRST', 'PRIV', 'REP', 'XGRP']   # minus type-shuffled expectation


def lines_of(d):
    lines, cur = [], []
    for x in d['toks']:
        if x[0] == 'L':
            if cur:
                lines.append(cur)
            cur = []
        else:
            cur.append(x)
    if cur:
        lines.append(cur)
    return lines


def nsign(t):
    return 1 if t.startswith('L:') or t.startswith('K:') else t.count('-') + 1


def shape(ln):
    s = ''.join('N' if x[0] == 'N' else 'T' for x in ln)
    return re.sub(r'T{3,}', 'TTT', re.sub(r'N{2,}', 'NN', s))


class Pool:
    def __init__(self, docs, groupkey='site'):
        self.df = collections.Counter()
        self.grp = collections.defaultdict(set)
        self.big = set(); self.lsh = set(); self.first = set()
        self.tok = []
        for d in docs:
            ts = set()
            for ln in lines_of(d):
                self.lsh.add(shape(ln))
                for a, b in zip(ln, ln[1:]):
                    self.big.add((a[1] if a[0] == 'T' else 'N', b[1] if b[0] == 'T' else 'N'))
                for x in ln:
                    if x[0] == 'T':
                        ts.add(x[1]); self.tok.append(x[1])
            for t in ts:
                self.df[t] += 1
                self.grp[t].add(d.get(groupkey))
            fw = next((x[1] for x in d['toks'] if x[0] == 'T'), None)
            if fw:
                self.first.add(fw)
        self.ids = {d['id'] for d in docs}


def raw_feats(d, P, gk='site', abbr_ok=True):
    lines = lines_of(d)
    T = [x[1] for x in d['toks'] if x[0] == 'T']
    Ns = [x for x in d['toks'] if x[0] == 'N']
    f = {}
    f['VOC'] = np.mean([P.df[t] > 0 for t in T])
    f['VOC3'] = np.mean([P.df[t] >= 3 for t in T])
    bg = [(a[1] if a[0] == 'T' else 'N', b[1] if b[0] == 'T' else 'N') for ln in lines for a, b in zip(ln, ln[1:])]
    bg = [b for b in bg if b != ('N', 'N')]
    f['BIG'] = np.mean([b in P.big for b in bg]) if bg else np.nan
    f['LSH'] = np.mean([shape(ln) in P.lsh for ln in lines])
    f['FIRST'] = float(T[0] in P.first) if T else np.nan
    lab = []
    for ln in lines:
        seenT = False
        for x in ln:
            if x[0] == 'T':
                seenT = True
            else:
                lab.append(seenT)
    f['LAB'] = np.mean(lab) if lab else np.nan
    f['HDR'] = float(not any(x[0] == 'N' for x in lines[0])) if lines else np.nan
    vals = [x[1] for x in Ns if x[1] is not None]
    tot = 0.0
    for i in range(2, len(vals)):
        if vals[i] > 0 and abs(vals[i] - sum(vals[:i])) < 1e-6:
            tot = 1.0
        for j in range(0, i - 1):
            if vals[i] > 0 and abs(vals[i] - sum(vals[j:i])) < 1e-6 and i - j >= 2:
                tot = 1.0
    f['TOT'] = tot
    f['ENT'] = math.log((len(T) + 0.5) / (len(Ns) + 0.5))
    f['PRIV'] = np.mean([nsign(t) == 1 and P.df[t] == 0 for t in T])
    c = collections.Counter(T)
    f['REP'] = np.mean([c[t] > 1 for t in T])
    f['BARE'] = np.mean([not any(x[0] == 'T' for x in ln) for ln in lines])
    g = d.get(gk)
    xs = [bool(P.grp[t] - {g}) for t in T]
    f['XGRP'] = np.mean(xs) if xs else np.nan
    return f


def null_doc(d, P, rng):
    """Same skeleton, word tokens drawn from the pool's token distribution (vocabulary-size null)."""
    if not P.tok:
        return d
    return dict(d, toks=[('T', P.tok[rng.randrange(len(P.tok))]) if x[0] == 'T' else x for x in d['toks']])


def doc_feats(d, P, rng, nnull=3):
    f = raw_feats(d, P)
    nf = [raw_feats(null_doc(d, P, rng), P) for _ in range(nnull)]
    out = {}
    for k in FEATS:
        if k in NULLADJ:
            out[k] = f[k] - np.nanmean([x[k] for x in nf]) if not np.isnan(f[k]) else np.nan
        else:
            out[k] = f[k]
    nT = sum(1 for x in d['toks'] if x[0] == 'T')
    out['_lt'] = math.log(nT)
    out['_ll'] = math.log(len(lines_of(d)))
    return out


def score_docs(docs, ref, rng, M=300, npools=4, exclude=None):
    """Score docs against random pools of M reference docs that never contain the scored doc."""
    ids = [d['id'] for d in docs]
    ref = [r for r in ref if not (exclude and exclude(r))]
    pools = []
    for k in range(npools):
        sub = rng.sample(ref, min(M, len(ref) - 1))
        pools.append((Pool(sub), {r['id'] for r in sub}))
    out = []
    for d in docs:
        cands = [p for p in pools if d['id'] not in p[1]]
        P = cands[rng.randrange(len(cands))][0] if cands else Pool([r for r in rng.sample(ref, M) if r['id'] != d['id']])
        out.append(doc_feats(d, P, rng))
    return out


def wrow(path, cells):
    with open(path, 'a') as fh:
        fh.write('| ' + ' | '.join(str(c).replace('\n', ' ') for c in cells) + ' |\n')
