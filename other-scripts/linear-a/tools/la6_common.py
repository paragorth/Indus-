#!/usr/bin/env python3
"""LA-6 shared parsers: 'decipher the economy, not the words'.

la_entries(): Linear A documents cut into entries. A new entry starts at every word
  token except the commodity word NI, and at every logogram that is NOT directly
  followed by a number (a label logogram, e.g. SI+SE). Commodity logograms followed by a
  number carry the current commodity; bare numbers inherit it (HT 30 '/ 14J').
  Each entry: doc, site, support, idx, label (word or None), role (entry/total/deficit/
  grand), commodities {base: value}, sub {full logo: value}, raw [(base, int, fracs)],
  zone (head/before/at/after relative to the first KU-RO of the document).
lb_docs(): Linear B (DAMOS) documents as {commodity: absolute value in major units},
  using the conventional metrological ratios (dry 1 = 10 T = 60 V = 240 Z;
  liquid 1 = 3 S = 18 V = 72 Z). Also per-line entries.
FRAC: conventional lineara.xyz fraction values (used only as a sensitivity choice;
  every test is also run with integers only).
"""
import json, os, re
from collections import defaultdict, Counter
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
FRAC = {'J': Fr(1, 2), 'E': Fr(1, 4), 'F': Fr(1, 8), 'K': Fr(1, 16), 'D': Fr(1, 5), 'B': Fr(1, 3),
        'A': Fr(1, 6), 'H': Fr(1, 6), 'JE': Fr(3, 4), 'L2': Fr(1, 16), 'DD': Fr(1, 10), 'X': Fr(1, 16),
        'L': Fr(1, 4), 'W': Fr(1, 16), 'Y': Fr(1, 8), 'L4': Fr(1, 32), 'L6': Fr(1, 16)}
MAIN = ['GRA', 'OLE', 'OLIV', 'VIN', 'NI', 'CYP', '*304', 'VIR', '*308', 'AROM', 'HIDE', '*86', '*305', 'CAP']
TOTALS = {'KU-RO': 'total', 'KI-RO': 'deficit', 'PO-TO-KU-RO': 'grand'}


def base_of(v):
    parts = [p.strip("'[] ") for p in v.split('+')]
    for p in parts:
        q = p.lstrip('*')
        if q in ('GRA', 'OLE', 'OLIV', 'VIN', 'VINb', 'CYP', 'VIR', 'AROM', 'HIDE', 'CAP'):
            return 'VIN' if q == 'VINb' else q
    p0 = parts[0]
    return p0 if p0.startswith('*') else p0


def qval(t, use_frac=True):
    v = Fr(t['v'])
    if use_frac:
        for f in t['frac']:
            v += FRAC.get(f, Fr(1, 16))
    return v


def la_entries(use_frac=True):
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        toks = ins['tokens']
        ents = []
        cur = None

        def new(label, role='entry'):
            e = {'doc': ins['id'], 'site': ins['site'], 'support': ins['support'], 'idx': len(ents),
                 'label': label, 'role': role, 'com': defaultdict(Fr), 'sub': defaultdict(Fr), 'raw': [], 'bare': []}
            ents.append(e)
            return e
        e = new(None, 'head')
        for i, t in enumerate(toks):
            nxt = next((u for u in toks[i + 1:] if u['t'] not in ('nl', 'div')), None)
            nxt_direct = toks[i + 1] if i + 1 < len(toks) else None
            if t['t'] == 'word':
                if t['s'] == ['NI']:
                    cur = ('NI', 'NI')
                    continue
                w = '-'.join(t['s'])
                if len(t['s']) == 1 and w.startswith('*') and nxt_direct is not None and nxt_direct['t'] == 'num':
                    cur = (w, w)  # single-sign *NNN 'word' used as a commodity logogram (*304 on HT 14)
                    continue
                e = new(w, TOTALS.get(w, 'entry'))
                cur = None
            elif t['t'] == 'logo':
                b = base_of(t['v'])
                if e['role'] in TOTALS.values() and e['bare'] and not e['raw']:
                    e = new(None, 'post')  # commodity lines after a bare KU-RO number (HT 27a)
                if nxt_direct is not None and nxt_direct['t'] == 'num':
                    cur = (b, t['v'])
                elif nxt is not None and nxt['t'] == 'num' and b in MAIN:
                    cur = (b, t['v'])
                else:
                    # label logogram (or commodity heading with no number)
                    if b in MAIN and e['label'] is not None and not e['raw']:
                        cur = (b, t['v'])
                    else:
                        e = new(t['v'], 'entry')
                        cur = (b, t['v']) if b in MAIN else None
            elif t['t'] == 'num':
                v = qval(t, use_frac)
                if cur is None:
                    e['bare'].append(v)
                else:
                    e['com'][cur[0]] += v
                    e['sub'][cur[1]] += v
                    e['raw'].append((cur[0], t['v'], list(t['frac'])))
        # zones
        first_total = next((x['idx'] for x in ents if x['role'] in ('total', 'grand')), None)
        for x in ents:
            if first_total is None:
                x['zone'] = 'nototal'
            elif x['idx'] < first_total:
                x['zone'] = 'before'
            elif x['idx'] == first_total:
                x['zone'] = 'at'
            else:
                x['zone'] = 'after'
            x['com'] = dict(x['com'])
            x['sub'] = dict(x['sub'])
        out.extend(ents)
    return out


LB_COM = {'GRA', 'HORD', 'FAR', 'OLE', 'VIN', 'NI', 'OLIV', 'CYP', 'AROM', 'ME±RI', 'CROC', 'LANA',
          'VIR', 'MUL', 'OVIS', 'CAP', 'SUS', 'BOS', 'TELA', 'KAPO', 'KA±PO', 'AES', 'AUR', 'CORN'}
LIQ_COM = {'OLE', 'VIN', 'ME±RI'}
DRYU = {'T': Fr(1, 10), 'V': Fr(1, 60), 'Z': Fr(1, 240)}
LIQU = {'S': Fr(1, 3), 'V': Fr(1, 18), 'Z': Fr(1, 72)}
NUM = re.compile(r'^\[?(\d+)\]?$')


def lb_docs():
    """Return list of docs: {'id','site','series','lines':[{com:value}], 'com':{com:value}}."""
    out = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        m = re.match(r'^([A-Z]{2,3})\s+([A-Z][a-z]?)', h)
        site, series = (m.group(1), m.group(2)) if m else (h[:2], '')
        lines = []
        for ln in (d.get('content') or '').split('\n'):
            toks = ln.split()
            com = defaultdict(Fr)
            cur = None
            for i, t in enumerate(toks):
                s = t.strip('[]⟦⟧')
                b = s.split('+')[0]
                if b in LB_COM:
                    cur = b
                    continue
                if cur is None:
                    continue
                mm = NUM.match(t)
                if mm and i > 0:
                    prev = toks[i - 1].strip('[]')
                    tab = LIQU if cur in LIQ_COM else DRYU
                    if prev in tab:
                        com[cur] += tab[prev] * int(mm.group(1))
                    elif prev.split('+')[0].strip('[]⟦⟧') in LB_COM:
                        com[cur] += int(mm.group(1))
                    elif prev in ('M', 'N', 'P', 'Q', 'L'):
                        pass
            if com:
                lines.append(dict(com))
        tot = defaultdict(Fr)
        for l in lines:
            for k, v in l.items():
                tot[k] += v
        out.append({'id': h, 'site': site, 'series': series, 'lines': lines, 'com': dict(tot)})
    return out


if __name__ == '__main__':
    E = la_entries()
    print(len(E), Counter(e['role'] for e in E), Counter(e['zone'] for e in E))
    m = [e for e in E if len([c for c in e['com'] if e['com'][c] > 0]) >= 2]
    print('multi-commodity entries', len(m))
    for e in m[:15]:
        print(e['doc'], e['label'], {k: float(v) for k, v in e['com'].items()})
    L = lb_docs()
    print('LB docs', len(L), 'with quantities', sum(1 for x in L if x['com']))
