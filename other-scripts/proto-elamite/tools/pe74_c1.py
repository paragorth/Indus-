#!/usr/bin/env python3
"""pe74 cycle 1: validate the restoration-aware parser and count restored / uncertain tokens.

Validation:
  (a) round trip: mode 'all' (fix_n off) reproduces pe_corpus.json token for token;
  (b) hand-checked gold lines (data/pe74_gold_lines.json);
  (c) planted control: random read tokens in clean lines are re-marked ([..], <..>, #, ?, #?, spans of
      2-3 tokens inside one bracket, part-restored compounds); the parser must return the planted status.
Counts by token kind, site and publication volume -> data/pe74_ckpt/c1.json
"""
import json, os, random, re, collections, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe74_parse as P

CK = os.path.join(P.DATA, 'pe74_ckpt')
os.makedirs(CK, exist_ok=True)
CODE = {'r': 'read', 'd': 'damaged', 'u': 'uncertain', 'R': 'restored', 'S': 'supplied'}


def gold():
    G = json.load(open(os.path.join(P.DATA, 'pe74_gold_lines.json')))['lines']
    ok = tok = 0
    bad = []
    for raw, es, en in G:
        s, ss, n, ns, _ = P.parse_body(raw)
        k = [i for i, x in enumerate(s) if x != 'n']
        ss = [ss[i] for i in k]
        exp_s = [CODE[c] for c in es]
        exp_n = [CODE[c] for c in en]
        for a, b in zip(ss + ns, exp_s + exp_n):
            tok += 1
            ok += a == b
        if ss != exp_s or ns != exp_n:
            bad.append((raw, ss, ns, exp_s, exp_n))
    return {'lines': len(G), 'lines_ok': len(G) - len(bad), 'tokens': tok, 'tokens_ok': ok, 'bad': bad}


def planted(seed=74, n=3000):
    rng = random.Random(seed)
    T = P.load_marked()
    clean = [l['raw'] for t in T for l in t['lines']
             if not re.search(r'[\[\]<>#?!]', l['raw']) and len(l['raw'].split()) >= 2]
    hits = tot = 0
    conf = collections.Counter()
    for _ in range(n):
        raw = rng.choice(clean)
        toks = raw.split(' ')
        idx = [i for i, x in enumerate(toks) if x not in (',', '') and '...' not in x]
        if not idx:
            continue
        kind = rng.choice(['R', 'S', 'd', 'u', 'du', 'span', 'part'])
        i = rng.choice(idx)
        expect = {}
        if kind == 'R':
            toks[i] = '[' + toks[i] + ']'; expect[i] = 'restored'
        elif kind == 'S':
            toks[i] = '<' + toks[i] + '>'; expect[i] = 'supplied'
        elif kind == 'd':
            toks[i] = toks[i] + '#'; expect[i] = 'damaged'
        elif kind == 'u':
            toks[i] = toks[i] + '?'; expect[i] = 'uncertain'
        elif kind == 'du':
            toks[i] = toks[i] + '#?'; expect[i] = 'uncertain'
        elif kind == 'span':
            j = i
            while j + 1 < len(toks) and j + 1 in idx and j - i < 2:
                j += 1
            toks[i] = '[' + toks[i]; toks[j] = toks[j] + ']'
            for k in range(i, j + 1):
                expect[k] = 'restored'
        elif kind == 'part':
            if not toks[i].startswith('|') or '+' not in toks[i]:
                continue
            a, b = toks[i].rsplit('+', 1)
            toks[i] = a + '+[' + b.rstrip('|') + ']|'; expect[i] = 'restored'
        body = ' '.join(toks)
        # map token index -> status via parse order (tokens are signs then numerals per side; rebuild order)
        got = []
        for piece, side, rfrac, sup in P._tokens(body):
            if '...' in piece:
                continue
            t = re.sub(r'[\[\]#?!*<>]', '', piece)
            if t in ('', '+'):
                continue
            got.append(P.status_of(piece, rfrac, sup))
        order = [k for k in idx]
        if len(got) != len(order):
            conf['len_mismatch'] += 1
            continue
        for pos, k in enumerate(order):
            e = expect.get(k, 'read')
            tot += 1
            hits += got[pos] == e
            if got[pos] != e:
                conf[(e, got[pos])] += 1
    return {'tokens': tot, 'correct': hits, 'errors': {str(k): v for k, v in conf.items()}}


def counts():
    T = P.load_marked()
    by = {'all': collections.Counter()}
    site = collections.defaultdict(collections.Counter)
    vol = collections.defaultdict(collections.Counter)
    tab = collections.Counter()
    slot = collections.Counter()
    for t in T:
        flags = set()
        num_lines = [i for i, l in enumerate(t['lines']) if l['numerals']]
        for i, l in enumerate(t['lines']):
            for kind, sts in (('sign', l['sign_status']), ('num', l['num_status'])):
                for s in sts:
                    key = kind + ':' + s
                    by['all'][key] += 1
                    site[t['site']][key] += 1
                    vol[t['volume']][key] += 1
                    flags.add(s)
            if l.get('unknown_numeral'):
                by['all']['bare_n_dropped'] += 1
            # slots used by earlier B results
            if 'M288' in l['signs']:
                for s in l['num_status']:
                    slot['M288_line_num:' + s] += 1
            if num_lines and i == num_lines[-1] and l['surface'] != 'obverse':
                for s in l['num_status']:
                    slot['total_line_num:' + s] += 1
            if i == 0 and not l['numerals']:
                for s in l['sign_status']:
                    slot['header_sign:' + s] += 1
        tab['tablets'] += 1
        for s in ('damaged', 'uncertain', 'restored', 'supplied'):
            if s in flags:
                tab['with_' + s] += 1
        if {'uncertain', 'restored', 'supplied'} & flags:
            tab['with_unc_or_restored'] += 1
    return by, site, vol, tab, slot


def share(c, kind):
    n = sum(v for k, v in c.items() if k.startswith(kind + ':'))
    r = c[kind + ':restored'] + c[kind + ':supplied']
    u = c[kind + ':uncertain']
    d = c[kind + ':damaged']
    return n, r, u, d


if __name__ == '__main__':
    out = {'roundtrip_mismatched_lines': P.check_against_old(), 'gold': gold(), 'planted': planted()}
    by, site, vol, tab, slot = counts()
    out['totals'] = dict(by['all'])
    out['tablets'] = dict(tab)
    out['slots'] = dict(slot)
    out['site'] = {k: dict(v) for k, v in site.items()}
    out['volume'] = {k: dict(v) for k, v in vol.items()}
    json.dump(out, open(os.path.join(CK, 'c1.json'), 'w'), indent=1)
    g = out['gold']
    print('gold lines %d/%d, tokens %d/%d' % (g['lines_ok'], g['lines'], g['tokens_ok'], g['tokens']))
    for b in g['bad']:
        print('  GOLD MISMATCH', b)
    print('planted', out['planted'])
    print('tablets', dict(tab))
    for kind in ('sign', 'num'):
        n, r, u, d = share(by['all'], kind)
        print('%s tokens %d: restored/supplied %d (%.2f%%), uncertain %d (%.2f%%), damaged %d (%.2f%%)'
              % (kind, n, r, 100 * r / n, u, 100 * u / n, d, 100 * d / n))
    print('slots', dict(slot))
    print('\nby site (sign n, R, U | num n, R, U)')
    for k in sorted(site, key=lambda k: -sum(site[k].values())):
        a = share(site[k], 'sign'); b = share(site[k], 'num')
        print('  %-22s %5d %4d %4d | %5d %3d %3d' % (k, a[0], a[1], a[2], b[0], b[1], b[2]))
    print('\nby volume')
    for k in sorted(vol, key=lambda k: -sum(vol[k].values()))[:20]:
        a = share(vol[k], 'sign'); b = share(vol[k], 'num')
        print('  %-22s %5d %4d %4d | %5d %3d %3d' % (k, a[0], a[1], a[2], b[0], b[1], b[2]))
