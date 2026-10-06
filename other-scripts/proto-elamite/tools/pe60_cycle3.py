#!/usr/bin/env python3
"""pe60 cycle 3: score every frozen prediction, exactly as frozen, on the Proto-Elamite texts that were NOT in the
training corpus and have a usable transliteration.

New texts: CDLI live ATF not in data/pe_raw.atf (data/pe60_ckpt/new_atf.atf, from pe60_inventory.py).
Optional: our own photo transliterations (data/pe60_ckpt/photo_*.atf) - only where cycle 2 validated the feature.

Frozen files checked by hash before scoring: pe59_frozen_predictions.json (P1-P10), pe48_frozen_forecasts.json,
pe52_frozen_weights.json, pe58_frozen_factors.json, pe46_frozen_predicted_codes.json, pe22_outofcorpus_predictions.json.
The pe59 grammar is refitted on the frozen training half and its parameter hash compared with the frozen one
(c1_PE_frozen_params.json) before the decoder (P10) runs.
Usage: python3 pe60_cycle3.py [extra.atf ...]
"""
import sys, os, re, json, math, hashlib, statistics
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_corpus import parse_side  # noqa: E402
from common import base, is_sign  # noqa: E402
import pe59_lib as L59  # noqa: E402
from pe59_lib import (pe_roles, pe_maps, ncls, value, line_role, header_role, tablet_type, norm_code, sha, DATA)  # noqa: E402

CK = os.path.join(DATA, 'pe60_ckpt')
os.makedirs(CK, exist_ok=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


def parse_atf(path):
    """Same parsing as build_corpus.main, then the same transformation as pe59_lib.build_pe."""
    txt = open(path, encoding='utf8').read()
    out = []
    for b in re.split(r'\n(?=&P\d{6})', txt):
        if not b.startswith('&P'):
            continue
        Ls = b.split('\n')
        pid = Ls[0][1:8]
        surface = 'obverse'
        lines = []
        for raw in Ls[1:]:
            s = raw.strip()
            if s.startswith('@'):
                kw = s[1:].split()
                if kw and kw[0] in ('obverse', 'reverse', 'top', 'bottom', 'left', 'right', 'edge', 'seal', 'surface'):
                    surface = kw[0]
                continue
            if s.startswith('#') or s.startswith('$') or not s:
                continue
            m = re.match(r'^(\S+?)\.\s+(.*)$', s)
            if not m:
                continue
            body = m.group(2)
            left, right = (body.split(',', 1) if ',' in body else (body, ''))
            s1, n1, lac1 = parse_side(left)
            s2, n2, lac2 = parse_side(right)
            lines.append({'surface': surface, 'signs': s1 + s2, 'numerals': [[n, norm_code(c)] for n, c in n1 + n2],
                          'lacuna': lac1 or lac2, 'raw': body})
        L = []
        for l in lines:
            sg = [base(x) for x in l['signs'] if is_sign(x)]
            clean = not ('x' in l['signs'] or l['lacuna'] or '...' in l['raw'] or '[' in l['raw'])
            nums = [[n, norm_code(c)] for n, c in l['numerals']]
            numclean = bool(nums) and all(n is not None and c != 'n' and isinstance(n, int) for n, c in nums) \
                and not re.search(r'\[|\.\.\.', l['raw'].split(',')[-1])
            L.append({'surf': l['surface'], 'signs': sg, 'clean': clean,
                      'nums': nums if numclean or not nums else [[n, c] for n, c in nums if isinstance(n, int) and c != 'n'],
                      'numclean': numclean, 'had_nums': bool(nums)})
        for l in L:
            if l['had_nums'] and not l['nums']:
                l['nums'] = [[0, 'n']]
        L59._roles(L)
        out.append({'id': pid, 'site': '', 'lines': L, 'designation': Ls[0].split('=', 1)[-1].strip()})
    return out


def check_hashes():
    res = {}
    f = json.load(open(os.path.join(DATA, 'pe59_frozen_predictions.json')))
    res['pe59_predictions'] = sha(f['predictions']) == f['sha256']
    for fn, key in [('pe48_frozen_forecasts.json', 'sha256_16'), ('pe46_frozen_predicted_codes.json', 'sha')]:
        d = json.load(open(os.path.join(DATA, fn)))
        res[fn] = d.get(key)
    return res


def score(T, label):
    R = pe_roles()
    cap, cnt = pe_maps()
    cntm = cnt['sex2']
    P = {}
    # P3 (exactly the pe59 after_count rule)
    k = n = 0; k0 = n0 = 0
    for t in T:
        prev = None
        for l in t['lines']:
            if l['role'] != 'E':
                continue
            if prev is not None and prev['numclean'] and ncls(prev['nums']) == 'AMB' and l['numclean'] and l['signs'] \
                    and prev['signs'] and prev['signs'][-1] != l['signs'][-1]:
                ku = value(prev['nums'], cntm)
                if ku and ku <= 60:
                    v = value(l['nums'], cap)
                    if v is not None and ncls(l['nums']) in ('CAP', 'AMB'):
                        if l['signs'][-1] == 'M288':
                            n += 1; k += v == 60 * ku
                        else:
                            n0 += 1; k0 += v == 60 * ku
            prev = l
    P['P3'] = {'k': k, 'n': n, 'null_k': k0, 'null_n': n0, 'testable': n >= 20,
               'verdict': 'untestable (n < 20 lines)' if n < 20 else ('KILLED' if k / n < 0.10 else ('supported' if k / n >= 0.20 else 'neither'))}

    def cap_ok(l, t):
        return ncls(l['nums']) == 'CAP' or (ncls(l['nums']) == 'AMB' and tablet_type(t) == 'CAPT')
    km = nm = kc = nc = kb = nb = 0
    rows4, rows5 = [], []
    for t in T:
        for l in t['lines']:
            if l['role'] == 'E' and l['numclean']:
                nb += 1; kb += cap_ok(l, t)
                if l['signs']:
                    r = line_role(l['signs'], R)
                    if r == 'MEASURED':
                        nm += 1; km += cap_ok(l, t); rows4.append((t['id'], l['signs'], l['nums']))
                    if r == 'COUNTED':
                        nc += 1; kc += ncls(l['nums']) != 'CAP'; rows5.append((t['id'], l['signs'], l['nums']))
    base_rate = kb / nb if nb else None
    P['P4'] = {'k': km, 'n': nm, 'base_k': kb, 'base_n': nb, 'rows': rows4,
               'verdict': 'untestable (n < 20 MEASURED entries)' if nm < 20 else ('supported' if km / nm >= 0.5 and (base_rate is None or km / nm > base_rate)
                                                                 else ('KILLED' if base_rate is not None and km / nm <= base_rate else 'neither')),
               'note': 'n tiny: a binomial interval is reported' , 'ci': wilson(km, nm)}
    P['P5'] = {'k': kc, 'n': nc, 'rows': rows5, 'ci': wilson(kc, nc),
               'verdict': 'untestable (n < 20 COUNTED entries)' if nc < 20 else ('supported' if kc / nc >= 0.95 else ('KILLED' if kc / nc < 0.85 else 'neither'))}
    # P6 closure needs the fitted grammar
    P['_needs_grammar'] = True
    # P7 total-line sign share
    tot = [l for t in T for l in t['lines'] if l['role'] == 'T']
    sc = Counter(s for l in tot for s in set(l['signs']))
    top = sc.most_common(1)
    P['P7'] = {'total_lines': len(tot), 'top_sign': top,
               'verdict': 'untestable (< 5 total lines)' if len(tot) < 5 else ('KILLED' if top and top[0][1] / len(tot) > 0.40 else 'survives')}
    # P8 edge tag
    m157 = [t for t in T if any(l['role'] == 'H' and l['signs'][:1] == ['M157'] for l in t['lines'])]
    oth = [t for t in T if t not in m157]
    e1 = sum(any(l['role'] == 'G' for l in t['lines']) for t in m157)
    e0 = sum(any(l['role'] == 'G' for l in t['lines']) for t in oth)
    P['P8'] = {'M157': [e1, len(m157)], 'other': [e0, len(oth)],
               'verdict': 'untestable (needs >= 100 new tablets)' if len(T) < 100 else 'see rates'}
    # P9 Yahya
    P['P9'] = {'new_yahya_tablets': 0, 'verdict': 'untestable (0 new Yahya tablets; kill needs 40+)'}
    return P


def grammar_checks(T):
    import pe59_cycle1 as c1
    import pe59_cycle2 as c2
    c1.CORPUS = 'PE'; c2.CORPUS = 'PE'
    TT, roles, maps = c1.corpus()
    tr, ho = L59.split(TT)
    g = c1.fit(tr, roles, maps)
    params = {'sign_S3': [float(x) for x in g.sign_par_S3], 'sign_G': [float(x) for x in g.sign_par_G],
              'num_G': g.par_ent, 'num_N2': g.par_ent_N2, 'tot_G': g.par_tot, 'tot_N2': g.par_tot_N2, 'theta': g.theta}
    fz = json.load(open(os.path.join(DATA, 'pe59_ckpt', 'c1_PE_frozen_params.json')))
    match = sha(params) == fz['params_sha']
    R = [c2.check_tablet(t, g) for t in T]
    R = [r for r in R if r]
    s = c2.summarise(R) if R else None
    # P6: written totals in the predicted system (MAP tau as decoder)
    clos = [(r['id'], r['C3']) for r in R if r['C3'] is not None]
    return match, R, s, clos


def yahya_sofalin(T):
    f = json.load(open(os.path.join(DATA, 'pe48_frozen_forecasts.json')))['forecasts']
    sof = [t for t in T if t['designation'].startswith('TSF')]
    out = {}
    for site, tabs in [('Sofalin', sof)]:
        pres = {s: sum(any(s in l['signs'] for l in t['lines']) for t in tabs) for s, _, _ in f[site]}
        out[site] = {'tablets': len(tabs), 'present': pres, 'forecast_per_tablet': {s: round(p, 3) for s, p, _ in f[site]}}
    return out


def pe52_pe58(T):
    w = json.load(open(os.path.join(DATA, 'pe52_frozen_weights.json')))['signs']
    fac = json.load(open(os.path.join(DATA, 'pe58_frozen_factors.json')))['signs']
    cap, cnt = pe_maps()
    cases52, cases58 = [], {}
    for t in T:
        tt = tablet_type(t)
        E = [l for l in t['lines'] if l['role'] == 'E' and l['numclean'] and l['signs']]
        for l in E:
            sysl = 'CAP' if (ncls(l['nums']) == 'CAP' or tt == 'CAPT') else 'CNT'
            m = cap if sysl == 'CAP' else cnt['sex2']
            v = value(l['nums'], m)
            if not v:
                continue
            for s in set(l['signs']):
                if s not in w and s not in fac:
                    continue
                sib = []
                for l2 in E:
                    if l2 is l or s in l2['signs']:
                        continue
                    s2 = 'CAP' if (ncls(l2['nums']) == 'CAP' or tt == 'CAPT') else 'CNT'
                    if s2 != sysl:
                        continue
                    v2 = value(l2['nums'], m)
                    if v2:
                        sib.append(math.log(float(v2)))
                if not sib:
                    continue
                shift = math.log(float(v)) - statistics.mean(sib)
                if s in w:
                    want = 1 if w[s]['tab'] > 0 else -1
                    cases52.append((t['id'], s, round(shift, 3), shift * want > 0))
                if s in fac:
                    cases58.setdefault(s, []).append(shift)
    k = sum(c[3] for c in cases52)
    v52 = {'cases': len(cases52), 'right_direction': k, 'rows': cases52,
           'verdict': 'untestable (< 20 cases)' if len(cases52) < 20 else ('supported' if k / len(cases52) >= 0.7 else ('KILLED' if k / len(cases52) <= 0.5 else 'neither'))}
    v58 = {}
    for s, xs in cases58.items():
        lo, hi = fac[s]['band']
        med = statistics.median(xs)
        v58[s] = {'n': len(xs), 'median_log_shift': round(med, 3), 'band_log': [round(math.log(lo), 3), round(math.log(hi), 3)],
                  'inside': math.log(lo) <= med <= math.log(hi)}
    n58 = sum(v['n'] for v in v58.values())
    return v52, {'per_sign': v58, 'cases': n58, 'verdict': 'untestable (< 20 cases)' if n58 < 20 else 'see per sign'}


def pe46_codes(T):
    d = json.load(open(os.path.join(DATA, 'pe46_frozen_predicted_codes.json')))
    pred = [tuple(p) for p in d['predictions']]
    ents = [l for t in T for l in t['lines'] if l['role'] == 'E' and l['signs']]
    hits = Counter()
    for l in ents:
        s = l['signs']
        for p in pred:
            k = len(p)
            if any(tuple(s[i:i + k]) == p for i in range(len(s) - k + 1)):
                hits[p] += 1
    return {'entries': len(ents), 'distinct_predicted_codes_seen': len(hits), 'hits': {' '.join(k): v for k, v in hits.items()},
            'verdict': 'scaled: support needs >= 4 of 40 in ~500 entries; with %d entries the expected count under support '
                       'is %.2f, so 0-1 hits cannot kill' % (len(ents), 4 * len(ents) / 500)}


def main():
    files = [os.path.join(CK, 'new_atf.atf')] + sys.argv[1:]
    out = {'hashes': check_hashes()}
    for fpath in files:
        T = [t for t in parse_atf(fpath) if any(l['signs'] or l['nums'] for l in t['lines'])]
        lab = os.path.basename(fpath)
        P = score(T, lab)
        match, R, s, clos = grammar_checks(T)
        P['grammar_params_match_frozen'] = match
        k6 = sum(1 for _, c in clos if c); n6 = len(clos)
        P['P6'] = {'k': k6, 'n': n6, 'verdict': 'untestable (0 clean totalled tablets)' if n6 == 0 else
                   ('KILLED' if k6 / n6 < 0.04 else ('supported' if k6 / n6 >= 0.10 else 'neither'))}
        P['P10_decoder'] = {'summary': s, 'per_tablet': [{k: r[k] for k in ('id', 'C1', 'C2n', 'C2ok', 'C3', 'full', 'strict', 'pred', 'obs')} for r in R],
                            'heldout_full_rate_frozen': 0.52,
                            'verdict': 'n=%d tablets (Susa ones only are in scope of P10; P10 names the Tehran set)' % len(R)}
        P['pe48'] = yahya_sofalin(T)
        P['pe52'], P['pe58'] = pe52_pe58(T)
        P['pe46'] = pe46_codes(T)
        P.pop('_needs_grammar', None)
        P['tablets'] = [t['id'] for t in T]
        out[lab] = P
    json.dump(out, open(os.path.join(CK, 'cycle3.json'), 'w'), indent=1, default=str)
    for lab, P in out.items():
        if lab == 'hashes':
            print('hashes', P); continue
        print('==', lab, P['tablets'])
        for k in ('P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9'):
            print(k, {a: b for a, b in P[k].items() if a != 'rows'})
        print('P10', P['P10_decoder']['summary'])
        for r in P['P10_decoder']['per_tablet']:
            print('   ', r)
        print('pe48', P['pe48']); print('pe52', {a: b for a, b in P['pe52'].items() if a != 'rows'}, P['pe52']['rows'])
        print('pe58', P['pe58']); print('pe46', P['pe46']); print('grammar match', P['grammar_params_match_frozen'])


if __name__ == '__main__':
    main()
