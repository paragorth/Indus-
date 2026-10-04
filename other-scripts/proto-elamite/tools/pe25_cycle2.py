"""pe25 cycle 2: where on the tablet could a second hand hide?

Regions: HDR (header line), BODY (other obverse lines), LAST (last obverse line),
REST (obverse minus last line), REV1 (reverse of tablets with a single reverse
line = a total), REVM (reverse with 2+ lines), REV on tablets with a top-edge
tag (EDGE) or an M157 header (M157).  Each region pair's hand index is compared
with the body-half index on the SAME tablets (pi = 1 - H_pair / H_half).
Plants: second hand confined to that region (5 checkers) must give pi ~ 1,
single hand ~ 0.  Comma channel: obverse vs reverse comma-omission concordance
against the stratum null.
"""
import sys, random, json
import numpy as np
from pe25_common import *
import pe25_cycle1 as c1

OUT = os.path.join(HERE, '..', 'loops', 'pe25_cycle2.txt')
d = c1.d
vb = c1.vb
rr = random.Random(252)

recs = []
for t in d:
    obv, rev, edge = faces(t)
    hdr = [l for l in obv[:1] if l.get('header_comment') or (l['signs'] and not l['numerals'])]
    body = [l for l in obv if l not in hdr]
    b1, b2 = split_halves(body)
    last = obv[-1:] if len(obv) >= 3 else []
    rest = obv[:-1] if last else []
    nrevl = len([l for l in rev if l['signs'] or l['numerals']])
    r = dict(id=t['id'], HDR=tokens(hdr, vb), BODY=tokens(body, vb), B1=tokens(b1, vb), B2=tokens(b2, vb),
             LAST=tokens(last, vb), REST=tokens(rest, vb), REV=tokens(rev, vb),
             edge=any(l['numerals'] for l in edge), m157=bool(hdr and hdr[0]['signs'] and base(hdr[0]['signs'][0]) == 'M157'),
             nrev=nrevl, comma=(comma_habit(obv), comma_habit(rev)))
    recs.append(r)


def region_test(recs, X, Y, subset, ctx=True, boot=200):
    eA, nA = c1.partners(recs, X, Y, random.Random(1))
    eA = [i for i in eA if subset(recs[i])]
    eW, nW = c1.partners(recs, 'B1', 'B2', random.Random(2))
    eW = [i for i in eW if subset(recs[i])]
    if len(eA) < 8 or len(eW) < 8:
        return dict(n=len(eA), nW=len(eW))
    A = c1.stat(recs, eA, nA, X, Y, ctx)
    W = c1.stat(recs, eW, nW, 'B1', 'B2', ctx)
    pi = 1 - A[2] / W[2] if W[2] > 0 else float('nan')
    pis = []
    r = random.Random(3)
    for _ in range(boot):
        sa = [r.choice(eA) for _ in eA]
        sw = [r.choice(eW) for _ in eW]
        a = c1.stat(recs, eA, nA, X, Y, ctx, sa)[2]
        w = c1.stat(recs, eW, nW, 'B1', 'B2', ctx, sw)[2]
        pis.append(1 - a / w if w > 0 else np.nan)
    return dict(n=len(eA), nW=len(eW), HA=A[2], HW=W[2], cells=A[3], pi=pi,
                ci=[float(np.nanpercentile(pis, 5)), float(np.nanpercentile(pis, 95))])


def plant_region(recs, regions, alpha, seed, second=True, subset=lambda r: True):
    """re-draw all variants from tablet hand; tokens in `regions` from a checker."""
    r = random.Random(seed)
    checkers = [c1.profile(alpha, r) for _ in range(5)]
    out = []
    for rec in recs:
        hand = c1.profile(alpha, r)
        chk = r.choice(checkers)
        use2 = second and subset(rec)
        n = dict(rec)
        # body tokens: B1+B2 = BODY order; LAST/REST overlap with BODY -> redraw lines consistently
        def red(toks, h):
            return [(b, c1.draw(h, b, r), k) for b, v, k in toks]
        n['HDR'] = red(rec['HDR'], chk if (use2 and 'HDR' in regions) else hand)
        body = red(rec['BODY'], hand)
        n['BODY'] = body
        n['B1'] = body[:len(rec['B1'])]
        n['B2'] = body[len(rec['B1']):]
        nl = len(rec['LAST'])
        if use2 and 'LAST' in regions:
            lastn = red(rec['LAST'], chk)
        else:
            lastn = body[len(body) - nl:] if nl else []
        n['LAST'] = lastn
        n['REST'] = (n['HDR'] + body)[:len(rec['REST'])]
        n['REV'] = red(rec['REV'], chk if (use2 and 'REV' in regions) else hand)
        out.append(n)
    return out


TESTS = [
    ('HDR-BODY', 'HDR', 'BODY', lambda r: True),
    ('LAST-REST', 'LAST', 'REST', lambda r: True),
    ('BODY-REV1', 'BODY', 'REV', lambda r: r['nrev'] == 1),
    ('BODY-REVM', 'BODY', 'REV', lambda r: r['nrev'] >= 2),
    ('BODY-REV|edge', 'BODY', 'REV', lambda r: r['edge']),
    ('BODY-REV|noedge', 'BODY', 'REV', lambda r: not r['edge']),
    ('BODY-REV|M157', 'BODY', 'REV', lambda r: r['m157']),
    ('BODY-REV|notM157', 'BODY', 'REV', lambda r: not r['m157']),
]

if __name__ == '__main__':
    res = {}
    for name, X, Y, sub in TESTS:
        real = region_test(recs, X, Y, sub)
        reg = X if X in ('HDR', 'LAST') else Y
        pl2 = [region_test(plant_region(recs, {reg}, 1.0, 10 + s, True, sub), X, Y, sub, boot=0 or 1).get('pi') for s in range(2)]
        pl1 = [region_test(plant_region(recs, {reg}, 1.0, 20 + s, False, sub), X, Y, sub, boot=1).get('pi') for s in range(2)]
        res[name] = dict(real=real, plant2=pl2, plant1=pl1)
        print(name, json.dumps(real), 'two-hand plant', pl2, 'one-hand plant', pl1, flush=True)
    # comma channel
    def comma_stat(pairs):
        xs, ys = [], []
        for (no, ko), (nr, kr) in pairs:
            xs.append(ko / no); ys.append(kr / nr)
        xs, ys = np.array(xs), np.array(ys)
        return float(np.mean((xs > 0) == (ys > 0)))
    el = [i for i, r in enumerate(recs) if r['comma'][0][0] and r['comma'][1][0]]
    real_c = comma_stat([recs[i]['comma'] for i in el])
    nulls = []
    for _ in range(500):
        pairs = []
        for i in el:
            j = null_partner(i, c1.groups, c1.keyof, rr, need=lambda j: recs[j]['comma'][1][0] > 0)
            if j is None:
                j = null_partner(i, c1.groups_s, c1.keyof_s, rr, need=lambda j: recs[j]['comma'][1][0] > 0)
            if j is None:
                continue
            pairs.append((recs[i]['comma'][0], recs[j]['comma'][1]))
        nulls.append(comma_stat(pairs))
    nulls = np.array(nulls)
    anyo = sum(1 for i in el if recs[i]['comma'][0][1]); anyr = sum(1 for i in el if recs[i]['comma'][1][1])
    both = sum(1 for i in el if recs[i]['comma'][0][1] and recs[i]['comma'][1][1])
    res['comma'] = dict(n=len(el), real=real_c, null_mean=float(nulls.mean()), p=float(np.mean(nulls >= real_c)), anyo=anyo, anyr=anyr, both=both)
    print(res['comma'], flush=True)
    dump('cycle2.json', res)
    for k, (name, X, Y, sub) in enumerate(TESTS):
        v = res[name]; r = v['real']
        if 'pi' not in r:
            row(OUT, 'PE-25.2%s' % 'abcdefgh'[k], 'Region %s (context-differing pairs); control: second hand planted only there / single hand' % name, 'too few tablets (n %d)' % r['n'], 'untestable')
            continue
        pl2 = [round(x, 2) if x is not None else None for x in v['plant2']]
        pl1 = [round(x, 2) if x is not None else None for x in v['plant1']]
        verdict = 'second hand' if r['ci'][0] > 0.3 else ('same hand (B)' if r['ci'][1] < 0.5 else 'open (C)')
        row(OUT, 'PE-25.2%s' % 'abcdefgh'[k], 'Region %s (context-differing pairs, same-tablet body-half reference); control: second hand planted only there (5 checkers) / single-hand plant' % name,
            'n %d tablets, %d cells; H_pair %.2f vs H_half %.2f; pi %.2f (90%% CI %.2f to %.2f); two-hand plant pi %s, one-hand plant pi %s' % (r['n'], r['cells'], r['HA'], r['HW'], r['pi'], r['ci'][0], r['ci'][1], pl2, pl1), verdict)
    c = res['comma']
    row(OUT, 'PE-25.2i', 'Comma-omission habit (sign+numeral line written without the separator) concordance obverse vs reverse; null: reverse from a same-stratum tablet, 500 draws',
        'n %d tablets; omission on obverse %d, reverse %d, both %d; concordance %.3f vs null %.3f, p %.3f' % (c['n'], c['anyo'], c['anyr'], c['both'], c['real'], c['null_mean'], c['p']),
        'same habit on both faces' if c['p'] < 0.05 else 'no signal')
