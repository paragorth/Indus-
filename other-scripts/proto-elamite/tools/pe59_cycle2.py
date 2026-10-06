"""pe59 cycle 2: DECODE. For every held-out tablet, write a structured gloss with the frozen reading and grammar
(fitted on the training half only) and check whether it explains the tablet fully:
  C1 system: the tablet type predicted from the signs alone is consistent with every numeral on the tablet
  C2 roles : every role-bearing entry carries a numeral its role allows (measured -> capacity, counted -> not
             capacity, person -> small count, allotment after a count line -> 60 or 120 N39C per unit)
  C3 total : the written total equals the sum of the entries in the predicted system
Controls: the same decoder with sign roles shuffled (refitted), and real roles on tablets whose numerals were
transplanted from another held-out tablet (same number of entries where possible).

usage: python3 pe59_cycle2.py [PE|PC] [n_shuffles]
"""
import sys, os, json, math, random, time
from collections import Counter
from multiprocessing import Pool
from fractions import Fraction as Fr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe59_lib import (READING, split, shuffle_roles, sha, seed, tablet_type, ncls, nkey, value, line_role,
                      header_role, CK, DATA, LOOPS, AMB)
import pe59_cycle1 as c1

CORPUS = sys.argv[1] if len(sys.argv) > 1 else 'PE'
NSH = int(sys.argv[2]) if len(sys.argv) > 2 else 8
c1.CORPUS = CORPUS

ROLE_GLOSS = {'MEASURED': 'grain-class commodity (measured)', 'COUNTED': 'counted item', 'PERSON': 'person-unit',
              'ALLOT': 'standard allotment', 'FRACLINE': 'fraction line', 'MEASURED_IN': 'string with a measured-class sign',
              'COUNTED_IN': 'string with a counted-class sign', 'OTHER': 'name / unclassified string', 'BARE': 'bare number'}
HDR_GLOSS = {'H_GEN': 'general opener', 'H_327': 'M327-family opener', 'H_CAP': 'capacity-account opener',
             'H_OTH': 'other opener', 'H_X': 'unclassified header', 'NONE': 'no header'}


def fmt_q(nums, tau, g):
    c = ncls(nums)
    if c == 'CAP' or (c == 'AMB' and tau == 'CAPT'):
        v = value(nums, g.capmap)
        if v is None:
            return '?', None
        if CORPUS == 'PE':
            lo, hi = float(v) * 0.6, float(v) * 0.8
            return '%s N39C (%.3g-%.3g l)' % (v, lo, hi), v
        return '%s N01-grain' % v, v
    m = g.cntmaps[g.cnt_main]
    v = value(nums, m)
    if v is None:
        return '?', None
    return '%s units' % v, v


def check_tablet(t, g, numerals_from=None):
    """Returns dict with gloss and checks. numerals_from: list of numeral groups to transplant onto the entries."""
    lines = [dict(l) for l in t['lines']]
    if numerals_from is not None:
        E = [i for i, l in enumerate(lines) if l['role'] in ('E', 'T')]
        for i, nums in zip(E, numerals_from):
            lines[i]['nums'], lines[i]['numclean'] = nums[0], nums[1]
    tt = dict(t); tt['lines'] = lines
    ents = [l for l in lines if l['role'] == 'E']
    clean = [l for l in ents if l['numclean']]
    if len(clean) < 2:
        return None
    S = g.roles['sets']
    ptau = g.tau_post(tt, with_nums=False)
    tau = 'CAPT' if ptau > 0.5 else 'CNTT'
    obs = tablet_type(tt)
    gl = []
    h = [l for l in lines if l['role'] == 'H']
    hr = header_role(h[0]['signs'][0], g.roles) if h and h[0]['signs'] else 'NONE'
    gl.append('header: %s%s' % (HDR_GLOSS[hr], (' [' + ' '.join(h[0]['signs']) + ']') if h and h[0]['signs'] else ''))
    if any(l['role'] == 'G' for l in lines):
        gl.append('edge tag: ' + ' '.join('%d(%s)' % (n, c) for l in lines if l['role'] == 'G' for n, c in l['nums']))
    gl.append('document: %s account (P=%.2f from signs alone)' % ('capacity' if tau == 'CAPT' else 'count',
                                                                  ptau if tau == 'CAPT' else 1 - ptau))
    # C1
    c1ok = True
    for l in clean:
        c = ncls(l['nums'])
        if tau == 'CAPT':
            if c in ('FRAC', 'BIS'):
                c1ok = False
            if c == 'AMB':
                d = Counter()
                for n, cc in l['nums']:
                    d[cc] += n
                if d['N01'] >= 6:
                    c1ok = False
        else:
            if c == 'CAP':
                c1ok = False
    # C2 + gloss
    c2n, c2ok, prev = 0, 0, None
    k = 0
    for l in lines:
        if l['role'] != 'E':
            continue
        k += 1
        sg = l['signs']
        role = line_role(sg, g.roles)
        pre = sg[0] if (len(sg) > 1 and sg[0] in S['PREFIX']) else None
        cls_s = sg[-1] if (sg and role in ('MEASURED', 'COUNTED', 'PERSON', 'ALLOT', 'FRACLINE')) else None
        mid = sg[(1 if pre else 0):(-1 if cls_s else len(sg))]
        q, v = fmt_q(l['nums'], tau, g) if l['numclean'] else ('[damaged]', None)
        parts = []
        if pre:
            parts.append('qualifier ' + pre)
        if mid:
            parts.append('name-string [' + ' '.join(mid) + ']')
        verdict = ''
        if l['numclean']:
            c = ncls(l['nums'])
            ok = None
            if role in ('MEASURED',):
                ok = c == 'CAP' or (c == 'AMB' and tau == 'CAPT')
            elif role in ('COUNTED', 'PERSON'):
                ok = c != 'CAP'
                if role == 'PERSON' and ok and v is not None and c == 'AMB' and tau == 'CNTT':
                    ok = v <= 60
            elif role == 'FRACLINE':
                ok = c != 'CAP'
            elif role == 'ALLOT' and prev is not None and prev['numclean'] and ncls(prev['nums']) == 'AMB':
                ku = value(prev['nums'], g.cntmaps[g.cnt_main])
                vv = value(l['nums'], g.capmap)
                if ku and vv is not None and ku <= 60:
                    ok = vv in (60 * ku, 120 * ku)
                    verdict = ' (allotment for %s units: %s)' % (ku, 'matches 60/120 N39C per unit' if ok else 'off-rate')
            if ok is not None:
                c2n += 1; c2ok += bool(ok)
                verdict = verdict or (' (role-consistent)' if ok else ' (ROLE CONFLICT)')
        gl.append('entry %d: %s%s %s %s%s' % (k, ' + '.join(parts) + (' ' if parts else ''),
                                              'receives' if role in ('ALLOT',) else '-',
                                              q, ('of ' + ROLE_GLOSS[role] + (' ' + cls_s if cls_s else '')), verdict))
        prev = l
    # C3
    c3 = g.closes(tt, tau)
    tot = [l for l in lines if l['role'] == 'T']
    if tot:
        if c3 is None:
            gl.append('total: present but not checkable (damage)')
        else:
            gl.append('total: %s' % ('closes in the %s system' % ('capacity' if tau == 'CAPT' else 'count') if c3
                                      else 'does not close under the predicted system'))
    else:
        gl.append('total: none written')
    full = c1ok and (c2ok == c2n) and (c3 is not False)
    strict = full and c2n >= 1 and c3 is True
    return {'id': t['id'], 'gloss': gl, 'C1': c1ok, 'C2n': c2n, 'C2ok': c2ok, 'C3': c3, 'full': full,
            'strict': strict, 'roles_checked': c2n >= 1, 'obs': obs, 'pred': tau}


def summarise(R):
    R = [r for r in R if r]
    n = len(R)
    s = {'n': n, 'C1': sum(r['C1'] for r in R) / n,
         'C2_lines': sum(r['C2ok'] for r in R) / max(1, sum(r['C2n'] for r in R)),
         'C2_tabs_with_checks': sum(r['roles_checked'] for r in R),
         'C2_all_ok': sum(r['C2ok'] == r['C2n'] and r['C2n'] > 0 for r in R) / max(1, sum(r['roles_checked'] for r in R)),
         'C3_n': sum(r['C3'] is not None for r in R), 'C3': sum(r['C3'] is True for r in R) / max(1, sum(r['C3'] is not None for r in R)),
         'full': sum(r['full'] for r in R) / n,
         'full_with_roles': sum(r['full'] and r['roles_checked'] for r in R) / n,
         'strict': sum(r['strict'] for r in R) / n, 'strict_n': sum(r['strict'] for r in R),
         'eligible_strict': sum(r['roles_checked'] and r['C3'] is not None for r in R)}
    return s


def transplant(ho, rng):
    """Numerals of each tablet replaced by the numerals of another held-out tablet with the same number of
    numeric (E+T) lines (role structure, signs and header kept)."""
    byn = {}
    for t in ho:
        E = [(l['nums'], l['numclean']) for l in t['lines'] if l['role'] in ('E', 'T')]
        byn.setdefault(len(E), []).append(E)
    out = []
    for t in ho:
        n = sum(1 for l in t['lines'] if l['role'] in ('E', 'T'))
        pool = byn.get(n, [])
        if len(pool) < 2:
            out.append(None); continue
        E = pool[rng.randrange(len(pool))]
        out.append(E)
    return out


def shuffled_job(args):
    k, = args
    T, roles, maps = c1.corpus()
    tr, ho = split(T)
    rng = random.Random(seed('pe59shuf%s%d' % (CORPUS, k)))
    R = shuffle_roles(roles, T, rng)
    g = c1.fit(tr, R, maps)
    return summarise([check_tablet(t, g) for t in ho])


def main():
    t0 = time.time()
    T, roles, maps = c1.corpus()
    tr, ho = split(T)
    g = c1.fit(tr, roles, maps)
    params = {'sign_S3': [float(x) for x in g.sign_par_S3], 'sign_G': [float(x) for x in g.sign_par_G],
              'num_G': g.par_ent, 'num_N2': g.par_ent_N2, 'tot_G': g.par_tot, 'tot_N2': g.par_tot_N2, 'theta': g.theta}
    fz = json.load(open(os.path.join(CK, 'c1_%s_frozen_params.json' % CORPUS)))
    out = {'params_sha': sha(params), 'cycle1_params_sha': fz['params_sha']}
    out['params_match'] = out['params_sha'] == fz['params_sha']
    R = [check_tablet(t, g) for t in ho]
    out['real'] = summarise(R)
    out['glosses'] = [r for r in R if r]
    rng = random.Random(seed('pe59transplant'))
    TS = []
    for d in range(20):
        nf = transplant(ho, rng)
        TS.append(summarise([check_tablet(t, g, numerals_from=n) if n is not None else None for t, n in zip(ho, nf)]))
    out['transplant'] = TS
    # trivial decoder: no roles at all (every entry OTHER), tau = majority
    with Pool(2) as pool:
        out['shuffled'] = pool.map(shuffled_job, [(k,) for k in range(NSH)])
    out['time'] = time.time() - t0
    json.dump(out, open(os.path.join(CK, 'c2_%s.json' % CORPUS), 'w'), indent=1, default=str)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
