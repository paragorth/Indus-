"""pe71 cycle 3: replicate the closing-line rule across volumes, test it on the CDLI transliterations that were not in the
training corpus (pe60 new_atf), measure find-lot clustering, and freeze predictions for the untransliterated pe60 tablets.

(a) per volume: |M153+X| / |M153+M342| tokens on the closing unnumbered line (c2b definition), with the within-tablet
    random-unnumbered-line expectation.  Hold-out: volumes other than MDP 06 (the volume whose context dump suggested the rule).
(b) find-lot: tablets with an M153 closing line, publication-number clustering within volume (mean gap to nearest other
    such tablet) vs random sets of the same size among the volume's tablets with any closing line (10,000).
(c) outside the corpus: every M153 compound in data/pe60_ckpt/new_atf.atf (6 live CDLI transliterations).
(d) frozen predictions for pe60 items (image only / catalogue only), with kill lines; sha256 of the JSON.
usage: pe71_c3.py -> data/pe71_ckpt/c3.json, data/pe71_frozen_predictions.json
"""
import json, os, re, hashlib, collections
import numpy as np
from pe71_lib import load, comps, CK, DATA
from pe71_c2b import closing
from common import base, is_sign

TGT = ('|M153+X|', '|M153+M342|')


def tok_rows(R):
    rows = []
    for r in R:
        k, _ = closing(r)
        bl = [i for i, l in enumerate(r['lines']) if l['signs'] and not l['nums']]
        for i, l in enumerate(r['lines']):
            for t in set(l['signs']):
                if t in TGT:
                    rows.append(dict(id=r['id'], vol=r['vol'], pub=r['pub'], tok=t, closing=int(i == k),
                                     p=(k is not None) / len(bl) if bl else 0.0, sealed=r['sealed'], n_lines=len(r['lines'])))
    return rows


def main():
    rng = np.random.default_rng(713)
    R = load()
    rows = tok_rows(R)
    res = {}
    byv = collections.defaultdict(list)
    for x in rows:
        byv[x['vol']].append(x)
    res['a'] = {v: dict(n=len(L), closing=sum(x['closing'] for x in L), expected=round(sum(x['p'] for x in L), 2)) for v, L in byv.items()}
    ho = [x for x in rows if x['vol'] != 'MDP 06']
    ps = np.array([x['p'] for x in ho]); obs = sum(x['closing'] for x in ho)
    sim = (rng.random((100000, len(ps))) < ps).sum(1)
    res['a_holdout'] = dict(n=len(ho), closing=obs, expected=float(ps.sum()), p=float((sim >= obs).mean()))
    print('a', res['a'], res['a_holdout'], flush=True)
    # (b) find-lot clustering
    cl = {r['id']: closing(r) for r in R}
    m153 = {r['id'] for r in R if cl[r['id']][1] and any(t in TGT for t in cl[r['id']][1])}
    out = {}
    for v in sorted({R[i]['vol'] for i in range(len(R)) if R[i]['id'] in m153}):
        pool = [r for r in R if r['vol'] == v and cl[r['id']][0] is not None and not np.isnan(r['pub'])]
        pubs = np.array([r['pub'] for r in pool]); isM = np.array([r['id'] in m153 for r in pool])
        if isM.sum() < 2:
            continue

        def gap(sel):
            p = np.sort(pubs[sel]); d = np.diff(p)
            return float(np.median(np.minimum(np.r_[d, np.inf], np.r_[np.inf, d])))
        real = gap(isM)
        nul = np.array([gap(rng.permutation(isM)) for _ in range(10000)])
        out[v] = dict(n=int(isM.sum()), pool=len(pool), pubs=sorted(pubs[isM].tolist()), median_nn_gap=real,
                      null=float(np.median(nul)), p=float((nul <= real).mean()))
    res['b'] = out
    print('b', out, flush=True)
    # (c) outside corpus
    cur, new = None, []
    for raw in open(os.path.join(DATA, 'pe60_ckpt', 'new_atf.atf'), encoding='utf-8'):
        if raw.startswith('&P'):
            cur = dict(id=raw[1:8], lines=[]); new.append(cur); continue
        m = re.match(r'^\s*([0-9]+\'?)\.\s+(.*)$', raw)
        if cur and m:
            body = m.group(2)
            sg = [base(re.sub(r'[#?!\[\]]', '', w)) for w in body.split(',')[0].split() if is_sign(re.sub(r'[#?!\[\]]', '', w))]
            nums = re.findall(r'\d+\(N', body)
            cur['lines'].append(dict(signs=sg, nums=nums))
    hits = []
    for t in new:
        k = max((i for i, l in enumerate(t['lines']) if l['signs']), default=None)
        for i, l in enumerate(t['lines']):
            for s in l['signs']:
                if s.startswith('|') and 'M153' in comps(s):
                    hits.append(dict(id=t['id'], tok=s, line=i + 1, last_signed=i == k, numbered=bool(l['nums']) or None,
                                     with_M288='M288' in l['signs'], n_lines=len(t['lines'])))
    res['c'] = dict(n_tablets=len(new), hits=hits)
    print('c', res['c'], flush=True)
    # (d) frozen predictions
    inv = json.load(open(os.path.join(DATA, 'pe60_inventory.json')))['items']
    # base rates from the corpus: short (<= 6 lines) M157-headed tablets, by volume
    rate = {}
    for v in ('MDP 06', 'MDP 26S', 'MDP 26', 'MDP 17', 'TCL 32'):
        S = [r for r in R if r['vol'] == v and r['hdr'] == 'M157' and len(r['lines']) <= 6]
        rate[v] = (sum(r['id'] in m153 for r in S), len(S))
    def local(des):
        m = re.match(r'(MDP \w+), 0*(\d+)', des)
        if not m:
            return None
        v, n = m.group(1), int(m.group(2))
        W = [r for r in R if r['vol'] == v and not np.isnan(r['pub']) and abs(r['pub'] - n) <= 15]
        return '%d of %d tablets within 15 numbers have an M153 closing line' % (sum(r['id'] in m153 for r in W), len(W))
    pred = dict(
        rule='|M153+X| and |M153+M342| sit on the tablet\'s last signed line, unnumbered (corpus 24/34; within-tablet chance 13.2/34)',
        base_rates_short_M157_tablets=rate,
        P1=dict(claim='Of the next >= 5 newly transliterated |M153+X| or |M153+M342| tokens (any source), >= 3 sit on the last signed line '
                      'and >= 2 of those are unnumbered', kill='<= 1 of 5 on the last signed line', grade='B'),
        P2=dict(claim='No new |M153+M342| / |M153+X| token stands in a numbered capacity (C-system) entry', kill='>= 2 such tokens', grade='B'),
        P3=dict(claim='Sealing is NOT predicted by the M153 closing line once the PES0329/PES0334 series is counted once: new tablets '
                      'with the closing line are sealed at the rate of short M157-headed tablets of their volume, not above',
                kill='>= 4 of the next 5 such tablets sealed by distinct seals', grade='B (negative)'),
        P4=dict(claim='Any new tablet impressed with PES0334 (boat) carries |M153+X| or |M153+M342| on its last line; any new PES0329 '
                      'tablet carries |M153+M342|', kill='2 new PES0334/PES0329 tablets without either compound', grade='C'),
        P5=dict(claim='The M153 closing partner stays office-like: new tokens bring at most 1 new partner value per 5 tokens '
                      '(V/n <= 0.2)', kill='>= 3 new partner values among the next 5 closing M153 compounds', grade='C'),
        pe60_targets=[dict(P=i['P'], designation=i['designation'], status=i['status'],
                           expect='if short and M157-headed: P(M153 closing line) ~ %d/%d in the volume' % rate.get(i['designation'].split(',')[0], (0, 0)),
                           local=local(i['designation']))
                      for i in inv if i['type'] == 'tablet' and i['designation'].split(',')[0] in ('MDP 06', 'MDP 26S', 'MDP 26', 'MDP 17')],
        outside_test_already=res['c']['hits'])
    s = json.dumps(pred, indent=1, sort_keys=True)
    sha = hashlib.sha256(s.encode()).hexdigest()[:16]
    open(os.path.join(DATA, 'pe71_frozen_predictions.json'), 'w').write(s)
    res['d'] = dict(sha=sha, rates=rate, n_targets=len(pred['pe60_targets']))
    print('d', res['d'], [t['designation'] for t in pred['pe60_targets']], flush=True)
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
