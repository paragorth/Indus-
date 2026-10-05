"""pe42 cycle 2: thousands of SPECIFIC error hypotheses, scored automatically.

A hypothesis is a family plus a condition: OMIT/DOUBLE of entries carrying sign
X; CONV by factor f (of entries with sign X or of any entry); SYS base->alt of
entries with X or of the total; WHOLE alt on tablets carrying X; CARRY / XCARRY /
DIGIT at a given code.  Score = number of failing tablets it explains.
Null: 300 noise replicates (another tablet's real discrepancy added to this
tablet's entries); per-hypothesis p and a max-statistic (family-wise) p.
Survivors are re-tested split-half: selected on half A, counted on half B
(real vs noise B).  Run on PE, PC and MES, and on PE with planted
sign-conditioned errors (a planted 'X read in cap' link must be found)."""
import collections, json, random, sys
from multiprocessing import Pool
from pe42_common import *

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def efeats(e):
    s = [base(x) for x in e['signs']] if e['signs'] else []
    out = {'*'} | {'S:' + x for x in s}
    if s:
        out.add('L:' + s[-1])
    else:
        out.add('BARE')
    return out


def keys_for(case, corp, pool):
    """Set of specific hypothesis keys that make the failing tablet close."""
    cl, ex, _ = explain(case, corp, pool)
    if cl:
        return None
    K = set()
    h0 = case['hyps']
    tsig = set()
    for h in h0:
        for p in h:
            for x in p['E'] + [p['T']]:
                tsig |= {base(s) for s in x['signs']}
    for fam, dets in ex.items():
        for d in dets:
            k, mn = d[0], d[1]
            rest = d[2:]
            # locate the pair to read entry features (hyp ambiguity: use any hyp with that pair index)
            if rest and rest[0] == 'i':
                i = rest[1]
                feats = set()
                for h in h0:
                    if k < len(h) and i < len(h[k]['E']):
                        feats |= efeats(h[k]['E'][i])
                famk = 'OMIT' if fam.startswith('OMIT') else fam
                for f in feats:
                    if famk == 'CONV':
                        K.add(('CONV', rest[2], f))
                    elif famk == 'SYS':
                        K.add(('SYS', rest[2], rest[3], f))
                    else:
                        K.add((famk, f))
            elif fam == 'SYS':
                K.add(('SYS_T', rest[1], rest[2]))
            elif fam == 'CODE':
                K.add(('CODE', rest[0], rest[1], rest[2])); K.add(('CODE', '*', rest[1], rest[2]))
            elif fam == 'WHOLE':
                for s in tsig | {'*'}:
                    K.add(('WHOLE', mn, rest[1], s))
            elif fam in ('CARRY', 'XCARRY'):
                K.add((fam, rest[0])); K.add((fam, '*'))
            elif fam == 'DIGIT':
                K.add(('DIGIT', rest[0], rest[1])); K.add(('DIGIT', '*'))
            else:
                K.add((fam,))
    return K


def corpus_keys(C, corp):
    per = make_pool(C, corp)
    return [keys_for(c, corp, pool_excluding(per, k)) for k, c in enumerate(C)]


def job(args):
    name, seed, plant_sign = args
    rng = random.Random(seed)
    C = load_cases(name) if not plant_sign else planted_corpus(name, plant_sign, seed)
    corp = CORP[name]
    if seed >= 0:
        C = null_noise(C, corp, rng)
    KK = corpus_keys(C, corp)
    return [sorted(map(list, k)) if k is not None else None for k in KK]


def planted_corpus(name, sign, seed):
    """PE count tablets carrying `sign`: the total is rewritten as if the
    entries with that sign had been read with capacity ratios (N14 = 6 N01)."""
    rng = random.Random(seed + 77)
    C = load_cases(name); corp = CORP[name]
    out = []
    for c in C:
        c2 = json.loads(json.dumps(c))
        p = c2['hyps'][0][0]
        if p['cls'] == 'cnt' and closes(c, corp) and any(sign in [base(s) for s in e['signs']] for e in p['E']):
            m = corp.maps['dec2']
            S = sum(val(e['nums'], corp.maps['cap'], m) if sign in [base(s) for s in e['signs']] else val(e['nums'], m)
                    for e in p['E'])
            nm = canon(S, m) if S and S > 0 else None
            if nm:
                for h in c2['hyps']:
                    h[0]['T']['nums'] = nm
        out.append(c2)
    return out


def plant_candidates():
    C = load_cases('PE'); corp = CORP['PE']
    cnt = collections.Counter()
    for c in C:
        p = c['hyps'][0][0]
        if p['cls'] == 'cnt' and closes(c, corp):
            sg = set()
            for e in p['E']:
                if any(cc == 'N14' for _, cc in e['nums']):
                    sg |= {base(x) for x in e['signs']}
            cnt.update(sg)
    return [s for s, _ in cnt.most_common(2)]


PLANT_SIGNS = plant_candidates()

if __name__ == '__main__':
    print('plant signs', PLANT_SIGNS)
    tasks = []
    for name in ('PE', 'PC', 'UR3'):
        tasks.append((name, -1, None))
        tasks += [(name, s, None) for s in range(NREP)]
    for sg in PLANT_SIGNS:
        tasks.append(('PE', -1, sg))
        tasks += [('PE', s, sg) for s in range(100)]
    with Pool(2) as pool:
        res = pool.map(job, tasks, chunksize=4)
    out = collections.defaultdict(dict)
    for (name, seed, sg), r in zip(tasks, res):
        out[name + ('' if not sg else ':' + sg)][seed] = r
    json.dump(out, open(os.path.join(CK, 'c2_keys.json'), 'w'))
    print('done')
