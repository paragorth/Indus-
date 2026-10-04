"""PE-22 cycle 3, stage 2: verify the cycle-2 hash, open the recent tablets and score; then the
controls that could kill the predictions.
  (a) random held-out pre-2000 tablets of the same sign-token count (exchangeable truth);
  (b) shuffled publication dates: random CONTIGUOUS runs of pre-2000 tablets in publication order
      (a publication batch, as a recent edition is), same token count;
  (c) proto-cuneiform older (<= 1999) -> newer publications (ATU 6/7 2001-05, Uruk; CUSAS 1/21/31
      2007-16, unprovenanced), full size and cut to PE size;
  (d) Linear A, the la15 split (GORILA -> post-GORILA) run through this pipeline;
  (e) planted repertoire of known size (ABC posterior communities), train and held-out at PE sizes.
Coverage of 95% intervals and AUC of the known-sign ranking are compared to the real score.
"""
import sys, json, collections, random, time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from pe22_common import *
import la15_common as LA15

rng = np.random.default_rng(2203)
prng = random.Random(2203)
t0 = time.time()
res = {}
_prev = os.path.join(CK, 'cycle3.json')
if '--resume' in sys.argv and os.path.exists(_prev):
    res.update(json.load(open(_prev)))
pre = json.load(open(os.path.join(CK, 'cycle2_predictions.json')))
h = pre.pop('sha256')
res['hash_ok'] = sha(pre) == h
print('hash ok', res['hash_ok'], flush=True)
PE = load_pe()
tr, ho = pe_split(PE)
KEYS = ('signs', 'graphs', 'words', 'bigr')
sc = {}
for k in KEYS:
    sc[k] = scored(pre[k], truth_new(tr, ho, k))
seen = set(w for d in tr for w in d['signs'])
for grp, sites in (('newsites', ('Sofalin', 'Ozbaki')), ('Malyan', ('Malyan',)), ('Susa', ('Susa',))):
    H = [d for d in ho if d['site'] in sites]
    newt = sorted(set(w for d in H for w in d['signs']) - seen)
    sc['site_' + grp] = dict(scored(pre['site_' + grp], len(newt)), new_types=newt)
present = set(w for d in ho for w in d['signs'])
shell = [dict(site=d['site'], m=len(d['signs'])) for d in ho if d['signs']]
sc_ex, sc_hab, ec = known_scores(tr, shell, 'signs')
obs = collections.Counter(w for d in ho for w in d['signs'])
from scipy.stats import spearmanr
kn = [w for w in sc_ex]
sc['known'] = dict(n_present=len(present & seen), pred=pre['known_types_expected'],
                   auc_exch=auc(sc_ex, present), auc_hab=auc(sc_hab, present),
                   spearman_counts=float(spearmanr([ec[w] for w in kn], [obs.get(w, 0) for w in kn])[0]),
                   top30_hab_hits=sum(w in present for w in pre['top30_hab']),
                   top30_exch_hits=sum(w in present for w in pre['top30_exch']),
                   risky_absent_correct=sum(w not in present for w in pre['risky_absent_hab']),
                   risky_absent_n=len(pre['risky_absent_hab']),
                   new_sign_types=sorted(present - seen),
                   new_token_share=sum(c for w, c in obs.items() if w not in seen) / sum(obs.values()))
# risky-absent chance: 15 random mid-band signs
mid = [w for w in sc_hab if 0.4 < sc_ex[w] < 0.9]
ch = [sum(w not in present for w in prng.sample(mid, len(pre['risky_absent_hab']))) for _ in range(2000)]
sc['known']['risky_absent_chance'] = iv(ch)
sc['known']['risky_absent_p'] = float(np.mean(np.array(ch) >= sc['known']['risky_absent_correct']))
res['PE_recent'] = sc
print('SCORED', json.dumps(sc, default=float), round(time.time() - t0), flush=True)

mS = pre['signs']['m']


def take(pool, m, key='signs'):
    out, t = [], 0
    for d in pool:
        if t >= m:
            break
        if d[key]:
            out.append(d)
            t += len(d[key])
    return out


def rep_test(name, make, nrep, keys=('signs', 'graphs', 'words', 'bigr'), abc_reps=6):
    if name in res:
        print('skip', name, flush=True)
        return
    rows = collections.defaultdict(list)
    aucs = collections.defaultdict(list)
    for r in range(nrep):
        trn, hold = make(r)
        for key in keys:
            meta = [(d['site'], len(d[key])) for d in hold if d[key]]
            if not meta:
                continue
            P = predict_new(trn, meta, key, rng, B=200, abc=(r < abc_reps and key in ('signs', 'graphs')))
            for k, v in scored(P, truth_new(trn, hold, key)).items():
                rows[key + ':' + k].append((v['inside'], v['err'], v['truth'], v['pred'][0]))
        pr = set(w for d in hold for w in d['signs'])
        e, hh, _ = known_scores(trn, [dict(site=d['site'], m=len(d['signs'])) for d in hold if d['signs']], 'signs')
        a1 = auc(e, pr)
        if a1 is not None:
            aucs['exch'].append(a1)
            aucs['hab'].append(auc(hh, pr))
    out = {k: dict(cover=float(np.mean([x[0] for x in v])), mean_err=float(np.mean([x[1] for x in v])),
                   mean_truth=float(np.mean([x[2] for x in v])), mean_pred=float(np.mean([x[3] for x in v])), n=len(v))
           for k, v in sorted(rows.items())}
    out['auc'] = {k: iv(v) for k, v in aucs.items()}
    res[name] = out
    print(name, json.dumps(out), round(time.time() - t0), flush=True)
    json.dump(res, open(os.path.join(CK, 'cycle3.json'), 'w'), indent=1, default=float)


TRs = [d for d in tr if d['signs']]


def pe_random(r):
    pool = TRs[:]
    prng.shuffle(pool)
    hold = take(pool, mS)
    ids = set(d['id'] for d in hold)
    return [d for d in tr if d['id'] not in ids], hold


rep_test('PE_random_heldout', pe_random, 30)

order = sorted(TRs, key=lambda d: (d['pub'].split(',')[0], d['id']))


def pe_block(r):
    i = prng.randrange(len(order))
    rot = order[i:] + order[:i]
    hold = take(rot, mS)
    ids = set(d['id'] for d in hold)
    return [d for d in tr if d['id'] not in ids], hold


rep_test('PE_block_shuffled_dates', pe_block, 20)

# planted: ABC-posterior communities, PE-shaped tablets
A_tr = list(collections.Counter(w for d in tr for w in d['signs']).values())
post = abc_fit(A_tr, rng, n_sims=2000, keep=60)
sizes_tr = [len(d['signs']) for d in tr if d['signs']]
sizes_ho = [len(d['signs']) for d in ho if d['signs']]


def planted(r):
    _, S, a, q = post[r % len(post)]
    S = max(S, 800)  # repertoire at least ~Chao1 size
    p = zm(S, a, q)
    names = ['s%d' % i for i in range(S)]

    def mk(sizes, site):
        out = []
        for m in sizes:
            th = rng.dirichlet(30 * p + 1e-9)
            x = rng.multinomial(m, th)
            toks = [names[i] for i in np.repeat(np.arange(S), x)]
            out.append(dict(id='p', site=site, signs=toks))
        return out
    return mk(sizes_tr, 'Susa'), mk(sizes_ho, 'Susa')


rep_test('planted_repertoire', planted, 12, keys=('signs',), abc_reps=4)

# proto-cuneiform
PC = load_pc()
pc_old = [d for d in PC if d['year'] is not None and d['year'] <= 1999]
pc_atu = [d for d in PC if d['pub'].startswith(('ATU 6', 'ATU 7'))]
pc_cus = [d for d in PC if d['pub'].startswith('CUSAS')]
for nm, H in (('ATU6_7', pc_atu), ('CUSAS', pc_cus)):
    out = {}
    for key in KEYS:
        meta = [(d['site'], len(d[key])) for d in H if d[key]]
        P = predict_new(pc_old, meta, key, rng, B=200, abc=key in ('signs', 'graphs'))
        out[key] = scored(P, truth_new(pc_old, H, key))
    pr = set(w for d in H for w in d['signs'])
    e, hh, _ = known_scores(pc_old, [dict(site=d['site'], m=len(d['signs'])) for d in H if d['signs']], 'signs')
    out['auc'] = dict(exch=auc(e, pr), hab=auc(hh, pr))
    res['PC_full_' + nm] = out
    print('PC_full', nm, json.dumps(out), round(time.time() - t0), flush=True)
nTR = sum(len(d['signs']) for d in tr)
for nm, H in (('ATU6_7', pc_atu), ('CUSAS', pc_cus)):
    def pcs(r, H=H):
        a = [d for d in pc_old if d['signs']]
        prng.shuffle(a)
        b = [d for d in H if d['signs']]
        prng.shuffle(b)
        return take(a, nTR), take(b, mS)
    rep_test('PC_PEsize_' + nm, pcs, 8, abc_reps=3)

# Linear A through this pipeline (la15 split)
LAd = LA15.load_la()
la_tr = [d for d in LAd if d['pub'] != 'post']
la_ho = [d for d in LAd if d['pub'] == 'post']
out = {}
for key in ('words', 'signs'):
    meta = [(d['site'], len(d[key])) for d in la_ho if d[key]]
    P = predict_new(la_tr, meta, key, rng, B=500, abc=True)
    out[key] = scored(P, truth_new(la_tr, la_ho, key))
pr = set(w for d in la_ho for w in d['words'])
e, hh, _ = known_scores(la_tr, [dict(site=d['site'], m=len(d['words'])) for d in la_ho if d['words']], 'words')
out['auc_words'] = dict(exch=auc(e, pr), hab=auc(hh, pr))
res['LA_la15_check'] = out
print('LA', json.dumps(out), flush=True)
json.dump(res, open(os.path.join(CK, 'cycle3.json'), 'w'), indent=1, default=float)
print('done', round(time.time() - t0))
