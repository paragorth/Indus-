"""pe86 cycle 2: thousands of random header definitions scored by clay squareness, held-out batches + line art."""
import sys, os, json, collections
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import pe86_common as Q
import pe83_common as P
rng = np.random.default_rng(8602)
D = Q.build(True); R = D['R']

# line-art-only tablets with intact line 1 (no photo outline used) -> independent record
LA = json.load(open(os.path.join(P.CK, 'lineart_feats.json'))); cat = P.catalogue()
photo = set(json.load(open(os.path.join(P.CK, 'c1_feats.json'))))
RL = []
for t in common.load():
    la = LA.get(t['id']); c = cat.get(t['id'])
    if not la or not t['lines'] or not c or t['id'] in photo:
        continue
    l1 = t['lines'][0]
    if l1['lacuna'] or l1['damaged']:
        continue
    h, w, th = P.num(c['height']), P.num(c['width']), P.num(c['thickness'])
    RL.append(dict(id=t['id'], t=t, top=la['la_cf_top'], bot=la['la_cf_bot'], asp=np.log(la['la_aspect']) if la.get('la_aspect') else 0.0,
                   la=np.log(h * w) if h and w else np.nan, tw=(th / w) if th and w else np.nan, nl=np.log(len(t['lines'])),
                   dx=float(any(g == 'x' for l in t['lines'] for g in l['signs'])), rev=float(any(l['surface'] == 'reverse' for l in t['lines']))))
for r in RL:
    for k in ('la', 'tw'):
        if not np.isfinite(r[k]):
            r[k] = np.nanmedian([x[k] for x in RL])
ZL = np.column_stack([np.ones(len(RL))] + [np.array([r[k] for r in RL], float) for k in ('la', 'asp', 'tw', 'nl', 'dx', 'rev')])
topL = np.array([r['top'] for r in RL])
print('line art held-out', len(RL), flush=True)

ALL = [r['t'] for r in R] + [r['t'] for r in RL]
NP = len(R)

def tabfeat(T, plant=None):
    f = dict(ntok=[], num=[], leg=[], first=[], ntok2=[], num2=[], leg2=[], first2=[], std=[])
    for i, t in enumerate(T):
        for j, suf in ((0, ''), (1, '2')):
            if j < len(t['lines']):
                l = t['lines'][j]; tk = Q.l1_tokens(t, j)
                if plant is not None and j == 0 and i in plant: tk = ['PLANT'] + tk
                f['ntok' + suf].append(len(tk)); f['num' + suf].append(bool(l['numerals']))
                f['leg' + suf].append(bool([s for s in l['signs'] if common.is_sign(s)]) or (plant is not None and j == 0 and i in plant))
                f['first' + suf].append(tk[0] if tk else '')
            else:
                f['ntok' + suf].append(0); f['num' + suf].append(True); f['leg' + suf].append(False); f['first' + suf].append('')
        f['std'].append(common.header(t) is not None)
    return {k: np.array(v) for k, v in f.items()}

voc = [k for k, _ in collections.Counter([x for t in ALL for j in (0, 1) if j < len(t['lines']) for x in Q.l1_tokens(t, j)[:1]]).most_common(121) if k not in ('', 'x')][:120]

def rand_def(rng, length_only=False, extra_voc=()):
    v = list(voc) + list(extra_voc)
    d = dict(allow_num=bool(rng.random() < .3), lo=int(rng.choice([0, 1, 1, 2])), hi=int(rng.choice([1, 2, 3, 99])),
             leg=bool(rng.random() < .6), two=bool(rng.random() < .25), comb=str(rng.choice(['replace', 'or', 'and'])))
    if d['hi'] < d['lo']: d['hi'] = 99
    if length_only:
        d['mode'] = 'ignore'; d['S'] = []
    else:
        d['mode'] = str(rng.choice(['require', 'exclude', 'ignore'], p=[.5, .3, .2]))
        d['S'] = [str(x) for x in rng.choice(v, int(rng.integers(1, 16)), replace=False)]
    return d

def evaluate(d, f):
    def cond(suf):
        c = (d['allow_num'] | ~f['num' + suf]) & (f['ntok' + suf] >= d['lo']) & (f['ntok' + suf] <= d['hi'])
        if d['leg']: c &= f['leg' + suf]
        if d['mode'] != 'ignore':
            inS = np.isin(f['first' + suf], d['S'])
            c &= inS if d['mode'] == 'require' else ~inS
        return c
    h = cond('')
    if d['two']: h = h | (cond('2') & ~f['num'])  # line 2 counts only below a numeral-free line 1
    if d['comb'] == 'or': h = h | f['std']
    elif d['comb'] == 'and': h = h & f['std']
    return h.astype(float)

def pr(y, h, Z):
    if h.std() == 0 or h.mean() < .05 or h.mean() > .95: return np.nan
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(np.corrcoef(res(rankdata(y)), res(h))[0, 1])

ZP = D['Z']; batches = D['batch']; ub = np.unique(batches)

def run_split(sp, ytrain_shuffle=False, plant=False, length_only=False, K=20000, rep=0):
    r2 = np.random.default_rng(1000 + sp)
    trb = set(r2.choice(ub, len(ub) // 2, replace=False)); tr = np.isin(batches, list(trb)); te = ~tr
    y = D['top'].copy()
    pl = None
    if plant:
        U = np.where(D['hd'] == 0)[0]; ULa = NP + np.where(np.array([common.header(r['t']) is None for r in RL]))[0]
        yy = np.concatenate([C_rank(D['ytop'][U]), C_rank(topL[ULa - NP])])
        cand = np.concatenate([U, ULa]); w = yy ** 2; w /= w.sum()
        pl = set(int(i) for i in r2.choice(cand, 26, replace=False, p=w))
    f = tabfeat(ALL, pl)
    fP = {k: v[:NP] for k, v in f.items()}; fL = {k: v[NP:] for k, v in f.items()}
    ytr = y.copy()
    if ytrain_shuffle: ytr = Q.perm_within(y, D['strata'], np.random.default_rng(5000 + 10 * sp + rep))
    std = evaluate(dict(allow_num=False, lo=1, hi=99, leg=True, two=False, comb='replace', mode='ignore', S=[]), f)
    sc_std = dict(train=pr(ytr[tr], std[:NP][tr], ZP[tr]), test=pr(y[te], std[:NP][te], ZP[te]), la=pr(topL, std[NP:], ZL))
    defs = [rand_def(r2, length_only, ['PLANT'] if plant else []) for _ in range(K)]
    if plant:  # make sure the plant token is available in the vocabulary sample at the same rate as any token
        pass
    HH = [evaluate(d, f) for d in defs]
    tr_sc = np.array([pr(ytr[tr], h[:NP][tr], ZP[tr]) for h in HH])
    ag = np.array([(h[:NP][tr] == std[:NP][tr]).mean() for h in HH])
    order = np.argsort(-np.nan_to_num(tr_sc, nan=-9))[:50]
    diff = np.argsort(-np.nan_to_num(np.where(ag < 0.95, tr_sc, np.nan), nan=-9))[:20]
    top = []
    for i in list(order) + list(diff):
        h = evaluate(defs[i], f)
        top.append(dict(d=defs[i], train=round(float(tr_sc[i]), 3), test=round(pr(y[te], h[:NP][te], ZP[te]), 3), la=round(pr(topL, h[NP:], ZL), 3),
                        prev=round(float(h.mean()), 3), agree_std=round(float((h == std).mean()), 3)))
    return dict(split=sp, std=sc_std, top=top[:50], diff=top[50:])

def C_rank(v):
    return rankdata(v) / len(v)

def summ(res):
    t = res['top']
    g = lambda k: float(np.nanmean([x[k] for x in t]))
    return dict(split=res['split'], std_test=res['std']['test'], std_la=res['std']['la'], top50_test=round(g('test'), 3), top50_la=round(g('la'), 3),
                top10_test=round(float(np.nanmean([x['test'] for x in t[:10]])), 3), top10_la=round(float(np.nanmean([x['la'] for x in t[:10]])), 3),
                best_train=t[0]['train'], agree=round(g('agree_std'), 3),
                diff20_test=round(float(np.nanmean([x['test'] for x in res['diff']])), 3), diff20_la=round(float(np.nanmean([x['la'] for x in res['diff']])), 3),
                diff20_agree=round(float(np.mean([x['agree_std'] for x in res['diff']])), 3))

if __name__ == '__main__':
    out = dict(real=[], null=[], plant=[], length=[])
    for sp in range(5):
        a = run_split(sp); out['real'].append(dict(summary=summ(a), top=a['top'][:12], diff=a['diff'][:10])); print('real', summ(a), flush=True)
        b = run_split(sp, length_only=True); out['length'].append(dict(summary=summ(b), top=b['top'][:5], diff=b['diff'][:5])); print('length', summ(b), flush=True)
        for k in range(5):
            c = run_split(sp, ytrain_shuffle=True, rep=k); c['split'] = f'{sp}_{k}'; out['null'].append(summ(c)); print('null', summ(c), flush=True)
        p = run_split(sp, plant=True)
        hasP = [('PLANT' in x['d']['S']) and x['d']['mode'] == 'require' and x['d']['comb'] == 'or' for x in p['top'][:10]]
        hasPd = [('PLANT' in x['d']['S']) and x['d']['mode'] == 'require' for x in p['diff'][:10]]
        out['plant'].append(dict(summary=summ(p), plant_in_top10=int(sum(hasP)), plant_in_top1=bool(hasP[0]), plant_in_diff10=int(sum(hasPd)), top=p['top'][:3])); print('plant', sum(hasP), hasP[0], sum(hasPd), summ(p), flush=True)
        json.dump(out, open(os.path.join(Q.CK, 'c2.json'), 'w'), indent=1, default=str)
