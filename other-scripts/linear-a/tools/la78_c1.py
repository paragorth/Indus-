"""LA-78 cycle 1: time-travel scoring of 3,000 random sign/word role hypotheses.

Observables per clean word token (>= 2 syllabic signs): what follows it (number, logogram,
word, end) and whether it opens the document. Hypotheses = random partitions (final sign,
initial sign, first x last, final x length, whole word). Fit on docs published <= X.
Selection windows W1 (<=1950 -> 1951-1980), W2 (<=1980 -> 1981-1988); held-out re-test W3
(<=1988 -> after 1988). Controls: shuffled publication years (20), site+support-matched test,
label-shuffled test (family-wise null for the best hypothesis).
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la78_engine import *  # noqa

N_PER = int(os.environ.get('N_PER', 600))
docs = load()
T = tokens(docs)
if os.environ.get('TABLET'):
    T = [t for t in T if docs[t['doc']]['support'] == 'tablet']
pool, signs, words, F = make_pool(T, N_PER, seed=78)
y1 = np.array([t['y1'] for t in T]); y2 = np.array([t['init'] for t in T])
Y = [(y1, 4)] if os.environ.get('OBS', 'y1') == 'y1' else [(y1, 4), (y2, 2)]
MODE['mode'] = os.environ.get('SMODE', 'prior'); MODE['grp'] = np.array([docs[t['doc']]['site'] for t in T])
TAG = os.environ.get('SMODE', 'prior') + '_' + os.environ.get('OBS', 'y1') + ('_tab' if os.environ.get('TABLET') else '')
docid = np.array([t['doc'] for t in T])
year = np.array([docs[t['doc']]['year'] for t in T])
cert = np.array([docs[t['doc']]['cert'] for t in T])
site = np.array([docs[t['doc']]['site'] + '|' + docs[t['doc']]['support'] for t in T])
WIN = [(1950, 1976), (1976, 2100), (1988, 2100)]
out = dict(n_tokens=len(T), n_hyp=len(pool), windows=WIN,
           n_test=[int(((year > a) & (year <= b)).sum()) for a, b in WIN],
           n_train=[int((year <= a).sum()) for a, b in WIN])
print(out, flush=True)


def summarize(cv, fut, ins, tag):
    r = {}
    for wi in range(len(WIN)):
        r[f'W{wi+1}'] = dict(rho_cv_fut=round(spearman(cv[:, wi], fut[:, wi]), 3),
                             rho_ins_fut=round(spearman(ins[:, wi], fut[:, wi]), 3),
                             best_fut=round(float(np.nanmax(fut[:, wi])), 4),
                             frac_pos=round(float(np.mean(fut[:, wi] > 0)), 3),
                             mean_fut=round(float(np.nanmean(fut[:, wi])), 4))
    r['n_pos_all3'] = int(((fut > 0).all(1)).sum())
    # selection on W1 (pretend it is 1950, score the 1951-1976 finds); re-test on W2, W3
    top = np.argsort(-fut[:, 0])[:50]
    topcv = np.argsort(-cv[:, 0])[:50]
    topins = np.argsort(-ins[:, 0])[:50]
    for w in (1, 2):
        r[f'W{w+1}_top50_by_time'] = round(float(np.nanmean(fut[top, w])), 4)
        r[f'W{w+1}_top50_by_cv'] = round(float(np.nanmean(fut[topcv, w])), 4)
        r[f'W{w+1}_top50_by_ins'] = round(float(np.nanmean(fut[topins, w])), 4)
        r[f'W{w+1}_all'] = round(float(np.nanmean(fut[:, w])), 4)
        r[f'W{w+1}_top50_by_time_pos'] = round(float(np.mean(fut[top, w] > 0)), 3)
    fam = collections.Counter(pool[i]['fam'] for i in top)
    r['top50_families'] = dict(fam)
    print(tag, json.dumps(r), flush=True)
    return r, top


sitegrp = np.array([docs[t['doc']]['site'] for t in T])
cv, fut, ins, lo = evaluate(pool, Y, year, docid, WIN, grp=sitegrp)
out['real'], top = summarize(cv, fut, ins, 'real')
out['real_loso_rho'] = [round(spearman(lo[:, w], fut[:, w]), 3) for w in range(3)]
print('loso', out['real_loso_rho'], flush=True)
np.savez(os.path.join(CK, f'c1_{TAG}_real.npz'), cv=cv, fut=fut, ins=ins, lo=lo)
out['top10'] = [dict(name=pool[i]['name'], fut=[round(x, 4) for x in fut[i]], cv=[round(x, 4) for x in cv[i]]) for i in top[:10]]

# --- site-matched: cumulative future, test tokens only from sites already present in training
WINC = [(1950, 2100), (1976, 2100), (1988, 2100)]
masks = []
for a, b in WINC:
    seen = set(sitegrp[year <= a]); masks.append(np.array([s in seen for s in sitegrp]))
out['site_matched_n_test'] = [int((((year > a) & (year <= b)) & m).sum()) for (a, b), m in zip(WINC, masks)]
cvS, futS, insS = evaluate(pool, Y, year, docid, WINC, extra_mask=masks)
out['site_matched'], topS = summarize(cvS, futS, insS, 'site')
out['site_matched_overlap_top50'] = len(set(top) & set(topS))
out['rho_real_vs_site'] = [round(spearman(fut[:, w], futS[:, w]), 3) for w in range(3)]

# --- certain-dates only (drop estimated years from test)
mC = [cert != 'e'] * 3
cvC, futC, insC = evaluate(pool, Y, year, docid, WIN[:1], extra_mask=mC[:1])
out['certain_W1_rho'] = round(spearman(fut[:, 0], futC[:, 0]), 3)

# --- label-shuffled future: family-wise null for best gain per window
rng = np.random.RandomState(1)
nullbest = {0: [], 1: [], 2: []}
for r_ in range(int(os.environ.get('NLAB', 30))):
    Ys = []
    for y, ny in Y:
        ys = y.copy()
        for a, b in WIN:
            te = np.where((year > a) & (year <= b))[0]
            ys[te] = ys[rng.permutation(te)]
        Ys.append((ys, ny))
    for wi, (a, b) in enumerate(WIN):
        tr = year <= a; te = (year > a) & (year <= b)
        nullbest[wi].append(max(score(p['f'], Ys, tr, te) for p in pool))
out['labelshuf_best'] = {f'W{k+1}': [round(float(np.mean(v)), 4), round(float(np.percentile(v, 95)), 4)] for k, v in nullbest.items()}
out['labelshuf_P_best'] = {f'W{k+1}': round(float((1 + sum(x >= np.nanmax(fut[:, k]) for x in v)) / (1 + len(v))), 3) for k, v in nullbest.items()}
print('labshuf', out['labelshuf_best'], out['labelshuf_P_best'], flush=True)

# --- shuffled publication years (document-level permutation)
sh = []
dy = np.array([d['year'] for d in docs])
for r_ in range(int(os.environ.get('NSH', 20))):
    perm = rng.permutation(len(docs)); yd = dy[perm]
    yr = yd[docid]
    c2, f2, i2 = evaluate(pool, Y, yr, docid, WIN)
    s, _ = summarize(c2, f2, i2, f'shuf{r_}')
    sh.append(s)
out['shuffled'] = {k: [round(float(np.mean([s[k][m] for s in sh])), 4) for m in ('rho_cv_fut', 'best_fut', 'frac_pos', 'mean_fut')] for k in ('W1', 'W2', 'W3')}
out['shuffled_W2_time_cv_ins_all'] = [round(float(np.mean([s[k] for s in sh])), 4) for k in ('W2_top50_by_time', 'W2_top50_by_cv', 'W2_top50_by_ins', 'W2_all')]
out['shuffled_n_pos_all3'] = round(float(np.mean([s['n_pos_all3'] for s in sh])), 1)
json.dump(out, open(os.path.join(CK, f'c1_{TAG}.json'), 'w'), indent=1)
print('DONE', json.dumps(out)[:3000])
