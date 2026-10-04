"""LA-23 cycle 3: predictions outside the estimating set.

(A) How many new person-names should a newly found Hagia Triada tablet contain?
    Incidence extrapolation (expected new individuals on one more sampling unit = Q1/T, Chao et al.
    2014), with the per-tablet slot distribution, over thousands of membership/ambiguity draws.
    Calibration: (i) Linear B KN and PY LA-sized training draws predicting held-out tablets of the
    same site; (ii) Hagia Triada room hold-out (each findspot held out in turn, predicted from the
    others: a 'new tablet' from a new room); (iii) planted populations.
(B) Do the HT person-like words recur at other Linear A sites less than the HT non-person words?
    (A local population should be site-endemic.) Linear B KN->PY as the analogue; word-site
    shuffle as the null.
(C) The population interval against the excavated size of the settlement (outside data, check only).
"""
import sys, json
from la23_common import *
import la23_c2 as C2

rng = np.random.default_rng(seed('la23c3'))
P = C2.P
HT, TAB, WORDS = C2.HT, C2.TAB, C2.WORDS


def person_sets(rng, tabs_words, eta=None, nu=None):
    tabs, docf, docc, par = C2.la_draw(rng, boot=False, eta=eta, nu=nu)
    return tabs


# ---------------------------------------------------------------- (A) new names on a new tablet
def predict_new(train_tabs):
    T = len(train_tabs)
    cnt = collections.Counter(x for t in train_tabs for x in t)
    Q1 = sum(1 for v in cnt.values() if v == 1); Q2 = sum(1 for v in cnt.values() if v == 2)
    # Chao et al. 2014 one-step extrapolation (m=1): Q0hat * (1 - (1 - Q1/(T*Q0hat+Q1))^1)
    if Q2 > 0:
        Q0 = (T - 1) / T * Q1 * Q1 / (2 * Q2)
    else:
        Q0 = (T - 1) / T * Q1 * (Q1 - 1) / 2
    if Q0 <= 0: return 0.0, cnt
    return Q0 * (1 - (1 - Q1 / (T * Q0 + Q1))), cnt


def holdout_eval(tabs, test_idx):
    test_idx = set(test_idx)
    train = [t for i, t in enumerate(tabs) if i not in test_idx]
    test = [t for i, t in enumerate(tabs) if i in test_idx]
    per_new, cnt = predict_new(train)
    seen = set(cnt)
    # observed: new individuals across held-out tablets, counted per tablet (first appearance only)
    obs_per_tab = [len(t - seen) for t in test]
    # predicted for k sequential new tablets: one-step rate * k is the first-order expectation;
    # use the full Chao extrapolation for m = k
    T = len(train); Q1 = sum(1 for v in cnt.values() if v == 1); Q2 = sum(1 for v in cnt.values() if v == 2)
    Q0 = (T - 1) / T * Q1 * Q1 / (2 * Q2) if Q2 > 0 else (T - 1) / T * Q1 * (Q1 - 1) / 2
    k = len(test)
    pred_union = Q0 * (1 - (1 - Q1 / (T * Q0 + Q1)) ** k) if Q0 > 0 else 0.0
    obs_union = len(set().union(*test) - seen) if test else 0
    return dict(pred_one=per_new, obs_mean_one=float(np.mean(obs_per_tab)) if test else 0.0,
                pred_union=pred_union, obs_union=obs_union, k=k,
                frac_new_obs=(sum(obs_per_tab) / max(1, sum(len(t) for t in test))),
                slots_mean=float(np.mean([len(t) for t in train])))


def run_A(ndraw=1500):
    out = {}
    # ---- LA: full-data prediction for one new HT tablet, over draws
    preds, slots, fracs = [], [], []
    for i in range(ndraw):
        tabs = person_sets(rng, TAB)
        pn, cnt = predict_new(tabs)
        preds.append(pn)
        slots.append(np.mean([len(t) for t in tabs]))
        fracs.append(pn / max(1e-9, np.mean([len(t) for t in tabs])))
    out['LA_new_per_tablet'] = dict(med=float(np.median(preds)), lo=float(np.percentile(preds, 5)), hi=float(np.percentile(preds, 95)))
    out['LA_slots_per_tablet'] = dict(med=float(np.median(slots)), lo=float(np.percentile(slots, 5)), hi=float(np.percentile(slots, 95)))
    out['LA_frac_new'] = dict(med=float(np.median(fracs)), lo=float(np.percentile(fracs, 5)), hi=float(np.percentile(fracs, 95)))
    # among tablets that carry person-like words at all
    nz = []
    for i in range(300):
        tabs = person_sets(rng, TAB)
        L = [len(t) for t in tabs if len(t) > 0]
        nz.append(np.mean(L))
    out['LA_slots_nonempty'] = dict(med=float(np.median(nz)))
    print('LA one new tablet: new names', out['LA_new_per_tablet'], 'slots', out['LA_slots_per_tablet'], 'frac new', out['LA_frac_new'], flush=True)
    # ---- LA room hold-out
    rooms = C2.FS4
    rh = collections.defaultdict(list)
    for i in range(200):
        tabs = person_sets(rng, TAB)
        for r in rooms:
            idx = [j for j, f in enumerate(C2.DOCF) if f == r]
            e = holdout_eval(tabs, idx); rh[r].append(e)
        # random hold-out of the same sizes (null: tablets exchangeable)
        for r in rooms:
            k = sum(1 for f in C2.DOCF if f == r)
            idx = rng.choice(len(tabs), k, replace=False)
            rh['rand_' + r].append(holdout_eval(tabs, idx))
    out['LA_room_holdout'] = {r: {k: float(np.median([e[k] for e in v])) for k in ('pred_union', 'obs_union', 'pred_one', 'obs_mean_one', 'frac_new_obs', 'k')} for r, v in rh.items()}
    for r, v in out['LA_room_holdout'].items(): print('room holdout', r, {k: round(x, 2) for k, x in v.items()}, flush=True)
    # ---- LB calibration: LA-sized training draw, 30 held-out tablets from the same site
    LB = lb_docs(); L = lb_name_label(); target = len(C2.OCC)
    for site in ('KN', 'PY'):
        docs = [d for d in LB if d['site'] == site and any(k == 'W' for ln in d['lines'] for k, _ in ln)]
        ev = []
        for rep in range(200):
            idx = rng.permutation(len(docs)); sel, n = [], 0
            for i in idx:
                d = docs[i]; sel.append(d); n += sum(1 for ln in d['lines'] for t, _ in ln if t == 'W')
                if n >= target: break
            rest = [docs[i] for i in idx[len(sel):len(sel) + 30]]
            tabs = [set(v for ln in d['lines'] for k, v in ln if k == 'W' and v in L) for d in sel + rest]
            e = holdout_eval(tabs, range(len(sel), len(sel) + len(rest)))
            ev.append(e)
        out['LB_' + site] = {k: (float(np.median([e[k] for e in ev])), float(np.mean([e[k] for e in ev]))) for k in ('pred_union', 'obs_union', 'pred_one', 'obs_mean_one', 'frac_new_obs', 'slots_mean')}
        z = np.array([e['obs_union'] - e['pred_union'] for e in ev])
        out['LB_' + site]['bias_union_mean'] = float(z.mean()); out['LB_' + site]['bias_union_sd'] = float(z.std())
        print('LB', site, out['LB_' + site], flush=True)
    # ---- planted
    pl = {}
    for N in (150, 400, 1000, 3000):
        ev = []
        for rep in range(60):
            tabs = C2.planted(N, rng, sigma=1.2, local=0.7, hom=0.15, var=0.1)
            idx = rng.choice(len(tabs), 20, replace=False)
            ev.append(holdout_eval(tabs, idx))
        pl[N] = {k: float(np.mean([e[k] for e in ev])) for k in ('pred_union', 'obs_union', 'pred_one', 'obs_mean_one')}
        print('planted', N, pl[N], flush=True)
    out['planted'] = pl
    return out


# ---------------------------------------------------------------- (B) endemism of person-like words
def run_B(nperm=2000):
    ALL = la_docs(site=None, support=None)
    occ = occurrences(ALL)
    sites = collections.defaultdict(set)
    for o in occ: sites[o['w']].add(ALL[o['doc']]['site'])
    htw = [w for w in WORDS]
    pw = np.array([P.get(w, (0, 0))[0] for w in htw])
    away = np.array([len(sites[w] - {'Haghia Triada'}) > 0 for w in htw], float)
    isP = pw >= 0.5
    obs = dict(person_away=float(away[isP].mean()), nonperson_away=float(away[~isP].mean()),
               n_person=int(isP.sum()), n_non=int((~isP).sum()))
    # weighted by p (no threshold)
    obs['p_weighted_away_person'] = float((pw * away).sum() / pw.sum())
    obs['p_weighted_away_non'] = float(((1 - pw) * away).sum() / (1 - pw).sum())
    # null: permute the person label among HT words with the same tablet-frequency band
    freq = collections.Counter(w for t in TAB for w in t)
    band = np.array([min(freq[w], 4) for w in htw])
    diffs = []
    for _ in range(nperm):
        lab = isP.copy()
        for b in set(band):
            ii = np.where(band == b)[0]; lab[ii] = rng.permutation(lab[ii])
        diffs.append(away[lab].mean() - away[~lab].mean())
    d0 = obs['person_away'] - obs['nonperson_away']
    obs['diff'] = d0; obs['p_le'] = float((np.array(diffs) <= d0).mean()); obs['null_mean'] = float(np.mean(diffs))
    # Linear B analogue: KN words, labelled names vs others, found at PY
    LB = lb_docs(); L = lb_name_label()
    kn = set(v for d in LB if d['site'] == 'KN' for ln in d['lines'] for k, v in ln if k == 'W')
    py = set(v for d in LB if d['site'] == 'PY' for ln in d['lines'] for k, v in ln if k == 'W')
    nm = [w for w in kn if w in L]; non = [w for w in kn if w not in L]
    obs['LB_KN_names_at_PY'] = float(np.mean([w in py for w in nm])); obs['LB_KN_non_at_PY'] = float(np.mean([w in py for w in non]))
    print('B', obs, flush=True)
    return obs


if __name__ == '__main__':
    res = dict(A=run_A(), B=run_B())
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), indent=1)
