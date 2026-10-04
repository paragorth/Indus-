"""pe26 cycle 1: does clay appearance predict find site beyond photo batch and museum?
Contrasts are always inside one museum. Three model families:
  BATCH  : photo-batch covariates only (EXIF batch one-hot via within-batch label rates is not allowed; we use bg colour, size, fg_frac)
  RAW    : clay colour+texture, grouped CV by photo batch (test batches never seen in training)
  WB     : clay features centred within each photo batch (only within-batch contrast survives), grouped CV by batch
Nulls: labels permuted within photo batch (WB and RAW), 200 draws. Planted control: b* (+delta) and texture shift added to
random pseudo-imports drawn inside the same batches as the real minority, recovery AUC.
Controls: same-museum non-PE contrasts (NMI: Susa OB vs Palum OB; Louvre: Susa vs Mesopotamian Uruk III; Susa PE vs Susa Uruk V)."""
import sys, json, os, numpy as np, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe26_common import *
rng = np.random.default_rng(26)
rows = load()
rows = [r for r in rows if r.get('is_grey', 0) < 0.5]
print('rows (colour photos):', len(rows), collections.Counter((r['group'], r['museum'], r['site'], r['period'][:6]) for r in rows).most_common(40))
NPERM = int(os.environ.get('NPERM', 200))

def contrast(name, A, B, note=''):
    if len(A) < 3 or len(B) < 3:
        print(json.dumps(dict(name=name, nA=len(A), nB=len(B), skipped='fewer than 3 colour photos'))); return dict(name=name, nA=len(A), nB=len(B), skipped=True)
    sub = A + B
    y = np.array([0] * len(A) + [1] * len(B))
    bat = [r['batch'] for r in sub]
    mixed = [k for k in set(bat) if len(set(y[np.array(bat) == k])) == 2]
    nmix = sum(np.isin(bat, mixed)[y == 1])
    Xc = X_of(sub, CLAY); Xb = X_of(sub, BATCHCOV)
    out = dict(name=name, nA=len(A), nB=len(B), n_batches=len(set(bat)), mixed_batches=len(mixed), minority_in_mixed=int(nmix), note=note)
    out['strat_raw'] = strat_cv_auc(Xc, y)[0]
    out['grp_batchcov'] = cv_auc(Xb, y, bat)[0]
    out['grp_raw'] = cv_auc(Xc, y, bat)[0]
    out['grp_raw_plus_batchcov'] = cv_auc(np.hstack([Xc, Xb]), y, bat)[0]
    # within-batch centred features; folds by tablet (stratified) since batch info removed; test also grouped
    Xw = within_batch_centre(Xc, bat)
    out['wb_strat'] = strat_cv_auc(Xw, y)[0]
    # null: within-batch label permutation, same pipeline
    if NPERM:
        nul_raw, nul_wb = [], []
        for i in range(NPERM):
            yp = perm_within_batch(y, bat, rng)
            if len(set(yp)) < 2: continue
            nul_raw.append(cv_auc(Xc, yp, bat, seed=i)[0]); nul_wb.append(strat_cv_auc(Xw, yp, seed=i)[0])
        nul_raw = np.array(nul_raw); nul_wb = np.array(nul_wb)
        out['null_raw_q95'] = float(np.nanpercentile(nul_raw, 95)); out['p_raw'] = float((np.sum(nul_raw >= out['grp_raw']) + 1) / (len(nul_raw) + 1))
        out['null_wb_q95'] = float(np.nanpercentile(nul_wb, 95)); out['p_wb'] = float((np.sum(nul_wb >= out['wb_strat']) + 1) / (len(nul_wb) + 1))
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    return out

def planted(A, B, delta_b=3.0, reps=20):
    """replace the minority labels with pseudo-imports drawn from A inside the same batches as B, add a b*/a* shift."""
    res = []
    batA = collections.defaultdict(list)
    for i, r in enumerate(A): batA[r['batch']].append(i)
    for rep in range(reps):
        pick = []
        for r in B:
            c = batA.get(r['batch']) or list(range(len(A)))
            pick.append(int(rng.choice(c)))
        pick = sorted(set(pick))
        sub = [dict(r) for r in A]; y = np.zeros(len(sub), int); y[pick] = 1
        for i in pick:
            for k in ['b_p10', 'b_p25', 'b_med', 'b_p75', 'b_p90']: sub[i][k] += delta_b
            sub[i]['b_over_L'] = sub[i]['b_med'] / max(sub[i]['L_med'], 1)
        Xc = X_of(sub, CLAY); bat = [r['batch'] for r in sub]
        res.append((cv_auc(Xc, y, bat, seed=rep)[0], strat_cv_auc(within_batch_centre(Xc, bat), y, seed=rep)[0]))
    res = np.array(res)
    return dict(delta_b=delta_b, grp_raw=float(np.nanmean(res[:, 0])), wb=float(np.nanmean(res[:, 1])))

S = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]
results = {}
pe_nmi_susa = S(group='PE', museum='NMI', site='Susa')
pe_nmi_plat = [r for r in rows if r['group'] == 'PE' and r['museum'] == 'NMI' and r['site'] in ('Yahya', 'Malyan', 'Ozbaki')]
pe_lou_susa = S(group='PE', museum='Louvre', site='Susa')
pe_lou_sialk = [r for r in rows if r['museum'] == 'Louvre' and r['site'] == 'Sialk']
results['NMI_PE_Susa_vs_plateau'] = contrast('NMI PE Susa vs Yahya+Malyan+Ozbaki', pe_nmi_susa, pe_nmi_plat)
if len(S(group='PE', museum='NMI', site='Yahya')) >= 3:
    results['NMI_PE_Susa_vs_Yahya'] = contrast('NMI PE Susa vs Yahya', pe_nmi_susa, S(group='PE', museum='NMI', site='Yahya'))
if len(S(group='PE', museum='NMI', site='Malyan')) >= 3:
    results['NMI_PE_Susa_vs_Malyan'] = contrast('NMI PE Susa vs Malyan', pe_nmi_susa, S(group='PE', museum='NMI', site='Malyan'))
results['LOU_Susa_vs_Sialk'] = contrast('Louvre Susa PE vs Sialk (PE + Uruk III)', pe_lou_susa, pe_lou_sialk)
# controls
meso = [r for r in rows if r['museum'] == 'Louvre' and r['site'] in ('Girsu', 'JemdetNasr', 'Uruk', 'Larsa', 'Kish')]
if len(meso) >= 5: results['CTRL_LOU_Susa_vs_Meso'] = contrast('CTRL Louvre Susa PE vs Mesopotamian Uruk III', pe_lou_susa, meso, 'positive check')
ukv = [r for r in rows if r['group'] == 'CTRL' and r['museum'] == 'Louvre' and r['site'] == 'Susa']
if len(ukv) >= 5: results['CTRL_LOU_SusaPE_vs_SusaUrukV'] = contrast('CTRL Louvre Susa PE vs Susa Uruk V (same clay, earlier)', pe_lou_susa, ukv, 'same-site check')
ob = [r for r in rows if r['group'] == 'CTRL' and r['museum'] == 'NMI' and r['site'] == 'Susa' and r['period'].startswith('Old Bab')]
pal = [r for r in rows if r['group'] == 'CTRL' and r['site'] == 'Palum']
if len(pal) >= 5: results['CTRL_NMI_SusaOB_vs_Palum'] = contrast('CTRL NMI Susa OB vs Palum OB', ob, pal, 'same museum, same period, different site')
per = [r for r in rows if r['group'] == 'CTRL' and r['site'] == 'Persepolis']
if len(per) >= 5: results['CTRL_NMI_SusaOB_vs_Persepolis'] = contrast('CTRL NMI Susa OB vs Persepolis', ob, per, 'same museum, different site and period')

PER = ('Ur III', 'Old Akkadian', 'Old Babyloni', 'Lagash II (c', 'ED IIIb (ca.')
lsusa = [r for r in rows if r['group'] == 'CTRL' and r['museum'] == 'Louvre' and r['site'] == 'Susa' and r['period'].startswith(PER)]
lgir = [r for r in rows if r['group'] == 'CTRL' and r['museum'] == 'Louvre' and r['site'] == 'Girsu' and r['period'].startswith(PER)]
if len(lsusa) >= 5 and len(lgir) >= 5:
    results['CTRL_LOU_Susa_vs_Girsu_3rd2nd'] = contrast('CTRL Louvre Susa vs Girsu, 3rd-early 2nd mill.', lsusa, lgir, 'positive check: same museum, overlapping periods, different site')
    results['CTRL_LOU_SusaPE_vs_SusaLater'] = contrast('CTRL Louvre Susa PE vs Susa later periods', pe_lou_susa, lsusa, 'same site, different period')
lsusa_ne = [r for r in rows if r['group'] == 'CTRL' and r['museum'] == 'Louvre' and r['site'] == 'Susa' and r['period'].startswith('Neo-Elamite')]
# negative: random halves of NMI Susa PE by museum-number parity
par = [r for r in pe_nmi_susa if any(ch.isdigit() for ch in r['museum_no'] or '')]
odd = [r for r in par if int(''.join(ch for ch in r['museum_no'] if ch.isdigit())[-1]) % 2]
even = [r for r in par if r not in odd]
results['NEG_NMI_Susa_parity'] = contrast('NEG NMI Susa PE odd vs even museum no.', even, odd, 'negative check')
# planted
results['PLANT_NMI'] = {str(d): planted(pe_nmi_susa, pe_nmi_plat, d, reps=10) for d in (1.0, 2.0, 4.0)}
print('planted', results['PLANT_NMI'])
json.dump(results, open(os.path.join(CK, 'cycle1.json'), 'w'), indent=1)
