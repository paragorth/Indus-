"""pe17 cycle 3: power of the clay test, and what the five chemically foreign tablets record.

(a) Empirical 'import percentile': each real plateau tablet's out-of-group plateau score placed among the
    Susa cross-fitted scores (the same frozen file). This is how a genuine plateau-written tablet would rank if
    it were found at Susa. Power: draw k such percentiles (k = 1..20) and test them against uniform
    (one-sided Mann-Whitney vs 10,000 uniform draws); share of draws with p < 0.05.
    Control: the same with Susa tablets' own percentiles (must give ~5%).
(b) Content of ST-11, YT-01, YT-07, YT-02, YT-06: lines, numeral types, signs; compared with their find-site's
    other tablets (share of numeral types / signs shared with Susa vs with the find site).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe17_common import DATA, load, tablet_tokens

fr = json.load(open(os.path.join(DATA, 'pe17_frozen_ranking.json')))
sus = np.array(sorted(r['plateau_score'] for r in fr['susa']))
pl = [r for r in fr['plateau']]
perc = np.array([np.searchsorted(sus, r['plateau_score_out_of_group']) / len(sus) for r in pl])
res = {'import_percentile': dict(mean=float(perc.mean()), median=float(np.median(perc)),
                                 share_top10=float((perc >= 0.9).mean()), share_top25=float((perc >= 0.75).mean()),
                                 by_site={s: float(np.mean([p for p, r in zip(perc, pl) if r['site'] == s]))
                                          for s in sorted({r['site'] for r in pl})})}
rng = np.random.default_rng(3)
susp = (np.argsort(np.argsort(sus)) + 0.5) / len(sus)


def power(src, k, B=4000):
    # one-sided test: mean percentile of k imports vs the uniform null (exact by simulation)
    null = rng.random((20000, k)).mean(1)
    crit = np.quantile(null, 0.95)
    draws = rng.choice(src, (B, k)).mean(1)
    return float((draws > crit).mean())


res['power'] = {k: dict(real=power(perc, k), control_susa=power(susp, k)) for k in [1, 2, 3, 5, 8, 10, 15, 20]}
ST11 = [r for r in fr['susa'] if r['id'] == 'P009157'][0]
res['ST11_percentile'] = 1 - ST11['rank'] / len(sus)

# (b) contents
tabs = {t['id']: t for t in load()}
foreign = {'P009157': ('ST-11', 'Susa', 'Yahya'), 'P009536': ('YT-01', 'Yahya', 'Susa'), 'P009537': ('YT-07', 'Yahya', 'Susa'),
           'P009545': ('YT-02', 'Yahya', 'Malyan'), 'P009540': ('YT-06', 'Yahya', 'Malyan')}
bysite = {}
for t in tabs.values():
    tk, _ = tablet_tokens(t)
    bysite.setdefault(t['site'], []).append((t['id'], tk))


def doc_freq(site, key, excl):
    L = [tk for i, tk in bysite[site] if i != excl]
    c = {}
    for tk in L:
        for x in tk[key]:
            c[x] = c.get(x, 0) + 1
    return {x: v / len(L) for x, v in c.items()}


content = {}
for pid, (lab, found, clay) in foreign.items():
    t = tabs[pid]
    tk, _ = tablet_tokens(t)
    row = dict(sample=lab, found=found, clay=clay, des=t['des'], lines=[l['raw'] for l in t['lines']],
               numerals=sorted(tk['NUM']), signs=sorted(tk['SIGN']))
    for key in ['SIGN', 'NUM']:
        for site in ['Susa', 'Malyan', 'Yahya']:
            df = doc_freq(site, key, pid)
            row[f'{key}_mean_docfreq_{site}'] = round(float(np.mean([df.get(x, 0) for x in tk[key]])) if tk[key] else float('nan'), 3)
    content[pid] = row
res['content'] = content
json.dump(res, open(os.path.join(DATA, 'pe17_cycle3.json'), 'w'), indent=1)
print(json.dumps(res, indent=1, ensure_ascii=False))
