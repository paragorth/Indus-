"""pe69 cycle 3: freeze predictions from what survived (the per-line allotment: a bare 'M288 n' line carries
60 N39C, sometimes 120 N39C, per unit of the count line directly above it, best after PERSON-final lines),
for broken tablets; re-grade the pe68 team-sum frozen predictions; in-corpus self-check on broken lines.

3a self-check: bare M288 line clean, the count line above it damaged -> implied lost count = M288/60 (or /120)
   must be a whole number 1..30. Null: M288 values from random clean bare slots (5,000x).
3b frozen: lost M288 values (count line above clean) and lost counts (M288 clean, count above damaged), with
   kill lines; the pe68 team-sum predictions re-graded (those that differ from the per-line prediction are now
   expected to FAIL); the M056 -> M288 48-per-unit pairs (dossier list) as a C prediction.
"""
import os, sys, json, time
from collections import Counter
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe69_lib import *  # noqa


def main():
    T = tablets()
    S = slots(T)
    rng = np.random.default_rng(seed('pe69-c3'))
    res = {}
    bare = lambda s: s['m_signs'] == ['M288']
    pool = np.array([float(s['m']) for s in S if s['m'] is not None and bare(s)])
    # ---------- 3a
    cases = [s for s in S if s['m'] is not None and bare(s) and s['before'] and s['before'][-1]['k'] == 'BROKEN']

    def whole(v):
        out = []
        for k in (60, 120):
            q = v / k
            out.append(abs(q - round(q)) < 1e-9 and 1 <= round(q) <= 30)
        return any(out), out[0]

    obs_any = sum(whole(float(s['m']))[0] for s in cases)
    obs_60 = sum(whole(float(s['m']))[1] for s in cases)
    na, n6 = [], []
    for _ in range(5000):
        vv = rng.choice(pool, len(cases))
        na.append(sum(whole(v)[0] for v in vv))
        n6.append(sum(whole(v)[1] for v in vv))
    na, n6 = np.array(na), np.array(n6)
    res['3a'] = {'cases': len(cases), 'whole_60_or_120': obs_any, 'null_mean': round(float(na.mean()), 2),
                 'p': float(((na >= obs_any).sum() + 1) / 5001), 'whole_60': obs_60, 'null60_mean': round(float(n6.mean()), 2),
                 'p60': float(((n6 >= obs_60).sum() + 1) / 5001)}
    print('3a', json.dumps(res['3a']), flush=True)
    # ---------- 3b frozen predictions
    lostM, lostC = [], []
    for s in S:
        lb = s['before'][-1] if s['before'] else None
        if s['m'] is None and lb is not None and lb['k'] == 'CNT' and 'M288' in s['m_signs'][-1:] and len(s['m_signs']) <= 1:
            lostM.append({'tablet': s['id'], 'line': s['line'], 'count_above': str(lb['v']),
                          'count_above_final_sign': lb['fin'], 'predicted_N39C': str(60 * lb['v']),
                          'alternative_N39C': str(120 * lb['v']),
                          'grade': 'B-' if lb['fin'] in PERSON else 'C'})
        if s['m'] is not None and bare(s) and lb is not None and lb['k'] == 'BROKEN':
            q60, q120 = s['m'] / 60, s['m'] / 120
            pred = [str(q) for q in (q60, q120) if q.denominator == 1 and 1 <= q <= 30]
            lostC.append({'tablet': s['id'], 'damaged_line': lb['i'], 'M288_N39C': str(s['m']), 'predicted_count': pred,
                          'status': 'predicted' if pred else 'rule cannot apply (M288/60 not a whole count <= 30)'})
    # re-grade the pe68 team-sum predictions
    F68 = json.load(open(os.path.join(DATA, 'pe68_frozen_broken_predictions.json')))
    regrade = []
    Sd = {(s['id'], s['line']): s for s in S}
    for p in F68['team_sum_M288']:
        if p['kind'] != 'lost M288 value':
            regrade.append({'pe68': p, 'pe69': 'team count prediction: depends on the killed team-sum form; withdrawn'})
            continue
        s = Sd.get((p['tablet'], p['line']))
        lb = s['before'][-1] if s and s['before'] and s['before'][-1]['k'] == 'CNT' else None
        if lb is None:
            regrade.append({'pe68': p, 'pe69': 'no clean count line directly above; withdrawn'})
        elif Fr(p['predicted_N39C']) == 60 * lb['v']:
            regrade.append({'pe68': p, 'pe69': 'same as the per-line prediction; kept (B-/C)'})
        else:
            regrade.append({'pe68': p, 'pe69': 'differs from the per-line prediction %s (or %s) N39C; pe69 expects the '
                            'team value to FAIL' % (60 * lb['v'], 120 * lb['v'])})
    m056 = json.load(open(os.path.join(CK, 'c2.json')))['2e']['pairs']
    out = {'made': time.strftime('%Y-%m-%d %H:%M'),
           'rule': 'per-line allotment: a bare M288 line = 60 N39C (alt. 120) x the count line directly above it; '
                   'strongest after PERSON-final lines (pe69 cycle 2: 34/49 vs 5.5 by chance); NOT the team sum (killed, pe69 cycle 1)',
           'lost_M288_values': lostM, 'lost_counts': lostC, 'pe68_team_sum_regraded': regrade,
           'M056_48_pairs_seen': m056,
           'kill_lines': {
               'lost_M288_values': 'a collation or join restoring >= 5 of these: if fewer than 2 equal 60 or 120 x the count '
                                   'above, the per-line rule is wrong for broken tablets (in-corpus clean rate ~0.35 on bare lines)',
               'lost_counts': 'restored counts: if none of the first 5 equals M288/60 or M288/120, kill',
               'pe68_team_values': 'if >= 2 of the first 3 restored pe68 team values that differ from the per-line value are '
                                   'confirmed, the pe69 kill of the team rule is itself wrong',
               'M056_48': 'C: new M056 count lines followed by a bare M288 line in the P0087xx-P0088xx dossier carry 48 N39C '
                          'per unit; killed if < 2 of the first 5 new pairs do'}}
    out['sha256'] = sha(out)
    json.dump(out, open(os.path.join(DATA, 'pe69_frozen_predictions.json'), 'w'), indent=1)
    res['3b'] = {'lost_M288_values': len(lostM), 'by_grade': dict(Counter(x['grade'] for x in lostM)),
                 'lost_counts': len(lostC), 'lost_counts_predicted': sum(1 for x in lostC if x['predicted_count']),
                 'pe68_regrade': dict(Counter(r['pe69'].split(';')[0].split(' (')[0][:40] for r in regrade)),
                 'sha': out['sha256'][:16]}
    print('3b', json.dumps(res['3b']), flush=True)
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
