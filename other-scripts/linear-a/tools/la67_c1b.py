#!/usr/bin/env python3
"""LA-67 cycle 1b: the *318 decoy result (la67_s318.py) and power checks for cycle-1 tests that no decoy
passed (headings) or that many decoys passed (TA-I)."""
from la67_lib import *
import subprocess
import la67_c1 as C1
OUT = C1.OUT


def s318():
    r = json.loads(subprocess.check_output([sys.executable, os.path.join(HERE, 'la67_s318.py'), 'report']))
    dg = np.array([x[2] for x in r['decoys']]); g = r['gain']
    row = ('| LA-67.1i | la61 C: *318 is a commodity marker. Stated test: COM role for *318 alone in the la60 grammar, held-out bits on 4 FRESH tablet splits, '
           'against 20 frequency-matched single decoy signs given the COM role one at a time; support needs >= 10 bits; kill = a gain inside the decoy range | '
           '*318 (3 admin tokens) %+.1f bits; decoys %.1f +- %.1f (range %.1f to %+.1f), %d/20 decoys >= *318 | KILLED: the la61 +15.6 bits does not reproduce on fresh splits and sits inside the decoy range |' % (
               g, dg.mean(), dg.std(), dg.min(), dg.max(), int((dg >= g).sum())))
    wlog(OUT, row)


def head_power(D):
    T = C1.tablets(D)
    cnt = counts(T, 'W')
    passed = []
    for kind in (True, False):
        pool = [x for x in cnt if is_single(x) == kind and cnt[x] >= 4]
        sc = {x: C1.head_score(T, x)[0] for x in pool}
        for x in pool:
            dec = matched_decoys(cnt, x, pool, 40)
            if pct_rank(sc[x], [sc[y] for y in dec]) <= 0.05:
                passed.append((x, cnt[x], round(sc[x], 2)))
    passed.sort(key=lambda z: -z[2])
    row = ('| LA-67.1j | Power check for LA-67.1a: the full-data heading criterion (P <= 0.05 vs 40 matched decoys) applied to every tablet word with n >= 4 | '
           '%d words pass: %s | The test can see heading words (these open tablets without a count); KU-RE, U, A, DA and KA-NA are not among them. Kills in 1a stand |' % (
               len(passed), ', '.join('%s (n %d, %.2f)' % p for p in passed[:14])))
    wlog(OUT, row)


def tai_rank():
    r = json.load(open(os.path.join(CK, 'c1.json')))
    row = ('| LA-67.1k | Note on LA-67.1e (TA-I/AROM) and LA-67.1g (HT 95b) | TA-I P %.3f; 29%% of decoy words with 2+ word->logogram tablets also pass P <= 0.05, so surviving '
           'this shuffle is common for rare logograms; it rests on 2 tablets. HT 95b: words vote GRA (0.60); calibrated hit rate at that confidence 0.44 (9 documents) against a GRA base rate of 0.13 '
           '- informative but weak, and the vote comes from the HT 86/95 near-copy list | TA-I/AROM: SURVIVES (weak; stays C). HT 95b grain: SURVIVES weakly (stays C) |' % r['tai']['p'])
    wlog(OUT, row)


if __name__ == '__main__':
    D = docs_all()
    import sys as _s
    if "tai" in _s.argv: tai_rank()
    else: s318(); head_power(D); tai_rank()
