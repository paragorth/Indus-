#!/usr/bin/env python3
"""LA-72 cycle 3b: write the corrected prediction file (loops/la72_predictions.txt, new sha256).
Starts from the 37 frozen la70 predictions (+ the order note), keeps the la70 file untouched for the
record, and applies the corrections found in cycle 3a (data/la72_ckpt/c3_evidence.json) and the la71
grade changes. Each entry gets la72_status: unchanged / corrected / withdrawn, and a la72_note."""
import json, os, hashlib, copy
from la72_common import CK, HERE

OLD = os.path.join(HERE, '..', 'loops', 'la70_predictions.txt')
NEW = os.path.join(HERE, '..', 'loops', 'la72_predictions.txt')

js = open(OLD).read().split('=====JSON=====\n')[1]
old_h = hashlib.sha256(js.encode()).hexdigest()
old = json.loads(js)
ev = json.load(open(os.path.join(CK, 'c3_evidence.json')))
fz = json.load(open(os.path.join(CK, 'frozen_V2c.json')))
c2 = json.load(open(os.path.join(CK, 'c2_summary.json')))
rd = c2['rd']; rdn = rd['ntest']
full_pct = 100 * rd['V2c']['full'] / rdn; strict_pct = 100 * rd['V2c']['strict'] / rdn

C = {}   # id -> (status, new fields, note)
C['P-kuro-HT'] = ('corrected', dict(claim='HT: a new administrative tablet carries KU-RO with p 0.19 (27 of 144 tablets); a KU-RO section with >= 2 entries closes with p 0.44 (11 of 25)'),
                  'la70 counted HT 127b (292) as closing; its sum needs the erased line [[KI+MU]] 14, so on clean data the section is untestable (12/26 -> 11/25). Read-only tokens: 6 of 9.')
C['P-order-GRA-NI'] = ('corrected', dict(claim='on a new multi-commodity list GRA is written before NI (in corpus 16 of 19)'),
                       'the NI of HT 96b is erased (la70 counted it). Read-only tokens: 7 of 7.')
C['P-order-VIR-NI'] = ('corrected', dict(claim='on a new multi-commodity list VIR is written before NI (in corpus 8 of 10)'),
                       'KH 8 order changes once its erased logogram is removed. Read-only tokens: too few (no prediction made there); VIR is 43 of 58 damaged.')
C['P-order-OLE-NI'] = ('withdrawn', {}, 'rested on the erased NI of HT 96b: 12/17 -> 11/16 (0.69), below la70\'s own 0.7 threshold for making a prediction; read-only 6/10.')
C['P-order-note'] = ('corrected', dict(claim='pairs of the v2 order NOT turned into predictions (in-corpus agreement < 0.7): CYP<OLE 5/9; CYP<VIR 3/8; OLE<NI 11/16; OLIV<VIN 4/6; VIR<VIN 4/6'),
                     'OLE<NI added (withdrawn as a prediction by la72).')
for pid in ('P-order-CYP-QA2', 'P-order-OLE-QA2'):
    C[pid] = ('corrected', dict(grade='C (la50 staple order; every QA2 token is a ligature with an unread part)'),
              'count unchanged on rd; all 19 QA2 tokens are damaged, so the pair has no read-only support. Grade C+ -> C.')
C['P-order-GRA-VIR'] = ('corrected', dict(grade='C (la50 staple order; VIR mostly damaged)'),
                        'count unchanged on rd (6/8); on read-only tokens fewer than 5 lists: VIR is 43 of 58 damaged. C+ -> C.')
C['P-order-GRA-VIN'] = ('corrected', dict(grade='C (la50 staple order; read-only 5/8)'),
                        'count unchanged on rd (11/15); on read-only tokens 5 of 8 (below 0.7). C+ -> C.')
C['P-KI'] = ('corrected', dict(claim='KI before a number marks a reduced amount: smaller than the entry just above it (in corpus 8 of 9, 4 tablets; on cleanly read tokens 4 of 5, 3 of them HT 118)'),
             'rd count unchanged, but 4 of the 8 supporting cases involve a damaged KI or amount; read-only 4/5 (HT 97a, HT 118 x3; HT 49a against). Passes its train-only gate on rd (6/6, P 0.011), fails on read (too few cases). Stays C.')
C['P-head-*307'] = ('withdrawn', {}, 'the three uncounted first-line uses (HT 5, HT 27a, HT 32) are all damaged; on read-only tokens *307 stands directly before a number on 2 of 3 uses (HT 127b, PE Wy 5), which already meets this prediction\'s own kill line; *307 fails the read-only heading gate. C -> withdrawn.')
C['P-head-A-DU'] = ('corrected', dict(claim='A-DU opens a document (first line) without its own count (in corpus n 9; 6 cleanly read, 4 of them on the first line)'),
                    '3 of 9 uses damaged; read-only gate keeps it in 7/10 splits. C unchanged.')
C['P-PADE'] = ('corrected', dict(claim='PA-DE is a marked entry: own line, small amount (HT 9a, 122a cleanly read; HT 9b damaged)'), 'HT 9b\'s PA-DE is damaged. C unchanged.')
C['P-zakros-template'] = ('corrected', dict(claim='*28B-NU-MA-RE and SI-PI-KI occur only on the Zakros wine template (ZA 4, 5, 15; SI-PI-KI damaged on ZA 15)'),
                          'unchanged on rd; ZA 15\'s SI-PI-KI is damaged. C unchanged.')
C['P-sitedefault'] = ('corrected', dict(claim='a document with numbers but no commodity sign is CYP at Khania and GRA at Arkhalokhori', grade='B (KH, la54); C (ARKH, n 3)',
                                        would_kill='a different commodity on >= half of 5 such new documents at that site'),
                      'Zakros withdrawn: the train-only gate drops it (training ZA ties NI 2 / GRA 2) and over all documents ZA ties GRA 7 / VIN 7 on rd, VIN 5 > GRA 4 on read-only tokens. KH CYP holds (34 vs NI 21; read 15 vs 10).')
C['P-rooms'] = ('corrected', dict(grade='C+ (QA2, TE, A-DU; la51, la67 3c, la71); C (NI, la71 read-only P 0.057)'),
                'la71 demoted the NI-house link to C; every QA2 token is a ligature with an unread part.')
C['P-consonants'] = ('corrected', dict(claim='inside new Hagia Triada word types, signs of the same consonant series co-occur less than re-dealt signs', grade='C (la10, la67 2b, la71: HT only)',
                                       would_support='avoidance P < 0.05 on 100+ new HT word types', would_kill='rate at the re-dealt level on 100+ new HT word types'),
                     'la71: outside HT the cleanly read word types show no avoidance (P 0.17). C+ -> C, restricted to HT.')
C['P-decode'] = ('corrected', dict(claim='the frozen V2c decoder (sha256 %s...) explains new HT/KH/ZA administrative documents in full at 18-25 %% and STRICT at >= 4 %% (held-out on clean data %.0f %% / %.1f %%; read-only tokens %.0f %% / %.1f %%), above its own role shuffles'
                                         % (fz['sha256'][:8], full_pct, strict_pct, 100 * c2['read']['V2c']['full'] / c2['read']['ntest'], 100 * c2['read']['V2c']['strict'] / c2['read']['ntest']),
                                         grade='B (la70 2b, la72 2a)'),
                 'rates re-measured on rd with V2c (decodes every held-out document exactly as la70 V2); the la70 text had a doubled percent sign.')
UNCH = {'P-kuro-KH': 'unchanged on rd (0/78; read-only 0/56).', 'P-kuro-ZA': 'unchanged on rd (1/25).',
        'P-308': 'unchanged: 9 of 9 on rd (one amount damaged), 8 of 8 on read-only tokens.',
        'P-DB': 'unchanged: no D >= 5 or B >= 3 on rd; HT 115a DDDD is cleanly read.',
        'P-TAI': 'unchanged: TA-I before AROM 3/3 on rd (HT 9b damaged), 2/2 read-only.',
        'P-sara2': 'not re-tested here; no token of the claim was erased or bridged.',
        'P-head-*516': 'unchanged: 7 uses, 1 damaged; read-only 6.',
        'P-firstsign': 'la71: holds in every version.', 'P-sites': 'la71: la58 deposit sharing holds in every version.',
        'P-QA': 'not re-run on clean data (no erased or bridged QA token).', 'P-zakros-read': 'not re-run on clean data.',
        'P-VINunit': 'untouched by the cleaning.', 'P-wordcom': 'not re-run on clean data (C-, decoy level).',
        'P-sealings': 'la71: the sealing heading class holds in every version.'}
for pid in ('P-order-CYP-NI', 'P-order-CYP-VIN', 'P-order-GRA-OLE', 'P-order-GRA-OLIV', 'P-order-OLE-VIN', 'P-order-NI-VIN'):
    rc = ev['read']['regenerated'].get(pid, '')
    UNCH[pid] = 'count unchanged on rd; read-only: %s.' % (rc.split('(in corpus ')[-1].rstrip(')') if rc else 'no prediction')

active = []; withdrawn = []; tally = {'unchanged': 0, 'corrected': 0, 'withdrawn': 0}
for p in old['predictions']:
    q = copy.deepcopy(p)
    if p['id'] in C:
        s, f, note = C[p['id']]
        q.update(f)
    else:
        s, note = 'unchanged', UNCH[p['id']]
    q['la72_status'] = s; q['la72_note'] = note
    if p['id'] != 'P-order-note':
        tally[s] += 1
    (withdrawn if s == 'withdrawn' else active).append(q)
body = dict(name='Linear A frozen outside-corpus predictions, corrected on the cleaned corpus (la72)', date='2026-10-06',
            corpus='data/corpus_ra.json, version rd (read + damaged; restored and erased tokens removed); read-only as the check',
            reading_V2c_sha256=fz['sha256'], reading_V2_la70_sha256=old['reading_V2_sha256'],
            previous=dict(la70_predictions=old_h, la60_predictions=old['previous']['la60_predictions']),
            scope=old['scope'], rule=old['rule'], tally=tally, predictions=active,
            withdrawn_for_the_record=withdrawn)
js2 = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
h = hashlib.sha256(js2.encode()).hexdigest()
with open(NEW, 'w') as f:
    f.write('LA-72 corrected outside-corpus predictions for Linear A (6 Oct 2026), replacing loops/la70_predictions.txt (sha256 %s..., kept for the record).\n' % old_h[:8])
    f.write('sha256 of the JSON block below: %s\n' % h)
    f.write('Check: python3 -c "import hashlib,sys;s=open(sys.argv[1]).read().split(\'=====JSON=====\\n\')[1];print(hashlib.sha256(s.encode()).hexdigest())" loops/la72_predictions.txt\n')
    f.write('Of the 37 la70 predictions: %d unchanged, %d corrected, %d withdrawn. Reading: V2c sha256 %s.\n\n' % (tally['unchanged'], tally['corrected'], tally['withdrawn'], fz['sha256']))
    for p in active:
        f.write('%-24s %s [%s] {%s}\n%24s would support: %s | would kill: %s\n%24s la72: %s\n' % (
            p['id'], p['claim'], p['grade'], p['la72_status'], '', p['would_support'], p['would_kill'], '', p['la72_note']))
    f.write('\nWITHDRAWN (kept for the record; do not score)\n')
    for p in withdrawn:
        f.write('%-24s %s [%s]\n%24s la72: %s\n' % (p['id'], p['claim'], p['grade'], '', p['la72_note']))
    f.write('\n=====JSON=====\n' + js2)
json.dump(dict(sha256=h, tally=tally), open(os.path.join(CK, 'c3_predictions_hash.json'), 'w'))
print('PREDICTIONS', tally, h)
