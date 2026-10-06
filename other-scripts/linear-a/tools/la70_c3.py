#!/usr/bin/env python3
"""LA-70 cycle 3: (a) full graded glosses of 20 held-out documents under the frozen V2 (primary split),
for a specialist to check; (b) every frozen outside-corpus prediction refreshed into one hashed file with
kill lines (loops/la70_predictions.txt; JSON copy in data/la70_ckpt/)."""
import json, os, re, hashlib, random, statistics as st
from la70_common import *

ROLEG = {'TOT': 'TOTAL', 'RES': 'residue/sub-heading', 'COM': 'commodity sign', 'HDR': 'heading',
         'TRX': 'heading/transaction word', 'MRK': 'marked entry', 'TPL': 'Zakros wine-template word',
         'RED': 'reduced amount'}


def num_txt(t):
    s = str(t[1])
    if t[2]:
        k, u = frac_known(t[2])
        parts = frac_parts(t[2])
        val = ' '.join(('%s=1/5' % p if p == 'D' else '%s=1/3' % p if p == 'B' else '%s=?' % p) for p in parts)
        s += ' ' + t[2] + ' [' + val + (', D/B values C' if k else ', unvalued') + ']'
    return s


def gloss_graded(d, R, acc):
    ch, rs, secs, coms = decode_v2(d, R, acc)
    toks = d['toks']; out = []; line = []
    tot_at = {s[0]: s for s in secs}
    i = 0
    while i < len(toks):
        t = toks[i]; r = rs[i]
        if t[0] == 'NL':
            if line:
                out.append(' '.join(line)); line = []
            i += 1; continue
        if t[0] == 'N':
            line.append(num_txt(t)); i += 1; continue
        if t[0] == 'L':
            g = 'A' if r == 'COM' else 'B'
            line.append('%s[commodity logogram, %s]' % (t[1], g)); i += 1; continue
        w = t[1]
        if w in R['roles']:
            role = R['roles'][w]
            grade = CAND.get(w, ('', 'B', ''))[1]
            note = ANNOT.get(w)
            lab = '%s[%s, %s%s]' % (w, ROLEG.get(role, role), grade, '; ' + note if note else '')
            if role == 'TOT' and i in tot_at:
                s = tot_at[i]
                if s[3] is not None:
                    lab += ' {sum of %d entries = %d%s: %s; arithmetic A}' % (
                        len(s[2]), sum(e[1] for e in s[2]), '+fractions' if any(e[2] for e in s[2]) else '',
                        'CLOSES' if s[3] else 'does NOT close')
            line.append(lab)
        elif r in ('ENT', 'ENT1'):
            note = ANNOT.get(w)
            line.append('%s[entry name, unread; slot B%s]' % (w, '; ' + note if note else ''))
        elif r == 'LIB':
            line.append('%s[libation-register word, B]' % w)
        else:
            line.append('%s[?? unexplained]' % w)
        i += 1
    if line:
        out.append(' '.join(line))
    o = R['order']
    ordtxt = ''
    if coms:
        ordtxt = 'commodities written: %s' % ' > '.join(coms)
        if ch['n_order_agree'] or ch['n_order_viol']:
            ordtxt += ' (staple order C+: %d pair(s) agree, %d against)' % (ch['n_order_agree'], ch['n_order_viol'])
    elif R['site_default'].get(d['site']):
        ordtxt = 'no commodity written; site default %s (B, la54)' % R['site_default'][d['site']]
    fails = [k for k in ('cover', 'roles', 'totals', 'order', 'substantive', 'rules') if not ch[k]]
    status = 'FULL' + (' STRICT' if ch['strict'] else '') if ch['full'] else 'NOT FULL (fails: %s)' % ', '.join(fails)
    return status, out, ordtxt, ch


def pick20(te, R, acc):
    """20 held-out documents fixed by rule: every STRICT document, then FULL ones with most tokens,
    then non-FULL ones with a KU-RO, then the longest remaining; ties by id."""
    sc = []
    for d in te:
        ch = decode_v2(d, R, acc)[0]
        n = sum(t[0] != 'NL' for t in d['toks'])
        tot = any(t[0] == 'W' and t[1] == 'KU-RO' for t in d['toks'])
        key = (0 if ch['strict'] else 1 if ch['full'] and n >= 6 else 2 if tot else 3, -n, d['id'])
        sc.append((key, d))
    sc.sort(key=lambda x: x[0])
    return [d for _, d in sc[:20]]


def predictions(A, R):
    """Outside-corpus predictions, refreshed from all 387 administrative documents (in-sample rates)."""
    by_site = defaultdict(list)
    for d in A:
        by_site[d['site']].append(d)
    P = []

    def add(pid, claim, grade, support, kill, source):
        P.append(dict(id=pid, claim=claim, grade=grade, would_support=support, would_kill=kill, source=source))
    # KU-RO rates
    for s in ('HT', 'KH', 'ZA'):
        tabs = {d['tab'] for d in by_site[s]}
        kt = {d['tab'] for d in by_site[s] if any(t[0] == 'W' and t[1] == 'KU-RO' for t in d['toks'])}
        secs = [x for d in by_site[s] for x in sections_v2(d, R) if x[3] is not None]
        cl = sum(1 for x in secs if x[3])
        txt = '%s: a new administrative tablet carries KU-RO with p %.2f (%d of %d tablets)' % (s, len(kt) / max(1, len(tabs)), len(kt), len(tabs))
        if secs:
            txt += '; a KU-RO section with >= 2 entries closes with p %.2f (%d of %d)' % (cl / len(secs), cl, len(secs))
        add('P-kuro-' + s, txt,
            'A (KU-RO = total) / B (rates)', 'rate within the 95 % binomial range on >= 20 new tablets',
            'KU-RO on >= 3 of the first 10 new KH tablets (bound < 3/78)' if s == 'KH' else
            'KU-RO on >= 4 of the first 15 new ZA tablets' if s == 'ZA' else
            'a KU-RO section closing rate below half the stated rate on >= 10 new sections', 'la60 P1-P3, la70')
    # order pairs under the v2 staple order
    wins = Counter()
    for d in A:
        s = com_seq_v2(d, R['roles'])
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                wins[(s[i], s[j])] += 1
    o = R['order']; weak = []
    for a in o:
        for b in o:
            if o[a] < o[b] - 0.3 and wins[(a, b)] + wins[(b, a)] >= 5:
                n = wins[(a, b)] + wins[(b, a)]
                if wins[(a, b)] / n < 0.7:
                    weak.append('%s<%s %d/%d' % (a, b, wins[(a, b)], n)); continue
                add('P-order-%s-%s' % (a, b), 'on a new multi-commodity list %s is written before %s (in corpus %d of %d)' % (a, b, wins[(a, b)], n),
                    'C+ (la50 staple order; la20 A existence)', 'the next 5 new lists with both agree >= 4',
                    '%s after %s on >= 3 new lists' % (a, b), 'la20, la50, la67 2k, la70')
    add('P-order-note', 'pairs of the v2 order NOT turned into predictions (in-corpus agreement < 0.7): ' + '; '.join(weak),
        'n/a', 'n/a', 'n/a', 'la70')
    add('P-sara2', 'after SA-RA2 the first commodity is GRA or CYP, then NI, then VIN', 'B (la46)',
        'a new HT tablet with SA-RA2 in this order', 'SA-RA2 with a different order on 3+ new tablets', 'la46, la60 P5')
    add('P-308', '*308 never stands with a bare integer: every *308 amount carries a fraction sign (in corpus 9 of 9)',
        'C (la63, la67 1c)', '*308 with a fraction on 3+ new entries', '*308 with a bare integer on a new tablet', 'la63, la67, la70')
    add('P-KI', 'KI before a number marks a reduced amount: smaller than the entry just above it (in corpus 8 of 9, 4 tablets)',
        'C (la66, la70 1b)', '>= 7 of 10 new KI cases below the amount above', 'KI amounts >= the entry above in >= half of 10 new cases', 'la66, la70')
    add('P-zakros-template', '*28B-NU-MA-RE and SI-PI-KI occur only on the Zakros wine template (ZA 4, 5, 15)',
        'C (la65, la67 1f)', 'both together with a number on a new Zakros VIN tablet', 'either word on a tablet of another template or commodity', 'la65, la67')
    add('P-DB', 'D = 1/5 and B = 1/3: no amount writes D five or more times or B three or more times',
        'C (la27, la67 1h)', 'a sound DDD or DDDD on a new tablet; a new total that closes only with D = 1/5 or B = 1/3',
        'five Ds or three Bs in one amount; HT 115a re-read without DDDD', 'la27, la67')
    for w in ('*516', '*307', 'A-DU'):
        n = sum(1 for d in A for t in d['toks'] if t[0] == 'W' and t[1] == w)
        add('P-head-' + w, '%s opens a document (first line) without its own count (in corpus n %d)' % (w, n),
            'C (la67 1j; train gate stable %s)' % ('6/10' if w == 'A-DU' else '8/10' if w == '*307' else '10/10'),
            'opening 2 of the next 3 new documents it is on', 'a counted entry head (word directly before a number) on 2 of its next 3 uses', 'la67, la70')
    add('P-PADE', 'PA-DE is a marked entry: own line, small amount (HT 9a, 9b, 122a)', 'C (la53, la67 3h)',
        'PA-DE on its own line with an amount below the list median', 'PA-DE as an ordinary crowded entry on a new tablet', 'la53, la67')
    add('P-rooms', 'at Hagia Triada QA2 and TE are found in Villa magazine deposits; NI and A-DU in house deposits', 'C+ (la51, la67 3c)',
        'new HT finds of these words in the same kind of room', 'new finds of these words in the other kind of room (>= 3)', 'la51, la67')
    add('P-TAI', 'TA-I stands before AROM', 'C (la60, la67 1e)', 'TA-I AROM on a new tablet',
        'TA-I before another logogram on 2+ new tablets', 'la60, la67')
    add('P-firstsign', 'outside HT, consecutive entries share their first sign about twice as often as order shuffles (0.16 vs 0.08)',
        'C+ (la14, la67 2n)', '>= 0.12 on 30+ new non-HT entry pairs', '<= 0.08 on 30+ new non-HT entry pairs', 'la14, la67')
    add('P-consonants', 'inside new word types, signs of the same consonant series co-occur less than re-dealt signs', 'C+ (la10, la67 2b)',
        'avoidance P < 0.05 on 100+ new word types', 'rate at the re-dealt level on 100+ new word types', 'la10, la67')
    add('P-QA', 'QA patterns with the pure-vowel row in sign-neighbour grids', 'C+ (la21, la67 3j)',
        'the same placement with new word types added', 'QA leaves the vowel row when 100+ new word types are added', 'la21, la67')
    add('P-sitedefault', 'a document with numbers but no commodity sign is CYP at Khania and GRA at Zakros / Arkhalokhori', 'B (la54)',
        'the commodity of such documents shown by joins or seals', 'a different commodity on >= half of 5 such new documents', 'la54')
    add('P-sites', 'a new LM IB archive overlaps its own site deposits >= 2x more than other sites; it does not share words with HT at the HT-room level (< 0.08)',
        'C (la58)', 'overlap ratio >= 2', 'a new site sharing words with HT at >= 0.08', 'la58')
    add('P-zakros-read', 'Zakros stays the least readable site from other sites', 'C (la69)', 'same order with new Zakros tablets',
        'Zakros same-site readability above HT once 50+ Zakros documents exist', 'la69')
    add('P-VINunit', 'if HT 27a/89/100 amounts are day issues, the VIN unit is 7-24 l', 'C- (la64; untestable-now)',
        'ZA Zb 3-type vessels of 230-770 l', 'THE Zb 13 (VIN+TE 120) on a vessel under ~500 l, or KN Zb 35 (OLE 100) under ~20 l', 'la64')
    add('P-wordcom', 'entry words KU-PA3-NU, MA-DI (DI), SA-RU (GRA), SA-RO (VIN), DA-RE (VIR) return with the same commodity', 'C- (la60, la67 2m: decoy level)',
        '>= 3 of them with their commodity on new tablets', '< 1 in 3 right on 10+ new uses', 'la60 P6')
    add('P-decode', 'the frozen V2 decoder explains new HT/KH/ZA administrative documents in full at 20-25 %% and STRICT at >= 4 %% (held-out 23 %% / 5.5 %%), above its own role shuffles',
        'B (la70 2b)', 'STRICT rate above the 95th percentile of 100 role shuffles on >= 40 new documents',
        'STRICT rate at or below the shuffle mean on >= 40 new documents', 'la70')
    add('P-sealings', 'single signs on new roundels and nodules will not abbreviate tablet words of their own deposit (no positional rule)',
        'B (la68, negative)', 'no rule at P < 0.05', 'a positional rule covering >= half of new sealing tokens', 'la68')
    return P


def main():
    A = admin_docs(load_la())
    tr, te = split(A, 'la60-main')
    fz = json.load(open(os.path.join(CK, 'frozen_V2.json')))
    R = dict(name='V2', roles=fz['roles'], order=fz['order'], lib=set(fz['lib']), site_default=fz['site_default'],
             ent_slot_unknown=True, rules=fz['rules'])
    assert hash_v2(R) == fz['sha256']
    acc = acceptor_v2(tr, R)
    sel = pick20(te, R, acc)
    lines = ['LA-70 graded glosses of 20 held-out documents (frozen V2 sha256 %s; primary split la60-main;' % fz['sha256'],
             'selection rule fixed in code: STRICT documents, then FULL with >= 6 tokens, then non-FULL with KU-RO, then longest).',
             'Grades: A established, B likely, C guess. Sign names are the conventional transliteration labels only; no Linear B',
             'sound value or meaning is used. "entry name, unread" = a word in an entry slot whose referent is unknown.', '']
    stat = Counter()
    for d in sel:
        status, out, ordtxt, ch = gloss_graded(d, R, acc)
        stat[status.split(' (')[0]] += 1
        lines.append('%s (%s, %s) -- %s' % (d['id'], d['site'], d['support'], status))
        for k, l in enumerate(out):
            lines.append('   l.%d  %s' % (k + 1, l))
        if ordtxt:
            lines.append('   ' + ordtxt)
        lines.append('')
    open(os.path.join(CK, 'c3_glosses20.txt'), 'w').write('\n'.join(lines))
    # all held-out glosses
    with open(os.path.join(CK, 'c3_glosses_all.txt'), 'w') as f:
        for d in te:
            status, out, ordtxt, ch = gloss_graded(d, R, acc)
            f.write('%s\t%s\t%s | %s\n' % (d['id'], status, ' / '.join(out), ordtxt))
    print(stat)
    # predictions: reading re-gated on ALL documents (in-sample rates for outside data)
    Rall = gate_v2(A, 'V2-all')
    P = predictions(A, Rall)
    body = dict(name='Linear A frozen outside-corpus predictions (la70)', date='2026-10-06',
                reading_V2_sha256=fz['sha256'], reading_V2_all_docs_sha256=hash_v2(Rall),
                previous=dict(la60_predictions='538ad6fa64360eb8178f65a54ea78aa15d9dcee3532eb5181a704ec3063c50fb',
                              la66_factors='3127699d (data/la66_frozen_factors.json)'),
                scope='any Linear A administrative document not in lineara.xyz LinearAInscriptions.js as of 6 Oct 2026 (e.g. RILA-S1 2024, Anetaki II)',
                rule='score each prediction support / kill / untestable on the new documents before reading any la70 output on them',
                predictions=P)
    js = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
    h = hashlib.sha256(js.encode()).hexdigest()
    open(os.path.join(CK, 'c3_predictions.json'), 'w').write(js)
    out = os.path.join(HERE, '..', 'loops', 'la70_predictions.txt')
    with open(out, 'w') as f:
        f.write('LA-70 frozen outside-corpus predictions for Linear A (6 Oct 2026). sha256 of the JSON block below: %s\n' % h)
        f.write('Check: python3 -c "import hashlib,sys;s=open(sys.argv[1]).read().split(\'=====JSON=====\\n\')[1];print(hashlib.sha256(s.encode()).hexdigest())" loops/la70_predictions.txt\n\n')
        for p in P:
            f.write('%-24s %s [%s]\n%24s would support: %s | would kill: %s\n' % (p['id'], p['claim'], p['grade'], '', p['would_support'], p['would_kill']))
        f.write('\n=====JSON=====\n' + js)
    print('PREDICTIONS', len(P), h)


if __name__ == '__main__':
    main()
