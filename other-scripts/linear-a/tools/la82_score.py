"""la82: score frozen Linear A predictions on material outside data/corpus_ra.json (data/la82_new_material.json).
Every frozen file is hash-checked first (la82_verify). A prediction is scored only when the new material meets the
size its own kill line states; otherwise it is reported 'untestable' with what is missing. Where a count is made,
a shuffled-label null is computed (labels permuted within the document). Read-only on every frozen file.
usage: python3 la82_score.py"""
import os, sys, json, random, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, '..', 'data')
ver = json.loads(subprocess.run([sys.executable, os.path.join(HERE, 'la82_verify.py')], capture_output=True, text=True).stdout)
bad = [f for f, v in ver.items() if isinstance(v, dict) and v['claimed'] and not v['verified']]
assert not bad, f'frozen file altered: {bad}'
M = json.load(open(os.environ.get('LA82_MATERIAL', os.path.join(D, 'la82_new_material.json'))))
items = [x for x in M['items'] if x.get('usable_text')]
all_items = M['items']
rng = random.Random(82)

def count(pred):
    return sum(1 for x in items if pred(x))

admin = [x for x in items if x.get('numerals')]
nonHT_tokens = [s for x in items if not x['site'].startswith('Haghia') for s in (x.get('signs') or [])]
groups = [g for x in items for g in (x.get('sign_groups') or []) if len(g) >= 2 and all(s and s != '?' for s in g)]
rows = []
def row(pid, need, have, status, result):
    rows.append(dict(pred=pid, need=need, have=have, status=status, result=result))

# la72 KU-RO rates
for site, n_need in (('HT', 20), ('KH', 10), ('ZA', 15)):
    have = count(lambda x: x.get('sigla_prefix') == site and x['object_type'] == 'tablet')
    row(f'la72 P-kuro-{site}', f'>= {n_need} new {site} tablets', have, 'untestable' if have < n_need else 'test', '')
# la72 commodity order pairs: one 'list' = a document with >= 2 different commodity logograms in a fixed written order
PAIRS = [('CYP', 'NI'), ('CYP', 'VIN'), ('CYP', 'QA2'), ('GRA', 'OLE'), ('GRA', 'OLIV'), ('GRA', 'VIR'), ('GRA', 'NI'),
         ('GRA', 'VIN'), ('OLE', 'VIN'), ('OLE', 'QA2'), ('VIR', 'NI'), ('NI', 'VIN')]
for a, b in PAIRS:
    obs, null = [], []
    for x in items:
        seq = x.get('logogram_order') or []
        if a in seq and b in seq:
            obs.append(seq.index(a) < seq.index(b))
            sims = []
            for _ in range(10000):
                s = seq[:]; rng.shuffle(s); sims.append(s.index(a) < s.index(b))
            null.append(sum(sims) / len(sims))
    n = len(obs); agree = sum(obs)
    st = 'untestable' if n < 5 and n - agree < 3 else ('KILLED' if n - agree >= 3 else ('SUPPORTED' if agree >= 4 else 'not supported'))
    res = f'{agree}/{n} agree; shuffled-label null expects {sum(null):.2f}/{n}' if n else 'no new list with both'
    row(f'la72 P-order-{a}-{b}', '5 new lists with both (kill: 3 against)', n, st, res)
MINS = {'admin': 40, 'entries': 30, 'htwords': 100, 'KI': 10, '*308': 3, 'SA-RA2': 1, 'TA-I': 2, 'A-DU': 3, 'DB': 1, 'sealing': 5}
for pid, need, key in [('la72 P-decode', '>= 40 new admin documents (text)', 'admin'), ('la72 P-firstsign', '>= 30 new non-HT entry pairs', 'entries'),
                       ('la72 P-consonants / P-QA', '>= 100 new HT word types', 'htwords'), ('la72 P-KI', '10 new KI cases', 'KI'),
                       ('la72 P-308', '*308 with an amount on 3+ entries', '*308'), ('la72 P-sara2 / la75 SA-RA2', 'a new SA-RA2 entry', 'SA-RA2'),
                       ('la72 P-TAI', 'TA-I on 2+ new tablets', 'TA-I'), ('la72 P-head-A-DU / *516', '3 new uses', 'A-DU'),
                       ('la72 P-DB', 'a new amount with D or B (read)', 'DB'), ('la72 P-sealings', '>= 5 new sealing signs from a deposit with tablets (read)', 'sealing')]:
    if key == 'admin': have = len(admin)
    elif key == 'entries': have = sum(max(0, len(x.get('sign_groups') or []) - 1) for x in admin if not x['site'].startswith('Haghia'))
    elif key == 'htwords': have = len({tuple(g) for x in items if x['site'].startswith('Haghia') for g in (x.get('sign_groups') or [])})
    elif key == 'DB': have = count(lambda x: any(f in ('D', 'B') for f in (x.get('fraction_letters') or [])))
    elif key == 'sealing': have = count(lambda x: x['object_type'] in ('roundel', 'nodule') and x.get('signs'))
    else: have = count(lambda x: key in ['-'.join(g) for g in (x.get('sign_groups') or [])] + (x.get('logograms') or []))
    row(pid, need, have, 'untestable' if have < MINS[key] else 'test', '')
row('la73 seen-token share', '>= 50 new HT tablets with words', count(lambda x: x.get('sigla_prefix') == 'HT'), 'untestable', '')
bigfr = count(lambda x: x.get('fraction_letters') and (x.get('integer_with_fraction') or 0) >= 5)
row('la74 K/L2 on big integers', '30 new fractions on integers >= 5', bigfr, 'untestable' if bigfr < 30 else 'test', '')
row('la76 values / la66 factors', '>= 10 new same-document sibling entries per factor', len(admin), 'untestable', '')
row('la77 outflow vs phase', 'new dated near-variant families', 0, 'untestable', '')
row('la78 top-25 role survivors / la79 P2', '>= 50 new clean sign-groups', len(groups), 'untestable' if len(groups) < 50 else 'test', '')
# la79 P1: share of rising vs falling signs among new non-HT sign tokens, with the kill/support rule only at >= 200
P1 = json.load(open(os.path.join(D, 'la79_frozen_predictions.json')))['P1']
syl = [s for s in nonHT_tokens if s and s != '?']
r = sum(s in P1['rising'] for s in syl); f = sum(s in P1['falling'] for s in syl)
row('la79 P1 sign drift', '>= 200 new non-HT sign tokens', len(syl), 'untestable' if len(syl) < 200 else 'test',
    f'descriptive only: rising {r}/{len(syl)}, falling {f}/{len(syl)} (rule thresholds 0.271 / 0.385)')
row('la79 P3 backward transfer', 'a new non-HT archive', 0, 'untestable', '')
row('la79 P4 affixes / LB words', '>= 100-150 new word types', len({tuple(g) for g in groups}), 'untestable', '')
row('la80 c1-c3 blank shape', '50 new complete tablets with scaled photos', 0, 'untestable', '')
row('la81 c1/c2 cuts, Zakros last-gap', '10 new Zakros cuts / 10 HT repeats', 0, 'untestable', '')
row('la22 SigLA edge predictions', 'new SigLA documents', 0, 'untestable', '')
import collections
exist = collections.Counter(f"{x['sigla_prefix']} {x['object_type']}" for x in all_items)
out = dict(existence_counts=dict(exist), verified=sorted(f for f, v in ver.items() if isinstance(v, dict) and v['verified']), n_items=len(M['items']),
           n_usable_text=len(items), rows=rows)
print(json.dumps(out, indent=1, ensure_ascii=False))
