"""pe50 control corpus: Ur III Drehem (Puzrish-Dagan) and Umma tablets as capture events.

For each Ur III administrative tablet from the two archives, extract person names
by fixed syntactic frames (no lexicon, no reading of signs as meaning anything
beyond the frame):
  ki X-ta | giri3 X | X i3-dab5 | mu-kux(DU) X | kiszib3 X | X maszkim |
  X szu ba-ti | ugula X | X (line before 'szu ba-ti')
X must be a single ATF word, not a numeral, not a stop word, not broken.
Also kept per tablet: number of numeric lines, whether a total line (szu-nigin)
is present, the year line, and the numeric entry lines (for cycle-2 matching).
Seal blocks are skipped.
Output: data/pe50_ckpt/ur3_caps.json  {archive: [{pid, names, nnum, total, year, entries}]}
"""
import csv, json, os, re, collections
csv.field_size_limit(10**9)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'pe50_ckpt', 'ur3_caps.json')
STOP = {'lugal', 'ensi2', 'dub-sar', 'sukkal', 'nu-banda3', 'szabra', 'sanga', 'ugula', 'dumu', 'lu2', 'ba-zi',
        'ba-usz2', 'mu-kux(DU)', 'i3-dab5', 'szu', 'ba-ti', 'maszkim', 'giri3', 'kiszib3', 'e2', 'iti', 'mu', 'u4',
        'sza3', 'zi-ga', 'gurusz', 'geme2', 'erin2', 'udu', 'gu4', 'sila3', 'gur', 'x', 'kiszib', 'ARAD2-zu', 'ensi2-ka', 'dumu-lugal', 'sukkal-mah', 'lugal-ka'}


def ok(w):
    if not w or re.search(r'[\[\]]|\.\.\.|^x$|\bx\b', w):
        return None
    w = re.sub(r'[#!?*<>]', '', w)
    if re.match(r'^\d', w) or w in STOP or '(' in w and re.match(r'^\d', w):
        return None
    if re.search(r'(^|-)x(-|$)', w):
        return None
    return w


def names_in(lines):
    out = []
    for i, t in enumerate(lines):
        w = t.split()
        if not w:
            continue
        if w[0] == 'ki' and len(w) == 2 and w[1].endswith('-ta'):
            out.append(ok(w[1][:-3]))
        elif w[0] in ('giri3', 'kiszib3', 'ugula', 'mu-kux(DU)') and len(w) >= 2:
            out.append(ok(w[1]))
        elif len(w) == 2 and w[1] in ('i3-dab5', 'maszkim'):
            out.append(ok(w[0]))
        elif len(w) == 3 and w[1:] == ['szu', 'ba-ti']:
            out.append(ok(w[0]))
        elif w == ['szu', 'ba-ti'] and i > 0 and len(lines[i - 1].split()) == 1:
            out.append(ok(lines[i - 1].split()[0]))
    return sorted({x for x in out if x})


def main():
    arch = {}
    for r in csv.DictReader(open(f'{SCR}/cdli_cat.csv', newline='')):
        if r['period'].startswith('Ur III') and r['genre'] == 'Administrative':
            p = r['provenience']
            a = 'DREHEM' if p.startswith('Puzri') else 'UMMA' if p.startswith('Umma') else None
            if a:
                arch['P%06d' % int(r['id_text'])] = a
    res = collections.defaultdict(list)
    cur = None

    def flush(c):
        if not c or c['pid'] not in arch or not c['lines']:
            return
        L = c['lines']
        nums = [x for x in L if re.match(r'^\d+(/\d+)?\(', x)]
        yr = next((x for x in L if x.startswith('mu ')), '')
        res[arch[c['pid']]].append({'pid': c['pid'], 'names': names_in(L), 'nnum': len(nums), 'nlines': len(L),
                                    'total': int(any(x.startswith('szu-nigin') for x in L)), 'year': yr,
                                    'entries': [re.sub(r'[#!?*\[\]<>]', '', x) for x in nums][:80]})

    for raw in open(f'{SCR}/cdli.atf', encoding='utf-8', errors='replace'):
        line = raw.rstrip('\n')
        if line.startswith('&P'):
            flush(cur)
            cur = {'pid': line[1:8], 'lines': [], 'inseal': False}
            continue
        if cur is None or cur['pid'] not in arch:
            continue
        if line.startswith('@'):
            tag = line[1:].strip().split()[0] if line[1:].strip() else ''
            if tag == 'seal':
                cur['inseal'] = True
            elif tag in ('obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge', 'envelope', 'tablet'):
                cur['inseal'] = False
            continue
        if cur['inseal']:
            continue
        m = re.match(r"^\d+'?\.\s*(.*)$", line)
        if m:
            cur['lines'].append(m.group(1).strip())
    flush(cur)
    for a, v in res.items():
        nn = collections.Counter(n for t in v for n in t['names'])
        print(a, 'tablets', len(v), 'with names', sum(1 for t in v if t['names']), 'distinct names', len(nn),
              'top', nn.most_common(8))
    json.dump(res, open(OUT, 'w'))


if __name__ == '__main__':
    main()
