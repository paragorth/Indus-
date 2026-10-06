"""pe58 control corpus: Ur III capacity lines (rations, grain, flour, beer) from CDLI, all proveniences.
Line = <capacity numeral> <words...> (<= 6 words, no breaks). Each whole word is one token; tokens are
made opaque (U00001 ...) only in the fitting engine; the word key is kept for SCORING the control.
Also collects distributive lines '<count> <grade> <capacity>-ta' (N persons at R each) for the ration ladder truth.
Output: data/pe58_ckpt/ur3cap.json, ur3ta.json"""
import csv, json, os, re, collections
csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
CK = os.path.join(HERE, '..', 'data', 'pe58_ckpt')
CAP = {'disz': None, 'ban2': 10, 'barig': 60, 'asz': 300}
CNT = {'disz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600}


def clean(w):
    return re.sub(r'[#!?*<>\[\]]', '', w)


def parse_cap(toks):
    """leading capacity numeral (with optional 'sila3' after disz-count) -> (sila, rest)"""
    i = 0; v = 0.0; cnt = 0.0; seen = False; unit = False
    while i < len(toks):
        m = re.match(r"^(\d+)(?:/(\d+))?\(([a-z0-9']+)\)$", toks[i])
        if not m:
            break
        n = float(m.group(1)) / (float(m.group(2)) if m.group(2) else 1.0)
        u = m.group(3)
        if u in ('ban2', 'barig', 'asz'):
            v += n * CAP[u]; unit = True
        elif u in CNT:
            cnt += n * CNT[u]
        else:
            return None
        seen = True; i += 1
    if not seen:
        return None
    rest = toks[i:]
    if rest and rest[0] == 'sila3':
        v += cnt; cnt = 0; unit = True; rest = rest[1:]
    elif rest and rest[0] == 'gur':
        v = v + cnt * 300; cnt = 0; unit = True; rest = rest[1:]
    if not unit or cnt:
        return None
    return v, rest


def main():
    per = {}
    for r in csv.DictReader(open(f'{SCR}/cdli_cat.csv', newline='')):
        if r['period'].startswith('Ur III') and r['genre'] == 'Administrative':
            try:
                per['P%06d' % int(r['id_text'])] = r['provenience'].split()[0] if r['provenience'] else '-'
            except ValueError:
                pass
    out = []; ta = []; cur = None; inseal = False
    for raw in open(f'{SCR}/cdli.atf', encoding='utf-8', errors='replace'):
        line = raw.rstrip('\n')
        if line.startswith('&P'):
            cur = line[1:8] if line[1:8] in per else None; inseal = False
            continue
        if cur is None:
            continue
        if line.startswith('@'):
            tag = line[1:].strip().split()[0] if line[1:].strip() else ''
            if tag == 'seal':
                inseal = True
            elif tag in ('obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge', 'tablet'):
                inseal = False
            continue
        if inseal:
            continue
        m = re.match(r"^\d+'?\.\s*(.*)$", line)
        if not m:
            continue
        txt = m.group(1)
        if re.search(r'\.\.\.|\bx\b|\$|szu-nigin|\[|la2', txt):
            continue
        toks = [clean(w) for w in txt.split()]
        # distributive: <count> <grade words> <capacity>-ta
        mt = re.match(r"^((?:\d+\((?:disz|u|gesz2)\)\s*)+)(\S+(?:\s\S+)?)\s+((?:\d+(?:/\d+)?\((?:ban2|barig|disz)\)[^ ]*\s*)+)(sila3-ta)?", txt)
        if '-ta' in txt and re.match(r"^\d+\((disz|u|gesz2)\)", txt):
            ws = txt.split()
            k = [j for j, w in enumerate(ws) if w.endswith('-ta')]
            if k:
                j = k[0]
                cnum = []; i = 0
                while i < len(ws) and re.match(r"^\d+\((disz|u|gesz2)\)$", ws[i]):
                    cnum.append(ws[i]); i += 1
                body = ws[i:j + 1]
                # rate: capacity tokens inside body ending with -ta
                rate_toks = []; gw = []
                for w in body:
                    w2 = clean(w[:-3] if w.endswith('-ta') else w)
                    if re.match(r"^\d+(?:/\d+)?\((ban2|barig|disz)\)$", w2) or w2 in ('sila3', 'gur'):
                        rate_toks.append(w2)
                    elif not rate_toks:
                        gw.append(clean(w))
                pr = parse_cap(rate_toks) if rate_toks else None
                if pr and gw and pr[0] > 0:
                    n = sum(int(re.match(r'(\d+)', c).group(1)) * CNT[re.search(r'\((\w+)\)', c).group(1)] for c in cnum)
                    ta.append(dict(tab=cur, prov=per[cur], n=n, rate=pr[0], grade=gw, after=ws[j + 1:]))
        p = parse_cap(toks)
        if not p:
            continue
        v, rest = p
        if v <= 0 or not rest or len(rest) > 6 or any(re.match(r'^\d', w) for w in rest):
            continue
        out.append(dict(tab=cur, prov=per[cur], q=v, words=rest))
    print('cap lines', len(out), 'tablets', len({o['tab'] for o in out}))
    print(collections.Counter(w for o in out for w in o['words']).most_common(50))
    print('ta lines', len(ta))
    c = collections.defaultdict(list)
    for t in ta:
        c[' '.join(t['grade'][:2])].append(t['rate'])
    for k, v in sorted(c.items(), key=lambda x: -len(x[1]))[:30]:
        print(k, len(v), collections.Counter(v).most_common(5))
    json.dump(out, open(os.path.join(CK, 'ur3cap.json'), 'w'))
    json.dump(ta, open(os.path.join(CK, 'ur3ta.json'), 'w'))


if __name__ == '__main__':
    main()
