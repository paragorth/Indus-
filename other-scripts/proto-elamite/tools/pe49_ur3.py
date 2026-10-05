"""pe49 control corpus: Ur III tablets whose seal legend names a dub-sar.

Reads the CDLI ATF dump and catalogue in the scratchpad. A tablet qualifies when
it is Ur III, its @seal block has a line beginning 'dub-sar', and the line above
gives the seal owner (the 'scribe'). Output: data/pe49_ckpt/ur3_scribes.json with
per-tablet line records (seal block and seal-impression comments removed).
"""
import csv, json, os, re, sys, collections
csv.field_size_limit(10**9)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'pe49_ckpt')


def clean(s):
    s = re.sub(r'[#?!*\[\]<>]', '', s)
    return s.strip()


def main():
    meta = {}
    with open(f'{SCR}/cdli_cat.csv', newline='') as f:
        for r in csv.DictReader(f):
            if r['period'].startswith('Ur III'):
                meta['P%06d' % int(r['id_text'])] = (r['provenience'].split(' (')[0], r['genre'], r['designation'])
    print('ur3 catalogue', len(meta), flush=True)
    recs = []
    cur = None

    def flush(c):
        if not c or c['pid'] not in meta:
            return
        owner = None
        for blk in c['seals']:
            for i, l in enumerate(blk):
                if l.startswith('dub-sar') and i > 0:
                    owner = blk[i - 1]
                    break
            if owner:
                break
        if not owner or not c['lines']:
            return
        p, g, d = meta[c['pid']]
        recs.append({'pid': c['pid'], 'prov': p, 'genre': g, 'desig': d, 'scribe': owner, 'lines': c['lines']})

    for raw in open(f'{SCR}/cdli.atf', encoding='utf-8', errors='replace'):
        line = raw.rstrip('\n')
        if line.startswith('&P'):
            flush(cur)
            cur = {'pid': line[1:8], 'lines': [], 'seals': [], 'surf': None, 'inseal': False}
            continue
        if cur is None:
            continue
        if line.startswith('@'):
            tag = line[1:].strip().split()[0] if line[1:].strip() else ''
            if tag == 'seal':
                cur['inseal'] = True
                cur['seals'].append([])
            elif tag in ('obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge', 'column', 'envelope', 'tablet', 'surface'):
                if tag != 'column':
                    cur['inseal'] = False
                    cur['surf'] = tag
                else:
                    cur['lines'].append({'surf': cur['surf'], 'kind': 'column'})
            continue
        if cur['inseal']:
            m = re.match(r'^\d+\'?\.\s*(.*)$', line)
            if m:
                cur['seals'][-1].append(clean(m.group(1)))
            continue
        if line.startswith('$'):
            cur['lines'].append({'surf': cur['surf'], 'kind': 'dollar', 'txt': line[1:].strip()})
            continue
        if line.startswith('#'):
            if 'seal' in line:
                continue
            cur['lines'].append({'surf': cur['surf'], 'kind': 'comment', 'txt': line[1:].strip()})
            continue
        m = re.match(r'^(\d+\'?)\.\s*(.*)$', line)
        if m:
            cur['lines'].append({'surf': cur['surf'], 'kind': 'text', 'txt': m.group(2).strip()})
    flush(cur)
    cnt = collections.Counter((r['prov'], r['scribe']) for r in recs)
    print('tablets with dub-sar seal', len(recs), 'scribes', len(cnt), flush=True)
    for k, v in cnt.most_common(40):
        print(k, v)
    json.dump(recs, open(f'{OUT}/ur3_scribes.json', 'w'))


if __name__ == '__main__':
    main()
