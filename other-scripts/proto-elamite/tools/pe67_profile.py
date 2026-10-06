"""pe67: Proto-Elamite sign profile per site (outpost-specific signs, number systems, line roles)."""
from pe67_lib import *

def main():
    T = load()
    rows = defaultdict(lambda: dict(site=Counter(), sys=Counter(), role=Counter()))
    ntab = Counter()
    for t in T:
        site = PE_SITE.get(t['provenience'])
        if not site:
            continue
        ntab[site] += 1
        seen = set()
        for i, l in enumerate(t['lines']):
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            if not sg:
                continue
            role = 'HEAD' if i == 0 and not l['numerals'] else ('SOLO' if l['numerals'] and len(sg) == 1 else ('STRING' if len(sg) >= 3 else 'OTHER'))
            sy = system_of(l['numerals']) if l['numerals'] else 'none'
            for s in set(sg):
                rows[s]['sys'][sy] += 1; rows[s]['role'][role] += 1
                if s not in seen:
                    rows[s]['site'][site] += 1; seen.add(s)
    sites = ['Susa', 'Malyan', 'Yahya', 'Sialk', 'Sofalin', 'Ozbaki', 'Shahr-i Sokhta']
    out = ['sign\t' + '\t'.join(sites) + '\toff_susa_share\tsystems\troles']
    keyf = lambda s: -sum(v for k, v in rows[s]['site'].items() if k != 'Susa')
    for s in sorted(rows, key=lambda s: (keyf(s), s)):
        r = rows[s]
        off = sum(v for k, v in r['site'].items() if k != 'Susa')
        if off == 0:
            continue
        tot = sum(r['site'].values())
        out.append('\t'.join([s] + [str(r['site'].get(x, 0)) for x in sites] + ['%.2f' % (off / tot),
                   ','.join('%s:%d' % kv for kv in r['sys'].most_common()), ','.join('%s:%d' % kv for kv in r['role'].most_common())]))
    out.insert(0, '# tablets per site: ' + ', '.join('%s %d' % (x, ntab[x]) for x in sites) + '; only signs seen off Susa listed')
    open(os.path.join(DATA, 'pe67_sign_profile.tsv'), 'w').write('\n'.join(out) + '\n')
    print(len(out), 'rows'); print('\n'.join(out[:25]))

main()
