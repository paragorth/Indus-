"""pe12: compact table of real vs null family statistics from a cycle json."""
import json, sys
for path in sys.argv[1:]:
    R = json.load(open(path))
    for tag, slots in R.items():
        for slot, s in slots.items():
            print('%s %s n=%d tabs=%d H=%d' % (tag, slot, s['n'], s['ntab'], s['H']))
            for f in ['ARITH', 'SIZE', 'CTX', 'SEQ', 'PAIR']:
                k = f + ':top1'
                if k not in s['real']:
                    continue
                parts = []
                for m, d in s['null'].items():
                    parts.append('%s %+.3f (z %+.1f, p %.3f)' % (m, d[k]['mean'], d[k]['z'], d[k]['p']))
                print('   %-5s real %+.3f | %s | %s' % (f, s['real'][k], '; '.join(parts),
                                                   ', '.join('%s B%+.3f C%+.3f' % tuple(b) for b in s['best'][f][0][:2])))
