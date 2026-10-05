"""Print top line-end-consistent rules of a cycle-1 checkpoint, decoding opaque symbols."""
import sys, json
import v61_lib as L
name, d = sys.argv[1], sys.argv[2]
key = {}
if name in ('Sanskrit', 'Welsh', 'Italian'):
    ld = {'Sanskrit': L.load_sanskrit, 'Welsh': L.load_welsh, 'Italian': L.load_italian}[name]
    key = {v: k for k, v in L.opaque(ld())[1].items()}
dec = lambda s: ''.join(key.get(c, c) for c in s)
s = L.jload('c1_%s_%s.json' % (name.replace(':', '_'), d))['summary']
for x in s['le_top'][:int(sys.argv[3]) if len(sys.argv) > 3 else 40]:
    tag = 'CONS' if x['llN'] > max(x['llC'], 0) + 2 else ('FLIP' if x['llC'] > max(x['llN'], 0) + 2 else '-')
    S, B, C = dec(x['S']), dec(x['B']), dec(x['cls'])
    if d == 'P':
        S, B = S[::-1], B[::-1]
    print('%-5s %-6s->%-6s ctx[%s] ztr %.1f zte %.1f fin n=%d pS C/N/all/fin %.2f/%.2f/%.2f/%.2f xz %.1f' % (
        tag, S or '0', B or '0', C[:20], x['z_train'], x['z_test'], x['n'], x['pS_C'], x['pS_N'], x['pS_all'], x['pS_fin'], x['xz']))
