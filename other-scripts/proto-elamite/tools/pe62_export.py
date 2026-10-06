"""pe62: export the imputed class of every unmarked PE entry (cycle-2 ensembles) as data/pe62_imputed_classes.tsv."""
import json, os
import pe62_common as C
d = json.load(open(os.path.join(C.CK, 'c2_PE.json')))
T = {t['id']: t for t in C.build_pe()}
fn = os.path.join(C.DATA, 'pe62_imputed_classes.tsv')
with open(fn, 'w') as f:
    f.write('# pe62 imputed class of unmarked Proto-Elamite entries. imp/conf: ensemble with numerals; imp_blind/conf_blind:\n'
            '# number-blind ensemble. tablet_has_written_class: 1 if another entry on the tablet shows a class sign.\n'
            '# Grade C: 92% of blind imputations equal the commonest written class on the same tablet (see loops/pe62_cycle2.txt).\n')
    f.write('tablet\tentry_index\tkind\tsigns\tnumerals\timp\tconf\timp_blind\tconf_blind\ttablet_has_written_class\n')
    for r in d['imputation']:
        vis = int(any(e['label'] for e in T[r['tab']]['entries']))
        f.write('%s\t%d\t%s\t%s\t%s\t%s\t%.3f\t%s\t%.3f\t%d\n' % (r['tab'], r['line'], r['kind'], ' '.join(r['signs']) or '-',
                ' '.join('%s(%s)' % (n, c) for n, c in r['nums']) or '-', r['imp'], r['conf'], r['imp_blind'], r['conf_blind'], vis))
print(fn)
