"""Loop 55 cycle 4: robustness of cycle 1 to the sampling choices. (i) Indus-matched duplication (each distinct source text
replicated with a copy count drawn from the Indus die-regime copy-count distribution) for the language and code corpora;
(ii) site-structured Ur III / Linear B / proto-cuneiform samples (Indus site shares mapped onto the source's largest sites).
Same battery and chains as cycle 1 (tools/dark_loop55_c1.py). Usage: python3 tools/dark_loop55_c4.py run [nres] [nnull] | summary"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark_loop55_c1 as C1

DUP = ['ur3_words', 'ur3_names_syll', 'linb_syll', 'latin_edh', 'icd10', 'hts', 'aircraft_reg', 'unicode_names', 'proto_cuneiform']
SITES = ['ur3_words', 'ur3_syll', 'linb_syll', 'proto_cuneiform', 'latin_edh']

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'
    nres = int(sys.argv[2]) if len(sys.argv) > 2 else 3; nnull = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    if cmd == 'run':
        C1.run(nres, nnull, 'indus', DUP)
        C1.run(nres, nnull, 'sites', SITES)
    else:
        C1.summary('indus', os.path.join(C1.C.DARK, 'loop55_c4_dup.txt'))
        C1.summary('sites', os.path.join(C1.C.DARK, 'loop55_c4_sites.txt'))
