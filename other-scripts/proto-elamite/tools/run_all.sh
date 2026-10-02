#!/bin/sh
# Rebuild the corpus and rerun every test. Args: path to cdliatf_unblocked.atf and cdli_cat.csv
# (Git LFS files from github.com/cdli-gh/data; raw.githubusercontent.com only serves the LFS pointer).
set -e
cd "$(dirname "$0")"
python3 build_corpus.py "$1" "$2"
python3 test_a_slots.py
python3 test_b_goods.py
python3 test_c_totals.py --strict
python3 test_c_totals.py
python3 test_c2_rations.py
python3 test_d_headers.py
python3 test_e_names.py
python3 test_f_le_matches.py
python3 test_f_le_matches.py --drop-frame
