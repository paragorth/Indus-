#!/usr/bin/env bash
# Fetch the public raw data this project uses into data/raw/ (not committed).
#   - Yajnadevam corpus + published key  (github.com/yajnadevam/lipi)
#   - TamilVU dictionary headwords       (github.com/Ezhil-Language-Foundation/open-tamil)
#   - Monier-Williams Sanskrit (SLP1)    (github.com/sanskrit-lexicon/csl-orig)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw && cd data/raw
if [ ! -d lipi ]; then git clone -q --depth 1 https://github.com/yajnadevam/lipi.git; fi
cp lipi/src/assets/data/inscriptions.csv lipi/src/assets/data/xlits.csv .
if [ ! -f tamilvu_dictionary_words.txt ]; then
  curl -sSfL -o tamilvu_dictionary_words.txt \
    https://raw.githubusercontent.com/Ezhil-Language-Foundation/open-tamil/main/solthiruthi/data/tamilvu_dictionary_words.txt
fi
if [ ! -f mw.txt ]; then
  curl -sSfL -o mw.txt https://raw.githubusercontent.com/sanskrit-lexicon/csl-orig/master/v02/mw/mw.txt
fi
( cd lipi && echo "lipi commit: $(git rev-parse HEAD)" ) > SOURCES.txt
sha256sum inscriptions.csv xlits.csv tamilvu_dictionary_words.txt mw.txt >> SOURCES.txt
cat SOURCES.txt
