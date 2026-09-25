#!/bin/bash
# Sources not redistributed here for licensing reasons. Run once from the code/ folder.
set -e
mkdir -p ../data/external && cd ../data/external
curl -sL -o lipi.zip https://codeload.github.com/yajnadevam/lipi/zip/refs/heads/main && unzip -o -q lipi.zip 'lipi-main/src/assets/data/*' && rm -rf yajnadevam-data && mv lipi-main/src/assets/data yajnadevam-data && rm -rf lipi-main lipi.zip
curl -sL -o mw.txt https://raw.githubusercontent.com/sanskrit-lexicon/csl-orig/master/v02/mw/mw.txt
echo "Place DEDR_full.csv (Burrow & Emeneau via dsal.uchicago.edu) in data/external/ yourself; it is not redistributed."
