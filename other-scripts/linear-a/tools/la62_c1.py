#!/usr/bin/env python3
"""LA-62 cycle 1: inventory of Linear A inscriptions outside our corpus snapshot.

The snapshot is data/LinearAInscriptions.js (lineara.xyz) -> data/corpus.json. Sources checked on
6 Oct 2026 (open only): lineara.xyz live file and GitHub raw (byte-identical to the snapshot), SigLA
browse list (802 documents), Crossref (Linear A titles 2015-2026), Ariadne Suppl. 5 (Kanta, Nakassis,
Palaima & Perna, open PDF), Proceedings of the Danish Institute at Athens V (Khania, open PDF), Kadmos 2018
abstract (Khania), CNR/ISPC notices of RILA-S1 (Del Freo & Zurbach 2024), Del Freo's 'Panorama' note
(open PDF), IULM IRIS (Notti 2023, restricted), news pages on the Knossos ivory.
Positive control: the matcher must find known post-1985 items (KH 97-99, KH 104-105, KH Wc 2124,
INA Zb 1, KY Za 2, THE Zg 15, LACH Za 1, ZO 1, TEL Zb 1, MIL Zb 1-4) in the snapshot.
Negative control: SigLA documents missing after name normalisation are inspected one by one.
Output: data/la62_ckpt/inventory.json and a summary on stdout.
"""
import json, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data'); CK = os.path.join(D, 'la62_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
from la60_common import _pub_source

C = json.load(open(os.path.join(D, 'corpus.json')))
IDS = {d['id'] for d in C}
NORM = {re.sub(r'[\s.]', '', i).upper(): i for i in IDS}


def present(name):
    k = re.sub(r'[\s.]', '', re.sub(r'&lt;|&gt;|[<>`()]|-latus', '', name)).upper()
    k = re.sub(r'[\u03b1\u03b2\u03b3]$', '', name.strip()) and re.sub(r'(Α|Β|Γ)$', '', k)
    ks = {k, re.sub(r'[AB]$', '', k)}
    hits = [v for n, v in NORM.items() if any(n == x or re.sub(r'[AB]$', '', n) == x or n.startswith(x)
                                              or ('+' in n and x in n) or n.replace('Z', '') == x
                                              or re.sub(r'Z[A-Z]', '', n) == x for x in ks)]
    return hits


POS = ['KH97', 'KH98', 'KH99', 'KH104', 'KH105', 'KHWc2124', 'INZb1', 'KYZa2', 'THEZg15', 'LACHZa1', 'ZO1',
       'TELZb1', 'MILZb1', 'MILZb2', 'MILZb3', 'MILZb4', 'KNZg57', 'KNZg58']


def content(i):
    d = [x for x in C if x['id'] == i]
    if not d:
        return None
    d = d[0]
    return dict(words=d['words'], logos=d['logograms'], numbers=d['numbers'],
                n_tokens=sum(1 for t in d['tokens'] if t['t'] in ('word', 'logo', 'num')))


INVENTORY = [
    dict(id='KN Zg 58', site='Knossos, Anetaki plot (cult centre), Room 1 Ivory Repository', date='found 2011-2017; published 2024 (proceedings vol.), news 2025',
         type='ivory handle, 4 faces; ACCOUNTING text (numerals, six different fraction signs)',
         source='Kanta, Nakassis, Palaima & Perna, Ariadne Suppl. 5 (2024) 27-43, doi 10.26248/ariadne.vi.1841 (open PDF)',
         open_media='one overall photo (Fig. 7, Sapirstein), signs NOT legible at published resolution; prose description; no drawing, no transcription',
         in_snapshot='record exists, EMPTY (0 tokens)', usable='prose skeleton only'),
    dict(id='KN Zg 57 faces B, C, D (+ rest of A)', site='Knossos, Anetaki plot', date='as above',
         type='ivory ring, ~119 signs, no numerals; Face A 12 animals + ~10 vases (6 amphorae, rhyton A664, tripod; VAS+PA, VAS+RU); B six sign-groups then GRA, FAR?, OLIV; C >= 9 groups + 5 TELA (one TELA+KA); D ~9 HIDE (one HIDE+KA/QE/KO)',
         source='same article', open_media='same photo; prose', in_snapshot='KNZg57a partial (7 tokens), KNZg57b empty',
         usable='prose skeleton (logogram classes and order only; no sign-group readings published)'),
    dict(id='RILA-S1 (107 documents, 427 signs)', site='23 sites (8 new vs GORILA)', date='2024',
         type='the full post-1985 corpus to 2023 (tablets +7 %, roundels +10 %, stone vessels +50 %, clay vessels +70 %, metal +37 %)',
         source='Del Freo & Zurbach, Etudes Cretoises 21.6, EFA 2024, ISBN 978-2-86958-642-0 (not open)',
         open_media='none', in_snapshot='66 documents tagged post-1985 by image source + others with blank source; overlap unknown',
         usable='needs a human (library)'),
    dict(id='THE Zg (Akrotiri loomweight, 3 signs)', site='Akrotiri, Thera', date='2023',
         type='loomweight', source='Notti 2023, IULM IRIS 10808/48144 (PDF restricted to IULM network)',
         open_media='none', in_snapshot='probably THEZg15 (*164-RI-DA, loom weight)', usable='already in'),
    dict(id='KH 104, KH 105, KH Wc 2124 (+ 2 small fragments)', site='Khania, Kastelli', date='2007-2014 finds; Kadmos 57 (2018) 33-44',
         type='tablet fragments, roundel', source='Hallager & Andreadaki-Vlazaki 2018 (abstract open)',
         open_media='abstract', in_snapshot='yes (KH104 DA-RE GRA GRA J; KH105 DE-KI-TI; KHWc2124)', usable='already in (positive control)'),
    dict(id='KH 97, 98, 99, KH Wc 2123, KH Zb 1', site='Khania', date='PDIA V (2007)', type='tablets, roundel, vessel',
         source='Andreadaki-Vlazaki & Hallager 2007 (open PDF)', open_media='drawings 1:1', in_snapshot='yes', usable='already in'),
    dict(id='INA Zb 1', site='Inatos cave (Tsoutsouros)', date='2022', type='clay vessel', source='Perna in INSTAP 2022',
         open_media='book (not open)', in_snapshot='yes (INZb1 NU-RO)', usable='already in'),
    dict(id='KY Za 2 (DA-MA-TE)', site='Kythera, Ag. Georgios sto Vouno', date='post-1985', type='stone vessel',
         source="Del Freo 'Panorama' (open)", open_media='text mention', in_snapshot='yes', usable='already in'),
]


def main():
    src = _pub_source()
    pos = {p: [h for h in present(p)] for p in POS}
    post = sorted(i for i in IDS if src.get(i) == 'post')
    sigla = []
    fn = os.environ.get('SIGLA_LIST')
    if fn and os.path.exists(fn):
        for l in open(fn):
            s = l.strip()
            if s and not present(s):
                sigla.append(s)
    out = dict(date='2026-10-06', snapshot_entries=len(C), snapshot_post1985_tagged=len(post),
               snapshot_sites=Counter(d['site'] for d in C).most_common(),
               positive_control={p: [(h, content(h)['n_tokens']) for h in v] for p, v in pos.items()},
               sigla_unmatched=sigla, inventory=INVENTORY,
               knossos_ivory_snapshot={i: content(i) for i in ['KNZg57a', 'KNZg57b', 'KNZg58']})
    json.dump(out, open(os.path.join(CK, 'inventory.json'), 'w'), indent=1, ensure_ascii=False)
    print('snapshot entries', len(C), 'post-1985 tagged', len(post))
    print('positive control found', sum(1 for v in pos.values() if v), 'of', len(pos),
          {p: [(h, content(h)['n_tokens']) for h in v] for p, v in pos.items()})
    print('SigLA unmatched after normalisation', len(sigla), sigla)


if __name__ == '__main__':
    main()
