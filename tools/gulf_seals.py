"""Laursen 2010 Gulf Type seals with Indus inscriptions (Fig. 11 / Fig. 8-9): match to corpus,
write data/gulf_seals.csv, then test them as a 'Gulf/western' register against the frame grammar.
Readings are NOT invented: each seal's sign sequence is taken from the matched corpus entry and
checked by eye against Laursen's drawing (tools: f11 renders in scratchpad). Seals with no corpus
match are listed as missing, with the figure they are drawn in."""
import json,csv,collections,random,math
m=json.load(open('data/derived/merged-corpus-canonical.json'))
def find(site,seq):
    for r in m:
        if r['site']==site and r.get('seq_raw')==seq: return r
# seal_no: (group/find region, publication concordance, corpus site, corpus seq (reading order), match quality, note)
S=[
 (1,'Indus Valley (Chanhu-daro)','Mackay 1943 pl. LI/23; CISI C-32','Chanhu-daro',[632,390,400,375],'probable','drawing has one extra final stroke'),
 (3,'Indus Valley (Mohenjo-daro)','Marshall 1931 pl. CXIV/478; CISI M-1369','Mohenjo-daro',[820,60,2,832,140,592,55],'confirmed',''),
 (4,'Indus Valley (Mohenjo-daro)','Marshall 1931 pl. CXII/383; CISI M-417','Mohenjo-daro',[140],'probable','fragment; drawing shows an extra stroke before the figure'),
 (5,'Indus Valley (Mohenjo-daro)','Mackay 1937-38 pl. XCVI/500; CISI M-415','Mohenjo-daro',[832,256],'probable','drawing has an extra oblique stroke'),
 (2,'Indus Valley (Mohenjo-daro), Group 3 shape','Marshall 1931 pl. CX/309; CISI M-416','Mohenjo-daro',[16,405,1,255,324],'confirmed',''),
 (8,'Bahrain (Karzakkan cemetery, BBM 18839)','Srivastava 1991 fig. 55 left; Al-Sindi 1999 no. 180','Karzakan',[91,31,455,220],'confirmed',''),
 (9,'Bahrain (Saar cemetery)','Al-Sindi 1999 no. 182','Saar',[55,220,91,1,93,31],'probable','stroke block drawn as 2x4 (8) in Laursen, coded W55 (12) in corpus'),
 (11,'Bahrain (Karzakkan cemetery, BBM 14569)','this article (Laursen 2010 fig. 1b)','Karzakan',[91,32,1,33],'confirmed',''),
 (56,'Bahrain (Karzakkan cemetery, BBM 20362)','Al-Sindi 1999 no. 160','Karzakan',[740,740],'confirmed','middle is a motif, not script'),
 (10,'Bahrain (Janabiyah cemetery)','this article (Laursen 2010 fig. 1a)','Janabiyah',[71,31,831,55,121,99,55],'probable','last block drawn as ~3x3 (9), coded W55 (12); twins coded as two man signs W121+W99'),
 (7,'Bahrain (Qala\'at al-Bahrain)','Kjaerum 1994 fig. 1725; Al-Sindi 1999 no. 279',"Qala'at al-Bahrain",[190,60,55,90,160],'uncertain','drawing order reversed vs corpus; drawing ends in twins, corpus in 90+160; block drawn 2x4'),
 (12,'Failaka','Kjaerum 1983 no. 319','Failaka',[716,350,1,90],'probable','drawing order reversed vs corpus'),
 (13,'Failaka (technically Dilmun Type)','Kjaerum 1983 no. 279','Failaka',[90,817,317],'uncertain','drawing has extra signs and a scene'),
 (14,'Iran (Susa)','Amiet 1972 no. 1643','Susa',[365,90,407,604,700],'confirmed','one further damaged sign in drawing'),
 (15,'Iran','Amiet 1973 pl. 23a-b','Luristan',[91,840,413,831],'confirmed',''),
 (16,'Mesopotamia (Ur)','Gadd 1932 pl. I no. 2','Ur',[528,220,924,340,93],'confirmed',''),
 (18,'Mesopotamia (Ur)','Gadd 1932 pl. I no. 4','Ur',[90,405],'confirmed','fragment'),
 (19,'Mesopotamia (Ur)','Gadd 1932 pl. I no. 5','Ur',[90,1],'uncertain','fragment; only the man sign visible'),
 (21,'Mesopotamia (Ur)','Gadd 1932 pl. I no. 16; BM 123208; IM77 9846','Dilmun',[415,803,1,717,354],'confirmed','corpus site label "Dilmun"'),
 (22,'Mesopotamia (Girsu)','Sarzec & Heuzey 1884-1912 pl. 30.3; IM77 9852','Girsu',[255,13,744,740],'probable','drawing shows one more sign (5 vs 4)'),
 (23,'Near East (no find-spot)','Gadd 1932 pl. I no. 17; IM77 9901','Unknown',[467,550,1,740,740],'confirmed',''),
 (24,'Near East (no find-spot; PCA outlier)','Gadd 1932 pl. I no. 18','Unknown',[190,91,603],'probable','drawing has one more sign'),
 (26,'Mesopotamia?','Buchanan 1981 no. 1088 / Newell 23','Unknown',[160,384,133,1,90],'probable','drawing has one more initial sign'),
 (27,'Mesopotamia?','Buchanan 1981 no. 1089 / Newell 876','Unknown',[725,90,2],'probable','with animal motif'),
]
MISSING=[(6,'Bahrain (Qala\'at al-Bahrain)','Kjaerum 1994 fig. 1726; Al-Sindi 1999 no. 8','Fig. 11 no. 6 / Fig. 9: fragment, only twins legible'),
 (17,'Mesopotamia (Ur)','Gadd 1932 pl. I no. 3','Fig. 11 no. 17: no corpus match; 4-5 signs, first damaged'),
 (20,'Mesopotamia (Ur)','Gadd 1932 pl. I no. 15','Fig. 11 no. 20: no corpus match; 3 signs'),
 (25,'Mesopotamia?','Langdon 1932 p. 48 (J. Rosen collection)','Fig. 11 no. 25: no corpus match; about 6 signs'),
 (28,'Iran','Winkelmann 1999 Abb. 2','Linear Elamite signs, not Indus; excluded')]
rows=[]
for n,reg,conc,site,seq,conf,note in S:
    r=find(site,seq); src=f"merged corpus ({site}{', '+r['cisi'] if r and r['cisi'] not in ('-',None) else ''})" if r else 'NOT FOUND'
    rows.append(dict(seal_no=n,site=reg,concordance=conc,sequence_glyph_coding='-'.join(map(str,seq)),match_source=src,confidence=conf,note=note,
                     motif=(r or {}).get('symbol',''),found_abroad=not reg.startswith('Indus Valley')))
for n,reg,conc,fig in MISSING:
    rows.append(dict(seal_no=n,site=reg,concordance=conc,sequence_glyph_coding='',match_source='missing: '+fig,confidence='missing',note='',motif='',found_abroad=True))
rows.sort(key=lambda d:d['seal_no'])
with open('data/gulf_seals.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['seal_no','site','concordance','sequence_glyph_coding','match_source','confidence','note','motif','found_abroad']); w.writeheader(); w.writerows(rows)
print('written',len(rows),collections.Counter(r['confidence'] for r in rows))
print('not found in corpus:',[r['seal_no'] for r in rows if r['match_source']=='NOT FOUND'])
