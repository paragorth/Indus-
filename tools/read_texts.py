"""Render the 60 most repeated texts (>= 3 signs) and a sample of Gulf/abroad texts with the current working glosses
(WORKING-DICTIONARY, grades noted). Output: READINGS.md. Glosses are provisional labels, not translations."""
import json,collections
C=json.load(open('data/derived/merged-corpus-canonical.json'))
G={817:'SHIELD',861:'SHIELD(diamond)',820:'WHEEL-shield',920:'BRACKET-shield',692:'X-shield',2:'of',60:'of',
740:'JAR',741:'jar-holder?',742:'jar-holder?',400:'COMB',90:'MAN',91:'TWO MEN',520:'ARROW',
151:'MAN WITH LOADS',156:'MAN WITH JARS',527:'HATCHED BOX',176:'SEATED MAN',100:'THREE-HEADED MAN',760:'U-branch title',
923:'title-923',390:'TREE',405:'TREE',407:'BRANCH',700:'TABLET-unit',220:'FISH',240:'FISH-whiskers',
235:'FISH-hat',233:'FISH-bar',231:'FISH-stroke',226:'FISH-4',590:'box-on-stand',705:'U-stroke',706:'U-stroke',55:'TWELVE',56:'24',
1:'1',3:'3',4:'4',5:'5',16:'6',17:'7',18:'8',31:'1',32:'2',33:'3',34:'4',415:'pitchfork',803:'leaf-tree',806:'leaf-tree',
840:'-(name end)',416:'(name start)-',692:'OPENER(X)',125:'ARCHER',140:'WOMAN',142:'STAFF-MAN'}
def rd(s): return ' · '.join(G.get(a,f'[{a}]') for a in s)
ABROAD={'Failaka','Hajar','Kish','Nippur',"Qala'at al-Bahrain","Ra's al-Junayz",'Susa','Tell Umma','Tello','Tepe Yahya','Ur','Gonur Depe','Salut','Girsu','Karzakan','Saar','Janabiyah','Kalba','Dilmun'}
c=collections.Counter(tuple(r['seq_raw']) for r in C if r['seq_raw'] and len(r['seq_raw'])>=3)
out=['# Picture readings (each sign read as what it depicts)\n','Glosses from WORKING-DICTIONARY.md (grades A–C). Capitals mark functions or strong guesses; [n] is an unread sign (Wells number). These are labels for thinking, **not translations**.\n','## The 60 most repeated texts\n','| Copies | Signs | Working reading |','|---|---|---|']
for s,n in c.most_common(60): out.append(f'| {n} | {"-".join(map(str,s))} | {rd(s)} |')
out+=['\n## Texts found abroad\n','| Site | Signs | Working reading |','|---|---|---|']
seen=set()
for r in C:
    s=r['seq_raw']
    if r['site'] in ABROAD and s and len(s)>=2 and (r['site'],tuple(s)) not in seen:
        seen.add((r['site'],tuple(s))); out.append(f'| {r["site"]} | {"-".join(map(str,s))} | {rd(s)} |')
open('READINGS.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out[:30]))
