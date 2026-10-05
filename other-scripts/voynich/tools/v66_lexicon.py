"""v66: concept classes for medieval medical/herbal vocabulary (la, it, de/MHG, en, fr/OF).

Each class is a list of full-word regexes, applied to lowercased, accent-stripped tokens.
Built by hand from the texts' own frequent words; it is a meaning grouping, not a reading.
"""
import re

CLASSES = {
    # body
    'HEAD':    [r'caput|capit(is|i|e)|capill\w*|cerebr\w*', r'haupt\w*|houbt\w*|hirn\w*', r'head|heads|brain|brains|hair', r'chief|cervel\w*|teste', r'capo|testa|cervello'],
    'EYE':     [r'ocul\w*|lumina|caligin\w*', r'aug|augen|gesiht', r'eye|eyes|sight|eyesight', r'ieu?z|oilz|oeil\w*|veue', r'occhi\w*|vista'],
    'EAR':     [r'auri(s|um|bus)|aure(s|m)?|auribus', r'orn|oren', r'ear|ears|deafness|hearing', r'oreill\w*', r'orecch\w*'],
    'MOUTH':   [r'dent(es|is|ium|ibus|em)|gingiv\w*|lingua\w*|oris|ore', r'zand?|zend\w*|zungen?|mund|munt', r'teeth|tooth|gums|tongue|mouth', r'denz|dent|boche|langue', r'denti|bocca|lingua'],
    'CHEST':   [r'pect\w*|pulmo\w*|guttur\w*|faucium|fauces', r'prust|lungel?|kel', r'breast|chest|lungs?|throat|lights', r'piz|poumon\w*|gorge', r'petto|polmon\w*|gola'],
    'STOMACH': [r'stomach\w*|ventr\w*|alv(us|um|i|o)|intestin\w*|viscer\w*', r'magen|daerm\w*|darm\w*|bauch', r'stomach|belly|bowels|guts|entrails', r'estomac\w*|ventre|ventreil|boiaux|entraill\w*', r'stomac\w*|ventre|intestin\w*'],
    'LIVER':   [r'iecor\w*|jecor\w*|hepat\w*|epat\w*|splen\w*|lien(is|em)?', r'leber\w*|lebern|milz\w*', r'liver|spleen|jaundice|dropsy|ictericis', r'foie|esplen?\w*|l.espliem|hydrop\w*', r'fegato|milza|itterizia|idrope', r'icteric\w*|hydropic\w*'],
    'URINE':   [r'urin\w*|vesic\w*|ren(es|ibus|um)|lumb(i|orum)|calcul\w*', r'harm\w*|lenden|nieren?|blater', r'urine|bladder|kidneys?|reins|stone|gravel|strangury', r'urine|vessie|rains|pissier|piere', r'orina|vescica|reni|calcol\w*'],
    'WOMB':    [r'matri(x|c\w*)|menstru\w*|menses|partu\w*|abort\w*|secundas|mulier\w*|uter(us|o|i|um)', r'muoter|frawen|fraw|weib\w*|gepurt|kint\w*', r'womb|menses|courses|women|woman|childbirth|conception', r'marriz|fame|feme|fames|enfant\w*', r'utero|mestru\w*|donne|parto|isterismo'],
    'SKIN':    [r'vulner\w*|vulnus|ulcer\w*|plag\w*|cut(is|em)|scabi\w*|lepr\w*|macul\w*|cicatr\w*|scroph\w*', r'wund\w*|geswer\w*|apostem\w*|aussatz\w*|haut', r'wounds?|ulcers?|sores?|skin|scabs?|itch|leprosy|swellings?|tumou?rs?|scurf', r'plaie\w*|aposteme\w*|apostume|chancre|rogne|lepre', r'ferit\w*|piag\w*|ulcer\w*|pelle|rogna'],
    'BLOOD':   [r'sangu\w*|cruor\w*', r'pluot|bluot|blut', r'blood|bleeding', r'sanc|sang', r'sangue'],
    'JOINT':   [r'nerv\w*|iunctur\w*|junctur\w*|articul\w*|podagr\w*|sciat\w*|paralys\w*', r'gelider|gelid\w*|aderen|adern|gesuht', r'joints?|sinews?|nerves?|gout|sciatica|palsy|cramp', r'nerf\w*|goute|jointur\w*|paralisi\w*', r'nervi|gotta|giunture|paralisi'],
    'HEART':   [r'cord(is|i)|praecord\w*|syncop\w*', r'herz\w*|herzen', r'heart|swooning|fainting', r'cuer|cueur|cordial\w*', r'cuore'],
    # conditions
    'FEVER':   [r'febr\w*|quartan\w*|tertian\w*', r'fieber\w*|riten', r'fevers?|agues|quartan|tertian', r'fevre\w*|fievre\w*', r'febbr\w*|intermittent\w*'],
    'COUGH':   [r'tuss\w*|asthm\w*|raucedin\w*|catarrh\w*|rheum\w*', r'huost\w*|husten', r'cough\w*|hoarseness|wheezing|rheum|asthma|phlegm|flegm', r'toux|fleume|reume|rume', r'tosse|catarr\w*|asma'],
    'POISON':  [r'venen\w*|ven(enum|eni)|morsu\w*|serpent\w*|scorpi\w*|toxic\w*|viper\w*', r'gift\w*|vergift\w*|slang\w*|nater\w*|natern', r'poisons?|venom\w*|bitings?|serpents?|adders?|vipers?|scorpions?', r'venin\w*|serpent\w*|morsure\w*|escorpion\w*', r'veleno\w*|serpent\w*|morsicatur\w*'],
    'PAIN':    [r'dolor\w*|tormin\w*|colic\w*', r'smerz\w*|smerzen|swer\w*', r'pains?|aches?|aching|griping|colic|torments?', r'dolor|dolors|dolour|doleur\w*|colique', r'dolor\w*|dolori|colich\w*'],
    'WORM':    [r'lumbric\w*|verm(es|ibus|is)|tine\w*', r'wurm|würm\w*|wurme|wurmen', r'worms', r'vermi'],
    'MIND':    [r'melanchol\w*|insan\w*|phrenes\w*|frenes\w*|furor\w*|mania|lethar\w*|somn\w*|epilep\w*|caduc\w*', r'unsin\w*|slaf\w*|tobsuht', r'melancholy|madness|frenzy|sleep|lethargy|epilepsy|memory', r'melancol\w*|frenesi\w*|dormir|someil|epilenc\w*', r'melancon\w*|sonno|epilessia|pazzia'],
    # preparations / remedies
    'WINE':    [r'vin(um|i|o)|vinum|mer(o|um)|mulsa', r'wein|win', r'wine|claret|sack', r'vin|vins', r'vino'],
    'HONEY':   [r'mel|mell(e|is|i)|melle', r'honig|hönig|honigs', r'honey|hydromel|mead|oxymel', r'miel', r'miele'],
    'VINEGAR': [r'acet\w*', r'ezzeich|ezzich|ezzig|essich', r'vinegar|oxycrate', r'aisil|aissil|vinaigre', r'aceto'],
    'OIL':     [r'ole(um|o|i)|olivo|unguin\w*', r'ol|oel|öl|oles', r'oil|oils', r'oile|uile|huile', r'olio|oglio'],
    'WATER':   [r'aqu(a|ae|am|is)', r'wazzer|wasser|wazzers', r'water|waters', r'eve|l.eve|eaue', r'acqua'],
    'POWDER':  [r'pulv(is|ere|erem)|pulveriz\w*|tritu\w*|trit(a|um|o|ae|is)|contrit\w*', r'pulver\w*|gepulvert|pulvert|gestozzen', r'powder\w*|beaten|bruised|pounded', r'poudre|triblez|broiez|pilez', r'polvere|polveriz\w*|pesta\w*'],
    'COOK':    [r'coqu\w*|coct\w*|decoct\w*|cocta|coctum|elix\w*', r'sied\w*|seud\w*|gesoten|kochen|gekocht', r'boil\w*|decoction\w*|seethed|sodden|stewed', r'cuire|cuite|cuiz|bollir|boli\w*|decoction', r'bollit\w*|decott\w*|cuoc\w*|cotto|cotta|cotti'],
    'DRINK':   [r'bib(e|ere|it|itum|atur|itur|ant|endum|ita|itus)|pot(u|us|ui|a|um|io|ui)|haust\w*|sorb\w*', r'trink\w*|tranch|trank', r'drink\w*|drank|drunk|draught', r'boivre|boive|beivre|bevez|bevrage', r'bere|beva|bevut\w*|bevanda'],
    'SALVE':   [r'emplastr\w*|cataplasm\w*|ungu\w*|inung\w*|perung\w*|linim\w*|fomen\w*', r'pflaster|salb\w*|gesalbet', r'plaisters?|plasters?|ointments?|anointed|anoint|poultice|salve|liniment', r'emplastre|oigniment|oigniez|enoign\w*|cataplasme', r'cataplasma|unguento|impiastro'],
    'JUICE':   [r'suc(us|i|o|um)|succ(us|i|o|um)', r'saf|saft|safs', r'juice|juices|sap', r'jus|suc', r'sugo|succo'],
    'SYRUP':   [r'syrup\w*|sirup\w*|electuar\w*|confectio\w*|pilul\w*|trochisc\w*', r'latwerg\w*|sirop', r'syrups?|electuary|electuaries|pills?|troches|conserve', r'sirop\w*|electuaire|confection\w*|pilules?', r'sciropp\w*|elettuar\w*|pillol\w*'],
    'SUGAR_SPICE': [r'piper\w*|zinzib\w*|cinnam\w*|crocus|croc(i|o|um)|sacchar\w*', r'pfeffer|ingwer|zukker|zimet|saffran', r'pepper|ginger|cinnamon|saffron|sugar|cloves', r'poivre|gingembre|canele|cucre|safran', r'pepe|zenzero|cannella|zucchero|zafferano'],
    # plant parts
    'ROOT':    [r'radi(x|ce|cis|cem|ces|cibus|cum)', r'wurzel\w*|wurz', r'roots?', r'racine\w*', r'radic\w*'],
    'LEAF':    [r'foli(a|um|is|o|orum)|frond\w*', r'pleter|bleter|plat|blat|pletern|pletter', r'leaf|leaves', r'foille\w*|fueille\w*|fuelle\w*', r'fogli\w*|foglioline'],
    'SEED':    [r'sem(en|ine|inis|ina|inibus)', r'samen|kern\w*|korn', r'seeds?|kernels?', r'semence\w*|graine\w*', r'sem(e|i)|semente'],
    'FLOWER':  [r'flo(s|r|res|ris|re|ribus|rem)', r'pluom\w*|bluom\w*|bluet\w*', r'flowers?|blossoms?', r'flor|flors|fleur\w*', r'fior\w*'],
    'FRUIT':   [r'fruct\w*|pom(a|um|i|o)|bacc\w*', r'frucht\w*|obz|opfel\w*', r'fruits?|berries|berry|apples?', r'fruit\w*|pome\w*', r'frutt\w*|bacch?\w*|pomo'],
    'STALK':   [r'caul(is|e|es|ibus)|ram(us|i|is|os)|corte\w*|lign\w*', r'stengel\w*|stam|rind\w*|holz', r'stalks?|branch\w*|bark|stems?|wood', r'tige|branche\w*|escorce\w*|l.escorce', r'fusto|rami|ramoso|scorza|legno'],
    # qualities
    'HOT':     [r'calid\w*|calor\w*|fervid\w*|ignis', r'haiz|hitz\w*|warm\w*|hais', r'hot|heat|heating|warm|warming', r'chaud\w*|chaut|chalor|chauz', r'cald\w*|calore'],
    'COLD':    [r'frigid\w*|frig(us|ore)|refriger\w*', r'kalt\w*|kelt\w*', r'cold|cool|cooling|coldness', r'froid\w*|froidure|froiz', r'fredd\w*|refriger\w*'],
    'DRY':     [r'sicc\w*|arid\w*', r'trucken\w*|durr\w*|dürr\w*', r'dry|drying|dryness|dried', r'sech\w*|seiche\w*', r'secc\w*|asciutt\w*'],
    'MOIST':   [r'humid\w*|humor\w*|humores', r'fauht\w*|feuht\w*|fiuht\w*', r'moist\w*|humid\w*|humou?rs?|moisture', r'moiste\w*|humors?|humeur\w*|moitez', r'umid\w*|umori'],
    # cosmos / time
    'SUN':     [r'sol|sol(is|e|em)', r'sunne\w*|sunn', r'sun|sunny', r'soleil\w*', r'sole'],
    'MOON':    [r'lun(a|ae|am)', r'mond|monde|mondes', r'moon', r'lune', r'luna'],
    'SEASON':  [r'aestat\w*|estat\w*|autumn\w*|hiem\w*|hyem\w*|vern(o|um|a|al\w*)', r'sumer|sumers|winter\w*|herbst\w*|lenz\w*|maien', r'summer|winter|spring|autumn|harvest', r'yver|iver|printemps|autompne', r'estate|inverno|primavera|autunno'],
    'MONTH':   [r'mens(e|is|ibus)|ianuar\w*|februar\w*|martio|maio|iunio|iulio|augusto', r'manod\w*|maend\w*|merz\w*|aprill\w*', r'months?|january|february|april|june|july|august|september|october|november|december', r'mois|janvier|fevrier|avril|juing|juignet|aoust', r'mese|mesi|gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre'],
    'DAY_NIGHT': [r'dies|diebus|diem|noct\w*|nox|mane', r'tag\w*|naht\w*|morgen\w*', r'day|days|night|nights|morning|evening', r'jor|jorz|jours?|nuit|matin|soir', r'giorno|giorni|notte|mattina'],
    'GARDEN':  [r'hort\w*|sativ\w*|domestic\w*', r'garten|gerten|haimisch', r'gardens?|sown|planted', r'jardin\w*|domesche', r'giardin\w*|ort[io]|coltiva\w*'],
    'WILD':    [r'agrest\w*|silvestr\w*|montan\w*|mont(es|ibus|ium)|camp(o|i|is|os|us|estr\w*)|pal(us|udibus|ustr\w*)', r'wild\w*|velt|berg\w*|walden|wald', r'wild|woods|hedges|fields|mountains|hills|meadows|ditches|marshes', r'sauvage\w*|champs?|montaign\w*|bois', r'selvatic\w*|prati|siepi|monti|boschi|fossi|pascoli|paludos\w*'],
}

_RX = {c: re.compile('^(?:' + '|'.join(p) + ')$') for c, p in CLASSES.items()}
CLASS_NAMES = list(CLASSES)


def classify(w):
    for c in CLASS_NAMES:
        if _RX[c].match(w):
            return c
    return None
