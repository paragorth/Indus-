"""Rule-based screening and labelling (no model needed).

These rules decide which search hits are audits at all and give each one a
provisional specialty/subspecialty and UK/Ireland flag.  The Anthropic
extraction step later confirms or corrects the specialty and fills everything
that needs reading the paper.
"""
import re

from queries import NON_ORTHO, ORTHO, ORTHO_ANCHORS

CLOSED_LOOP = re.compile(
    r"closed[- ]loop|re-?audit|complete[d]? (?:the )?audit (?:cycle|loop)|full audit cycle|"
    r"second (?:audit )?cycle|two[- ]cycle|third cycle|cycle 2|cycle two|close[d]? the loop|"
    r"closing the loop|audit loop|completed loop|repeat audit", re.I)
AUDIT = re.compile(r"\baudit", re.I)
QI = re.compile(r"quality improvement|\bPDSA\b|plan[- ]do[- ]study[- ]act", re.I)
EXCLUDE_TYPES = {"Review", "Systematic Review", "Meta-Analysis", "Editorial", "Comment",
                 "Published Erratum", "Retracted Publication", "Retraction of Publication",
                 "Letter", "News", "Guideline", "Practice Guideline", "Scoping Review"}
EXCLUDE_TITLE = re.compile(r"systematic review|scoping review|meta-analysis|study protocol|"
                           r"\bprotocol for\b|erratum|correction to|retracted", re.I)
UK_IE = re.compile(
    r"\bUK\b|United Kingdom|England|Scotland|Wales|Northern Ireland|\bIreland\b|\bNHS\b|"
    r"London|Manchester|Birmingham|Leeds|Glasgow|Edinburgh|Cardiff|Belfast|Dublin|Cork|Galway|"
    r"Liverpool|Bristol|Oxford|Cambridge|Sheffield|Newcastle|Nottingham|Leicester|Southampton", re.I)
COUNTRY = [(re.compile(p, re.I), c) for p, c in [
    (r"Northern Ireland|Belfast", "UK"), (r"\bIreland\b|Dublin|Cork|Galway|Limerick", "Ireland"),
    (r"\bUK\b|United Kingdom|England|Scotland|Wales|\bNHS\b", "UK"),
    (r"Pakistan", "Pakistan"), (r"\bIndia\b", "India"), (r"Sudan", "Sudan"), (r"Egypt", "Egypt"),
    (r"Nigeria", "Nigeria"), (r"Saudi Arabia", "Saudi Arabia"), (r"Australia", "Australia"),
    (r"New Zealand", "New Zealand"), (r"Canada", "Canada"), (r"\bUSA\b|United States", "USA"),
    (r"Malaysia", "Malaysia"), (r"Sri Lanka", "Sri Lanka"), (r"Oman", "Oman"), (r"Qatar", "Qatar"),
    (r"United Arab Emirates|\bUAE\b", "UAE"), (r"Bahrain", "Bahrain"), (r"Kuwait", "Kuwait"),
    (r"Nepal", "Nepal"), (r"Bangladesh", "Bangladesh"), (r"South Africa", "South Africa"),
    (r"Kenya", "Kenya"), (r"Ghana", "Ghana"), (r"Netherlands", "Netherlands"),
    (r"Germany", "Germany"), (r"Italy", "Italy"), (r"Spain", "Spain"), (r"France", "France"),
    (r"Singapore", "Singapore"), (r"Hong Kong", "Hong Kong"), (r"China", "China"),
    (r"Iraq", "Iraq"), (r"Jordan", "Jordan"), (r"Turkey|Türkiye", "Turkey"),
]]


def _has(term, text):
    t = term.strip()
    if len(t) <= 4 and t.isupper():          # acronyms: case-sensitive, whole word
        return re.search(rf"\b{re.escape(t)}\b", text) is not None
    return re.search(rf"\b{re.escape(t.lower())}", text.lower()) is not None


def screen(rec, fulltext=""):
    """Return (keep: bool, reason, audit_kind).  fulltext (optional) rescues papers whose
    abstract never says "audit" but whose body reports an audit cycle."""
    title, abst = rec.get("title", ""), rec.get("abstract", "")
    text = f"{title}\n{abst}"
    if EXCLUDE_TITLE.search(title):
        return False, "excluded: review/protocol/erratum by title", None
    bad = EXCLUDE_TYPES.intersection(rec.get("pub_types") or [])
    if bad and not CLOSED_LOOP.search(title):
        return False, f"excluded: publication type {sorted(bad)[0]}", None
    if not AUDIT.search(text):
        if fulltext and CLOSED_LOOP.search(fulltext) and len(AUDIT.findall(fulltext)) >= 3:
            return True, "kept", "closed-loop or re-audit (stated in full text only)"
        return False, "excluded: no audit wording in title/abstract", None
    if CLOSED_LOOP.search(text):
        return True, "kept", "closed-loop or re-audit"
    if QI.search(text) or AUDIT.search(title):
        return True, "kept", "audit or QI, loop not stated in abstract"
    return False, "excluded: audit mentioned only in passing", None


def is_ortho(rec):
    text = f"{rec.get('title', '')}\n{rec.get('abstract', '')}\n{rec.get('journal', '')}".lower()
    return any(a in text for a in ORTHO_ANCHORS)


def label(rec, areas, query_labels):
    """First area whose terms appear in the title, then abstract, then the query that found it."""
    for field in ("title", "abstract"):
        text = rec.get(field, "")
        for name, terms in areas.items():
            if any(_has(t, text) for t in terms):
                return name
    return re.sub(r" \(.*\)$", "", query_labels[0]) if query_labels else "unclassified"


def ortho_subspecialty(rec, query_labels):
    return label(rec, ORTHO, query_labels)


def nonortho_specialty(rec, query_labels):
    return label(rec, NON_ORTHO, query_labels)


def country_from_affiliations(affs):
    text = " ".join(affs or [])
    for rx, c in COUNTRY:
        if rx.search(text):
            return c
    return "not reported"


def uk_ireland(affs):
    return bool(UK_IE.search(" ".join(affs or [])))
