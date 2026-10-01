#!/usr/bin/env python3
"""Build the compact data files used by the web app in app/.

Reads (never writes) ai4qi-library.json, new_audits/*_new_audits.json,
new_audits/templates/*.csv, standards/standards.json and figures/.
Writes app/data/*.json, app/data/audits/*.json, app/templates/*.csv, app/figures/.
The PWA files (app/sw.js, app/manifest.webmanifest, app/icons/) and app/config.json (feedback and
optional Supabase settings) are kept; missing config keys are added empty. The service worker's
cache VERSION is re-stamped from a hash of the built files, config.json and sw.js itself and written
to app/data/version.json, so installed copies of the app pick up the new data and code.

Standard library only. Re-runnable: output folders are rebuilt each time.

    python3 build_app_data.py
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / "app"
DATA = APP / "data"
SHARD = 100  # published audits per detail file

OPEN_LICENCES = {"cc by", "cc0"}
UK_WORDS = re.compile(r"\b(UK|United Kingdom|England|Scotland|Wales|Northern Ireland|Ireland|Irish|British)\b", re.I)
UK_HOSTS = re.compile(r"(\.nhs\.uk|\.nhs\.scot|\.hse\.ie|\.ac\.uk|\.gov\.uk|\.org\.uk|\.co\.uk|\.ie)(/|$)", re.I)

# Editorial wording for the product: the site calls the collection "the library".
WORDING = [
    (re.compile(r"\bin the corpus\b"), "in the library"),
    (re.compile(r"\bin corpus\b"), "in the library"),
    (re.compile(r"\bThe corpus\b"), "The library"),
    (re.compile(r"\bthe corpus\b"), "the library"),
    (re.compile(r"\bcorpus\b"), "library"),
    (re.compile(r"\bCorpus\b"), "Library"),
]


def tidy(text):
    if not isinstance(text, str):
        return text
    for pat, rep in WORDING:
        text = pat.sub(rep, text)
    return text


def tidy_all(obj):
    if isinstance(obj, str):
        return tidy(obj)
    if isinstance(obj, list):
        return [tidy_all(x) for x in obj]
    if isinstance(obj, dict):
        return {k: tidy_all(v) for k, v in obj.items()}
    return obj


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))


def clean(value):
    """Drop empty values so the files stay small."""
    if isinstance(value, dict):
        out = {k: clean(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [clean(v) for v in value if v not in (None, "", [], {})]
    return value


def is_uk(a):
    paper = a.get("paper") or {}
    if paper.get("uk_ireland") is True:
        return True
    if paper.get("uk_ireland") is False:
        return False
    country = str((a.get("detail") or {}).get("country") or "")
    if country and country != "not reported":
        return bool(UK_WORDS.search(country))
    src = str(a.get("source") or "")
    host = re.sub(r"^https?://", "", src).split("/")[0]
    return bool(host and UK_HOSTS.search(host + "/"))


def loop_closed(detail):
    v = (detail or {}).get("loop_closed")
    if v is True or (isinstance(v, str) and v.lower().startswith("yes")):
        return 1
    return 0


def spec_group(s):
    return (s or "Other").split(" – ")[0].strip()


def build_library(lib):
    index, shards = [], {}
    figures_needed = set()
    for a in lib["audits"]:
        d = a.get("detail") or {}
        p = a.get("paper") or {}
        c1 = d.get("cycle1") or {}
        c2 = d.get("cycle2") or {}
        country = d.get("country") if d.get("country") not in (None, "not reported") else ""
        if not country and p.get("country_from_affiliation"):
            country = p["country_from_affiliation"]
        row = {
            "id": a["id"],
            "t": a.get("title"),
            "s": re.sub(r"^Orthopaedics\b", "Trauma and orthopaedics", a.get("specialty") or "") or a.get("specialty"),
            "g": spec_group(a.get("specialty")),
            "tp": a.get("topic"),
            "st": a.get("standard"),
            "f": a.get("finding"),
            "c": a.get("change"),
            "r1": c1.get("result") if c1.get("result") != a.get("finding") else "",
            "iv": d.get("intervention") if d.get("intervention") != a.get("change") else "",
            "r2": c2.get("result"),
            "co": country,
            "y": p.get("year"),
            "uk": 1 if is_uk(a) else 0,
            "lc": loop_closed(d),
            "dt": 1 if d else 0,
            "fx": d.get("fix_type") if d.get("fix_type") not in ("not reported", "unclassified") else "",
            "ss": a.get("status"),
        }
        index.append(clean(row))

        full = {k: a.get(k) for k in ("id", "title", "specialty", "topic", "status", "standard",
                                       "finding", "change", "next_audit", "source", "figure_link")}
        if d:
            full["detail"] = {k: v for k, v in d.items() if k not in ("fix_type_note",)}
        if p:
            keep = ("authors", "journal", "year", "doi", "pmid", "pmcid", "licence", "uk_ireland", "citation", "title")
            full["paper"] = {k: p.get(k) for k in keep}
            if str(p.get("licence") or "").lower() in OPEN_LICENCES:
                full["paper"]["abstract"] = p.get("abstract")
        if a.get("figures"):
            full["figures"] = [{k: f.get(k) for k in ("label", "caption", "credit", "licence", "source_url", "file")}
                               for f in a["figures"]]
            figures_needed.update(f["file"] for f in a["figures"] if f.get("file"))
        shards.setdefault(a["id"] // SHARD, []).append(clean(full))
    return index, shards, figures_needed


def build_proposed():
    st = HERE / "new_audits" / "short_titles.json"          # 3-4 word names, e.g. for email subjects
    short = json.loads(st.read_text(encoding="utf-8")) if st.exists() else {}
    sp = HERE / "new_audits" / "pass_short.json"            # the pass criterion in 14 words or fewer
    pass_short = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    hp = HERE / "new_audits" / "pp_heads.json"              # 2-5 word headings for each pitfall and pearl
    pp_heads = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else {}
    out = []
    for fname, group in (("ortho_new_audits.json", "Trauma and orthopaedics"), ("nonortho_new_audits.json", "")):
        for p in json.load(open(HERE / "new_audits" / fname, encoding="utf-8")):
            q = tidy_all(dict(p))
            # Evidence lines are shown only when they cite a library entry.
            q["evidence"] = [e for e in q.get("evidence") or [] if re.search(r"\[\d+", e)]
            q["group"] = group or q.get("area")
            q["template_file"] = "templates/" + Path(p.get("template_file") or (p["id"] + ".csv")).name
            if short.get(p["id"]):
                q["short"] = short[p["id"]]
                if pass_short.get(q["id"]):
                    q["pass_short"] = pass_short[q["id"]]
            h = pp_heads.get(q["id"]) or {}
            for k, hk in (("pitfalls", "pitfall_heads"), ("pearls", "pearl_heads")):
                if q.get(k) and len(h.get(k) or []) == len(q[k]):
                    q[hk] = h[k]
            out.append(clean(q))
    return out


def build_standards(proposed):
    stds = json.load(open(HERE / "standards" / "standards.json", encoding="utf-8"))
    # standards/standards.json is rebuilt by new_audits/compile.py from the audits; standards added on
    # their own (not yet used by an audit) live in extra_standards.json so a recompile never drops them.
    extra = HERE / "standards" / "extra_standards.json"
    if extra.exists():
        stds = stds + json.load(open(extra, encoding="utf-8"))
    by_key = {}
    for s in stds:
        s = dict(s)
        s["used_by"] = list(dict.fromkeys(s.get("used_by") or []))
        by_key.setdefault((s.get("wording") or "").strip(), s)
    result = list(by_key.values())
    for p in proposed:
        st = p.get("standard") or {}
        key = (st.get("wording") or "").strip()
        if not key:
            continue
        s = by_key.get(key)
        if s is None:
            s = {"source": st.get("source"), "wording": st.get("wording"), "url": st.get("url", ""), "used_by": []}
            by_key[key] = s
            result.append(s)
        if p["id"] not in s["used_by"]:
            s["used_by"].append(p["id"])
    result.sort(key=lambda s: (s.get("source") or "").lower())
    return [clean(s) for s in result]


def build_cards(lib):
    cards = []
    for c in lib["topic_knowledge"]:
        card = {k: c.get(k) for k in ("topic", "audits_in_library", "countries", "years_seen", "usual_baseline",
                                      "fix_that_works", "fix_that_fails", "consultant_advice", "evidence_ids",
                                      "status", "revises_seed_card")}
        card["countries"] = [x for x in card.get("countries") or [] if x and x != "not reported"]
        card["draft"] = 1 if c.get("status") else 0
        cards.append(clean(card))
    return cards


SHELL = ("index.html", "app.js", "export.js", "styles.css", "manifest.webmanifest")
CONFIG_DEFAULTS = {"feedback_url": "", "supabase_url": "", "supabase_anon_key": "",
                   "analytics": "", "plausible_script": "", "plausible_domain": "",
                   "plausible_host": "https://plausible.io", "cloudflare_token": "", "build_url": "", "claude_link": "", "nice_ai_permission": False,
                   "legal": {"owner_name": "", "postal_address": "", "contact_email": "", "security_email": "", "site_url": "", "ico_number": "", "updated": ""}}
VERSION_LINE = re.compile(r"^var VERSION = '[^']*';", re.M)


def keep_config():
    """Keep app/config.json as the owner set it; only add keys that are missing."""
    path = APP / "config.json"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("config.json must hold an object")
    except FileNotFoundError:
        cfg = {}
    missing = {k: v for k, v in CONFIG_DEFAULTS.items() if k not in cfg}
    if missing or not path.is_file():
        cfg.update(missing)
        path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return cfg


def stamp_version():
    """Hash everything the app serves and stamp it into sw.js so clients refresh their caches."""
    h = hashlib.sha256()
    files = [APP / n for n in SHELL] + [APP / "config.json"] + sorted((APP / "icons").glob("*.png"))
    for sub in ("data", "templates", "figures", "vendor"):
        files += sorted(f for f in (APP / sub).rglob("*") if f.is_file() and f.name != "version.json")
    for f in files:
        if f.is_file():
            h.update(f.relative_to(APP).as_posix().encode())
            h.update(f.read_bytes())
    sw = APP / "sw.js"
    if sw.is_file():  # the worker's own code, minus the stamped VERSION line
        h.update(VERSION_LINE.sub("", sw.read_text(encoding="utf-8")).encode())
    version = h.hexdigest()[:12]
    dump(DATA / "version.json", {"version": version})
    if sw.is_file():
        text = sw.read_text(encoding="utf-8")
        new = VERSION_LINE.sub(f"var VERSION = '{version}';", text, count=1)
        if new != text:
            sw.write_text(new, encoding="utf-8")
    return version


LEGAL = [("terms", "TERMS_OF_USE.md", "Terms of use"), ("privacy-notice", "PRIVACY_NOTICE.md", "Privacy notice"),
         ("cookies", "COOKIES_AND_STORAGE.md", "Cookies and storage"), ("accessibility", "ACCESSIBILITY_STATEMENT.md", "Accessibility"),
         ("security", "VULNERABILITY_DISCLOSURE.md", "Reporting a security problem")]


BEACON = re.compile(r"\s*<!-- analytics -->.*?<!-- /analytics -->", re.S)


def write_beacon(cfg):
    """Cloudflare Web Analytics is a manual snippet (automatic injection is off in Cloudflare): write it
    into the head of every page from config.json, or remove it when analytics is not Cloudflare."""
    token = str(cfg.get("cloudflare_token") or "").strip()
    on = str(cfg.get("analytics") or "").lower() == "cloudflare" and re.fullmatch(r"[A-Za-z0-9]{16,64}", token)
    tag = ('\n<!-- analytics -->\n<script defer src="https://static.cloudflareinsights.com/beacon.min.js" '
           f'data-cf-beacon=\'{{"token": "{token}", "spa": false}}\'></script>\n<!-- /analytics -->') if on else ""
    for name in ("index.html", "404.html"):
        path = APP / name
        if not path.exists():
            continue
        html = BEACON.sub("", path.read_text(encoding="utf-8"))
        html = html.replace("</head>", tag.lstrip("\n") + "\n</head>", 1) if tag else html
        path.write_text(html, encoding="utf-8")


def write_security_txt(cfg):
    """app/.well-known/security.txt (RFC 9116), only once the owner has set site_url and an email."""
    L = (cfg or {}).get("legal") or {}
    site, mail = (L.get("site_url") or "").rstrip("/"), L.get("security_email") or L.get("contact_email")
    src = HERE / "governance" / "security.txt"
    if not (site and mail and src.exists()):
        return
    txt = src.read_text(encoding="utf-8").replace("[security contact email]", mail)
    txt = txt.replace("https://[your domain]/security", site + "/#/security").replace("https://[your domain]", site)
    out = APP / ".well-known" / "security.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8")


def build_legal(cfg):
    """Public legal pages from governance/*.md (draft banners and review notes stripped)."""
    L = (cfg or {}).get("legal") or {}
    fill = {"OWNER LEGAL NAME": L.get("owner_name"), "Owner name": L.get("owner_name"), "ADDRESS": L.get("postal_address"),
            "Postal address": L.get("postal_address"), "CONTACT EMAIL": L.get("contact_email"), "Contact email": L.get("contact_email"),
            "ICO REG NO.": L.get("ico_number"), "ICO registration number": L.get("ico_number"), "DATE": L.get("updated"),
            "security contact email": L.get("security_email") or L.get("contact_email"),
            "your domain": (L.get("site_url") or "").replace("https://", "").rstrip("/")}
    try:
        import markdown
    except ImportError:
        print("legal pages skipped: pip install markdown")
        return {}
    out = {}
    for slug, fn, title in LEGAL:
        p = HERE / "governance" / fn
        if not p.exists():
            continue
        raw = p.read_text(encoding="utf-8").splitlines()
        first_h2 = next((i for i, l in enumerate(raw) if l.startswith("## ")), len(raw))
        lines = [l for i, l in enumerate(raw)            # owner notes are blockquotes above the first section
                 if not (l.lstrip().startswith(">") and (i < first_h2 or re.match(r"\s*>\s*\**\s*draft", l, re.I)))]
        text = re.sub(r"<!--.*?-->", "", "\n".join(lines), flags=re.S)
        text = re.split(r"\n#+\s*sources checked", text, flags=re.I)[0]
        for ph, val in fill.items():                    # placeholders filled from app/config.json "legal"
            if val:
                text = re.sub(r"\[" + re.escape(ph) + r"\]", val, text, flags=re.I)
        html = markdown.markdown(text, extensions=["tables", "sane_lists"])
        html = re.sub(r"<script.*?</script>", "", html, flags=re.S | re.I)
        out[slug] = {"title": title, "html": html}
    return out


def main():
    lib = json.load(open(HERE / "ai4qi-library.json", encoding="utf-8"))

    for sub in ("data", "templates", "figures"):
        shutil.rmtree(APP / sub, ignore_errors=True)
    cfg = keep_config()

    index, shards, figs = build_library(lib)
    proposed = build_proposed()
    standards = build_standards(proposed)
    cards = build_cards(lib)

    dump(DATA / "library.json", {"shard": SHARD, "audits": index})
    for key, rows in shards.items():
        dump(DATA / "audits" / f"{key}.json", rows)
    dump(DATA / "proposed.json", proposed)
    dump(DATA / "cards.json", cards)
    dump(DATA / "standards.json", standards)
    dump(DATA / "legal.json", build_legal(cfg))
    og = HERE / "standards" / "organisations.json"          # NHS trusts, health boards, HSE regions (drop-down)
    dump(DATA / "organisations.json", json.loads(og.read_text()) if og.exists() else {})
    nt = HERE / "standards" / "nice_titles.json"            # NICE titles for plain-words sources
    dump(DATA / "nice_titles.json", json.loads(nt.read_text()) if nt.exists() else {})
    write_security_txt(cfg)
    write_beacon(cfg)

    (APP / "templates").mkdir(parents=True, exist_ok=True)
    for csv in sorted((HERE / "new_audits" / "templates").glob("*.csv")):
        shutil.copy2(csv, APP / "templates" / csv.name)

    copied = 0
    for rel in sorted(figs):
        src = HERE / rel
        if src.is_file():
            dst = APP / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    version = stamp_version()

    size = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
    print(f"library index: {len(index)} audits, {len(shards)} detail files")
    print(f"proposed: {len(proposed)}  cards: {len(cards)}  standards: {len(standards)}")
    print(f"templates: {len(list((APP / 'templates').glob('*.csv')))}  figures copied: {copied}")
    print(f"app/data total: {size / 1024:.0f} KB")
    print(f"offline cache version: {version}")
    print("analytics: " + (cfg.get("analytics") or "off"))
    print("backend: " + ("Supabase configured" if cfg.get("supabase_url") and cfg.get("supabase_anon_key")
                         else "Supabase not configured (feedback stays on the device" +
                         (" or goes to feedback_url)" if cfg.get("feedback_url") else ")")))


if __name__ == "__main__":
    main()
