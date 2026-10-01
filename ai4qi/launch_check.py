"""Pre-launch check for the hosted site. Lists what is still missing before Ai4Qi goes live.

    python3 launch_check.py            # exit code 1 while anything blocks launch

Checks the repository only. Owner steps done in dashboards (ICO fee, DPAs, two-factor, DPIA
signature) are listed as reminders because they cannot be seen from here.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE / "app"
blockers, warnings, ok = [], [], []


def need(cond, good, bad, soft=False):
    (ok if cond else warnings if soft else blockers).append(good if cond else bad)


cfg = json.loads((APP / "config.json").read_text())
# Older Safari (before 16.4) cannot parse look-behind patterns; one in the code stops the whole site loading there.
import re as _re
_lb = [f for f in ("app.js", "export.js", "sw.js") if _re.search(r"\(\?<[=!]", (APP / f).read_text(encoding="utf-8"))]
need(not _lb, "no look-behind patterns (older Safari can load the site)", "look-behind regex in " + ", ".join(_lb) + ": older Safari cannot load the site")
L = cfg.get("legal") or {}
site = (L.get("site_url") or "").strip()
need(site.startswith("https://"), f"site address set ({site})", "config.json legal.site_url: the live address, e.g. https://ai4qi.org")
for k, label in (("owner_name", "controller's legal name"), ("postal_address", "postal address"),
                 ("contact_email", "contact email"), ("ico_number", "ICO registration number"), ("updated", "date the legal pages take effect")):
    need(bool((L.get(k) or "").strip()), f"legal: {label} set", f"config.json legal.{k}: {label}")
need(bool(cfg.get("supabase_url")) == bool(cfg.get("supabase_anon_key")), "Supabase URL and key set together",
     "config.json: supabase_url and supabase_anon_key must both be set (or both empty)")
key = cfg.get("supabase_anon_key") or ""
need(not key or key.startswith("sb_publishable_") or key.count(".") == 2, "Supabase key is a public key",
     "config.json supabase_anon_key: use the publishable key, never the secret key")
need(not re.search(r"service_role|sb_secret_", json.dumps(cfg)), "no secret keys in config.json",
     "config.json contains a secret key: remove it now and rotate it in Supabase")
need(bool(cfg.get("supabase_url")), "sign-in, reminders and feedback collection switched on",
     "config.json supabase_url empty: the site works, but sign-in, reminders and feedback stay off", soft=True)
need(bool(cfg.get("build_url")), "'Build an audit' connected to the build-audit function",
     "config.json build_url empty: 'Build an audit' cannot write new protocols on the hosted site", soft=True)
need(bool(cfg.get("analytics")), f"analytics: {cfg.get('analytics')}", "analytics off (optional)", soft=True)
need(cfg.get("nice_ai_permission") is False or cfg.get("nice_ai_permission") is True, "NICE AI setting present", "nice_ai_permission missing")
if not cfg.get("nice_ai_permission"):
    ok.append("NICE wording is kept out of AI prompts until NICE gives permission")

legal = json.loads((APP / "data" / "legal.json").read_text()) if (APP / "data" / "legal.json").exists() else {}
need(len(legal) >= 5, f"{len(legal)} legal pages built", "data/legal.json missing: run build_app_data.py")
for slug, page in legal.items():
    left = sorted(set(re.findall(r"\[[A-Z][A-Za-z .]{2,40}\]", page["html"])))
    need(not left, f"{slug}: no placeholders left", f"{slug} page still shows {', '.join(left)}")
need((APP / ".well-known" / "security.txt").exists(), "security.txt published",
     "app/.well-known/security.txt: generated once legal.site_url and an email are set")

big = [p for p in APP.rglob("*") if p.is_file() and p.stat().st_size > 25 * 1024 * 1024]
need(not big, "every file under 25 MB (Cloudflare Pages limit)", "files over 25 MB: " + ", ".join(str(p.relative_to(APP)) for p in big))
n = sum(1 for p in APP.rglob("*") if p.is_file())
need(n < 20000, f"{n} files (Cloudflare Pages limit 20,000)", f"{n} files: over the Cloudflare Pages limit of 20,000")
hdr = (APP / "_headers").read_text() if (APP / "_headers").exists() else ""
need("Content-Security-Policy" in hdr and "frame-ancestors 'none'" in hdr, "security headers and CSP in _headers", "app/_headers missing the CSP")
for f in ("index.html", "robots.txt", "404.html", "manifest.webmanifest", "sw.js"):
    need((APP / f).exists(), f"{f} present", f"app/{f} missing")
idx = (APP / "index.html").read_text()
ext = [s for s in re.findall(r'src="(https?://[^"]+)"', idx)
       if "plausible.io" not in s and s != "https://static.cloudflareinsights.com/beacon.min.js"]
need(not ext, "no outside scripts in index.html", "outside scripts in index.html: " + ", ".join(ext))
if str(cfg.get("analytics") or "").lower() == "cloudflare":
    tok = cfg.get("cloudflare_token") or ""
    for f in ("index.html", "404.html"):
        page = (APP / f).read_text() if (APP / f).exists() else ""
        need("static.cloudflareinsights.com/beacon.min.js" in page and tok in page, f"{f}: Cloudflare Web Analytics beacon with the token",
             f"app/{f}: Cloudflare beacon missing (run build_app_data.py)")
    need("https://static.cloudflareinsights.com" in hdr and "https://cloudflareinsights.com" in hdr,
         "CSP allows the Cloudflare beacon (script-src and connect-src)", "app/_headers CSP: add static.cloudflareinsights.com (script-src) and cloudflareinsights.com (connect-src)")
sw = (APP / "sw.js").read_text() if (APP / "sw.js").exists() else ""
shell_list = re.search(r"SHELL_FILES\s*=\s*\[(.*?)\]", sw, re.S)
need(bool(shell_list) and not re.search(r"['\"](\./|/)?index\.html['\"]", shell_list.group(1)),
     "sw.js does not precache index.html (Cloudflare Pages redirects it to /)",
     "app/sw.js precaches index.html: returning visitors get ERR_FAILED from the redirected response")
need("redirected" in sw, "sw.js rebuilds redirected responses before caching or returning them",
     "app/sw.js: redirected responses are cached as-is (Chrome refuses them for navigations)")
code = (APP / "app.js").read_text() + (APP / "export.js").read_text()
need(not re.search(r"sk-ant-|sb_secret_|service_role", code), "no API keys in the site code", "an API key appears in app.js or export.js")

reminders = [
    "Domain bought and connected to Cloudflare Pages (runbook §1–2)",
    "Supabase project in London, migrations 001–009 run, admin email added (runbook §3)",
    "Anthropic key and ALLOWED_ORIGIN in Supabase secrets; build-audit deployed; spend limit set (runbook §4)",
    "Resend domain verified; SMTP and reminder secrets set; send-reminders deployed and scheduled (runbook §5)",
    "ICO fee paid; number added to config.json legal.ico_number (runbook §6)",
    "DPAs accepted and saved as PDFs; two-factor on every account (runbook §8, §10)",
    "DPIA read and signed; terms and privacy notice read once (runbook §10)",
    "After deploy: securityheaders.com shows A; sign in with a magic link; build one audit; download each file",
]

print("READY" if not blockers else f"NOT READY: {len(blockers)} item(s) block launch")
for title, items, mark in (("Blocks launch", blockers, "✗"), ("Optional or later", warnings, "!"), ("Done", ok, "✓")):
    if items:
        print(f"\n{title}:")
        for i in items:
            print(f"  {mark} {i}")
print("\nOwner steps to confirm by hand (not visible from the repository):")
for r in reminders:
    print(f"  ☐ {r}")
sys.exit(1 if blockers else 0)
