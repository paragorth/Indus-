# Launch day (Friday 2 October 2026): checklist

Live site: https://ai4qi.com. `python3 ai4qi/launch_check.py` prints READY (29 Sep). Tested on the
live site on 29 Sep, desktop and phone: every page loads with no errors; a whole audit runs end to end
in demo mode; the data sheet, results slides, records, backup, calendar and Word proposal all download
and open; uploading a sheet works; a fresh audit builds in about 25 seconds.

## Before Friday: the owner (Claude in Chrome can do most of it; it stops for logins and payments)

1. **NHS web filters.** New domains are often blocked on Trust computers as "newly registered" or
   "uncategorised". Submit `ai4qi.com` for categorisation as **Health and Medicine** (or Education)
   today; each takes 1–3 working days:
   - Forcepoint (many NHS Trusts): https://csi.forcepoint.com/
   - Palo Alto Networks: https://urlfiltering.paloaltonetworks.com/
   - Zscaler: https://sitereview.zscaler.com/
   - Cisco Talos: https://talosintelligence.com/reputation_center/
   - Broadcom (Blue Coat) WebPulse: https://sitereview.bluecoat.com/
   - Fortinet FortiGuard: https://www.fortiguard.com/faq/wfratingsubmit
   Then open https://ai4qi.com on a Trust computer before Friday. If it is blocked, ask the Trust IT
   service desk to allow it (it is a clinical audit tool; no patient data leaves the device).
2. **Sign-in emails can keep up.** Supabase → Authentication → Emails → SMTP: custom SMTP must be on
   (Resend). Without it Supabase sends only a few emails an hour. Then Authentication → **Rate limits**:
   raise "emails sent per hour" to 200 for launch day.
3. **Resend allowance.** The free plan sends 100 emails a day. If more than about 80 people might sign
   in on Friday, move to Resend Pro for the month (about $20).
4. **Branded sign-in emails.** Paste the two templates from `ai4qi/backend/supabase/email-templates/`
   (Magic Link, Confirm signup); see that folder's README.
5. **Claude spending limit.** Anthropic Console → Settings → Limits: a monthly limit you are comfortable
   with (a build costs a few pence; each person is capped at 20 new builds a day).
6. **Your own account.** Sign in, put your name in Account → "Your name" (the circle then shows PG),
   and check the Admin menu appears (your email must be in the `admins` table).
7. **Sign the DPIA** as director of Paraggarg Limited and save the PDF with the processors' DPAs.

## On the day

- Put `ai4qi/demo/ai4qi-qr.png` (QR code for https://ai4qi.com, tagged "launch" in the visitor
  statistics) on the first and last slides.
- Open https://ai4qi.com once on the presenting laptop beforehand, then reload, so the offline copy is
  current; install it (address bar → Install) so it still opens if the Wi-Fi drops.
- Live demo: `ai4qi/demo/README.md` (demo mode: account menu → Demo mode, then the dashed Demo buttons,
  one per stage). Build one audit live on a fresh topic; have NNA-074 (sepsis) open in another tab in
  case the Wi-Fi is slow.
- Turn demo mode off afterwards (account menu → Turn off demo mode).

## If something goes wrong

| Problem | What to do |
|---|---|
| "Build an audit" fails or is slow | Use a ready-made audit (Proposed audits or Suggested audits); check Supabase → Edge Functions → build-audit → Logs |
| Sign-in email does not arrive | Check junk; Supabase → Authentication → Logs; Resend → Emails (bounces, daily limit) |
| Site blocked on a Trust computer | Use a phone or personal device on the day; ask Trust IT to allow ai4qi.com |
| "This site can't be reached" after an update | Reload once (the offline copy updates itself) |
| Anything else | The site works without sign-in: library, proposed audits, My audits, all downloads |

## After Friday

- The feedback review runs on the 1st of each month and emails you; nothing changes without your approval.
- Look at Cloudflare Web Analytics and the Admin statistics page after a week.
