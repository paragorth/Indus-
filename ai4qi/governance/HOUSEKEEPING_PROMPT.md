# Prompt: finish the launch housekeeping (for Claude in Chrome)

Paste everything in the box below into Claude in Chrome (the browser extension). It does the clicking.
It stops for anything only the owner can do.

---

> You are helping me, the owner of Ai4Qi (a free clinical audit website for UK and Irish clinicians),
> finish the paperwork and set-up so the site can go live. Work through the steps below in order.
> After each step, tell me in one line what you did and what is next.
>
> **Ground rules**
> 1. **STOP and ask me** before any payment, before ticking anything that accepts terms, a contract
>    or a data processing agreement, before any declaration, and whenever a password, two-factor
>    code or passkey is needed. Never click through a CAPTCHA; ask me to do it.
> 2. **Secrets** (API keys, the database password, the Supabase secret key, CRON_SECRET) go straight
>    from one dashboard into another dashboard's secret field. Never paste them into a chat,
>    document, email or GitHub, and never show them in your summaries.
> 3. Read files from GitHub: repository `paragorth/Indus-`, branch
>    `claude/ai4qi-audit-library-pipeline-8yph4w`, folder `ai4qi/governance/`. Use the **Raw** button.
> 4. If a page looks different from what the files describe, describe what you see and ask me.
>
> **Step 1: decisions (ask me, one message)**
> Ask me for: (a) the domain (suggest checking `ai4qi.org`, `ai4qi.co.uk` and `ai4qi.com` in the
> Cloudflare registrar; I may want all three), (b) whether the data controller is me as an
> individual or my company (Surgeon Led AI Ltd), (c) the postal address to publish, (d) the contact
> email to publish (suggest `privacy@<domain>`), (e) Plausible or Cloudflare Web Analytics.
>
> **Step 2: ICO data protection fee**
> 1. Open `ai4qi/governance/ICO_REGISTRATION_ANSWERS.md` and read it.
> 2. Go to https://ico.org.uk/for-organisations/data-protection-fee/ and start the registration
>    ("Register (pay fee)").
> 3. Fill in every answer from the file, using my decisions from Step 1 for the controller name,
>    address and contact email. Choose tier 1 and pay by direct debit (£47) if offered.
> 4. **STOP** on the final review page. Show me the answers you entered, one per line, so I can
>    check them. I will make the declaration and pay.
> 5. When I have paid, note the registration number (it starts ZA or ZB) for Step 7.
>
> **Step 3: domain and hosting**
> Follow sections 1 and 2 of `ai4qi/governance/LAUNCH_RUNBOOK.md`: buy the domain in Cloudflare
> (**STOP** for payment), then create the Cloudflare Pages project from the GitHub repository with
> build output directory `ai4qi/app` and production branch as above. Add the custom domain. Check
> the home page loads.
>
> **Step 4: Supabase, Anthropic, Resend**
> Follow sections 3, 4 and 5 of the runbook exactly. **STOP** at each sign-up for me to accept the
> terms and data processing agreements and to turn on two-factor authentication.
>
> **Step 5: analytics**
> Section 7 of the runbook, with my choice from Step 1.
>
> **Step 6: save the paperwork**
> For each company I signed up with (Cloudflare, Supabase, Anthropic, Resend, Plausible if used),
> open its data processing agreement page (links in `ai4qi/governance/PROCESSORS_AND_TRANSFERS.md`)
> and help me save it as a PDF named `DPA-<company>-<date>.pdf` in a folder called
> "Ai4Qi paperwork". Remind me to read and sign `ai4qi/governance/DPIA.md` and save a signed PDF
> in the same folder.
>
> **Step 7: hand back (public values only)**
> Give me one short message, which I will paste to Claude Code, with exactly these lines and
> nothing secret:
> ```
> site_url: https://...
> owner_name: ...
> postal_address: ...
> contact_email: ...
> security_email: ...
> ico_number: ...
> legal pages take effect: <today's date>
> supabase_url: https://<ref>.supabase.co
> supabase_publishable_key: sb_publishable_...
> build_url: https://<ref>.supabase.co/functions/v1/build-audit
> analytics: plausible <domain>  |  cloudflare <token>  |  none
> ```
>
> **Step 8: final checks after Claude Code has pushed the update**
> 1. Open the live site. Check that the Terms, Privacy, Cookies and Accessibility pages show no
>    square-bracket placeholders.
> 2. Run the address through https://securityheaders.com and tell me the grade.
> 3. Sign in with a magic link (I will open the email), build one audit, and download the Excel
>    sheet and the Word proposal.
> 4. Tell me anything that failed.

---

## After Step 7 (in Claude Code)

Paste the Step 7 message into Claude Code with: "Fill these into ai4qi/app/config.json, rebuild,
run launch_check.py and push." `python3 ai4qi/launch_check.py` must print READY before launch.
