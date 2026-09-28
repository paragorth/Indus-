# Ai4Qi — sponsorship and advertising policy

> **Draft for the owner's review — not legal advice.** Internal document for the owner, and for
> anyone who asks. Version 0.1, 28 September 2026. It describes how Ai4Qi could take sponsorship
> while it stays free, independent and safe for users' data. No sponsor exists yet.

## 1. Principles

1. **Ai4Qi stays free.** Sponsorship pays running costs. No feature is locked behind a sponsor.
2. **Sponsors never see user data.** The only exception: users who tick "Email me occasional offers
   from Ai4Qi's sponsors" (`consent_sponsors`) receive messages that **Ai4Qi sends**. Even then, no
   email address is given to a sponsor.
3. **No sponsor influence over content.** Sponsors have no say over audits, standards, library
   entries, rankings, the AI's instructions or which topics are proposed.
4. **Everything sponsored is labelled** as sponsored, and it is never mixed with audit content.

## 2. What a sponsor can have

| Offer | Detail |
|---|---|
| Name and logo on a **"Supporters"** page, and in the site footer | Labelled "Supported by". No product claims |
| **Aggregate figures** | Sign-ups by grade, specialty, region, work setting and audit purpose; number of NHS/HSE email sign-ups; numbers who opted in to news and sponsor offers. Taken from `signup_breakdown()`, which only admins can run and which hides groups smaller than five |
| **Sponsor messages to opted-in users** | Written or approved by Ai4Qi, sent by Ai4Qi from the `mailing_list('sponsors')` list, clearly marked as sponsored (§5) |
| Named support of a **non-clinical feature or event** (for example a webinar on audit method) | Labelled; content controlled by Ai4Qi |

**Not offered:** email addresses, names, profiles, individual usage, feedback comments, built-audit
themes, or any targeting of individuals. Sponsors get no access to the admin pages or the database.
Nothing that sponsors do affects the app's scoring or ranking.

## 3. Where sponsor material may appear

- **Allowed:** the Supporters page, the site footer, sponsor emails to opted-in users, and event
  pages Ai4Qi runs.
- **Not allowed:** library pages, standards, proposed or built audits, the "run an audit" pages,
  reminder emails, and the sign-in email. Two reasons:
  - It keeps audit content visibly independent.
  - **The NICE UK Open Content Licence forbids showing NICE content "next to any advertising or
    promotional text".** Most Ai4Qi audit pages quote NICE.
- Nothing about a medicine or device appears in the app itself.

## 4. Editorial independence (written agreement)

Every sponsor signs a short agreement before any money or benefit is received. It must say:

- the sponsor has **no right to review, approve or change** any Ai4Qi content, and Ai4Qi can end
  the arrangement if it tries;
- the sponsor receives **aggregate data only**, as listed in §2, and never personal data;
- the sponsor will not send, or ask anyone else to send, marketing to Ai4Qi users. Sponsor
  messages go only through Ai4Qi;
- how the sponsorship is described in public (wording agreed, and no suggestion that NICE, any
  college or the NHS endorses the sponsor);
- the amount, the term, and the declarations each side must make (§6, §7).

## 5. Labelling (ASA/CAP Code)

The CAP Code applies to marketing on websites and in emails. Section 2 requires:

- 2.1 — "Marketing communications must be obviously identifiable as such."
- 2.2 — unsolicited marketing emails "must be obviously identifiable as marketing communications
  without the need to open them".
- 2.3 — marketing must "make clear their commercial intent, if that is not apparent from the
  context".
- 2.4 — advertorials must be made clear, "for example, by heading them 'advertisement feature'".

**Ai4Qi's practice:**

- Supporter logos carry "Supported by".
- Any sponsored page or article is headed "**Sponsored by [Company]**" and says that the sponsor
  paid for it.
- Every sponsor email has a subject line starting "**Sponsored:**" and names the sponsor in the
  first line.
- Ai4Qi never presents sponsor material as its own recommendation or as clinical guidance.

## 6. Email consent (PECR and UK GDPR)

- **The law.** Regulation 22 of PECR prohibits sending, or instigating, unsolicited direct
  marketing emails to individuals unless the recipient has consented "to such communications being
  sent by, or at the instigation of, the sender". The ICO says consent must be freely given,
  specific, informed and unambiguous, must "give your name in the consent request", and must not
  use pre-ticked boxes. People can withdraw consent at any time, and every message must give a way
  to opt out. A sponsor that asks Ai4Qi to send its message is likely to be an **instigator**, so
  both are responsible.
- **What the app already does.** Separate, unticked boxes for news and for sponsor offers, at
  sign-in and on the account page. The time of the last change is stored (`consent_updated_at`).
  Only admins can pull the list (`mailing_list('sponsors')`).
- **To add before the first sponsor email:**
  - List current sponsors by name next to the consent box and on the Supporters page. The ICO says
    consent that names only "trusted partners" or similar is not valid when relied on by another
    organisation. Naming them is the safer course.
  - Put an unsubscribe link and a contact address in every sponsor email, and act on opt-outs
    straight away (untick `consent_sponsors`).
  - Keep a record of each send: date, sponsor, text, and number of recipients.
  - Send no more than [one message a month].
- Reminder and sign-in emails are service messages and must never carry sponsor content.
- **UK GDPR.** The lawful basis for sponsor emails is consent (already stated in the privacy
  notice). Sponsors receive no personal data, so no data sharing agreement is needed.

## 7. Pharmaceutical sponsors (ABPI Code of Practice 2024)

The ABPI Code binds ABPI members and companies that have agreed to it, not Ai4Qi. But a compliant
company will insist on these points, and they protect Ai4Qi too.

- **Donations and grants cannot go to an individual.** Clause 23.1: "Donations and grants to
  individuals are prohibited." Under clause 1.22, sponsorship is support for an activity "performed,
  organised, created, etc. by a healthcare organisation, patient organisation or other independent
  organisation". **Ai4Qi is run by an individual clinician today, so it needs a legal entity (for
  example a company or community interest company) before taking pharmaceutical money.**
- **Declaration.** Clause 5.6: material on human health or diseases that a company sponsors "must
  clearly indicate the role of that pharmaceutical company". The supplementary information says
  the declaration must be "sufficiently prominent" so that readers know about it "at the outset".
- **No disguised promotion.** Clause 15.6: "Promotional material and activities must not be
  disguised." Material that a company pays for "must not resemble independent editorial matter".
- **No advertising of prescription-only medicines on Ai4Qi.** Clause 26.1: "Prescription only
  medicines must not be advertised to the public." Ai4Qi pages are public, with no check that the
  reader is a health professional. Clause 16.1 (supplementary information) requires a company
  website or company-sponsored website to keep health-professional sections clearly separate from
  public sections. **Ai4Qi's policy is corporate sponsorship only, with no product or brand
  advertising.**
- **Emails.** Clause 15.5: emails "must not be used for promotional purposes, except with the prior
  permission of the recipient". Ai4Qi's sponsor emails from pharmaceutical companies are limited
  to non-promotional items, such as courses or events, and never mention a medicine.
- **Transparency.** Under clause 23.2 a written agreement is needed, and donations and grants must
  be publicly disclosed each year (clauses 28 and 29). Expect the sponsor to publish the payment.

## 8. Medical device and health technology sponsors (ABHI Code)

The ABHI Code of Ethical Business Practice (July 2019 edition, the current one on the ABHI site)
binds ABHI members:

- "A Member Company shall not provide Grants or Charitable Donations to individual Healthcare
  Professionals." Payment goes "directly to the qualifying organisation", in its name. **This is
  another reason for Ai4Qi to become a legal entity first.**
- Grants need a written request, a signed agreement, and an independent review inside the company.
- Educational grants are "restricted" to a stated purpose, the company may check how they are used,
  and they are publicly disclosed.
- Advertising packages (such as advertisement space) are treated as normal commercial transactions.
  Ai4Qi's no-product-advertising rule still applies.

## 9. Conflicts of interest

The owner declares any personal link with a sponsor (employment, consultancy, shares, hospitality)
on the Supporters page and in the register. Anyone who writes or reviews Ai4Qi content declares the
same, and does not review content in an area where they have a sponsor link.

## 10. Sponsor register (template; publish the first five columns)

| Sponsor (legal name) | Type (pharma / device / education / other) | What they receive | Amount or value | Term (from–to) | Agreement signed (date) | Code notes (ABPI/ABHI clause, disclosure) | Owner's conflicts declared | Emails sent (dates, recipients) | Reviewed by / date |
|---|---|---|---|---|---|---|---|---|---|
| [Name] | | Supporters page; aggregate figures | £ | | | | None / [details] | | |

Review the register every 6 months and whenever a sponsor is added or leaves.

## Sources checked (28 Sep 2026)

- CAP Code, section 2 (Recognition of marketing communications): https://www.asa.org.uk/type/non_broadcast/code_section/02.html
- PECR, regulation 22 (latest revised version): https://www.legislation.gov.uk/uksi/2003/2426/regulation/22
- ICO, Electronic mail marketing (Guide to PECR; marked as under review after the Data (Use and Access) Act 2025): https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/guide-to-pecr/electronic-and-telephone-marketing/electronic-mail-marketing/
- ICO, How do we comply with the PECR electronic mail marketing rules?: https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/guidance-on-direct-marketing-using-electronic-mail/how-do-we-comply-with-the-pecr-electronic-mail-marketing-rules/
- ICO, Direct marketing guidance (latest update 28 April 2026): https://ico.org.uk/for-organisations/direct-marketing-and-privacy-and-electronic-communications/direct-marketing-guidance/
- ABPI Code of Practice 2024 (in operation from 1 October 2024), clauses 1.22, 5.6, 15.5, 15.6, 16.1, 23, 26.1, 28: https://www.pmcpa.org.uk/media/r0anf5ya/2024-abpi-code.pdf
- ABHI Code of Ethical Business Practice, July 2019, Chapter 4: https://www.abhi.org.uk/media/0y3fmw2s/abhi-code-of-business-practice-july-2019-final.pdf (listed at https://www.abhi.org.uk/code-of-ethical-business-practice/coebp-documents/)
- NICE UK Open Content Licence (advertising restriction): https://www.nice.org.uk/reusing-our-content/nice-uk-open-content-licence
- App code checked: `backend/supabase/migrations/006_consent_profile.sql` (consent fields, `signup_breakdown`, `mailing_list`), `app/app.js` (consent boxes).
