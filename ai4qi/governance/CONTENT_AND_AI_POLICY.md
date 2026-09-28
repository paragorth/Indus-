# Ai4Qi — content and AI policy

> **Draft for the owner's review — not legal advice.** Internal document for the owner, and for
> anyone who asks. Version 0.1, 28 September 2026. It covers how library content is sourced and shown,
> how proposed audits are designed, how AI is used, and how errors and takedown requests are
> handled.

## 1. Library content: sources and presentation

| Content | How it is shown | Rule |
|---|---|---|
| Published audits (papers, conference abstracts) | A summary in our own words (setting, sample, results, change, re-audit) with the citation and a DOI, PubMed or publisher link | Facts and numbers are not protected by copyright. We write our own sentences and always link to the source |
| Abstract text, verbatim | Only when the paper's licence is CC BY or CC0. The page states: "Abstract reproduced verbatim under the article's [licence] licence." | Anything else is summarised and linked, never copied |
| Figures | Only CC BY or CC0 figures (held in `figures/`), each shown with its credit line | The credit line must always be shown |
| Topic cards | Seed cards from the team; draft cards labelled "Draft, needs consultant sign-off" until signed off | Draft status stays visible |
| National standards (NICE, royal colleges, national audits, NHS England, HSE) | The exact wording, marked as a quotation, with the source named and a link | See §2 |

"Not reported" in a record means the paper did not say. We never fill it in.

## 2. Quoting standards: the legal position

**The quotation exception.** Under section 30(1ZA) of the Copyright, Designs and Patents Act 1988,
quoting a work does not infringe copyright if: (a) the work has been made available to the public;
(b) the use is fair dealing; (c) the quotation is no longer than the specific purpose needs; and
(d) it has a sufficient acknowledgement. Section 30(4) makes contract terms that try to stop such
quotation unenforceable. So Ai4Qi quotes **only the sentence or recommendation that defines the
audit standard**, names the source and links to it. It does not reproduce whole guidelines,
tables or algorithms. Fair dealing is judged case by case, and a large, systematic collection of
quotations is harder to defend than a single one. That is why the licences below matter too.

**NICE.** NICE content can be reused in the UK under the **NICE UK Open Content Licence**. It is
free, and it covers commercial and non-commercial use without an application. Its conditions:

- **Wording as published.** Do not "amend or adapt the wording or structure" of NICE
  recommendations or quality statements. Ai4Qi quotes them exactly.
- **Attribution and disclaimer in the product.** The required form is: "© NICE [YEAR] TITLE.
  Available from www.nice.org.uk/guidance/ngXX All rights reserved. Subject to Notice of rights.
  NICE guidance is prepared for the National Health Service in England. All NICE guidance is
  subject to regular review and may be updated or withdrawn. NICE accepts no responsibility for
  the use of its content in this product/publication." Where possible, link to the licence and the
  source. **Action:** the app does not show this statement yet. Add it next to each NICE standard,
  or once on each page that quotes NICE plus on an "About the standards" page.
- **No advertising next to NICE content.** The licence forbids showing the information "next to
  any advertising or promotional text". This limits where sponsor material can go (see
  `SPONSORSHIP_POLICY.md`).
- **No implied endorsement** by NICE, and no use of NICE logos.
- **UK only.** Outside the UK, "except for personal use, study or personal research", NICE content
  needs NICE's written agreement, which may involve a licence and fee. Irish users can see the
  site. **Action:** ask NICE about this.
- **AI use is not covered.** The licence says requests to use NICE content "for artificial
  intelligence (AI) purposes in the United Kingdom and internationally are not covered". Such use
  needs written permission (email reuseofcontent@nice.org.uk and explain how AI is used, how the
  outputs are used and where they appear). NICE's terms say every such request goes through an
  approval process. **Ai4Qi's "Build an audit" sends library excerpts, which can include NICE
  standard wording, to an AI model, so this applies.** **Action before launch:** either send the
  request to NICE now and leave NICE wording out of build prompts until NICE replies, or rely on
  the quotation exception only after taking advice. Whichever you choose, record it in the
  register (§7).

**Royal colleges: RCOG as the example.** RCOG's rights and permissions policy says RCOG material
"must not be adapted, reproduced, or translated without the written permission of the RCOG".
Reuse for "charitable, non-commercial and educational purposes is usually granted free of charge".
Its reproduction page asks people to request permission from copyright@rcog.org.uk and promises a
reply within 2 weeks. RCOG logos must not be used, and no endorsement may be implied. **Ai4Qi's
approach:** short quotations under s30(1ZA), with "Source: RCOG, [title], [year]" and a link. Ask
RCOG for written permission, because Ai4Qi quotes several standards systematically. Take the same
approach with other colleges (for example RCEM, whose code of conduct also requires written
permission to reproduce college educational material). Check each college's terms before adding
its standards.

## 3. How proposed audits are designed

- Every proposed audit (ONA-xxx, NNA-xxx) and every built audit has **one plain, measurable
  question**, the **exact standard wording** fetched from the source, the **source named and
  linked**, a pass definition, sample, timeline, change and re-audit plan.
- Where no national standard exists, the standard line says **"Local standard"**. A local target
  is never presented as national.
- Standards are quoted from `standards/standards.json`, whose entries were fetched from the source
  pages. We never overstate them (for example, NICE says "regularly", not "daily").
- **Proposed means not yet run.** The labels "Proposed – not yet run" and "Built for you – not
  yet run" appear on every such audit. Proposed audits are ideas, never evidence. Evidence lines
  cite only real library entries; lines citing ids that are not in the library are removed
  automatically.
- User feedback (thumbs up/down and fixed reasons) is used to rank and improve proposed audits.

## 4. AI use: what users are told

Put this text (or a close version) on every built audit and on an "About Ai4Qi" page:

> **How this was made.** This protocol was drafted by an AI model (Claude, made by Anthropic) from
> the topic you typed, using published audits and standards from the Ai4Qi library. It is a
> draft. Check the standard against its linked source, and ask your supervisor to review the
> protocol before you collect data. Your audit records are never sent to the AI or to Ai4Qi.

What is true today, and must stay true:

- **What the AI builds:** audit protocols (question, standard, sample, data fields, change,
  re-audit, pitfalls). It does not see, analyse or write about audit records.
- **What is sent:** the typed topic plus excerpts from the Ai4Qi library, through Ai4Qi's server
  function, to Anthropic's API. No email address, no user id and no patient data are sent. The
  function refuses requests that are not about audits.
- **Inside Claude (claude.ai):** building uses the viewer's own Claude account, under their own
  terms with Anthropic.
- **Output is a draft.** People check it. Ai4Qi does not require any extra approval from users;
  their normal audit registration and supervision apply.
- **Review:** the owner reviews saved builds (`built_audits`) and adds good ones to the library
  only after checking them.
- **Gap to fix:** built pages show "Built for you – not yet run" but do not yet say that AI wrote
  them or name Anthropic. Add the text above before launch.

## 5. Errors: reporting and correction

**How users report:**

- The thumbs-down button on any proposed or built audit, with a reason and an optional comment of
  up to 500 characters.
- Email to [Contact email], linked from every page footer, with the subject "Correction".

**How the owner corrects:**

| Severity | Example | Target |
|---|---|---|
| Serious | Wrong or out-of-date standard wording, a standard presented as national when it is not, an invented source | Hide or fix within **2 working days** of the report |
| Moderate | Wrong numbers in a summary, a broken link, a misattributed paper | Fix within **10 working days** |
| Minor | Typos, wording | Next routine update |

Library ids never change, so a corrected entry keeps its id. Each correction is recorded in a
corrections log (date, item, what changed, who reported it, if they agree to be named). Where a
standard has changed since it was fetched, update the wording and the "accessed" date.

## 6. Takedown and correction procedure (rights holders and authors)

1. **Contact:** [Contact email] or [Postal address]. Please say what the content is, where it
   appears, and why (copyright, inaccuracy, or other).
2. **Acknowledge** within **2 working days**.
3. **Interim action:** if the request is from a rights holder or an author and is not plainly
   unfounded, **hide the item within 5 working days** while it is reviewed. If a clear error could
   mislead clinicians, hide it at once.
4. **Decide** within **10 working days** of the request: remove, replace the verbatim text with
   a summary and link, correct, or keep with reasons. Tell the requester the outcome.
5. **Record** it in the register (§7). Remove the item from the data files as well as the page,
   so later builds do not bring it back.

## 7. Register (keep as a spreadsheet)

| Date | Source or item | Type (licence / permission / correction / takedown) | Decision | Done by | Date closed |
|---|---|---|---|---|---|
| | NICE UK Open Content Licence | licence | Attribution to add; AI-use request sent [date] | | |
| | RCOG | permission | Request sent [date] | | |

## Sources checked (28 Sep 2026)

- Copyright, Designs and Patents Act 1988, s30 (latest revised version): https://www.legislation.gov.uk/ukpga/1988/48/section/30
- NICE, Terms and conditions, Notice of rights, clauses 13 and 18: https://www.nice.org.uk/terms-and-conditions
- NICE UK Open Content Licence: https://www.nice.org.uk/reusing-our-content/nice-uk-open-content-licence
- NICE, Reusing our content: https://www.nice.org.uk/re-using-our-content
- RCOG, Rights and permissions policy 2024: https://www.rcog.org.uk/about-us/policies/rights-and-permissions/
- RCOG, Reproducing RCOG guidance and patient information: https://www.rcog.org.uk/guidance/reproducing-rcog-guidance-and-patient-information/
- RCOG, Terms and conditions: https://www.rcog.org.uk/legal/terms-conditions/
- RCEM, Code of conduct (search summary only; page not read in full): https://rcem.ac.uk/code-of-conduct/
- App code checked: `app/app.js` (licence note, badges, feedback reasons), `backend/supabase/functions/build-audit/index.ts`.
