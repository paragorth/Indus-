# Microsoft Copilot in the NHS: what Ai4Qi can use (researched 27 Sep 2026)

Claims marked [snippet] come from search-engine snippets of NHSmail support pages that are now
behind a login (supporthub.nhs.net); [inferred] is our reading, not a published statement.

## Who has what

- **Microsoft 365 Copilot Chat (free)**: on for every NHS.net Connect (NHSmail) user. Web-grounded
  (Bing), enterprise data protection, file upload including .xlsx/.csv. Prompts stored in the UK for
  180 days. Source: NHS.net Connect Copilot Chat DPIA v1.0 (Oct 2025)
  https://comms-mat.s3.eu-west-1.amazonaws.com/Comms-Archive/Data+Protection+Impact+Assessment+-+Copilot+Chat.pdf
- **Paid Microsoft 365 Copilot** (in Word, Excel, PowerPoint, Teams): national deal 8 Jun 2026,
  505,000 staff, each trust typically starting with ~2,000 licences, rollout expected by Oct 2026.
  https://www.england.nhs.uk/2026/06/500000-nhs-staff-to-get-new-artificial-intelligence-tools-to-help-free-up-more-time-for-patients/
- **Copilot in Excel** needs the paid licence (Microsoft removed in-app Copilot for unlicensed users
  in large organisations from 15 Apr 2026). Free Copilot Chat can still read an uploaded workbook.
- Trusts on their own Microsoft 365 tenants set their own rules. Scotland/Wales: no public policy
  found. Northern Ireland: 1,000 licences issued (BSO annual report 2025–26). HSE Ireland: Copilot
  Chat webinars for all staff [inferred availability].

## Rules for staff

- Acceptable use policy v1.2 (Dec 2025): Copilot may be used for "administrative and business support
  purposes including Clinical Administration activities with sensitive information"; it "must not be
  used for any clinical decision-making or direct patient care"; outputs must be labelled and reviewed.
  https://comms-mat.s3.eu-west-1.amazonaws.com/Comms-Archive/M365+Copilot+Acceptable+Use+Policy+v1.1.pdf
- Agents: "Not allowed: Patient data and clinical records"; non-clinical and administrative use only.
  https://comms-mat.s3.eu-west-1.amazonaws.com/Comms-Archive/Copilot+Extensibility+Acceptable+Use+Policy.pdf
- Local trust IG policy applies on top; many trusts are stricter [inferred].

## Agents and integrations in NHS.net

| Route | Status | Approval needed |
|---|---|---|
| Paste a prompt into Copilot Chat | Works for everyone | None |
| Copilot reads a public web page (Bing grounding) | On | None, but not guaranteed to fetch a given page |
| Upload a (non-patient) file into Copilot Chat | Works for everyone | None |
| Personal agent via Agent Builder ("Create an agent") | Paid licence only [snippet] | None for a personal agent without connectors [inferred] |
| SharePoint / declarative agents | Enabled, paid licence needed to use | Local governance |
| Third-party agent from the Agent Store / Teams app | Admin only: NHSmail Technical Design Authority "application hurdle assessment" [snippet] | National + likely local IG |
| API plugins, custom connectors, MCP | Blocked by default; DLP exception with business case [snippet] | Local admin + national |
| Copilot Studio | Separate onboarding: licences, credits, environment, DPIA, DCB0160 [snippet] | Organisation-level |

## What this means for Ai4Qi

1. **Now, no approval:** a copy-paste "Ai4Qi audit builder" prompt for Copilot Chat, public plain-HTML
   pages for each protocol and standard (so Copilot can find and cite them), and downloadable
   non-patient knowledge files (template, standard, instructions) users can upload into a chat.
2. **Paid-licence users (growing to ~505,000):** instructions to make a personal Ai4Qi agent in
   Agent Builder from our instructions + public URL + knowledge file, with no connectors.
3. **Later, with approval:** a declarative agent package submitted through Microsoft Partner Center and
   the NHSmail application assessment. Slow; only worth it once Ai4Qi has users asking for it.
4. **Always:** patient data never goes into Copilot via Ai4Qi. Trainees fill the Excel data sheet on
   Trust systems; Copilot in Excel (paid) can then summarise it inside the NHS tenant. Ai4Qi only needs
   totals. Present Ai4Qi as clinical audit administration, not decision support.
