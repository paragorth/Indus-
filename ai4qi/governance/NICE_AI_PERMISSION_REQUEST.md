# Email to NICE: permission to use NICE content with AI

> Owner's draft. Send from your own email to **reuseofcontent@nice.org.uk**. When NICE agrees, set
> `"nice_ai_permission": true` in `ai4qi/app/config.json`, rebuild and push. Until then, the app
> never puts NICE wording into an AI prompt.

**Subject:** Request to use NICE recommendations with AI in a free clinical audit tool (Ai4Qi)

Dear NICE Reuse of Content team,

I am a UK clinician and I run Ai4Qi ([site address]), a free, non-commercial website. It helps
doctors in training and other clinicians plan and run clinical audits against national standards.

**How we use NICE content now, under the NICE UK Open Content Licence**
- The site shows the exact wording of individual NICE recommendations and quality statements that
  audits measure against. Each one carries NICE's attribution statement and disclaimer, and links
  to the source.
- We do not change the wording, imply endorsement or use NICE logos. No advertising appears next
  to NICE content.

**What we would like permission for (AI use)**
- **What it does.** "Build an audit" lets a clinician type a topic, for example "sepsis antibiotics
  in the emergency department". An AI model (Claude, made by Anthropic, used through its commercial
  API) then drafts an audit protocol: one measurable question, the standard, a pass definition, a
  data collection template, a timeline and a re-audit plan.
- **What we would send.** The prompt would include the exact wording of the few NICE
  recommendations most relevant to the topic, so the model picks the right standard. Today we send
  only the recommendation number and URL.
- **How the output is used and shown.**
  - The protocol is shown to that clinician on screen, clearly labelled as AI-drafted, with the
    NICE recommendation quoted exactly and attributed.
  - The NICE wording on screen always comes from our stored copy, never from the AI's output.
  - Clinicians are told to check the standard against the NICE source and to have their
    supervisor approve the protocol.
- **What we never send or do.**
  - No patient data is ever sent.
  - NICE content is not used to train any model. Under Anthropic's commercial terms, API inputs are
    not used for training.
- **Users and scale.** UK and Irish clinicians. We expect [number] builds a month at first.

We would also like to ask about access from Ireland. Some of our users work in Ireland; is further
permission needed for them to view NICE wording on the site?

Please let me know if you need anything else, or if there are conditions you would like us to
follow.

Kind regards,
[Owner name]
[Role, organisation]
[Contact email]
