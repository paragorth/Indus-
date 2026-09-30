# Sign-in email templates (Supabase Auth)

Supabase sends the sign-in emails, so their look is set in the Supabase dashboard, not in the app.

Supabase → **Authentication** → **Emails** → **Templates**:

| Template | Subject | Message body |
|---|---|---|
| **Magic Link** (someone who already has an account) | `Your Ai4Qi sign-in code` | the whole of `magic-link.html` |
| **Confirm signup** (first sign-in of a new email address) | `Your Ai4Qi code` | the whole of `confirm-signup.html` |

Paste into the **Source** (HTML) view and save. Leave `{{ .Token }}` exactly as it is.

The emails carry **only the 6-digit code: no link, no button, no image**. ai4qi.com was registered on
28 Sep 2026, and Microsoft filtering (NHSmail and Trust Microsoft 365 tenants) put the earlier emails,
which had a sign-in button, into admin-only quarantine as suspected phishing, although Resend showed
them delivered. A code-only message carries none of those signals. People type the code on the
sign-in page, which also lets them read the email on a phone and sign in on a Trust computer.

The app still accepts `?token_hash=` links (see AUTH_RETURN in app.js) if a link is ever added back
once the domain has a sending reputation.
