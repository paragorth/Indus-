# Sign-in email templates (Supabase Auth)

Supabase sends the sign-in emails, so their look is set in the Supabase dashboard, not in the app.

Supabase → **Authentication** → **Emails** → **Templates**:

| Template | Subject | Message body |
|---|---|---|
| **Magic Link** (someone who already has an account) | `Sign in to Ai4Qi` | the whole of `magic-link.html` |
| **Confirm signup** (first sign-in of a new email address) | `Create your Ai4Qi account` | the whole of `confirm-signup.html` |

Paste into the **Source** (HTML) view and save. Leave `{{ .TokenHash }}` and `{{ .Token }}` exactly as they are.

The email has a **Sign me in** button and, smaller, the code. The button opens
`https://ai4qi.com/?token_hash=...`; that page signs the person in only when they press **Sign me in**
there. Email security scanners (Microsoft Defender on NHSmail) open every link in an incoming email,
which used up one-time links before people clicked them; scanners never press buttons, so the sign-in
waits for the person. The code is for reading the email on a phone and signing in on another computer.
The link points to ai4qi.com (the sender's own domain), not to supabase.co.
