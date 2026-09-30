# Sign-in email templates (Supabase Auth)

Supabase sends the sign-in emails, so their look is set in the Supabase dashboard, not in the app.

Supabase → **Authentication** → **Emails** → **Templates**:

| Template | Subject | Message body |
|---|---|---|
| **Magic Link** (someone who already has an account) | `Welcome back: sign in to Ai4Qi` | the whole of `magic-link.html` |
| **Confirm signup** (first sign-in of a new email address) | `Welcome to Ai4Qi: create your account` | the whole of `confirm-signup.html` |

Paste into the **Source** (HTML) view and save. The button links to
`https://ai4qi.com/?token_hash={{ .TokenHash }}&type=email`, not to Supabase's own address: the site
checks the token itself. NHSmail (Microsoft) treats a sign-in email whose link goes to a different
domain from the sender as possible phishing, and its link scanner can use up a one-time Supabase
link before the person clicks it. Leave `{{ .TokenHash }}` and `{{ .Token }}` exactly as they are. The emails use only inline styles and tables, so they look the same in
Outlook, NHSmail, Gmail and Apple Mail. They go out through Resend (custom SMTP, sender
`Ai4Qi <signin@mail.ai4qi.com>`).

Both emails show the button and a 6-digit code (`{{ .Token }}`). The code can be typed on the sign-in page, so
people can open the email on their phone and still sign in on a Trust computer.
