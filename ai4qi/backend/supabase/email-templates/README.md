# Sign-in email templates (Supabase Auth)

Supabase sends the sign-in emails, so their look is set in the Supabase dashboard, not in the app.

Supabase → **Authentication** → **Emails** → **Templates**:

| Template | Subject | Message body |
|---|---|---|
| **Magic Link** | `Your Ai4Qi sign-in link` | the whole of `magic-link.html` |
| **Confirm signup** (first sign-in of a new email address) | `Confirm your email for Ai4Qi` | the whole of `confirm-signup.html` |

Paste into the **Source** (HTML) view and save. `{{ .ConfirmationURL }}` is Supabase's own placeholder:
leave it exactly as it is. The emails use only inline styles and tables, so they look the same in
Outlook, NHSmail, Gmail and Apple Mail. They go out through Resend (custom SMTP, sender
`Ai4Qi <signin@mail.ai4qi.com>`).
