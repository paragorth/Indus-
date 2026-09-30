# Cookies and storage

Last updated: [DATE]

This page explains exactly what Ai4Qi stores on your device, and why.

## The short version

- Ai4Qi sets **no cookies** of its own.
- There are **no advertising, tracking or social media cookies**, and nothing that follows you
  across other websites.
- Ai4Qi uses your browser's storage only for the features you choose to use. Most of it is how
  Ai4Qi keeps your audit records **on your device instead of on our servers**.
- Our visitor statistics are cookieless and store nothing on your device.

Because everything we store is needed for a feature you have asked for, we do not show a cookie
banner.

## What Ai4Qi stores in your browser

Browsers offer two kinds of storage. **Local storage** stays until it is deleted. **Session
storage** is cleared when you close the browser (or the tab).

| Name | What it holds | Why | How long |
|---|---|---|---|
| `ai4qi_vault_v1` | How your audits are protected: the device key they are encrypted with (or, for audits protected by an older version's passcode, a random "salt" and a check value, never the passcode itself, until they are next opened); and whether you chose shared-computer mode | To unlock your audits with your passcode | Until you erase your audits. In shared-computer mode: session storage, cleared when the browser closes |
| `ai4qi_runs_enc_v1` | Your audits and their de-identified records, **encrypted** with the device key | To keep your audit on your device, not on our servers | As above |
| `ai4qi_runs_v1` | Audits saved by an older version of Ai4Qi, before encryption was added | Moved into the encrypted store, and then deleted, when you next open My audits | Until you next open My audits or erase your audits |
| `ai4qi_built_v1` | The last 30 audit protocols you built (no patient data) | So you can go back to them | Until you clear your browser's site data |
| `ai4qi_feedback_v1` | Your thumbs up or down, reasons and any comment on proposed audits | To send your feedback when you are online, and show that you have already given it | Until you clear your browser's site data |
| `ai4qi_device_v1` | A random device number. It is created only when feedback is sent | To limit spam and count one vote per device | Until you clear your browser's site data |
| `ai4qi_consent_pending` | Whether you ticked the news and sponsor email boxes when asking for a sign-in link | To save those choices to your account once you open the link | Deleted once your choices are saved to your account |
| `ai4qi_me_v1` | Your name, role, email and hospital, and your supervisor's name and email, if you type them into "Send to your supervisor" | So you do not have to type them again | Until you clear your browser's site data |
| `ai4qi_demo` | The value "1" when demo mode is on | To show the "Demo" buttons until you turn demo mode off | Until you turn demo mode off |

**Demo mode.** Example audits made in demo mode are fictitious and are saved like any other audit.
If you use the demo passcode, they are kept in session storage and cleared when the browser closes.
You can remove them at any time with **Clear example audits**.

## If you sign in

Signing in is optional. If you sign in, our account provider's code (Supabase) stores:

| Name | What it holds | Why | How long |
|---|---|---|---|
| `sb-…-auth-token` | Your sign-in session: access and refresh tokens and basic account details, such as your email address | To keep you signed in | Until you sign out, or delete your account |
| `sb-…-auth-token-code-verifier` | A one-time secret used while you sign in | To check that the sign-in link is completed in the same browser that asked for it | Removed when sign-in completes |

The "…" is our project's name at Supabase.

## Files the app keeps for offline use

Ai4Qi saves a copy of its own pages, scripts, fonts and library data in your browser's cache (using
a "service worker"). This lets it load quickly and work without a connection. The cache holds no
personal data. It is replaced when Ai4Qi is updated.

## Visitor statistics

We count visits with Cloudflare Web Analytics. It sets **no cookies** and stores nothing in your
browser, and it does not follow you across other websites. It records the page viewed, the referring
website, browser and device type, the country and how quickly the page loaded.

## No other cookies

Ai4Qi loads no advertising, social media or tracking code. All the scripts and fonts that Ai4Qi
runs are served from our own website, and the site's security settings block scripts from anywhere
else, except our visitor statistics service.

If you use Ai4Qi inside Claude (claude.ai), Claude's own website may store its own data in your
browser. That is covered by Anthropic's own policies, not by this page.

## Why we do not ask for consent

The law on storing information on your device is set out in the Privacy and Electronic
Communications Regulations (PECR). Consent is not needed where storage is **strictly necessary** to
provide a service you have asked for. Keeping your audit records, your protection settings, your
built protocols, your feedback, your sign-in and your demo-mode choice all fall within this: each
one exists only because you used that feature, and the feature cannot work without it.

Our visitor statistics store nothing on your device, so these rules do not require consent for them.

## How to remove what is stored

- **Your audits:** open an audit and choose **Delete this audit and its data from this device**,
  choose **This is a shared computer** under Protection on My audits to delete them when the browser
  closes, or clear your browser's site data for Ai4Qi.
- **Demo mode:** choose **Turn off** in the demo bar, or **Turn off demo mode** in the account menu.
- **Sign-in:** choose **Sign out** on your account page.
- **Everything:** clear your browser's "site data" (or "cookies and site data") for Ai4Qi. This also
  deletes your audit records. **Download a backup first** if you need them.

## Questions

Email [CONTACT EMAIL]. You can read more about how we use personal data in our
[privacy notice](#/privacy-notice).
