# Putting Ai4Qi live (Cloudflare Pages, free)

The site is the folder `ai4qi/app/`: plain files, no build server needed.

1. Build the data: `python3 ai4qi/build_app_data.py` (already done in the repo).
2. Cloudflare dashboard → Workers & Pages → Create → Pages → Connect to Git → pick this repository and
   branch. Build command: none. Build output directory: `ai4qi/app`. Deploy.
   (Or upload the folder directly: Pages → Upload assets → drag `ai4qi/app`.)
3. Custom domain: Pages project → Custom domains → add `ai4qi.<your domain>` or the root domain.
   Cloudflare issues the HTTPS certificate automatically.
4. `_headers` in the folder sets the security headers (strict content policy, HSTS, no framing).
5. After Supabase is set up, fill `ai4qi/app/config.json` (`supabase_url`, `supabase_anon_key`,
   `build_url`), rebuild, and push; Pages redeploys on every push.
6. In Supabase → Authentication → URL configuration, set Site URL to the live address and add it to
   Redirect URLs, so sign-in links return to the site.

Netlify works the same way: publish directory `ai4qi/app`, no build command; it also reads `_headers`.
