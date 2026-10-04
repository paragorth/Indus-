# Ai4Qi mobile apps

## Android (Google Play): `android/`

A Trusted Web Activity: the Play app opens https://ai4qi.com full screen in Chrome's engine, so every
change to the website is in the app at once (no app update needed). Package `com.ai4qi.app`.

Build (needs JDK 17+ and the Android SDK; `local.properties` has `sdk.dir=`):

    cd ai4qi/mobile/android
    gradle bundleRelease      # app/build/outputs/bundle/release/app-release.aab  -> upload to Play
    gradle assembleRelease    # app/build/outputs/apk/release/app-release.apk     -> install on a phone to test

Signing: `keystore.properties` (never committed) points at the upload key `upload.jks`. The owner keeps
both in a password manager. Play App Signing holds the real app key; if the upload key is ever lost,
Google can reset it.

Before each new Play release, raise `versionCode` (and `versionName`) in `app/build.gradle`.

**Full screen without a browser bar** needs `app/.well-known/assetlinks.json` on the website to list the
key fingerprints. It lists the upload key now. After the first upload, copy the **App signing key
certificate SHA-256** from Play Console > Test and release > App integrity, add it to the
`sha256_cert_fingerprints` list, and push.

## iPhone (App Store): `ios-shell/`

A Capacitor shell (`com.ai4qi.app`) that opens https://ai4qi.com and adds what a website cannot do on an
iPhone, so it is more than a wrapper (Apple guideline 4.2):
- **Phone reminders** for each audit step (local notifications at 9am on the due date; `nativeRemind` in app.js).
- **Files through the share sheet** (Excel, Word, PowerPoint, calendar): save to Files, AirDrop, email (`nativeSave`).

Built in the cloud, no Mac: GitHub Actions > "Ai4Qi iPhone build" (`.github/workflows/ai4qi-ios.yml`) archives,
signs automatically and uploads to App Store Connect (TestFlight). Needs repository secrets ASC_KEY_ID,
ASC_ISSUER_ID, ASC_KEY_P8 and APPLE_TEAM_ID. iPhone only (no iPad screenshots needed).

## Store material: `store/`

`LISTING.md` (names, descriptions, data-safety answers, content rating), store screenshots
(`play-1..6.png` 1080x1920 for Google Play, `ios-1..6.png` 1290x2796 for the App Store; source `shots.html`),
`feature-graphic-1024x500.png` and `icon-512.png`.
