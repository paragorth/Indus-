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

## iPhone (App Store)

Apple does not accept a plain website in a wrapper (guideline 4.2). Plan: a small native shell
(Capacitor) that opens ai4qi.com and adds phone notifications for due audit steps, built and uploaded
from the cloud (no Mac needed) once the Apple Developer account is approved.

## Store material: `store/`

`LISTING.md` (names, descriptions, data-safety answers, content rating), phone screenshots,
`feature-graphic-1024x500.png` and `icon-512.png`.
