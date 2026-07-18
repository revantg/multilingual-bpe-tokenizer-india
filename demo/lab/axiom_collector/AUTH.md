# Axiom Learning OS — headless auth map

Host: `https://axiom.theschoolofai.in`  
Course (objective): `cmq97i5kn032208o8xu5dab4q`

## How auth works

| Piece | Detail |
|-------|--------|
| Stack | Next.js app + **NextAuth** |
| Browser login | Google OAuth (`revantgupta2@gmail.com` → student role) |
| API auth model | **Same-origin cookie**, not `Authorization: Bearer …` |
| Required cookie | `__Secure-next-auth.session-token` (HttpOnly, Secure) |
| Optional cookies | `__Host-next-auth.csrf-token`, `__Secure-next-auth.callback-url` appear in some browser curls; **not required** for the JSON GETs below |
| Headers that matter | `Cookie: __Secure-next-auth.session-token=…` and `Accept: application/json` |
| Headers that do **not** matter | `User-Agent`, `Referer`, `Sec-*`, CSRF cookie (proven via minimal curl) |

Session cookie value is a JWE (`alg=dir`, `enc=A256GCM`). It is opaque client-side; the server decrypts it. Do not commit live tokens.

## Probe endpoint

`GET /api/auth/session`

| Cookie | HTTP | Body |
|--------|------|------|
| Valid session-token | 200 | `{ "user": { name, email, image, id, role }, "expires", "loginAt" }` |
| Missing / invalid | 200 | `{}` (empty object — **not** 401) |

Use this as a soft probe: empty object ⇒ treat as unauthenticated.

## Collect endpoints (course page)

Browser course page fires (among others):

1. `GET /api/auth/session`
2. `GET /api/courses`
3. `GET /api/courses/{courseId}`
4. **`GET /api/courses/{courseId}/sessions`** ← sessions + lessons + nested quizzes/assignments stubs
5. **`GET /api/assignments?courseId={courseId}`** ← full assignment list for the course
6. `GET /api/quizzes?courseId={courseId}` (out of scope)
7. Per-assignment `…/submission` and `…/grade` (out of scope)

Cleanest headless path for the objective: **only (4) and (5)**, after resolving a valid cookie. Optional: (1) for a clear “who am I” check before collect.

### Auth behavior on collect endpoints

| Cookie | Sessions / Assignments |
|--------|-------------------------|
| Valid session-token only | **200** + JSON array |
| Missing cookie | **401** `{"error":"Unauthorized"}` |
| Garbage token | **401** `{"error":"Unauthorized"}` |

Same cookie authenticates every same-origin `/api/*` call; no per-request signature or CSRF for these GETs.

## Obtaining a cookie (one-time, browser)

1. Sign in at `https://axiom.theschoolofai.in` (Google).
2. Export `__Secure-next-auth.session-token` from Firefox:
   - DevTools → Storage → Cookies, or
   - Profile `cookies.sqlite` (`moz_cookies` where host `axiom.theschoolofai.in`), or
   - Browser network panel → any `/api/*` request Cookie header.
3. Store **outside git**, e.g.:
   - env `AXIOM_SESSION_TOKEN=<token>`
   - or file path in `AXIOM_COOKIE_FILE` / `--cookie-file` containing either the raw token or a full `Cookie:` header line.

Tokens expire (`expires` in session JSON, ~30 days from last refresh in practice). On 401, refresh via browser and re-export.

## Non-goals (auth)

- Automating Google OAuth / storing Google passwords.
- Using `login.axiom.co` (observability product) cookies for Learning OS.
- Committing live session tokens or personal course payloads.
