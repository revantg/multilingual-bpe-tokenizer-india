# Axiom headless course collector

Fetch **sessions** and **assignments** for an Axiom Learning OS course using only HTTP + a NextAuth session cookie (no browser at collect time).

## Auth

See [AUTH.md](./AUTH.md). Required secret:

```bash
export AXIOM_SESSION_TOKEN='…'   # value of __Secure-next-auth.session-token
# or
export AXIOM_COOKIE_FILE=~/.config/axiom/session.token
```

## Collect

```bash
cd demo/lab/axiom_collector
python collect.py \
  --course-id cmq97i5kn032208o8xu5dab4q \
  --out-dir ../../../results/axiom_course_cmq97i5kn032208o8xu5dab4q
```

Writes `collect.json`, `sessions.json`, `assignments.json`.

## Tests

```bash
python -m unittest tests/test_schema.py -v
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success, non-empty validated arrays (unless `--allow-empty`) |
| 2 | Auth / cookie / session problem |
| 1 | Other API or schema error |
