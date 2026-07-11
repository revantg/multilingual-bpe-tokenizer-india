# Structure (post-cleanup)

## Top level of `grok_2` only

| Folder | Role |
|--------|------|
| `approaches/` | Methods a–e + champion (only place for approach code) |
| `datasets/` | Corpora + HTML→MD builders + result JSONs |
| `demo/` | slides + widget + lab UI |
| `harness/` | score helpers |
| `results/` | PRIMARY score notes |
| `archive/` | obsolete vocabs |

## No external symlinks

Every file used at runtime is a real path under `data/grok_2/…`.

Outer `data/full_page_md` or `data/submission*` may still exist as older copies elsewhere in the repo, but **this agent does not link to them**; citizens of `grok_2` use `datasets/` and `demo/` only.

## Renames

| Old | New |
|-----|-----|
| `approach_b/` | `approaches/b/` |
| `full_page_rerun/` | `datasets/build_full_page/` |
| `full_rerun/` | `datasets/build_full_md/` |
| `corpora/*` symlinks | removed → real `datasets/*` |
| `delivery/*` symlinks | removed → real `demo/slides`, `demo/widget` |
