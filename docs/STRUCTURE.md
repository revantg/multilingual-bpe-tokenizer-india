# Structure (post-cleanup)

## Top level of this repo (`tokenizer-assignment`)

| Folder | Role |
|--------|------|
| `approaches/` | Methods a–f + champion (only place for approach code) |
| `datasets/` | Corpora + HTML→MD builders + result JSONs |
| `demo/` | slides + widget + lab UI |
| `harness/` | score helpers |
| `results/` | PRIMARY score notes |
| `archive/` | obsolete vocabs |

## No external symlinks

Every file used at runtime is a real path under this repo. Corpora live in `datasets/`; demos in `demo/`.

## Renames

| Old | New |
|-----|-----|
| `approach_b/` | `approaches/b/` |
| `full_page_rerun/` | `datasets/build_full_page/` |
| `full_rerun/` | `datasets/build_full_md/` |
| `corpora/*` symlinks | removed → real `datasets/*` |
| `delivery/*` symlinks | removed → real `demo/slides`, `demo/widget` |
