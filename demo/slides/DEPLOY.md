# Deploy — tokenizer slideshow (Approach B optimized)

Static Netlify site for the presentation deck in this folder.

## Publish directory

```
data/submission_slides/
```

Contents: `index.html` + tokenizer artifacts (no build step).

## Create new site + production deploy

```bash
cd data/submission_slides

# Create a NEW site (unique name required)
npx netlify-cli sites:create --name tokenizer-india-b-slides-<suffix>

# Deploy production (use site slug or site id from create)
npx netlify-cli deploy --prod --dir=. --site <slug-or-id> --no-build
```

`--no-build` is required so Netlify does not look for a framework build.

## Production deploy (2026-07-11)

| Field | Value |
|-------|--------|
| Site name / slug | `tokenizer-india-b-slides-rev-55142` |
| Site id | `e805d619-8f09-48cc-a6fb-1687d8f1b2c1` |
| Production URL | https://tokenizer-india-b-slides-rev-55142.netlify.app |
| Admin URL | https://app.netlify.com/projects/tokenizer-india-b-slides-rev-55142 |
| Account | Revant Gupta / revantgupta2@gmail.com |

### Commands used

```bash
cd data/submission_slides
npx netlify-cli sites:create --name tokenizer-india-b-slides-rev-55142
npx netlify-cli deploy --prod --dir=. --site e805d619-8f09-48cc-a6fb-1687d8f1b2c1 --no-build
```

## Notes

- This is a **new** site, separate from the interactive widget  
  `https://tokenizer-india-b-opt-rev-28915.netlify.app`
- Tokenizer downloads are **self-contained** on the slides site  
  (`tokenizer.json`, `tokens.txt`, `scores.json`, `vocab_optimized_fullpage.json`)
- Do not print or commit Netlify auth tokens
