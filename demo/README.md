# Demo & delivery (one place)

Everything you **open in a browser** or **upload to Netlify** for this agent:

```
demo/
├── README.md
├── slides/          Netlify slideshow (index.html, tokenizer.json, scores, dataset, …)
├── widget/          Netlify metrics widget
└── lab/             older local HTML explainers / export scripts
```

## Deploy

```bash
# slideshow (current story)
npx netlify-cli deploy --prod --dir=data/grok_2/demo/slides

# widget
npx netlify-cli deploy --prod --dir=data/grok_2/demo/widget --site=<widget-site-id>
```

## Local preview

```bash
cd data/grok_2/demo/slides && python3 -m http.server 8765
```

## Note

`slides/` and `widget/` are **real file trees inside `grok_2`**, not symlinks to `data/submission*`.  
After retraining Approach B, update model files under `demo/slides/` (and `widget/`) before redeploying.
