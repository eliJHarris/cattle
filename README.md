# Cattle Futures Dashboard

A Python-generated, self-contained cattle-futures dashboard designed for GitHub Pages. It carries the analytical views from `cattleFutures.ipynb` into an interactive website with:

- global date-range and topic controls;
- feeder-cattle price, drawdown, supply, drought, curve, positioning, seasonality, event, and risk views;
- a transparent five-factor tactical recommendation;
- light and dark themes;
- source and methodology notes beside the data they support;
- weekday refreshes through GitHub Actions.

## Build locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python build_dashboard.py
python -m http.server 8000 --directory docs
```

Open <http://localhost:8000>. The output is a single `docs/index.html`; no application server is required.

## Deploy

1. Push this repository to GitHub with `main` as the default branch.
2. In **Settings → Pages**, choose **GitHub Actions** as the source.
3. Run **Build and deploy cattle dashboard** once, or push to `main`.

The workflow also rebuilds the published snapshot at 23:37 UTC on weekdays. A failed optional source (for example USDA or drought) produces a clearly labeled unavailable section; a failed core Yahoo feeder-cattle download stops the build rather than publishing a misleading dashboard.

## Data and caveats

The dashboard uses Yahoo Finance continuous and listed futures data, USDA Cattle on Feed releases, CFTC Commitments of Traders, and the U.S. Drought Monitor. The recommendation is a transparent heuristic for general market research, not individualized trading, hedging, tax, or legal advice. Futures are leveraged and can lose more than initial margin.

