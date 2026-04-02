# ⚾ MLB Analytics Dashboard

> Hitter vs pitcher matchup predictions · Park-adjusted analysis · Game predictions · Built free with Python

---

## What this project does

- Pulls live MLB data (schedule, probable pitchers, lineups) every day for free
- Builds hitter vs pitcher matchup profiles using pitch-level Statcast data
- Applies park factors to adjust every stat for the ballpark being played in
- Predicts outcomes: HR probability, K probability, over/under, win probability
- Displays everything in a Streamlit dashboard you can share publicly

---

## Quick start — Google Colab (no install needed)

### Step 1 — Fork this repo on GitHub
1. Go to `github.com/YOUR_USERNAME/mlb-analytics` (your fork)
2. Click the green **Code** button → **Download ZIP** if you want a local copy, or just work from Colab

### Step 2 — Open the notebook in Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open notebook → GitHub**
3. Paste your repo URL and select `MLB_Analytics_Pipeline.ipynb`
4. Or: **File → Upload notebook** and upload the `.ipynb` file directly

### Step 3 — Run the pipeline
Run the cells **top to bottom** in order:

| Cell | What it does | Time |
|------|-------------|------|
| 1 | Install packages | ~2 min |
| 2 | Mount Google Drive | ~30 sec |
| 4 | Set config + imports | instant |
| 5 | Load pipeline module | instant |
| 6 | **Full pipeline** (first time only) | ~15–20 min |
| 7+ | Explore data interactively | instant |

### Step 4 — Daily use (every morning)
Just run **Cell 8** (`nightly_update()`). Takes ~2 minutes and refreshes today's schedule + recent Statcast data.

---

## Project structure

```
mlb-analytics/
├── MLB_Analytics_Pipeline.ipynb   ← Start here (Google Colab)
├── requirements.txt
├── pipeline/
│   └── fetch_data.py              ← All data fetching logic
├── models/
│   └── matchup_model.py           ← Coming in Phase 2
├── dashboard/
│   └── app.py                     ← Coming in Phase 3 (Streamlit)
└── data/
    └── mlb_analytics.db           ← SQLite database (auto-created)
```

---

## Data sources (all free)

| Source | Data | How we access it |
|--------|------|-----------------|
| **MLB Stats API** | Schedule, lineups, probable pitchers | `mlb-statsapi` Python package |
| **Baseball Savant** | Statcast pitch-level data (spin, velo, exit velocity) | `pybaseball.statcast()` |
| **FanGraphs** | Season stats: wOBA, xwOBA, FIP, xFIP, K%, Barrel% | `pybaseball.batting_stats()` |
| **Baseball Reference** | Park factors (1yr, 3yr, HR factor) | `pybaseball.park_factors()` |

---

## Key concepts

### Park factors
A park factor above 100 means the park inflates offense. Below 100 = pitcher-friendly.

- **Coors Field (Colorado)** — HR factor ~130 = 30% more home runs hit here
- **Petco Park (San Diego)** — HR factor ~85 = 15% fewer home runs
- **Fenway Park (Boston)** — Very hitter-friendly for doubles (Green Monster)

We use park factors to:
1. Adjust expected stats for tonight's game venue
2. Weight the over/under model (more runs expected at Coors vs Petco)
3. Adjust HR probability in the matchup model

### Statcast metrics we use
| Metric | What it measures |
|--------|----------------|
| `xwOBA` | Expected weighted on-base average based on exit velo + launch angle |
| `xBA` | Expected batting average |
| `barrel_pct` | % of batted balls that are "barreled" (hardest, best angle) |
| `hard_hit_pct` | % of balls hit 95+ mph |
| `whiff_rate` | % of swings that miss |
| `release_spin_rate` | Pitch spin — higher spin = more movement |
| `pfx_x / pfx_z` | Horizontal and vertical pitch movement in inches |

### FanGraphs metrics
| Metric | What it measures |
|--------|----------------|
| `FIP` | Fielding Independent Pitching — ERA based only on K, BB, HR (no luck) |
| `xFIP` | Like FIP but normalizes HR rate — best single predictor of future ERA |
| `SIERA` | Most advanced ERA predictor — includes batted ball data |
| `K-BB%` | Strikeout minus walk percentage — elite pitchers are 15%+ |
| `Stuff+` | Pitch quality relative to league average (100 = average, 110 = 10% better) |

---

## Roadmap

- [x] **Phase 1** — Data pipeline (you are here)
- [ ] **Phase 2** — Matchup + game prediction models (XGBoost)
- [ ] **Phase 3** — Streamlit dashboard with park factor visualizations
- [ ] **Phase 4** — Automated daily updates via GitHub Actions
- [ ] **Phase 5** — Public deployment + monetization

---

## Tools used (all free)

| Tool | Purpose | Cost |
|------|---------|------|
| Google Colab | Run Python in browser | Free |
| GitHub | Code storage + version control | Free |
| pybaseball | MLB data wrapper | Free / open source |
| mlb-statsapi | Official MLB API wrapper | Free |
| SQLite | Local database | Free |
| Supabase | Cloud database (Phase 3+) | Free tier |
| Streamlit | Dashboard UI | Free |
| Streamlit Cloud | Deploy dashboard publicly | Free |

---

## Contributing / Contact
Built as a sports analytics portfolio project.
