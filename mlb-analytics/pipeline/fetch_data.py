"""
MLB Analytics — Data Pipeline
==============================
Pulls everything needed for the matchup + game prediction models:
  • Today's schedule + probable pitchers  (MLB Stats API)
  • Season batting stats w/ Statcast metrics  (FanGraphs via pybaseball)
  • Season pitching stats w/ FIP/xFIP       (FanGraphs via pybaseball)
  • Park factors (1yr + 3yr + HR)            (Baseball Reference via pybaseball)
  • Pitch-level Statcast data                (Baseball Savant via pybaseball)

Run in Google Colab — all data is stored in an SQLite DB on Google Drive.
"""

# ── CELL 1: Install packages (run once) ─────────────────────────────────────
# !pip install pybaseball mlb-statsapi pandas requests sqlalchemy -q

# ── CELL 2: Imports ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import sqlite3
import requests
import os
import time
from datetime import datetime, date, timedelta

import statsapi

from pybaseball import (
    statcast,
    batting_stats,
    pitching_stats,
    team_batting,
    team_pitching,
    playerid_lookup,
    playerid_reverse_lookup,
)
from pybaseball import park_factors
from pybaseball import cache

# Speed up pybaseball by caching responses locally
cache.enable()

# ── CELL 3: Config ───────────────────────────────────────────────────────────

CURRENT_SEASON = datetime.now().year

# If using Google Drive, change this to:
# DB_PATH = "/content/drive/MyDrive/mlb_analytics/mlb_analytics.db"
DB_PATH = "mlb_analytics.db"

# How many days of Statcast pitch data to pull on first run
# 30 days ≈ 5–10 min fetch time, ~500k rows
# Increase to 90 for a richer model training set (takes longer)
STATCAST_INITIAL_DAYS = 30


# ── CELL 4: Database helpers ─────────────────────────────────────────────────

def get_conn():
    """Return a SQLite connection to the analytics DB."""
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create all tables on first run."""
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS schedule (
            game_id         TEXT PRIMARY KEY,
            game_date       TEXT,
            away_team       TEXT,
            home_team       TEXT,
            venue_name      TEXT,
            game_time_utc   TEXT,
            away_pitcher    TEXT,
            home_pitcher    TEXT,
            away_pitcher_id INTEGER,
            home_pitcher_id INTEGER,
            fetched_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS batting_stats (
            player_id       INTEGER,
            name            TEXT,
            team            TEXT,
            season          INTEGER,
            pa              INTEGER,
            avg             REAL,
            obp             REAL,
            slg             REAL,
            ops             REAL,
            woba            REAL,
            xwoba           REAL,
            xba             REAL,
            xslg            REAL,
            barrel_pct      REAL,
            hard_hit_pct    REAL,
            k_pct           REAL,
            bb_pct          REAL,
            sprint_speed    REAL,
            sweet_spot_pct  REAL,
            launch_angle    REAL,
            exit_velocity   REAL,
            fetched_at      TEXT,
            PRIMARY KEY (player_id, season)
        );

        CREATE TABLE IF NOT EXISTS pitching_stats (
            player_id       INTEGER,
            name            TEXT,
            team            TEXT,
            season          INTEGER,
            ip              REAL,
            era             REAL,
            fip             REAL,
            xfip            REAL,
            siera           REAL,
            k_pct           REAL,
            bb_pct          REAL,
            k_bb_pct        REAL,
            hr9             REAL,
            whip            REAL,
            lob_pct         REAL,
            gb_pct          REAL,
            fb_pct          REAL,
            hard_pct        REAL,
            stuff_plus      REAL,
            location_plus   REAL,
            pitching_plus   REAL,
            fetched_at      TEXT,
            PRIMARY KEY (player_id, season)
        );

        CREATE TABLE IF NOT EXISTS park_factors (
            team            TEXT,
            venue_name      TEXT,
            basic_1yr       REAL,
            basic_3yr       REAL,
            hr_factor       REAL,
            runs_factor     REAL,
            h_factor        REAL,
            doubles_factor  REAL,
            triples_factor  REAL,
            season          INTEGER,
            fetched_at      TEXT,
            PRIMARY KEY (team, season)
        );

        CREATE TABLE IF NOT EXISTS statcast_pitches (
            pitch_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date           TEXT,
            game_pk             INTEGER,
            pitcher_id          INTEGER,
            batter_id           INTEGER,
            pitch_type          TEXT,
            pitch_name          TEXT,
            release_speed       REAL,
            release_spin_rate   REAL,
            pfx_x               REAL,
            pfx_z               REAL,
            plate_x             REAL,
            plate_z             REAL,
            zone                INTEGER,
            launch_speed        REAL,
            launch_angle        REAL,
            hit_distance        REAL,
            xwoba               REAL,
            events              TEXT,
            description         TEXT,
            stand               TEXT,
            p_throws            TEXT,
            home_team           TEXT,
            away_team           TEXT,
            bb_type             TEXT,
            barrel              INTEGER,
            fetched_at          TEXT
        );

        CREATE TABLE IF NOT EXISTS team_stats (
            team            TEXT,
            season          INTEGER,
            runs_scored     REAL,
            runs_allowed    REAL,
            avg             REAL,
            obp             REAL,
            slg             REAL,
            woba            REAL,
            era             REAL,
            fip             REAL,
            fetched_at      TEXT,
            PRIMARY KEY (team, season)
        );
    """)

    conn.commit()
    conn.close()
    print("✓ Database ready:", os.path.abspath(DB_PATH))


# ── CELL 5: Schedule fetcher ─────────────────────────────────────────────────

def fetch_schedule(target_date: str = None) -> pd.DataFrame:
    """
    Fetch MLB schedule for a given date (defaults to today).
    Includes probable pitchers and venue names.

    Args:
        target_date: 'YYYY-MM-DD' string. Defaults to today.

    Returns:
        DataFrame with one row per game.
    """
    target_date = target_date or date.today().strftime("%Y-%m-%d")
    print(f"\nFetching schedule for {target_date}...")

    games = statsapi.schedule(date=target_date)

    if not games:
        print(f"  No games found for {target_date}")
        return pd.DataFrame()

    rows = []
    for g in games:
        rows.append({
            "game_id":          str(g.get("game_id", "")),
            "game_date":        target_date,
            "away_team":        g.get("away_name", ""),
            "home_team":        g.get("home_name", ""),
            "venue_name":       g.get("venue_name", ""),
            "game_time_utc":    g.get("game_datetime", ""),
            "away_pitcher":     g.get("away_probable_pitcher", "TBD"),
            "home_pitcher":     g.get("home_probable_pitcher", "TBD"),
            "away_pitcher_id":  g.get("away_probable_pitcher_id"),
            "home_pitcher_id":  g.get("home_probable_pitcher_id"),
            "fetched_at":       datetime.now().isoformat(),
        })

    df = pd.DataFrame(rows)
    conn = get_conn()
    df.to_sql("schedule", conn, if_exists="replace", index=False)
    conn.close()

    print(f"  ✓ {len(df)} games saved")
    for _, row in df.iterrows():
        print(f"    {row['away_team']:25s} @ {row['home_team']:25s}  |  "
              f"SP: {row['away_pitcher']} vs {row['home_pitcher']}")

    return df


# ── CELL 6: Batting stats fetcher ────────────────────────────────────────────

# Map FanGraphs column names → our clean schema names
BATTING_COL_MAP = {
    "IDfg":         "player_id",
    "Name":         "name",
    "Team":         "team",
    "PA":           "pa",
    "AVG":          "avg",
    "OBP":          "obp",
    "SLG":          "slg",
    "OPS":          "ops",
    "wOBA":         "woba",
    "xwOBA":        "xwoba",
    "xBA":          "xba",
    "xSLG":         "xslg",
    "Barrel%":      "barrel_pct",
    "HardHit%":     "hard_hit_pct",
    "K%":           "k_pct",
    "BB%":          "bb_pct",
    "Sprint Speed": "sprint_speed",
    "SwSp%":        "sweet_spot_pct",
    "LA":           "launch_angle",
    "EV":           "exit_velocity",
}


def fetch_batting_stats(season: int = None, min_pa: int = 50) -> pd.DataFrame:
    """
    Fetch season batting stats with Statcast metrics from FanGraphs.

    Args:
        season:  MLB season year. Defaults to current year.
        min_pa:  Minimum plate appearances filter.

    Returns:
        DataFrame of batter stats.
    """
    season = season or CURRENT_SEASON
    print(f"\nFetching {season} batting stats (min {min_pa} PA)...")

    try:
        df = batting_stats(season, qual=min_pa)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return pd.DataFrame()

    # Rename columns we care about, drop the rest
    available = {k: v for k, v in BATTING_COL_MAP.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)
    df["season"] = season
    df["fetched_at"] = datetime.now().isoformat()

    # Convert percentage columns from 0-100 to 0-1 if needed
    for col in ["k_pct", "bb_pct", "barrel_pct", "hard_hit_pct", "sweet_spot_pct"]:
        if col in df.columns and df[col].median() > 1:
            df[col] = df[col] / 100

    conn = get_conn()
    df.to_sql("batting_stats", conn, if_exists="replace", index=False)
    conn.close()

    print(f"  ✓ {len(df)} batters saved")
    print(f"  Top 5 by xwOBA:\n{df.nlargest(5, 'xwoba')[['name','team','pa','xwoba','barrel_pct']].to_string(index=False)}")

    return df


# ── CELL 7: Pitching stats fetcher ───────────────────────────────────────────

PITCHING_COL_MAP = {
    "IDfg":       "player_id",
    "Name":       "name",
    "Team":       "team",
    "IP":         "ip",
    "ERA":        "era",
    "FIP":        "fip",
    "xFIP":       "xfip",
    "SIERA":      "siera",
    "K%":         "k_pct",
    "BB%":        "bb_pct",
    "K-BB%":      "k_bb_pct",
    "HR/9":       "hr9",
    "WHIP":       "whip",
    "LOB%":       "lob_pct",
    "GB%":        "gb_pct",
    "FB%":        "fb_pct",
    "Hard%":      "hard_pct",
    "Stuff+":     "stuff_plus",
    "Location+":  "location_plus",
    "Pitching+":  "pitching_plus",
}


def fetch_pitching_stats(season: int = None, min_ip: int = 20) -> pd.DataFrame:
    """
    Fetch season pitching stats with FIP, xFIP, SIERA from FanGraphs.

    Args:
        season:  MLB season year. Defaults to current year.
        min_ip:  Minimum innings pitched filter.

    Returns:
        DataFrame of pitcher stats.
    """
    season = season or CURRENT_SEASON
    print(f"\nFetching {season} pitching stats (min {min_ip} IP)...")

    try:
        df = pitching_stats(season, qual=min_ip)
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return pd.DataFrame()

    available = {k: v for k, v in PITCHING_COL_MAP.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)
    df["season"] = season
    df["fetched_at"] = datetime.now().isoformat()

    for col in ["k_pct", "bb_pct", "k_bb_pct", "gb_pct", "fb_pct", "hard_pct", "lob_pct"]:
        if col in df.columns and df[col].median() > 1:
            df[col] = df[col] / 100

    conn = get_conn()
    df.to_sql("pitching_stats", conn, if_exists="replace", index=False)
    conn.close()

    print(f"  ✓ {len(df)} pitchers saved")
    print(f"  Top 5 starters by FIP:\n{df.nsmallest(5, 'fip')[['name','team','ip','fip','xfip','k_pct']].to_string(index=False)}")

    return df


# ── CELL 8: Park factors fetcher ─────────────────────────────────────────────

def fetch_park_factors_data(season: int = None) -> pd.DataFrame:
    """
    Fetch park factors from Baseball Reference via pybaseball.
    Pulls single-year and 3-year averages for run, HR, H, 2B, 3B factors.

    Note: Park factors above 100 = hitter-friendly, below 100 = pitcher-friendly.
    E.g. Coors Field HR factor ~130 means 30% more HRs hit there vs neutral park.

    Args:
        season: MLB season. Defaults to current year.

    Returns:
        DataFrame with one row per MLB park.
    """
    season = season or CURRENT_SEASON
    print(f"\nFetching park factors for {season}...")

    try:
        pf = park_factors(season, min_year=season - 2)
    except Exception as e:
        print(f"  ✗ Error fetching park factors: {e}")
        # Fall back to a manually curated table for the current season
        pf = _get_fallback_park_factors(season)
        if pf is None:
            return pd.DataFrame()

    pf.columns = [c.strip() for c in pf.columns]
    pf["season"] = season
    pf["fetched_at"] = datetime.now().isoformat()
    pf.columns = [c.lower().replace(" ", "_") for c in pf.columns]

    conn = get_conn()
    pf.to_sql("park_factors", conn, if_exists="replace", index=False)
    conn.close()

    print(f"  ✓ Park factors saved for {len(pf)} parks")
    print("\n  Most hitter-friendly parks (by basic factor):")

    # Show top/bottom parks if the column exists
    basic_col = next((c for c in pf.columns if "basic" in c or "1yr" in c or "factor" in c.lower()), None)
    if basic_col:
        display_cols = [c for c in ["team", "venue", basic_col] if c in pf.columns]
        print(pf.nlargest(5, basic_col)[display_cols].to_string(index=False))

    return pf


def _get_fallback_park_factors(season: int) -> pd.DataFrame:
    """
    Hardcoded fallback park factors if pybaseball fails.
    Based on 3-year averages. All values are relative to 100 (neutral).
    """
    print("  Using fallback park factor table...")
    data = [
        # team,              venue,                   basic, hr,    runs, h
        ("Colorado Rockies",  "Coors Field",            115,   130,   119,  112),
        ("Cincinnati Reds",   "Great American Ball Park",112,  122,   111,  109),
        ("Texas Rangers",     "Globe Life Field",       108,   115,   109,  107),
        ("Boston Red Sox",    "Fenway Park",            107,   104,   107,  111),
        ("Philadelphia Phillies","Citizens Bank Park",  107,   114,   107,  106),
        ("Chicago Cubs",      "Wrigley Field",          106,   108,   105,  107),
        ("Atlanta Braves",    "Truist Park",            104,   108,   103,  103),
        ("Milwaukee Brewers", "American Family Field",  103,   105,   103,  102),
        ("Toronto Blue Jays", "Rogers Centre",          102,   103,   101,  101),
        ("Houston Astros",    "Minute Maid Park",       101,   100,   101,  101),
        ("New York Yankees",  "Yankee Stadium",         101,   106,   101,  100),
        ("Arizona Diamondbacks","Chase Field",          100,   100,   100,  100),
        ("St. Louis Cardinals","Busch Stadium",         99,    97,     99,   99),
        ("Baltimore Orioles", "Camden Yards",           99,   100,    99,  100),
        ("Kansas City Royals","Kauffman Stadium",       98,    95,    98,   99),
        ("Los Angeles Dodgers","Dodger Stadium",        97,    97,    98,   97),
        ("Cleveland Guardians","Progressive Field",     97,    93,    97,   97),
        ("Detroit Tigers",    "Comerica Park",          96,    90,    96,   97),
        ("Chicago White Sox", "Guaranteed Rate Field",  96,    96,    96,   96),
        ("Pittsburgh Pirates","PNC Park",               95,    88,    95,   96),
        ("San Francisco Giants","Oracle Park",          94,    85,    93,   95),
        ("Minnesota Twins",   "Target Field",           94,    91,    93,   95),
        ("Tampa Bay Rays",    "Tropicana Field",        93,    89,    93,   94),
        ("Oakland Athletics", "Oakland Coliseum",       93,    88,    92,   94),
        ("Seattle Mariners",  "T-Mobile Park",          93,    87,    92,   94),
        ("Los Angeles Angels","Angel Stadium",          92,    91,    92,   93),
        ("San Diego Padres",  "Petco Park",             91,    85,    91,   92),
        ("Miami Marlins",     "loanDepot park",         90,    83,    89,   91),
        ("New York Mets",     "Citi Field",             89,    83,    88,   90),
        ("Washington Nationals","Nationals Park",       88,    82,    87,   89),
    ]
    return pd.DataFrame(data, columns=["team", "venue_name", "basic_3yr", "hr_factor",
                                        "runs_factor", "h_factor"])


# ── CELL 9: Statcast pitch-level fetcher ─────────────────────────────────────

STATCAST_COLS = [
    "game_date", "game_pk", "pitcher", "batter",
    "pitch_type", "pitch_name",
    "release_speed", "release_spin_rate",
    "pfx_x", "pfx_z", "plate_x", "plate_z", "zone",
    "launch_speed", "launch_angle", "hit_distance_sc",
    "estimated_woba_using_speedangle",
    "events", "description", "stand", "p_throws",
    "home_team", "away_team", "bb_type", "barrel",
]


def fetch_statcast_data(
    start_dt: str = None,
    end_dt: str = None,
    days_back: int = None,
    append: bool = True,
) -> pd.DataFrame:
    """
    Fetch pitch-level Statcast data from Baseball Savant.

    This is the richest dataset — every pitch thrown with spin rate,
    movement, and outcome. Used to build pitch-type matchup features.

    Args:
        start_dt:  'YYYY-MM-DD'. Overrides days_back.
        end_dt:    'YYYY-MM-DD'. Defaults to today.
        days_back: Convenience — how many days back from today.
        append:    If True, append to existing table. If False, replace.

    Returns:
        DataFrame of pitch-level events.

    Notes:
        • First run: use days_back=30 (fast) or days_back=90 (richer features)
        • Nightly update: use days_back=2 with append=True
        • Full season: ~800k rows, takes ~15 min in Colab
    """
    end = date.today()
    if end_dt:
        end = datetime.strptime(end_dt, "%Y-%m-%d").date()

    if start_dt:
        start = datetime.strptime(start_dt, "%Y-%m-%d").date()
    elif days_back:
        start = end - timedelta(days=days_back)
    else:
        start = end - timedelta(days=STATCAST_INITIAL_DAYS)

    print(f"\nFetching Statcast data: {start} → {end}")
    print("  (This takes 2–5 min depending on date range. Don't close the tab.)")

    try:
        df = statcast(
            start_dt=start.strftime("%Y-%m-%d"),
            end_dt=end.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        print(f"  ✗ Statcast error: {e}")
        return pd.DataFrame()

    # Keep only needed columns
    keep = [c for c in STATCAST_COLS if c in df.columns]
    df = df[keep].copy()

    # Rename to match our schema
    df = df.rename(columns={
        "pitcher":                              "pitcher_id",
        "batter":                               "batter_id",
        "hit_distance_sc":                      "hit_distance",
        "estimated_woba_using_speedangle":      "xwoba",
    })

    df["fetched_at"] = datetime.now().isoformat()

    # Remove rows with no pitch type (warmup pitches, data errors)
    df = df.dropna(subset=["pitch_type", "pitcher_id", "batter_id"])

    conn = get_conn()
    mode = "append" if append else "replace"
    df.to_sql("statcast_pitches", conn, if_exists=mode, index=False)
    conn.close()

    print(f"  ✓ {len(df):,} pitches saved (mode: {mode})")

    # Quick summary of pitch type distribution
    if "pitch_type" in df.columns:
        pt = df["pitch_type"].value_counts().head(8)
        print("\n  Pitch type breakdown:")
        for pt_name, count in pt.items():
            bar = "█" * int(count / pt.iloc[0] * 20)
            print(f"    {pt_name:5s}  {bar:20s}  {count:,}")

    return df


# ── CELL 10: Matchup preview builder ─────────────────────────────────────────

def build_todays_matchup_preview() -> pd.DataFrame:
    """
    Build a quick matchup preview for today's games by joining:
    schedule + batting_stats + pitching_stats + park_factors.

    Returns a DataFrame with key metrics for each probable SP matchup.
    This is the foundation that the dashboard and models will use.
    """
    conn = get_conn()

    try:
        schedule  = pd.read_sql("SELECT * FROM schedule",       conn)
        batting   = pd.read_sql("SELECT * FROM batting_stats",  conn)
        pitching  = pd.read_sql("SELECT * FROM pitching_stats", conn)
        parks     = pd.read_sql("SELECT * FROM park_factors",   conn)
    except Exception as e:
        print(f"  ✗ Tables not ready yet: {e}")
        conn.close()
        return pd.DataFrame()

    conn.close()

    if schedule.empty:
        print("  No schedule data. Run fetch_schedule() first.")
        return pd.DataFrame()

    print("\n" + "═" * 70)
    print("  TODAY'S MATCHUP PREVIEW")
    print("═" * 70)

    # Identify the basic park factor column
    park_basic_col = next(
        (c for c in parks.columns if "basic" in c or "3yr" in c), None
    )

    rows = []
    for _, game in schedule.iterrows():
        away_sp_name = game["away_pitcher"]
        home_sp_name = game["home_pitcher"]
        venue        = game["venue_name"]

        # Lookup park factor for home team
        pf_row = parks[parks["venue_name"] == venue] if "venue_name" in parks.columns else pd.DataFrame()
        if not pf_row.empty and park_basic_col:
            park_factor = pf_row.iloc[0][park_basic_col]
            hr_factor   = pf_row.iloc[0].get("hr_factor", 100)
        else:
            park_factor = 100
            hr_factor   = 100

        # Find away SP stats
        away_sp = _lookup_pitcher(pitching, away_sp_name)
        home_sp = _lookup_pitcher(pitching, home_sp_name)

        row = {
            "away_team":        game["away_team"],
            "home_team":        game["home_team"],
            "venue":            venue,
            "park_factor":      park_factor,
            "hr_factor":        hr_factor,
            "away_sp":          away_sp_name,
            "away_sp_fip":      away_sp.get("fip"),
            "away_sp_xfip":     away_sp.get("xfip"),
            "away_sp_k_pct":    away_sp.get("k_pct"),
            "home_sp":          home_sp_name,
            "home_sp_fip":      home_sp.get("fip"),
            "home_sp_xfip":     home_sp.get("xfip"),
            "home_sp_k_pct":    home_sp.get("k_pct"),
        }
        rows.append(row)

        print(f"\n  {game['away_team']:28s} @ {game['home_team']}")
        print(f"  Venue: {venue:<30s}  Park factor: {park_factor:.0f}  HR factor: {hr_factor:.0f}")
        print(f"  Away SP: {away_sp_name:<25s}  FIP: {away_sp.get('fip', 'N/A')}  xFIP: {away_sp.get('xfip', 'N/A')}  K%: {_fmt_pct(away_sp.get('k_pct'))}")
        print(f"  Home SP: {home_sp_name:<25s}  FIP: {home_sp.get('fip', 'N/A')}  xFIP: {home_sp.get('xfip', 'N/A')}  K%: {_fmt_pct(home_sp.get('k_pct'))}")

    print("\n" + "═" * 70)
    return pd.DataFrame(rows)


def _lookup_pitcher(pitching_df: pd.DataFrame, name: str) -> dict:
    """Fuzzy-match a pitcher name in the stats table."""
    if not name or name == "TBD" or pitching_df.empty:
        return {}
    name_lower = name.lower()
    # Try exact match first, then partial
    mask = pitching_df["name"].str.lower() == name_lower
    if not mask.any():
        parts = name_lower.split()
        mask = pitching_df["name"].str.lower().str.contains(parts[-1]) if parts else mask
    if mask.any():
        row = pitching_df[mask].iloc[0]
        return row.to_dict()
    return {}


def _fmt_pct(val) -> str:
    """Format a 0-1 decimal as a percentage string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.1%}"


# ── CELL 11: Pitch-type matchup analyzer ─────────────────────────────────────

def analyze_batter_vs_pitch_type(
    batter_name: str,
    pitcher_name: str = None,
    season: int = None,
) -> pd.DataFrame:
    """
    Analyze how a batter performs against each pitch type.
    If pitcher_name is given, filters to that specific pitcher.

    This is the core of the matchup model — understanding HOW a batter
    struggles or excels against specific pitch types (e.g. Shohei vs sliders).

    Args:
        batter_name:  Full name e.g. "Juan Soto"
        pitcher_name: Optional — filter to specific pitcher
        season:       Season year. Defaults to current.

    Returns:
        DataFrame with xwOBA, whiff rate, barrel rate by pitch type.

    Example:
        analyze_batter_vs_pitch_type("Aaron Judge", "Gerrit Cole")
    """
    season = season or CURRENT_SEASON
    print(f"\nAnalyzing: {batter_name}" + (f" vs {pitcher_name}" if pitcher_name else ""))

    conn = get_conn()
    try:
        pitches = pd.read_sql("SELECT * FROM statcast_pitches", conn)
    except Exception as e:
        print(f"  ✗ Statcast table not ready: {e}")
        conn.close()
        return pd.DataFrame()
    conn.close()

    if pitches.empty:
        print("  No Statcast data yet. Run fetch_statcast_data() first.")
        return pd.DataFrame()

    # Look up batter ID by name
    try:
        lookup = playerid_lookup(
            batter_name.split()[-1],
            batter_name.split()[0],
        )
        if lookup.empty:
            print(f"  Could not find player ID for '{batter_name}'")
            return pd.DataFrame()
        batter_id = lookup.iloc[0]["key_mlbam"]
    except Exception as e:
        print(f"  Player lookup error: {e}")
        return pd.DataFrame()

    # Filter pitches for this batter
    df = pitches[pitches["batter_id"] == batter_id].copy()

    if pitcher_name:
        try:
            p_lookup = playerid_lookup(
                pitcher_name.split()[-1],
                pitcher_name.split()[0],
            )
            if not p_lookup.empty:
                pitcher_id = p_lookup.iloc[0]["key_mlbam"]
                df = df[df["pitcher_id"] == pitcher_id]
        except Exception:
            pass

    if df.empty:
        print(f"  No pitch data found for {batter_name} in current dataset.")
        print("  Try fetching more Statcast data (increase days_back).")
        return pd.DataFrame()

    # Compute per-pitch-type metrics
    df["is_swing"]  = df["description"].str.contains("swinging|foul|hit_into_play", na=False)
    df["is_whiff"]  = df["description"].str.contains("swinging_strike", na=False)
    df["is_barrel"] = df["barrel"] == 1
    df["is_contact"]= df["description"].str.contains("hit_into_play|foul", na=False)

    summary = df.groupby("pitch_type").agg(
        pitches      = ("pitch_type",  "count"),
        avg_velo     = ("release_speed","mean"),
        avg_spin     = ("release_spin_rate","mean"),
        xwoba        = ("xwoba",        "mean"),
        whiff_rate   = ("is_whiff",     "mean"),
        swing_rate   = ("is_swing",     "mean"),
        barrel_rate  = ("is_barrel",    "mean"),
    ).reset_index()

    summary = summary[summary["pitches"] >= 5].sort_values("xwoba", ascending=False)

    # Round for display
    for col in ["avg_velo", "avg_spin"]:
        summary[col] = summary[col].round(1)
    for col in ["xwoba", "whiff_rate", "swing_rate", "barrel_rate"]:
        summary[col] = summary[col].round(3)

    print(f"\n  {batter_name} — performance by pitch type:")
    print(summary.to_string(index=False))
    print(f"\n  Biggest vulnerability: {summary.nsmallest(1, 'xwoba').iloc[0]['pitch_type']} "
          f"(xwOBA: {summary.nsmallest(1, 'xwoba').iloc[0]['xwoba']:.3f})")

    return summary


# ── CELL 12: Nightly update function ─────────────────────────────────────────

def nightly_update():
    """
    Lightweight update to run daily.
    Refreshes today's schedule and appends the last 2 days of Statcast data.
    Run this every morning before checking today's slate.

    In the future, this will be automated via GitHub Actions.
    """
    print("Running nightly update...")
    fetch_schedule()
    fetch_statcast_data(days_back=2, append=True)
    print("✓ Nightly update complete")


# ── CELL 13: Full pipeline runner ────────────────────────────────────────────

def run_full_pipeline(statcast_days: int = STATCAST_INITIAL_DAYS):
    """
    Run the complete first-time data pipeline.
    Takes ~10–20 minutes total depending on Statcast date range.

    Call this once to seed your database, then use nightly_update() daily.

    Args:
        statcast_days: How many days of pitch data to pull (default: 30).
    """
    print("=" * 70)
    print("  MLB ANALYTICS — FULL PIPELINE")
    print(f"  Season: {CURRENT_SEASON}  |  Statcast window: {statcast_days} days")
    print("=" * 70)

    init_db()

    fetch_schedule()
    fetch_batting_stats()
    fetch_pitching_stats()
    fetch_park_factors_data()
    fetch_statcast_data(days_back=statcast_days, append=False)
    build_todays_matchup_preview()

    print("\n" + "=" * 70)
    print("  ✓ PIPELINE COMPLETE")
    print(f"  Database: {os.path.abspath(DB_PATH)}")
    print("  Next step: open analysis/matchup_explorer.ipynb")
    print("=" * 70)


# ── Run directly (or import and call individual functions) ───────────────────
if __name__ == "__main__":
    run_full_pipeline(statcast_days=30)
