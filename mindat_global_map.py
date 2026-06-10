"""
pyrite_global_heatmap.py
========================
Wraps mindat_collector() to query pyrite mine localities for every country on
Earth, then plots a choropleth heatmap (plotly) coloured by locality count.

Requirements
------------
    pip install pycountry plotly pandas pymindat

Usage
-----
    python pyrite_global_heatmap.py

The script is resumable: per-country results are persisted to
`collection_log.json` after each API call, so you can Ctrl-C and restart
without re-querying already-processed countries.

Output
------
    pyrite_heatmap.html   — interactive choropleth (open in any browser)
    collection_log.json   — raw per-country results (progress + cache)
    summary.csv           — tidy country × count table
"""

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import pycountry

# ── paste or import your mindat_collector here ─────────────────────────────────
from data_utils import mindat_collector
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# 1.  Country list & name-mapping
# =============================================================================

# pycountry uses ISO-3166 official names; Mindat often uses shorter common names.
# Extend this dict whenever a country query returns zero results unexpectedly.
MINDAT_NAME_OVERRIDES: dict[str, str] = {
    "United Kingdom": "UK",
    "United States": "USA",
    "Svalbard and Jan Mayen": "Svalbard, Norway",
    "Türkiye": "Turkey",
    "United States": "USA",
    "Congo, The Democratic Republic of the": "DR Congo",
    "Puerto Rico": "Puerto Rico, USA",
    "Guadeloupe": "Guadeloupe, France",
    "French Guiana": "French Guiana, France",
    "Bolivia, Plurinational State of":          "Bolivia",
    "Congo":                                    "Republic of the Congo",
    "Côte d'Ivoire":                            "Ivory Coast",
    "Czechia":                                  "Czech Republic",
    "Eswatini":                                 "Swaziland",
    "Holy See (Vatican City State)":            "Vatican",
    "Hong Kong":                                "Hong Kong SAR",
    "Iran, Islamic Republic of":                "Iran",
    "Korea, Democratic People's Republic of":   "North Korea",
    "Korea, Republic of":                       "South Korea",
    "Lao People's Democratic Republic":         "Laos",
    "Libya":                                    "Libya",
    "Macao":                                    "Macau SAR",
    "Moldova, Republic of":                     "Moldova",
    "Palestine, State of":                      "Palestinian Territory",
    "Russian Federation":                       "Russia",
    "Syrian Arab Republic":                     "Syria",
    "Tanzania, United Republic of":             "Tanzania",
    "Taiwan, Province of China":                "Taiwan",
    "Venezuela, Bolivarian Republic of":        "Venezuela",
    "Viet Nam":                                 "Vietnam",
}

# Build the master lookup: pycountry_name → ISO-alpha3 (needed by plotly)
ALL_COUNTRIES: dict[str, str] = {c.name: c.alpha_3 for c in pycountry.countries}


# =============================================================================
# 2.  Batch collector
# =============================================================================

def collect_all_countries(
    path_str:        str   = "../data/",
    mindat_api_str:  str   = "mindat_API_key.txt",
    material_id:     int   = 3314,
    mineral_strings: str   = "(Fe|S)",
    material_name:   str   = "pyrite",
    sleep_between:   float = 1.5,
    log_path:        str   = "../data/mindat datacollection_log.json",
) -> dict:
    """
    Calls mindat_collector() for every country and returns a progress dict:

        { pycountry_name: {"iso3": "...", "count": N, "mindat_name": "..."} }

    count = -1 signals an API / parsing error for that country.

    The log_path JSON is written after every country so interrupted runs
    can be resumed safely — already-logged countries are skipped on restart.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # ── Resume from previous run ───────────────────────────────────────────────
    progress: dict = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            progress = json.load(f)
        log.info("Resuming: %d / %d countries already logged.",
                 len(progress), len(ALL_COUNTRIES))

    # ── Iterate over all countries ─────────────────────────────────────────────
    for pycountry_name, iso3 in ALL_COUNTRIES.items():
        if pycountry_name in progress:
            continue                                          # already done

        mindat_name = MINDAT_NAME_OVERRIDES.get(pycountry_name, pycountry_name)
        log.info("Querying %-50s (mindat: '%s')", pycountry_name, mindat_name)

        try:
            mindat_collector(
                region=mindat_name,
                material_id=material_id,
                mineral_strings=mineral_strings,
                material_name=material_name,
                path_str=path_str,
                mindat_api_str=mindat_api_str,
            )

            # The function saves a CSV: {path_str}mindat_data/{region}_{material}.csv
            csv_path = (
                Path(path_str) / "mindat_data" / f"{mindat_name}_{material_name}.csv"
            )
            count = len(pd.read_csv(csv_path)) if csv_path.exists() else 0

        except Exception as exc:
            log.warning("  ✗  %s — %s", pycountry_name, exc)
            count = -1          # sentinel: error, treat as 0 when plotting

        progress[pycountry_name] = {
            "iso3":        iso3,
            "count":       count,
            "mindat_name": mindat_name,
        }

        # Persist after every country so progress is never lost
        with open(log_path, "w") as f:
            json.dump(progress, f, indent=2)

        time.sleep(sleep_between)    # be polite to the Mindat API

    log.info("Collection complete. %d countries processed.", len(progress))
    return progress


# =============================================================================
# 3.  Aggregate results into a tidy DataFrame
# =============================================================================

def build_summary(progress: dict) -> pd.DataFrame:
    """
    Converts the progress dict into a DataFrame ready for plotting.

    Columns
    -------
    country      pycountry English name
    iso3         ISO-3166 alpha-3 (used by plotly locations=)
    count        number of pyrite-mine localities  (errors → 0)
    status       'ok' | 'error' | 'empty'
    """
    rows = []
    for country, info in progress.items():
        raw = info["count"]
        rows.append({
            "country": country,
            "iso3":    info["iso3"],
            "count":   max(raw, 0),                            # clip errors → 0
            "status":  "error" if raw < 0
                       else ("empty" if raw == 0 else "ok"),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return df


# =============================================================================
# 4.  Choropleth heatmap
# =============================================================================

def plot_heatmap(
    df:          pd.DataFrame,
    log_scale:   bool = True,
    output_html: str  = "pyrite_heatmap.html",
) -> "plotly.graph_objects.Figure":
    """
    Renders an interactive choropleth map coloured by locality count.

    Parameters
    ----------
    log_scale : bool
        Apply log₁₀(count + 1) to the colour axis.  Strongly recommended:
        pyrite occurrence is highly skewed (USA/Russia/China dominate), so
        linear scale washes out most of the world.
    output_html : str
        If given, the interactive figure is saved here.
    """
    import numpy as np

    plot_df = df.copy()

    if log_scale:
        plot_df["colour_val"] = np.log10(plot_df["count"] + 1)
        colour_label = "log₁₀(localities + 1)"
        tickvals = [0, 0.5, 1, 1.5, 2, 2.5, 3]
        ticktext = [f"{10**v - 1:.0f}" for v in tickvals]   # back to raw counts
    else:
        plot_df["colour_val"] = plot_df["count"]
        colour_label = "Localities"
        tickvals = ticktext = None

    fig = px.choropleth(
        plot_df,
        locations="iso3",
        color="colour_val",
        hover_name="country",
        hover_data={
            "count":      True,          # show raw count in tooltip
            "colour_val": False,         # hide log value in tooltip
            "iso3":       False,
            "status":     True,
        },
        color_continuous_scale="YlOrRd",  # white-yellow-orange-red
        range_color=(0, plot_df["colour_val"].max()),
        labels={"colour_val": colour_label, "count": "Localities"},
        title="Pyrite Mine Localities per Country — Mindat",
    )

    # Colour-bar: show raw count labels even when using log scale
    cb_kwargs = dict(title=colour_label)
    if log_scale and tickvals:
        cb_kwargs.update(tickvals=tickvals, ticktext=ticktext)

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
        ),
        coloraxis_colorbar=cb_kwargs,
        margin={"r": 0, "t": 60, "l": 0, "b": 0},
        title_x=0.5,
    )

    if output_html:
        fig.write_html(output_html)
        print(f"Saved interactive heatmap → {output_html}")

    fig.show()
    return fig


# =============================================================================
# 5.  Entry point
# =============================================================================

if __name__ == "__main__":
    # ── Config ─────────────────────────────────────────────────────────────────
    PATH_STR        = "../data/"
    MINDAT_API_FILE = "mindat_API_key.txt"
    LOG_PATH        = "../data/mindat_data/collection_log.json"
    SUMMARY_CSV     = "../data/mindat_data/summary.csv"
    HEATMAP_HTML    = "../data/mindat_datapyrite_heatmap.html"

    # ── Step 1: collect (skips countries already cached on disk) ───────────────
    progress = collect_all_countries(
        path_str=PATH_STR,
        mindat_api_str=MINDAT_API_FILE,
        log_path=LOG_PATH,
        sleep_between=1.5,   # increase if you hit rate-limit errors
    )

    # ── Step 2: summarise ──────────────────────────────────────────────────────
    summary = build_summary(progress)

    n_ok    = (summary["status"] == "ok").sum()
    n_empty = (summary["status"] == "empty").sum()
    n_err   = (summary["status"] == "error").sum()

    print(f"\n{'='*55}")
    print(f"  Countries with data : {n_ok}")
    print(f"  Countries empty     : {n_empty}")
    print(f"  Countries errored   : {n_err}")
    print(f"{'='*55}\n")
    print(summary.head(25).to_string(index=False))

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSaved summary table → {SUMMARY_CSV}")

    # ── Step 3: plot ───────────────────────────────────────────────────────────
    fig = plot_heatmap(summary, log_scale=True, output_html=HEATMAP_HTML)