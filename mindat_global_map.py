import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import pycountry


from data_utils import mindat_collector


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

ALL_COUNTRIES: dict[str, str] = {c.name: c.alpha_3 for c in pycountry.countries}


def collect_all_countries(
    path_str = "../data/",
    mindat_api_str = "mindat_API_key.txt",
    material_id = 3314,
    mineral_strings = "(Fe|S)",
    material_name = "pyrite",
    sleep_between = 1.5,
    log_path = "../data/mindat datacollection_log.json",
):
    """
    Calls mindat_collector() for every country and returns a progress dict:

    count = -1 signals an API / parsing error for that country.

    The log_path JSON is written after every country so interrupted runs
    can be resumed safely — already-logged countries are skipped on restart.

    Parameters:
    --------------
    path_str : str
        Base path for storing the CSVs, optional, by default "../data/".
    mindat_api_str : str
        Path to the Mindat API key file, optional, by default "mindat_API_key.txt".
    material_id : int
        Mindat material ID for a mineral, optional, by default 3314 (pyrite).
    mineral_strings : str
        Regex string to match mineral names, optional, by default "(Fe|S)".
    material_name : str
        Name of the mineral, optional, by default "pyrite".
    sleep_between : float
        Seconds to sleep between API calls, optional, by default 1.5.
    log_path : str
        Path to the JSON log file, optional, by default "../data/mindat_data/collection_log.json".
    
    Returns:
    --------------
    dict
        Progress dictionary with country names as keys and a dictionary of
        ISO3 code, count of localities, and Mindat name as values: { pycountry_name: {"iso3": "...", "count": N, "mindat_name": "..."} }
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # resume after possible restarts / craches
    progress: dict = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            progress = json.load(f)
        log.info("Resuming: %d / %d countries already logged.",
                 len(progress), len(ALL_COUNTRIES))

    # loop over countries
    for pycountry_name, iso3 in ALL_COUNTRIES.items():
        if pycountry_name in progress:
            continue # already done

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

            # saving
            csv_path = (
                Path(path_str) / "mindat_data" / f"{mindat_name}_{material_name}.csv"
            )
            count = len(pd.read_csv(csv_path)) if csv_path.exists() else 0

        except Exception as exc:
            log.warning("  ✗  %s — %s", pycountry_name, exc)
            count = -1 

        progress[pycountry_name] = {
            "iso3":        iso3,
            "count":       count,
            "mindat_name": mindat_name,
        }

        
        with open(log_path, "w") as f:
            json.dump(progress, f, indent=2)

        time.sleep(sleep_between)

    log.info("Collection complete. %d countries processed.", len(progress))
    return progress


def build_summary(progress):
    """
    Converts the progress dict into a DataFrame ready for plotting.

    Parameters:
    ----------------
    progress : dict
        Progress dictionary returned by collect_all_countries().
    
    Returns:
    -----------
    pd.DataFrame
        DataFrame with columns: country, iso3, count, status.
        Columns
        -------
        country: pycountry English name
        iso3: ISO-3166 alpha-3 (used by plotly locations=)
        count: number of pyrite-mine localities  (errors → 0)
        status: 'ok' | 'error' | 'empty'
    """
    rows = []
    for country, info in progress.items():
        raw = info["count"]
        rows.append({
            "country": country,
            "iso3":    info["iso3"],
            "count":   max(raw, 0),                          
            "status":  "error" if raw < 0
                       else ("empty" if raw == 0 else "ok"),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return df


def plot_heatmap(
    df,
    log_scale = True,
    output_html ="pyrite_heatmap.html",
):
    """
    Renders an interactive choropleth map coloured by locality count.

    Parameters:
    ---------------
    df : pd.DataFrame
        DataFrame returned by build_summary().
    log_scale : bool
        Whether to use log₁₀(count + 1) for the colour scale, optional, by default True.
    output_html : str
        Path to save the interactive HTML, optional, by default "pyrite_heatmap.html".

    Returns:
    -------------
    fig : plotly.graph_objects.Figure
        The interactive choropleth figure.
    """
    import numpy as np

    plot_df = df.copy()

    if log_scale:
        plot_df["colour_val"] = np.log10(plot_df["count"] + 1)
        colour_label = "log₁₀(localities + 1)"
        tickvals = [0, 0.5, 1, 1.5, 2, 2.5, 3]
        ticktext = [f"{10**v - 1:.0f}" for v in tickvals] 
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
            "count": True,        
            "colour_val": False,        
            "iso3": False,
            "status": True,
        },
        color_continuous_scale="YlOrRd",  # white-yellow-orange-red
        range_color=(0, plot_df["colour_val"].max()),
        labels={"colour_val": colour_label, "count": "Localities"},
        title="Pyrite Mine Localities per Country — Mindat",
    )

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


if __name__ == "__main__":
    # config
    PATH_STR        = "../data/"
    MINDAT_API_FILE = "mindat_API_key.txt"
    LOG_PATH        = "../data/mindat_data/collection_log.json"
    SUMMARY_CSV     = "../data/mindat_data/summary.csv"
    HEATMAP_HTML    = "../data/mindat_datapyrite_heatmap.html"

    # collect
    progress = collect_all_countries(
        path_str=PATH_STR,
        mindat_api_str=MINDAT_API_FILE,
        log_path=LOG_PATH,
        sleep_between=1.5,   # increase if you hit rate-limit errors
    )

    # summarise
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

    # plot
    fig = plot_heatmap(summary, log_scale=True, output_html=HEATMAP_HTML)