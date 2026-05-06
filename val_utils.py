# validation utilities for AMDFLOW
import xarray as xr
import pandas as pd
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
import os
import scores as sc
from sklearn.metrics import r2_score

def load_datasets(amd_path, caravan_path, MGL_TO_UGL = 1e3):
    amd = xr.open_dataset(amd_path, chunks={})
    iron_vars = [v for v in amd.data_vars if "iron" in v]
    # amd = amd.assign({v: amd[v] * MGL_TO_UGL for v in iron_vars})
    # print(f"  Converted {iron_vars} from mg/L → µg/L")
    print("\nLoading Caravan-Qual Lite …")

    caravan = xr.open_dataset(caravan_path, engine="zarr",
                               chunks={})
    print(f"  {caravan.sizes['wqms_id']} stations, "
      f"{caravan.sizes['time']} time steps")
    return amd, caravan

def wqms_stations_domain_filter(amd, caravan):
    stations = pd.DataFrame({
        "wqms_id": caravan["wqms_id"].values,
        "lat":     caravan["wqms_lat"].values.astype(float),
        "lon":     caravan["wqms_lon"].values.astype(float),
        }).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    
    print(f"\n{len(stations)} stations with valid coordinates")

    # Cheap bounding-box pre-filter before the UTM projection
    lat_min, lat_max = float(amd.lat.min()), float(amd.lat.max())
    lon_min, lon_max = float(amd.lon.min()), float(amd.lon.max())
    pad = 0.1   # degrees of padding before the exact 5 km check

    in_bbox = (
        (stations["lat"] >= lat_min - pad) & (stations["lat"] <= lat_max + pad) &
        (stations["lon"] >= lon_min - pad) & (stations["lon"] <= lon_max + pad)
        )

    candidates = stations[in_bbox].reset_index(drop=True)
    print(f"{len(candidates)} stations fall inside (or near) the AMD domain "
        f"[lat {lat_min:.2f}–{lat_max:.2f}, lon {lon_min:.2f}–{lon_max:.2f}]")

    if len(candidates) == 0:
        raise SystemExit(
            "\nNo Caravan stations overlap the AMD model domain. "
            "Check that the datasets cover the same geographic region."
        )

    return candidates

def valid_masking(amd, VAR_MAP):
    VAR_MAP = {
        "pH":    "pH",
        "Fe-Dis": ["ferrous_iron", "ferric_iron"],
        "Fe-Tot": ["ferrous_iron", "ferric_iron",
                "iron_III_hydroxide"],
        }
    
    print("\nComputing valid-cell mask …")
    # Use the first AMD variable from VAR_MAP; unwrap list if composite.
    _ref_spec = list(VAR_MAP.values())[0]
    ref_var   = _ref_spec[0] if isinstance(_ref_spec, list) else _ref_spec

    # Sample 5 time steps evenly spread across the full run.
    # Robust to spin-up: flow-path cells that are NaN early will be caught
    # by later samples. Reads ~3.8 MB instead of the full 2 GB variable.
    _sample_idx = np.linspace(0, amd.sizes["time"] - 1, 10, dtype=int)

    valid_mask = (
        amd[ref_var]
        .isel(time=_sample_idx)
        .notnull()
        .any(dim="time")
        .compute()
    )

    iron_mask = (
        ((amd["ferrous_iron"] + amd["ferric_iron"] + amd["iron_III_hydroxide"])
        .isel(time=_sample_idx) > 0)
        .any(dim="time")
        .compute()
    )

    # Flatten grid to 1-D arrays of lat/lon for valid cells
    lat_vals = amd.lat.values
    lon_vals = amd.lon.values
    lat_2d, lon_2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")

    flat_valid = valid_mask.values.ravel()
    valid_lat  = lat_2d.ravel()[flat_valid]
    valid_lon  = lon_2d.ravel()[flat_valid]
    valid_ilat = np.where(flat_valid)[0] // len(lon_vals)
    valid_ilon = np.where(flat_valid)[0]  % len(lon_vals)

    flat_iron  = iron_mask.values.ravel()
    iron_lat   = lat_2d.ravel()[flat_iron]
    iron_lon   = lon_2d.ravel()[flat_iron]
    iron_ilat  = np.where(flat_iron)[0] // len(lon_vals)
    iron_ilon  = np.where(flat_iron)[0]  % len(lon_vals)

    print(f"  {flat_valid.sum():,} valid cells (pH mask)")
    print(f"  {flat_iron.sum():,} iron cells (iron data present mask)")
    return valid_ilat, valid_ilon, valid_lat, valid_lon, \
            iron_ilat, iron_ilon, iron_lat, iron_lon
def _safe_kdtree(lon_arr, lat_arr, to_utm):
    if len(lat_arr) == 0:
        return cKDTree(np.empty((0, 2)))
    east, north = to_utm.transform(lon_arr, lat_arr)
    return cKDTree(np.column_stack([east, north]))

def build_kdtree(amd, candidates, utm_crs, valid_lon, valid_lat,
                 iron_lon, iron_lat, ):
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

    st_east, st_north = to_utm.transform(
        candidates["lon"].values, candidates["lat"].values
    )

    tree = _safe_kdtree(valid_lon, valid_lat, to_utm)
    tree_iron = _safe_kdtree(iron_lon, iron_lat, to_utm)

    return tree, tree_iron, st_east, st_north

def match_stations(candidates, tree, ilat_arr, ilon_arr, lat_arr, lon_arr, 
                   st_east, st_north, MAX_DIST_M = 5000):
    
    if len(ilat_arr) == 0:
        empty = candidates.copy()
        empty["dist_m"] = np.inf
        empty["matched"] = False
        empty["ilat"] = -1
        empty["ilon"] = -1
        empty["cell_lat"] = np.nan
        empty["cell_lon"] = np.nan
        return empty[empty["matched"]].reset_index(drop=True)
    
    dist, idx = tree.query(
        np.column_stack([st_east, st_north]),
        k=1,
        distance_upper_bound=MAX_DIST_M,
        workers=-1,
    )
    matched_mask = dist < np.inf
    c = candidates.copy()
    c["dist_m"]   = dist
    c["matched"]  = matched_mask
    safe_idx = np.where(matched_mask, idx, 0)
    c["ilat"]     = np.where(matched_mask, ilat_arr[safe_idx], -1)
    c["ilon"]     = np.where(matched_mask, ilon_arr[safe_idx], -1)
    c["cell_lat"] = np.where(matched_mask, lat_arr[safe_idx],  np.nan)
    c["cell_lon"] = np.where(matched_mask, lon_arr[safe_idx],  np.nan)
    return c[c["matched"]].reset_index(drop=True)

def extract_and_align(matches, amd, caravan, caravan_var, amd_var, resample_freq):
    """Vectorized extraction 
    """
    # ── Overlapping period ────────────────────────────────────────────────────
    t_start = max(pd.Timestamp(amd.time.values[0]),
                  pd.Timestamp(caravan.time.values[0]))
    t_end   = min(pd.Timestamp(amd.time.values[-1]),
                  pd.Timestamp(caravan.time.values[-1]))

    if t_start >= t_end:
        raise ValueError(
            f"No overlapping period.\n"
            f"  AMD:     {amd.time.values[0]} – {amd.time.values[-1]}\n"
            f"  Caravan: {caravan.time.values[0]} – {caravan.time.values[-1]}"
        )
    print(f"  Overlapping period: {t_start.date()} → {t_end.date()}")

    # ── Step A: vectorized AMD extraction ─────────────────────────────────────
    # Deduplicate cells so identical model grid cells are only read once.
    cell_df = (matches[["ilat", "ilon"]]
               .drop_duplicates()
               .reset_index(drop=True))

    # Build DataArrays of integer indices — this is the key to vectorized isel.
    # xarray treats these as "pointwise" (not outer-product) indexing, so the
    # result is shape (time, n_unique_cells) rather than (time, nlat, nlon).
    ilat_da = xr.DataArray(cell_df["ilat"].values.astype(int), dims="cell")
    ilon_da = xr.DataArray(cell_df["ilon"].values.astype(int), dims="cell")

    # amd_var may be a single string or a list of strings to sum.
    amd_vars = [amd_var] if isinstance(amd_var, str) else amd_var
    label    = "+".join(amd_vars)   # e.g. "ferrous_iron+ferric_iron"

    print(f"  Reading {len(cell_df)} unique AMD cells "
          f"(covers {len(matches)} stations) …")
    print(f"  AMD variable(s): {label}")

    # Extract and optionally sum composite components — all in one compute call.
    def _extract_one(var: str) -> xr.DataArray:
        return (
            amd[var]
            .sel(time=slice(t_start, t_end))
            .isel(lat=ilat_da, lon=ilon_da)   # (time, cell)
        )

    if len(amd_vars) == 1:
        amd_extract = _extract_one(amd_vars[0]).compute()
    else:
        # Sum components lazily, then compute once — one task graph, one I/O pass
        amd_extract = sum(_extract_one(v) for v in amd_vars).compute()

    amd_times = pd.to_datetime(
        amd["time"].sel(time=slice(t_start, t_end)).values
    )

    # Long-form AMD DataFrame: one row per (cell, time)
    amd_df = (
        amd_extract
        .to_series()
        .rename("modelled")
        .reset_index()                     # columns: time, cell, modelled
    )
    amd_df["time"] = pd.to_datetime(amd_df["time"])

    # Map cell index back to (ilat, ilon) and then to wqms_id(s)
    # A single cell can be matched by multiple stations, so we merge on both.
    cell_df = cell_df.reset_index().rename(columns={"index": "cell"})
    amd_df  = amd_df.merge(cell_df, on="cell")   # adds ilat, ilon columns

    # Match → cell mapping (handles multiple stations sharing a cell)
    station_cell = matches[["wqms_id", "ilat", "ilon", "dist_m"]]
    amd_df = amd_df.merge(station_cell, on=["ilat", "ilon"])
    # amd_df now has: time, wqms_id, modelled, dist_m

    # ── Step B: Caravan observations ──────────────────────────────────────────
    obs_ids = matches["wqms_id"].values
    caravan_sliced = (
        caravan[caravan_var]
        .sel(wqms_id=obs_ids, time=slice(t_start, t_end))
    )
    if resample_freq:
        caravan_resampled = (
            caravan_sliced
            .resample(time=resample_freq)
            .mean(skipna=True)
            .compute()
        )
    else:
        caravan_resampled = caravan_sliced.compute()

    # Long-form Caravan DataFrame
    obs_df = (
        caravan_resampled
        .to_series()
        .rename("observed")
        .reset_index()
    )
    obs_df["time"] = pd.to_datetime(obs_df["time"])

    # ── Step C: time alignment via merge ──────────────────────────────────────
    # AMD is weekly; Caravan (after resampling) is also weekly but timestamps
    # may differ by a few days depending on the week anchor.
    # merge_asof requires the "on" key (time) to be globally monotonically
    # increasing across the ENTIRE DataFrame. Sorting by ["wqms_id", "time"]
    # resets the time sequence for every station, breaking that requirement.
    # The by="wqms_id" parameter handles per-station grouping internally.
    amd_df = amd_df.sort_values("time").reset_index(drop=True)
    obs_df = obs_df.sort_values("time").reset_index(drop=True)

    result = pd.merge_asof(
        amd_df,
        obs_df,
        on="time",
        by="wqms_id",
        direction="nearest",
        tolerance=pd.Timedelta("14D"),
    )
    # result columns: time, wqms_id, modelled, dist_m, observed

    n_obs = result["observed"].notna().sum()
    print(f"  {n_obs:,} / {len(result):,} timesteps have paired observations "
          f"({100 * n_obs / len(result):.1f} %)")

    return result[["wqms_id", "time", "observed", "modelled", "dist_m"]]

def full_run(VAR_MAP, MIN_PAIRED_OBS = 3, resample_freq="W"):
    all_results = {}
    IRON_VARS = {"Fe-Dis", "Fe-Tot"}

    for caravan_var, amd_var in VAR_MAP.items():
        matches = matches_iron if caravan_var in IRON_VARS else matches_ph

        if len(matches) == 0:
            print(f"\n{caravan_var}: No matched stations found; skipping variable")
            continue

        ts = extract_and_align(matches, amd, caravan, caravan_var, amd_var, resample_freq)
        
        # for sid, grp in ts.groupby("wqms_id"):
        #     paired = grp.dropna(subset=["observed", "modelled"])
        #     if len(paired) <= 3:
        #         continue
        #     else:
        #         print(sid,
        #             f"n={len(paired)}",
        #             f"obs_std={paired['observed'].std():.4f}",
        #             f"mod_std={paired['modelled'].std():.4f}",
        #             f"obs_mean={paired['observed'].mean():.4f}")


        # ── Drop stations with too few paired observations ────────────────────────
        # Count non-NaN pairs per station; remove any below MIN_PAIRED_OBS.
        n_paired = (
            ts.dropna(subset=["observed", "modelled"])
            .groupby("wqms_id")
            .size()
            .rename("n_paired")
        )
        keep = n_paired[n_paired >= MIN_PAIRED_OBS].index
        dropped = n_paired[n_paired < MIN_PAIRED_OBS]
        if len(dropped):
            print(f"  Dropped {len(dropped)} station(s) with < {MIN_PAIRED_OBS} "
                f"paired observations: {dropped.to_dict()}")
            
        # drop stations where all modelled values are 0
        all_zero = (
            ts.groupby("wqms_id")["modelled"]
            .apply(lambda x: (x == 0).all())
        )
        zero_stations = all_zero[all_zero].index
        if len(zero_stations):
            print(f"  Dropped {len(zero_stations)} station(s) with all-zero "
                f"modelled values: {zero_stations.tolist()}")
            keep = keep.difference(zero_stations)
        ts = ts[ts["wqms_id"].isin(keep)].reset_index(drop=True)

        n_stations = ts["wqms_id"].nunique()
        all_results[caravan_var] = ts
    
    return all_results, matches

def compute_metrics(obs: np.ndarray, mod: np.ndarray) -> dict:
    o = obs.resample("W").mean()
    o = xr.DataArray(o.values, coords = {"time": o.index}, dims = ["time"])
    m = xr.DataArray(mod.values, coords = {"time": mod.index}, dims = ["time"])
    o, m = xr.align(o, m, join = "inner")

    valid = np.isfinite(o.values) & np.isfinite(m.values)
    o = o.isel(time = valid)
    m = m.isel(time = valid)
    n = int(o.sizes["time"])
    if n < 3:
        return dict(n=n, RMSE=np.nan, bias=np.nan, NSE=np.nan, KGE=np.nan, R=np.nan)
    if o.values.std() == 0:
        return dict(n=n, RMSE=np.nan, bias=np.nan,
                    NSE=np.nan, KGE=np.nan, R=np.nan)
    if m.values.std() == 0:
        return dict(n=n, RMSE=np.nan, bias=np.nan,
                    NSE=np.nan, KGE=np.nan, R=np.nan)
    
    rmse = sc.continuous.rmse(m, o).compute().values
    bias = np.mean(m - o).values
    nse = sc.continuous.nse(o, m).compute().values
    r = r2_score(o.values, m.values)
    kge = sc.continuous.kge(o, m).compute().values
    return dict(n=n, RMSE=rmse, bias=bias, NSE=nse, KGE=kge, R=r)

def validation_metrics(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, grp in ts.groupby("wqms_id"):
        grp_t = grp.set_index("time")
        m = compute_metrics(grp_t["observed"], grp_t["modelled"])
        m["wqms_id"] = sid
        rows.append(m)

    df = pd.DataFrame(rows).set_index("wqms_id")

    # Drop station rows where every metric is NaN (n < 3 after pairing).
    # Keep the "ALL" pooled row regardless — it should always have enough data.
    metric_cols = ["RMSE", "bias", "NSE", "KGE", "R"]
    station_rows = df.index != "ALL"
    all_nan = df.loc[station_rows, metric_cols].isna().all(axis=1)
    if all_nan.any():
        dropped = all_nan[all_nan].index.tolist()
        print(f"  Excluded {len(dropped)} station(s) with all-NaN metrics "
              f"(n < 3): {dropped}")
        df = df[~(station_rows & all_nan)]

    return df

if __name__ == "__main__":
    case_study_nr = "CSIII"
    times = ("1960", "2015")
    caravan_path = "../data/validation data/Caravan-Qual_lite.zarr"   # adjust extension if .nc
    amd_path = f"../data/validation data/AMDFLOW_{case_study_nr}_{times[0]}-{times[1]}_W.nc"
    output_path = f"../data/validation data/{case_study_nr}/"
    utm_crs = "EPSG:6312"  # "EPSG:5641" (Brazil), "EPSG:32650" (China), "EPSG:25833" (N europe), "EPSG:6312" (Cyprus) # "EPSG: 3761" (Canada), "EPSG:24378" (India), "EPSG:26712" (USA)
    resample_freq = "W"
    VAR_MAP = {
        "pH":    "pH",
        "Fe-Dis": ["ferrous_iron", "ferric_iron"],
        "Fe-Tot": ["ferrous_iron", "ferric_iron",
                "iron_III_hydroxide"],
        }
    print(f"Validating AMDFLOW Case Study {case_study_nr} ({times[0]}–{times[1]}), \nagainst Caravan-Qual Lite resampled to weekly frequency with bi-weekly time tolerance and 5km pairing distance.")
    amd, caravan = load_datasets(amd_path, caravan_path)

    candidates = wqms_stations_domain_filter(amd, caravan)

    valid_ilat, valid_ilon, valid_lat, valid_lon, \
            iron_ilat, iron_ilon, iron_lat, iron_lon = valid_masking(amd, VAR_MAP)
    
    tree, tree_iron, st_east, st_north = build_kdtree(amd, candidates, utm_crs, valid_lon, valid_lat,
                                         iron_lon, iron_lat)
    
    matches_ph = match_stations(candidates, tree, valid_ilat, valid_ilon,
                                valid_lat, valid_lon, st_east, st_north)
    
    matches_iron = match_stations(candidates,tree_iron, iron_ilat, iron_ilon, 
                                  iron_lat, iron_lon, st_east, st_north)
    
    print(f"  pH mask:   {len(matches_ph)} stations matched")
    print(f"  Iron mask: {len(matches_iron)} stations matched")

    all_results, matches = full_run(VAR_MAP, MIN_PAIRED_OBS=3, resample_freq=resample_freq)

    for var, ts in all_results.items():
        print(f"\n── {var} ──────────────────────────────────────────")
        metrics = validation_metrics(ts)
        
        coords = matches[["wqms_id", "lat", "lon", "cell_lat", "cell_lon"]].set_index("wqms_id")
        results_df = metrics.join(coords)
        coord_cols = ["lat", "lon", "cell_lat", "cell_lon"]
        station_rows = results_df.index != "ALL"
        missing_coords = results_df.loc[station_rows, coord_cols].isna().any(axis=1)
        if missing_coords.any():
            dropped = missing_coords[missing_coords].index.tolist()
            print(f"  Warning: {len(dropped)} station(s) have missing coordinates "
                  f"in the metrics output: {dropped}")
            results_df = results_df.loc[~results_df.index.isin(dropped)]
        
        os.makedirs(output_path, exist_ok=True)
        results_df.to_csv(f"{output_path}metrics_{var}.csv")
        print(f"  Saved → {output_path}metrics_{var}.csv")
