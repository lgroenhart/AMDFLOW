# validation utilities for AMDFLOW
import xarray as xr
import pandas as pd
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
import os
import scores as sc
from sklearn.metrics import r2_score
import rasterio
from shapely.geometry import Point
import networkx as nx
import warnings
import geopandas as gpd

# Suppress warnings if desired
warnings.filterwarnings("ignore")


def load_datasets(amd_path, caravan_path, acc_path, rivers_path):
    """
    Load AMDFLOW NetCDF, Caravan-Qual Zarr, HydroSHEDS ACC raster, and river network.
    Returns: amd, caravan, acc_array, acc_transform, rivers_gdf, cell_area_func
    """
    print("\nLoading all datasets.....")
    amd = xr.open_dataset(amd_path, chunks={})
    
    # open ACC raster (keep dataset open, read band once)
    acc_ds = rasterio.open(acc_path)
    acc_array = acc_ds.read(1)          
    acc_transform = acc_ds.transform
    acc_nodata = acc_ds.nodata
    
    rivers = gpd.read_file(rivers_path)
    required = ['HYRIV_ID', 'NEXT_DOWN', 'LENGTH_KM']
    for col in required:
        if col not in rivers.columns:
            raise ValueError(f"River network missing column: {col}")
    
    caravan = xr.open_dataset(caravan_path, engine="zarr", chunks={})
    
    def cell_area_km2(lat, lon):
        """Return approximate area (km²) of a 30-arc-second cell at given latitude."""
        deg_to_km = 111.32
        cell_width_deg = 1/120.0   # 30 arc-seconds
        km_per_deg_lon = deg_to_km * np.cos(np.radians(lat))
        return (deg_to_km * cell_width_deg) * (km_per_deg_lon * cell_width_deg)
    
    return amd, caravan, acc_array, acc_transform, acc_nodata, rivers, cell_area_km2

def wqms_stations_domain_filter(amd, caravan):
    stations = caravan[["wqms_id", "wqms_lat", "wqms_lon", "merged_LINKNO"]].to_dataframe().reset_index()
    stations = stations.rename(columns={
        "wqms_lat": "lat",
        "wqms_lon": "lon",
        "merged_LINKNO": "linkno"
    })

    # drop rows with missing needed data
    stations = stations.dropna(subset=["lat", "lon", "linkno"]).reset_index(drop=True)

    area = caravan["DSContArea"].to_dataframe().reset_index()  # columns: LINKNO, DSContArea
    area = area.rename(columns={"DSContArea": "area_sqkm"})

    # merge on LINKNO
    stations = stations.merge(area, left_on="linkno", right_on="LINKNO", how="inner")
    stations = stations.drop(columns=["linkno", "LINKNO"])

    if 'index' in stations.columns:
        stations = stations.drop(columns=['index'])
    
    # convert DSContArea from m2 to km2
    stations["area_sqkm"] = stations["area_sqkm"] / 1e6

    # bounding‑box pre‑filter
    lat_min, lat_max = float(amd.lat.min()), float(amd.lat.max())
    lon_min, lon_max = float(amd.lon.min()), float(amd.lon.max())
    pad = 0.1
    in_bbox = ((stations["lat"] >= lat_min - pad) & (stations["lat"] <= lat_max + pad) &
               (stations["lon"] >= lon_min - pad) & (stations["lon"] <= lon_max + pad))
    candidates = stations[in_bbox].reset_index(drop=True)
    print(f"{len(candidates)} stations fall inside (or near) the AMD domain")

    if len(candidates) == 0:
        raise SystemExit("\nNo Caravan stations overlap the AMD model domain.")
    return candidates

def valid_masking(amd):
    """
    Compute boolean masks for cells that have valid data for pH and for iron.
    """
    print("\nComputing valid-cell mask …")
    _sample_idx = np.linspace(0, amd.sizes["time"] - 1, 10, dtype=int)
    
    # pH mask: any non-null pH value in any time sample
    valid_mask = (amd["pH"].isel(time=_sample_idx).notnull().any(dim="time").compute())
    
    # Iron mask: any positive sum of iron components
    iron_sum = (amd["ferrous_iron"] + amd["ferric_iron"] + amd["iron_III_hydroxide"])
    iron_mask = (iron_sum.isel(time=_sample_idx) > 0).any(dim="time").compute()
    
    # 2D lat/lon grids
    lat_vals = amd.lat.values
    lon_vals = amd.lon.values
    lat_2d, lon_2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    
    # flattened indices for later use (deprecated)
    flat_valid = valid_mask.values.ravel()
    valid_lat  = lat_2d.ravel()[flat_valid]
    valid_lon  = lon_2d.ravel()[flat_valid]
    valid_ilat = np.where(flat_valid)[0] // len(lon_vals)
    valid_ilon = np.where(flat_valid)[0]  % len(lon_vals)
    
    flat_iron = iron_mask.values.ravel()
    iron_lat  = lat_2d.ravel()[flat_iron]
    iron_lon  = lon_2d.ravel()[flat_iron]
    iron_ilat = np.where(flat_iron)[0] // len(lon_vals)
    iron_ilon = np.where(flat_iron)[0]  % len(lon_vals)
    
    print(f"  {flat_valid.sum():,} valid cells (pH mask)")
    print(f"  {flat_iron.sum():,} iron cells")
    
    return (valid_ilat, valid_ilon, valid_lat, valid_lon,
            iron_ilat, iron_ilon, iron_lat, iron_lon,
            valid_mask, iron_mask)

def build_river_graph(rivers_gdf):
    """Build directed graph of river segments (nodes: HYRIV_ID)."""
    G = nx.DiGraph()
    for _, row in rivers_gdf.iterrows():
        sid = row["HYRIV_ID"]
        nid = row["NEXT_DOWN"]
        length = row["LENGTH_KM"]
        G.add_node(sid)
        if nid != -1 and not pd.isna(nid):
            G.add_edge(sid, nid, length=length)
    return G

def snap_cells_to_river(amd_lat_2d, amd_lon_2d, valid_mask, rivers_gdf, target_crs):
    """
    For each valid AMDFLOW cell, find nearest point on river network.
    Returns dict: (ilat, ilon) -> {HYRIV_ID, point_proj, distance_m, point_deg}
    """
    print("Snapping cells to river(s)")
    # reproject rivers to target CRS
    rivers_proj = rivers_gdf.to_crs(target_crs)
    sindex = rivers_proj.sindex
    
    # prepare cell points in projected CRS
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    cell_to_river = {}
    
    cells = list(zip(*np.where(valid_mask)))
    for ilat, ilon in cells:
        lon = amd_lon_2d[ilat, ilon]
        lat = amd_lat_2d[ilat, ilon]
        pt_deg = Point(lon, lat)
        pt_proj = Point(transformer.transform(lon, lat))
        
        # query river segments within 5 km buffer
        buffer_m = 5000.0
        possible = list(sindex.intersection(pt_proj.buffer(buffer_m).bounds))
        best_dist = np.inf
        best_seg = None
        best_proj = None
        for idx in possible:
            seg_geom = rivers_proj.iloc[idx].geometry
            proj_pt = seg_geom.interpolate(seg_geom.project(pt_proj))
            dist_m = pt_proj.distance(proj_pt)
            if dist_m < best_dist:
                best_dist = dist_m
                best_seg = rivers_proj.iloc[idx]['HYRIV_ID']
                best_proj = proj_pt
        if best_seg is not None:
            cell_to_river[(ilat, ilon)] = {
                'HYRIV_ID': best_seg,
                'point_proj': best_proj,
                'distance_m': best_dist,
                'point_deg': Point(transformer.transform(best_proj.x, best_proj.y, direction='INVERSE'))
            }
    return cell_to_river

def assign_uparea_from_acc(amd_lat_2d, amd_lon_2d, valid_mask, acc_array, acc_transform, acc_nodata, cell_area_func):
    """
    Read upstream area (km²) for each valid AMDFLOW cell from ACC grid.
    acc_array is a 2D numpy array (already read).
    """
    print("Assigning upstream watershed area to cells from flow accumulation")
    uparea_dict = {}
    for ilat, ilon in zip(*np.where(valid_mask)):
        lon = amd_lon_2d[ilat, ilon]
        lat = amd_lat_2d[ilat, ilon]
        # convert lon/lat to row, col in ACC grid (assuming same CRS: WGS84)
        col, row = ~acc_transform * (lon, lat)   
        row_int, col_int = int(round(row)), int(round(col))
        if 0 <= row_int < acc_array.shape[0] and 0 <= col_int < acc_array.shape[1]:
            acc_val = acc_array[row_int, col_int]
            if acc_val > 0 and acc_val != acc_nodata:
                cell_area_km2 = cell_area_func(lat, lon)
                uparea_km2 = acc_val * cell_area_km2
                uparea_dict[(ilat, ilon)] = uparea_km2
            else:
                uparea_dict[(ilat, ilon)] = np.nan
        else:
            uparea_dict[(ilat, ilon)] = np.nan
    return uparea_dict

def compute_network_distance(seg_id_a, point_a_proj, seg_id_b, point_b_proj, rivers_proj, graph):
    """
    Compute shortest path distance (km) along river network between two points.
    Assumes all geometries are in the same projected CRS (metres).
    """
    if seg_id_a == seg_id_b:
        # Same segment: linear distance along the line
        seg_geom = rivers_proj[rivers_proj['HYRIV_ID'] == seg_id_a].geometry.iloc[0]
        pos_a = seg_geom.project(point_a_proj)
        pos_b = seg_geom.project(point_b_proj)
        return abs(pos_a - pos_b) / 1000.0   # metres -> km
    
    # Different segments: need shortest path through graph.
    # We approximate by using the endpoints of each segment, 
    # which requires that we know the nodes of the network.
    # For simplicity, we fall back to Euclidean distance if no path found,
    # with a warning.
    try:
        # Attempt to find path using segment IDs as nodes.
        # This is incomplete (should use node-to-node distances) but works
        # if the river network graph is built with segments as nodes and edge lengths.
        path = nx.shortest_path(graph, source=seg_id_a, target=seg_id_b, weight='length')
        # We ignore the on-ramp/off-ramp distances at the ends (simplification)
        total_km = sum(graph[path[i]][path[i+1]]['length'] for i in range(len(path)-1))
        return total_km
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Fallback: Euclidean distance between projected points (warning)
        warnings.warn(f"No network path between {seg_id_a} and {seg_id_b}; using Euclidean distance")
        return point_a_proj.distance(point_b_proj) / 1000.0

def snap_stations_hydrosheds(candidates, amd_lat_2d, amd_lon_2d, valid_mask,
                             uparea_dict, cell_to_river, rivers_gdf, river_graph,
                             target_crs, max_network_km=5.0,
                             area_ratio_min=0.5, area_ratio_max=1.5):
    """
    Match Caravan stations to AMDFLOW cells using network distance and area ratio.
    Returns DataFrame with matched stations and cell indices.
    """
    print("Snapping stations to rivers and then cells")
    # reproject rivers and stations to target CRS
    rivers_proj = rivers_gdf.to_crs(target_crs)
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    
    # build spatial index for projected rivers
    sindex = rivers_proj.sindex
    
    # snap each station to nearest river point (projected)
    station_snaps = {}
    for _, row in candidates.iterrows():
        wqms_id = row['wqms_id']
        pt_deg = Point(row['lon'], row['lat'])
        pt_proj = Point(transformer.transform(row['lon'], row['lat']))
        
        # query rivers within 50 km 
        possible = list(sindex.intersection(pt_proj.buffer(50000).bounds))
        best_dist = np.inf
        best_seg = None
        best_proj = None
        for idx in possible:
            seg_geom = rivers_proj.iloc[idx].geometry
            proj_pt = seg_geom.interpolate(seg_geom.project(pt_proj))
            dist_m = pt_proj.distance(proj_pt)
            if dist_m < best_dist:
                best_dist = dist_m
                best_seg = rivers_proj.iloc[idx]['HYRIV_ID']
                best_proj = proj_pt
        if best_seg is not None and best_dist / 1000.0 <= 50.0:
            station_snaps[wqms_id] = {
                'HYRIV_ID': best_seg,
                'point_proj': best_proj,
                'point_deg': Point(transformer.transform(best_proj.x, best_proj.y, direction='INVERSE')),
                'distance_to_river_km': best_dist / 1000.0
            }
    
    # match each station to the best cell
    matches = []
    for _, row in candidates.iterrows():
        wqms_id = row['wqms_id']
        station_area = row['area_sqkm']
        if wqms_id not in station_snaps:
            continue
        sta_snap = station_snaps[wqms_id]
        sta_seg = sta_snap['HYRIV_ID']
        sta_pt_proj = sta_snap['point_proj']
        
        best_cell = None
        best_net_dist = np.inf

        for (ilat, ilon), cell_info in cell_to_river.items():
            cell_seg = cell_info['HYRIV_ID']
            cell_pt_proj = cell_info['point_proj']
            uparea = uparea_dict.get((ilat, ilon), np.nan)
            if np.isnan(uparea):
                continue

            # compute network distance
            net_dist_km = compute_network_distance(sta_seg, sta_pt_proj,
                                                    cell_seg, cell_pt_proj,
                                                    rivers_proj, river_graph)
            if net_dist_km > max_network_km:
                continue

            # area ratio check
            ratio = station_area / uparea
            if not (area_ratio_min <= ratio <= area_ratio_max):
                continue

            if net_dist_km < best_net_dist:
                best_net_dist = net_dist_km
                best_cell = (ilat, ilon, uparea, ratio)
        
        if best_cell is not None:
            ilat, ilon, uparea, ratio = best_cell
            matches.append({
                'wqms_id': wqms_id,
                'dist_m': best_net_dist * 1000, 
                'ilat': ilat,
                'ilon': ilon,
                'cell_lat': amd_lat_2d[ilat, ilon],
                'cell_lon': amd_lon_2d[ilat, ilon],
                'station_area_km2': station_area,
                'cell_uparea_km2': uparea,
                'area_ratio': ratio
            })
        
    return pd.DataFrame(matches)

def extract_and_align(matches, amd, caravan, caravan_var, amd_var, resample_freq):
    """Extract and align model and observed time series for matched stations."""
    # Overlapping period
    t_start = max(pd.Timestamp(amd.time.values[0]),
                  pd.Timestamp(caravan.time.values[0]))
    t_end   = min(pd.Timestamp(amd.time.values[-1]),
                  pd.Timestamp(caravan.time.values[-1]))
    if t_start >= t_end:
        raise ValueError("No overlapping period.")
    print(f"  Overlapping period: {t_start.date()} → {t_end.date()}")
    
    # deduplicate cells
    cell_df = matches[["ilat", "ilon"]].drop_duplicates().reset_index(drop=True)
    ilat_da = xr.DataArray(cell_df["ilat"].values.astype(int), dims="cell")
    ilon_da = xr.DataArray(cell_df["ilon"].values.astype(int), dims="cell")
    
    amd_vars = [amd_var] if isinstance(amd_var, str) else amd_var
    label = "+".join(amd_vars)
    print(f"  Reading {len(cell_df)} unique AMD cells (covers {len(matches)} stations) …")
    print(f"  AMD variable(s): {label}")
    
    def _extract_one(var):
        return (amd[var].sel(time=slice(t_start, t_end))
                .isel(lat=ilat_da, lon=ilon_da))
    
    if len(amd_vars) == 1:
        amd_extract = _extract_one(amd_vars[0]).compute()
    else:
        amd_extract = sum(_extract_one(v) for v in amd_vars).compute()
    
    amd_df = (amd_extract.to_series().rename("modelled").reset_index())
    amd_df["time"] = pd.to_datetime(amd_df["time"])
    
    # map cell index back to coordinates and station ids
    cell_df = cell_df.reset_index().rename(columns={"index": "cell"})
    amd_df = amd_df.merge(cell_df, on="cell")
    station_cell = matches[["wqms_id", "ilat", "ilon", "dist_m"]]
    amd_df = amd_df.merge(station_cell, on=["ilat", "ilon"])
    
    # Caravan observations
    obs_ids = matches["wqms_id"].values
    caravan_sliced = caravan[caravan_var].sel(wqms_id=obs_ids, time=slice(t_start, t_end))
    if resample_freq:
        caravan_resampled = caravan_sliced.resample(time=resample_freq).mean(skipna=True).compute()
    else:
        caravan_resampled = caravan_sliced.compute()
    
    obs_df = caravan_resampled.to_series().rename("observed").reset_index()
    obs_df["time"] = pd.to_datetime(obs_df["time"])
    
    # merge
    amd_df = amd_df.sort_values("time").reset_index(drop=True)
    obs_df = obs_df.sort_values("time").reset_index(drop=True)
    result = pd.merge_asof(amd_df, obs_df, on="time", by="wqms_id",
                           direction="nearest", tolerance=pd.Timedelta("14D"))
    
    n_obs = result["observed"].notna().sum()
    print(f"  {n_obs:,} / {len(result):,} timesteps have paired observations ({100 * n_obs / len(result):.1f} %)")
    
    return result[["wqms_id", "time", "observed", "modelled", "dist_m"]]

def full_run(var_map, matches_ph, matches_iron, amd, caravan,
             min_paired_obs=3, resample_freq="W"):
    """Run extraction and alignment for all variables."""
    all_results = {}
    iron_vars = {"Fe-Dis", "Fe-Tot"}
    
    for caravan_var, amd_var in var_map.items():
        if caravan_var in iron_vars:
            matches = matches_iron
        else:
            matches = matches_ph
        
        if len(matches) == 0:
            print(f"\n{caravan_var}: No matched stations found; skipping")
            continue
        
        ts = extract_and_align(matches, amd, caravan, caravan_var, amd_var, resample_freq)
        
        # filter stations with too few paired observations
        n_paired = ts.dropna(subset=["observed", "modelled"]).groupby("wqms_id").size()
        keep = n_paired[n_paired >= min_paired_obs].index
        dropped = n_paired[n_paired < min_paired_obs]
        if len(dropped):
            print(f"  Dropped {len(dropped)} station(s) with < {min_paired_obs} paired observations")
        
        # drop stations where modelled all zero
        all_zero = ts.groupby("wqms_id")["modelled"].apply(lambda x: (x == 0).all())
        zero_stations = all_zero[all_zero].index
        if len(zero_stations):
            print(f"  Dropped {len(zero_stations)} station(s) with all-zero modelled values")
            keep = keep.difference(zero_stations)
        
        ts = ts[ts["wqms_id"].isin(keep)].reset_index(drop=True)
        all_results[caravan_var] = ts
    
    return all_results

def compute_metrics(obs, mod):
    """Compute RMSE, bias, NSE, KGE, R for two pandas Series (aligned by time)."""
    o = obs.resample("W").mean()
    m = mod.resample("W").mean()

    common = o.index.intersection(m.index)
    o = o.loc[common]
    m = m.loc[common]
    valid = np.isfinite(o) & np.isfinite(m)
    o = o[valid]
    m = m[valid]
    n = len(o)
    if n < 3 or o.std() == 0 or m.std() == 0:
        return {"n": n, "RMSE": np.nan, "bias": np.nan, "NSE": np.nan, "KGE": np.nan, "R": np.nan}
    
    # Use xarray for compat with scores
    o_da = xr.DataArray(o.values, dims=["time"])
    m_da = xr.DataArray(m.values, dims=["time"])
    rmse = sc.continuous.rmse(m_da, o_da).compute().values.item()
    bias = np.mean(m - o)
    nse = sc.continuous.nse(o_da, m_da).compute().values.item()
    r = r2_score(o, m)
    kge = sc.continuous.kge(o_da, m_da).compute().values.item()
    return {"n": n, "RMSE": rmse, "bias": bias, "NSE": nse, "KGE": kge, "R": r}

def validation_metrics(ts):
    """Compute per-station metrics and return DataFrame."""
    rows = []
    for sid, grp in ts.groupby("wqms_id"):
        grp = grp.set_index("time").sort_index()
        metrics = compute_metrics(grp["observed"], grp["modelled"])
        metrics["wqms_id"] = sid
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("wqms_id")
    
    # drop rows where all metrics are NaN
    metric_cols = ["RMSE", "bias", "NSE", "KGE", "R"]
    all_nan = df[metric_cols].isna().all(axis=1)
    if all_nan.any():
        print(f"  Excluded {all_nan.sum()} station(s) with all-NaN metrics (n<3)")
        df = df[~all_nan]
    return df

if __name__ == "__main__":
    case_study_nr = "CSIII"
    times = ("1960", "2015")
    caravan_path = "../data/validation data/Caravan-Qual_lite.zarr"
    amd_path = f"../data/validation data/AMDFLOW_{case_study_nr}_{times[0]}-{times[1]}_W.nc"
    output_path = f"../data/validation data/{case_study_nr}/"
    acc_path = "../data/validation data/hyd_eu_acc_30s.tif"
    rivers_path = "../data/validation data/HydroRIVERS_v10_eu_shp/HydroRIVERS_v10_eu.shp"
    
    # choose UTM zone appropriate for your domain 
    utm_crs = "EPSG:6312"   
    resample_freq = "W"
    var_map = {
        "pH": "pH",
        "Fe-Dis": ["ferrous_iron", "ferric_iron"],
        "Fe-Tot": ["ferrous_iron", "ferric_iron", "iron_III_hydroxide"],
    }
    
    print(f"Validating AMDFLOW Case Study {case_study_nr} ({times[0]}–{times[1]}),")
    print("against Caravan-Qual Lite using HydroSHEDS network + area ratio snapping.")
    
    # load everything
    amd, caravan, acc_array, acc_transform, acc_nodata, rivers, cell_area_func = load_datasets(
        amd_path, caravan_path, acc_path, rivers_path
    )
    
    candidates = wqms_stations_domain_filter(amd, caravan)
    
    # get masks
    (valid_ilat, valid_ilon, valid_lat, valid_lon,
     iron_ilat, iron_ilon, iron_lat, iron_lon,
     valid_mask, iron_mask) = valid_masking(amd)
    
    # build river graph
    river_graph = build_river_graph(rivers)
    
    # 2D lat/lon grids
    lat_vals = amd.lat.values
    lon_vals = amd.lon.values
    amd_lat_2d, amd_lon_2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    
    # upstream areas from ACC
    uparea_dict = assign_uparea_from_acc(amd_lat_2d, amd_lon_2d, valid_mask,
                                         acc_array, acc_transform, acc_nodata, cell_area_func)
    
    # snap AMD cells to river network (once)
    cell_to_river = snap_cells_to_river(amd_lat_2d, amd_lon_2d, valid_mask, rivers, utm_crs)
    print(f"Cells snapped to river: {len(cell_to_river)}")
    # snap stations for pH mask
    matches_ph = snap_stations_hydrosheds(
        candidates, amd_lat_2d, amd_lon_2d, valid_mask,
        uparea_dict, cell_to_river, rivers, river_graph, utm_crs,
        max_network_km=5.0, area_ratio_min=0.5, area_ratio_max=2.0
    )
    
    # snap stations for iron mask
    iron_cells = [(ilat, ilon) for ilat, ilon in zip(*np.where(iron_mask))]
    iron_uparea_dict = {k: uparea_dict.get(k, np.nan) for k in iron_cells}
    iron_cell_to_river = {k: v for k, v in cell_to_river.items() if k in iron_cells}
    
    matches_iron = snap_stations_hydrosheds(
        candidates, amd_lat_2d, amd_lon_2d, iron_mask,
        iron_uparea_dict, iron_cell_to_river, rivers, river_graph, utm_crs,
        max_network_km=5.0, area_ratio_min=0.5, area_ratio_max=2.0
    )
    
    print(f"  pH mask:   {len(matches_ph)} stations matched")
    print(f"  Iron mask: {len(matches_iron)} stations matched")
    
    # run full validation
    all_results = full_run(var_map, matches_ph, matches_iron, amd, caravan,
                           min_paired_obs=3, resample_freq=resample_freq)
    
    for var, ts in all_results.items():
        print(f"\n── {var} ──────────────────────────────────────────")
        metrics = validation_metrics(ts)
        
        # add coordinates for matched stations (pH and iron may differ)
        if var in ["Fe-Dis", "Fe-Tot"]:
            match_df = matches_iron
        else:
            match_df = matches_ph
        
        coords = match_df[["wqms_id", "cell_lat", "cell_lon"]].set_index("wqms_id")
        results_df = metrics.join(coords)
        
        os.makedirs(output_path, exist_ok=True)
        results_df.to_csv(f"{output_path}metrics_{var}.csv")
        print(f"  Saved → {output_path}metrics_{var}.csv")