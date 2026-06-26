# validation utilities for AMDFLOW
import xarray as xr
import pandas as pd
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.stats import iqr, pearsonr
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

def extract_caravan_stations(input_path = "../data/validation data/Caravan-Qual_lite.zarr", 
                             output_path = "../data/validation data/stations_"):
    """Helper function to get and save the global water quality observation stations of Caravan-Qual Lite,
        useful for plotting, mapping, looking for case studies, etc.

    Parameters
    ----------
    input_path : str, optional
        Path to Caravan-Qual Lite dataset in zarr format, by default "../data/validation data/Caravan-Qual_lite.zarr"
    output_path : str, optional
        Path to save the extracted station data as csv files (per variable), by default "../data/validation data/stations_"
    """
    ds = xr.open_zarr(input_path)

    target_vars = ["pH", "Fe-Dis", "Fe-Tot"]

    # load lat/lon once, both are (wqms_id,) arrays
    lats = ds["wqms_lat"].compute()
    lons = ds["wqms_lon"].compute()

    for var in target_vars:
        if var not in ds.data_vars:
            print(f"Skipping '{var}': not found in dataset")
            continue

        da = ds[var]  

        # boolean mask: stations with at least one valid observation
        has_data = da.notnull().any(dim="time").compute()

        valid_mask = has_data.values

        df = pd.DataFrame({
            "wqms_id":   ds["wqms_id"].values[valid_mask],
            "latitude":  lats.values[valid_mask],
            "longitude": lons.values[valid_mask],
        })

        filename = f"{output_path}{var.replace('-', '_')}.csv"
        df.to_csv(filename, index=False)
        print(f"{var}: {len(df)} stations → saved to '{filename}'")
    return 

def load_datasets(amd_path, caravan_path, acc_path, rivers_path):
    """ Load AMDFLOW NetCDF, Caravan-Qual Zarr, HydroSHEDS ACC raster, and river network.
        all dataset should have WG84 lat/lon coordinates

    Parameters
    ----------
    amd_path : str
        path to AMDFLOW netcdf file
    caravan_path : str
        path to Caravan-Qual Lite zarr file
    acc_path : str
        path to HydroSHEDS flow accumulation raster (ACC), .tif format
    rivers_path : str
        path to HydroRIVERS river network shapefile, .shp file

    Returns
    -------
    amd : xr.Dataset
        dataset of AMDFLOW output
    caravan : xr.Dataset
        Caravan-Qual Lite dataset
    acc_array : np.ndarray
        2d array of flow accumulation (ACC) raster values, values are amount of upstream cells
    acc_transform : rio.transform
        transform object of ACC raster
    acc_nodata : rio.nodata 
        nodata value of ACC raster
    rivers : Geodataframe
        river network with columns HYRIV_ID, NEXT_DOWN, LENGTH_KM, geometry
    cell_area_func : function (deprecated)
        function(lat, lon) that calculates area of raster cell, no longer used, 1 km2 cell assumed 
    Raises
    ------
    ValueError
        River network missing one of required columns
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
    
    return amd, caravan, acc_array, acc_transform, acc_nodata, rivers

def cell_area(amd):

    # adapted from: https://gis.stackexchange.com/questions/317392/determine-area-of-cell-in-raster-qgis 
    lats_1d = amd["lat"].values
    lons_1d = amd["lon"].values

    rows = len(lats_1d)
    cols = len(lons_1d)


    pix_height = float(lats_1d[1] - lats_1d[0])
    pix_width  = float(lons_1d[1] - lons_1d[0])


    ulY_edge = float(lats_1d[0])  - pix_height / 2
    lrY_edge = float(lats_1d[-1]) + pix_height / 2

    a = 6378137
    b = 6356752.3142

    lats = np.linspace(ulY_edge, lrY_edge, rows + 1)
    lats = lats * np.pi / 180

    e       = np.sqrt(1 - (b/a)**2)
    sinlats = np.sin(lats)
    zm = 1 - e * sinlats
    zp = 1 + e * sinlats

    q = pix_width / 360

    areas_to_equator   = np.pi * b**2 * (2*np.arctanh(e*sinlats) / (2*e) + sinlats / (zp*zm)) / 10**6
    areas_between_lats = np.diff(areas_to_equator)
    areas_cells        = np.abs(areas_between_lats) * q

    areagrid = np.transpose(np.matlib.repmat(areas_cells, cols, 1))
    return areagrid

def wqms_stations_domain_filter(amd, caravan):
    """bounding box filter of caravan stations to be within general AMD region 

    Parameters
    ----------
    amd : xr.Dataset
        AMDFLOW output dataset
    caravan : xr.Dataset
        Caravan-Qual Lite dataset

    Returns
    -------
    candidates : pd.DataFrame
        Dataframe of candidate stations, columns: wqms_id, lat, lon, area_sqkm

    Raises
    ------
    SystemExit
        If no stations fall within the AMD domain, exits with message
    """
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
    """Compute boolean masks for cells that have valid data for pH and for iron.

    Parameters
    ----------
    amd : xr.Dataset
        AMDFLOW output dataset

    Returns
    -------
    tuple
        Tuple of boolean masks and corresponding indices
    """
    print("\nComputing valid-cell mask …")
    _sample_idx = np.linspace(0, amd.sizes["time"] - 1, 10, dtype=int)
    
    # pH mask: any non-null pH value in any time sample
    valid_mask = (amd["pH"].isel(time=_sample_idx).notnull().any(dim="time").compute())
    
    # Iron mask: any positive sum of iron components
    iron_sum = (amd["ferrous_iron"] + amd["ferric_iron"] + amd["ferric_oxyhydroxide"])
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
    """Build directed graph of river segments (nodes: HYRIV_ID).

    Parameters
    ----------
    rivers_gdf : gpd.GeoDataFrame
        GeoDataFrame of river network (HydroRIVERS), must contain columns HYRIV_ID, NEXT_DOWN, LENGTH_KM

    Returns
    -------
    G : networkx.DiGraph
        Directed graph of river segments, with length edges and HYRIV_ID nodes
    """
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
    """For each valid AMDFLOW cell, find nearest point on river network.
    Returns dict: (ilat, ilon) -> {HYRIV_ID, point_proj, distance_m, point_deg}

    Parameters
    ----------
    amd_lat_2d : np.ndarray
        2D array of latitudes for AMDFLOW grid cells, shape (nlat, nlon)
    amd_lon_2d : np.ndarray
        2D array of longitudes for AMDFLOW grid cells, shape (nlat, nlon)
    valid_mask : np.ndarray (bool)
        2D boolean array indicating valid cells (shape (nlat, nlon))
    rivers_gdf : gpd.GeoDataFrame
        GeoDataFrame of river network (HydroRIVERS), must contain column HYRIV_ID and geometry
    target_crs : str
        CRS string for reprojection (e.g. "EPSG:3857"), should be a projected CRS in metres for distance calculations

    Returns
    -------
    dict: (ilat, ilon) -> {HYRIV_ID, point_proj, distance_m, point_deg}
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

def assign_uparea_from_acc(amd, amd_lat_2d, amd_lon_2d, valid_mask, acc_array, acc_transform, acc_nodata, cell_area_func):
    """ Read upstream area (km²) for each valid AMDFLOW cell from ACC grid.
    acc_array is a 2D numpy array (already read).

    Parameters
    ----------
    amd : xr.dataset
        AMDFLOW output dataset
    amd_lat_2d : np.ndarray
        2D array of latitudes for AMDFLOW grid cells, shape (nlat, nlon)
    amd_lon_2d : np.ndarray
        2D array of longitudes for AMDFLOW grid cells, shape (nlat, nlon)
    valid_mask : np.ndarray (bool)
        2D boolean array indicating valid cells (shape (nlat, nlon))
    acc_array : np.ndarray
        2D numpy array of flow accumulation values
    acc_transform : affine.Affine
        Affine transformation for converting coordinates to grid indices
    acc_nodata : float
        No-data value for the flow accumulation array
        _description_

    Returns
    -------
    dict: (ilat, ilon) -> uparea_km2
    """
    print("Assigning upstream watershed area to cells from flow accumulation")
    uparea_dict = {}
    cell_areas = cell_area(amd)
    for ilat, ilon in zip(*np.where(valid_mask)):
        lon = amd_lon_2d[ilat, ilon]
        lat = amd_lat_2d[ilat, ilon]
        # convert lon/lat to row, col in ACC grid (assuming same CRS: WGS84)
        col, row = ~acc_transform * (lon, lat)   
        row_int, col_int = int(round(row)), int(round(col))
        if 0 <= row_int < acc_array.shape[0] and 0 <= col_int < acc_array.shape[1]:
            acc_val = acc_array[row_int, col_int]
            if acc_val > 0 and acc_val != acc_nodata:
                cell_area_km2 = cell_areas[ilat, ilon]
                uparea_km2 = acc_val * cell_area_km2
                uparea_dict[(ilat, ilon)] = uparea_km2
            else:
                uparea_dict[(ilat, ilon)] = np.nan
        else:
            uparea_dict[(ilat, ilon)] = np.nan
    return uparea_dict

def compute_network_distance(seg_id_a, point_a_proj, seg_id_b, point_b_proj, rivers_proj, graph):
    """ Compute shortest path distance (km) along river network between two points.
    Assumes all geometries are in the same projected CRS (metres).

    Parameters
    ----------
    seg_id_a : str
        HYRIV_ID of segment A
    point_a_proj : shapely.geometry.Point
        projected point of cell A
    seg_id_b : str
        HYRIV_ID of segment B
    point_b_proj : shapely.geometry.Point
        projected point of cell B
    rivers_proj : gpd.GeoDataFrame
        projected river network GeoDataFrame, must contain HYRIV_ID and geometry
    graph : networkx.DIGraph
        network graph of river network, with HYRIV_ID as nodes and edge lengths in km

    Returns
    -------
    float
        shortest path distance in kilometers
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
    """ Match Caravan stations to AMDFLOW cells using network distance and area ratio.

    Parameters
    ----------
    candidates : pd.DataFrame
        DataFrame of candidate stations after domain filter, columns: wqms_id, lat, lon, area_sqkm
    amd_lat_2d : np.ndarray
        2D array of latitudes for AMDFLOW grid cells, shape (nlat, nlon)
    amd_lon_2d : np.ndarray
        2D array of longitudes for AMDFLOW grid cells, shape (nlat, nlon)
    valid_mask : np.ndarray (bool)
        2D boolean array indicating valid cells (shape (nlat, nlon))
    uparea_dict : dict
        dict mapping (ilat, ilon) to upstream area in km² for valid cells
    cell_to_river : dict
        dict mapping (ilat, ilon) to the nearest river segment ID and projected point
    rivers_gdf : gpd.GeoDataFrame
        GeoDataFrame of river network, must contain HYRIV_ID and geometry
    river_graph : networkx.DIGraph
        network graph of river network, with HYRIV_ID as nodes and edge lengths in km
    target_crs : str
        CRS string for reprojection (e.g. "EPSG:3857"), should be a projected CRS in metres for distance calculations
    max_network_km : float, optional
        maximum network distance between stations and cells snapped to a river in kilometers, by default 5.0
    area_ratio_min : float, optional
        minimum ratio bound of station area to cell area, by default 0.5
    area_ratio_max : float, optional
        maximum ratio bound of station area to cell area, by default 1.5

    Returns
    -------
    pd.DataFrame
        DataFrame with matched stations and cell indices.
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
    """Extract and align model and observed time series for matched stations

    Parameters
    ----------
    matches : pd.DataFrame
        Dataframe of matched stations and cell indices
    amd : xr.Dataset
        AMDFLOW output dataset
    caravan : xr.Dataset
        Caravan-Qual Lite dataset
    caravan_var : str
        variable name in caravan dataset (e.g. "pH", "Fe-Dis")
    amd_var : str or list of str
         variable name(s) in AMD dataset to extract (e.g. "pH" or ["ferrous_iron", "ferric_iron", "ferric_oxyhydroxide"])
         if list, will sum the variables (after unit conversion if needed) to compare with the caravan variable
    resample_freq : str
        resampling frequency for caravan observations (e.g. "W" for weekly), or None to keep original frequency

    Returns
    -------
    pd.DataFrame
         DataFrame with columns: wqms_id, time, observed, modelled, dist_m, where each row is a paired observation-model timestep for a station

    Raises
    ------
    ValueError
        If no overlapping period between model and observations, or if caravan_var not in caravan dataset, or if amd_var(s) not in AMD dataset
    """
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
        _feoh3_to_fe = {"ferric_oxyhydroxide": (55.845 / 106.87)}
        scale = _feoh3_to_fe.get(var, 1.0)
        return (amd[var].sel(time=slice(t_start, t_end))
                .isel(lat=ilat_da, lon=ilon_da)) * scale
    
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
             min_paired_obs=3, resample_freq="W", output_path=None):
    """Run extraction and alignment for all variables

    Parameters
    ----------
    var_map : dict
        Dictionary mapping caravan variables to AMD variables
    matches_ph : pd.DataFrame
        DataFrame of matched stations for pH variables
    matches_iron : pd.DataFrame
        DataFrame of matched stations for iron variables
    amd : xr.Dataset
        AMD dataset
    caravan : xr.Dataset
        Caravan dataset
    min_paired_obs : int, optional
        Minimum number of paired observations required, by default 3
    resample_freq : str, optional
        Frequency for resampling observations, by default "W"

    Returns
    -------
    dict
        Dictionary of results for each variable
    """
    all_results = {}
    dropped_info = {}
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
        ts_all = ts.copy()

        # filter stations with too few paired observations
        n_paired = ts.dropna(subset=["observed", "modelled"]).groupby("wqms_id").size()
        keep = set(n_paired[n_paired >= min_paired_obs].index)
        dropped = n_paired[n_paired < min_paired_obs]

        # prepare dropped-info dataframe (stations that failed the paired-observation threshold)
        if len(dropped):
            dropped_df = matches[matches["wqms_id"].isin(dropped.index)].copy()
            dropped_df = dropped_df.drop_duplicates(subset=["wqms_id"]).reset_index(drop=True)
            # add n_paired column (may be NaN for stations present in matches but not in n_paired)
            dropped_map = dropped.to_dict()
            dropped_df["n_paired"] = dropped_df["wqms_id"].map(dropped_map).fillna(0).astype(int)
            dropped_df["drop_reason"] = "few_obs"
            dropped_info[caravan_var] = dropped_df
            print(f"  Recording {len(dropped_df)} station(s) with < {min_paired_obs} paired observations")

        # drop stations where modelled all zero
        all_zero = ts.groupby("wqms_id")["modelled"].apply(lambda x: (x == 0).all())
        zero_stations = all_zero[all_zero].index
        if len(zero_stations):
            # record zero stations too
            zero_df = matches[matches["wqms_id"].isin(zero_stations)].copy()
            zero_df = zero_df.drop_duplicates(subset=["wqms_id"]).reset_index(drop=True)
            zero_df["n_paired"] = n_paired.reindex(zero_df["wqms_id"]).fillna(0).astype(int).values
            zero_df["drop_reason"] = "all_zero"
            # merge with existing dropped_info if present
            if caravan_var in dropped_info:
                dropped_info[caravan_var] = pd.concat([dropped_info[caravan_var], zero_df], ignore_index=True)
            else:
                dropped_info[caravan_var] = zero_df
            print(f"  Recording {len(zero_df)} station(s) with all-zero modelled values")

        # finalize keep set by removing zero stations
        keep = set(keep).difference(zero_stations)

        ts = ts[ts["wqms_id"].isin(keep)].reset_index(drop=True)
        # keep the full (unfiltered) timeseries in all_results so CSVs include stations with few observations
        all_results[caravan_var] = ts_all

        # optionally save dropped stations to CSV in provided output path
        if output_path is not None and caravan_var in dropped_info:
            os.makedirs(output_path, exist_ok=True)
            outfn = os.path.join(output_path, f"dropped_stations_{caravan_var.replace(' ', '_')}.csv")
            dropped_info[caravan_var].to_csv(outfn, index=False)
            print(f"  Saved {len(dropped_info[caravan_var])} dropped stations → {outfn}")
    
    return all_results

def compute_metrics(obs, mod):
    """Compute RMSE, bias, NSE, KGE, R, FZE for two pandas Series (aligned by time)

    Parameters
    ----------
    obs : pd.Series
        observed values (Caravan-Qual Lite), indexed by time
    mod : pd.Series
        modelled values (AMDFLOW), indexed by time, must be aligned with obs

    Returns
    -------
    dict
        Dictionary of metrics: n, RMSE, bias, NSE, KGE, R, FZE
    """
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
        return {"n": n, "RMSE": np.nan, "bias": np.nan, "NSE": np.nan, "KGE": np.nan, "R": np.nan, "FZE": np.nan}
    
    # Use xarray for compat with scores
    o_da = xr.DataArray(o.values, dims=["time"])
    m_da = xr.DataArray(m.values, dims=["time"])
    rmse = sc.continuous.rmse(m_da, o_da).compute().values.item()
    bias = np.mean(m - o)
    nse = sc.continuous.nse(o_da, m_da).compute().values.item()
    r = r2_score(o, m)
    kge = sc.continuous.kge(o_da, m_da).compute().values.item()

    beta = np.mean(mod) / np.mean(obs)
    gamma = iqr(mod) / iqr(obs)
    p_r, _ = pearsonr(obs, mod)
    fze = 1 - np.sqrt((r - 1)**2 + (gamma - 1)**2 + (beta - 1) **2)

    return {"n": n, "RMSE": rmse, "bias": bias, "NSE": nse, "KGE": kge, "R": r, "FZE": fze}

def validation_metrics(ts, min_n=3):
    """Compute per-station metrics and return DataFrame.

    Parameters
    ----------
    ts : pd.DataFrame
        DataFrame with columns: wqms_id, time, observed, modelled, dist_m


    Returns
    -------
    pd.DataFrame
        DataFrame with columns: wqms_id, n, RMSE, bias, NSE, KGE, R, FZE
    """
    if ts.empty:
        return pd.DataFrame(columns=["n", "RMSE", "bias", "NSE", "KGE", "R", "FZE"]).set_index(pd.Index([], name="wqms_id"))

    rows = []
    for sid, grp in ts.groupby("wqms_id"):
        grp = grp.set_index("time").sort_index()

        # align weekly and compute valid pairs
        o = grp["observed"].resample("W").mean()
        m = grp["modelled"].resample("W").mean()
        common = o.index.intersection(m.index)
        o = o.loc[common]
        m = m.loc[common]
        valid = np.isfinite(o) & np.isfinite(m)
        o = o[valid]
        m = m[valid]
        n = len(o)

        # compute bias even for small n
        bias = np.nan
        if n > 0:
            bias = np.mean(m - o)

        # other metrics only when n >= min_n and both series have variance
        RMSE = np.nan
        NSE = np.nan
        KGE = np.nan
        R = np.nan
        FZE = np.nan
        if n >= min_n and o.std() != 0 and m.std() != 0:
            # Use xarray for compatibility with scores
            o_da = xr.DataArray(o.values, dims=["time"])
            m_da = xr.DataArray(m.values, dims=["time"])
            try:
                RMSE = sc.continuous.rmse(m_da, o_da).compute().values.item()
            except Exception:
                RMSE = np.nan
            try:
                NSE = sc.continuous.nse(o_da, m_da).compute().values.item()
            except Exception:
                NSE = np.nan
            try:
                KGE = sc.continuous.kge(o_da, m_da).compute().values.item()
            except Exception:
                KGE = np.nan
            try:
                R = r2_score(o, m)
            except Exception:
                R = np.nan
            try:
                beta = np.mean(m) / np.mean(o)
                gamma = iqr(m) / iqr(o)
                r, _ = pearsonr(o, m)
                fze = 1 - np.sqrt((r - 1)**2 + (gamma - 1)**2 + (beta - 1)**2)
            except:
                fze = np.nan

        rows.append({"wqms_id": sid, "n": n, "RMSE": RMSE, "bias": bias, "NSE": NSE, "KGE": KGE, "R": R, "FZE": fze})

    df = pd.DataFrame(rows).set_index("wqms_id")
    
    # drop stations with n=0 (no paired observations)
    n_zero = (df["n"] == 0).sum()
    if n_zero:
        df = df[df["n"] > 0]
        print(f"  Dropped {n_zero} station(s) with zero paired observations")
    
    # report how many stations had 0 < n < min_n
    few = (df["n"] < min_n).sum()
    if few:
        print(f"  Included {few} station(s) with 0 < n < {min_n} — bias computed, other metrics NaN")
    return df

if __name__ == "__main__":
    case_study_nr = "CSIII"
    times = ("1960", "2015")
    hydrobasins_region_code = "eu"
    caravan_path = "../data/validation data/Caravan-Qual_lite.zarr"
    amd_path = f"../data/validation data/AMDFLOW_{case_study_nr}_{times[0]}-{times[1]}_W.nc"
    output_path = f"../data/validation data/{case_study_nr}/"
    acc_path = f"../data/validation data/hyd_{hydrobasins_region_code}_acc_30s.tif"
    rivers_path = f"../data/validation data/HydroRIVERS_v10_{hydrobasins_region_code}_shp/HydroRIVERS_v10_{hydrobasins_region_code}.shp"
    
    utm_crs = "EPSG:6312" #"EPSG:6312" (Cyprus),  "EPSG:24378" (India) "EPSG: 3761" #(Canada)
    resample_freq = "W"
    var_map = {
        "pH": "pH",
        "Fe-Dis": ["ferrous_iron", "ferric_iron"],
        "Fe-Tot": ["ferrous_iron", "ferric_iron", "ferric_oxyhydroxide"],
    }
    
    print(f"Validating AMDFLOW Case Study {case_study_nr} ({times[0]}–{times[1]}),")
    print("against Caravan-Qual Lite using HydroSHEDS network + area ratio snapping.")
    
    # load everything
    amd, caravan, acc_array, acc_transform, acc_nodata, rivers = load_datasets(
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
    uparea_dict = assign_uparea_from_acc(amd, amd_lat_2d, amd_lon_2d, valid_mask,
                                         acc_array, acc_transform, acc_nodata)
    
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