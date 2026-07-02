# data_utils py file for AMDFLOW
# contains all helper functions to clip, transform and treat data before model run
from openmindat import LocalitiesRetriever, GeomaterialRetriever
import os
import json
import pandas as pd
import re
import xarray as xr
import geopandas as gpd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import rasterio as rio
import rasterio.features
import rioxarray
import pyflwdir
from scipy.interpolate import CubicHermiteSpline

def mindat_collector(region, material_id = 3314, mineral_strings = "(Fe|S)", material_name = "pyrite", path_str = "../data/", mindat_api_str = "mindat_API_key.txt"):
    """Mindat data collector, queries mindat API in specific region for a specific mineral and checks if those minerals are located at a mine/quarry, saves locations at lat/lon csv file

    Parameters
    ----------
    region : str
        string matching the mindat region to query
    material_id : int, optional
        mindat material id corresponding to a specific material, by default 3314 (pyrite id)
    mineral_strings : str, optional
        mindat string corresponding to specific chemicals using regex expression, specify full compound with string1|string2, always place full string in brackets (), by default "(Fe|S)"
    material_name : str, optional
        mindat name of material, by default "pyrite"
    path_str : str, optional
        string of where function should place "mindat_data/" folder wherein output csv files are places and checked, by default "../data/"
    mindat_api_str : str, optional
        string of where mindat API key file with API key is located, by default "mindat_API_key.txt"
    """
    # set API key
    with open(mindat_api_str) as f:
        key = f.read()
        os.environ["MINDAT_API_KEY"] = key

    # try opening material file containing worldwide material localities
    try:
        with open(f"{path_str}mindat_data/v1_geomaterials_{material_id}/v1_geomaterials.json") as file:
            material = json.load(file)

    # if no file available get data from mindat and save and load to json in data
    except:
        gr = GeomaterialRetriever()
        gr.expand("locality").id_in(material_id)
        gr.saveto(f"{path_str}mindat_data/v1_geomaterials_{material_id}")
        with open(f"{path_str}mindat_data/v1_geomaterials_{material_id}/v1_geomaterials.json") as file:
            material = json.load(file)

    # try to open region file containing region localities
    try:
        with open(f"{path_str}mindat_data/localities_{region}/v1_localities.json", "r") as f:
            region_json = json.load(f)
            # return if file/data already exists
            print("File already exists, delete if you want new data")
            return
    
    # if no file avaible get data from mindat and save/load as json in data
    except:
        # get region specific localities and save and load as json in data
        lr = LocalitiesRetriever()
        lr.country(region).page_size(100) # explicit page_size to circumvent brocli errors
        lr.saveto(f"{path_str}mindat_data/localities_{region}")
        with open(f"{path_str}mindat_data/localities_{region}/v1_localities.json") as file:
            region_json = json.load(file)

    # start filtering data for strings in mineral_strings and mine|mining|quarry
    df_region = pd.json_normalize(region_json["results"])
    df_region = df_region[
    # check for mine/mining/quarry in description (case insensitive)
    df_region["description_short"].str.contains(r"\b(mine|mining|quarry)\b", 
                                        case=False, na=False, regex=True) &
    # check for Fe or S in elements string
    df_region["elements"].str.contains(f"{mineral_strings}", na=False, regex=True)
                            ]
    # filter region dataframe localities by only keeping the ids that are also in the material localities
    df_material_ids = pd.DataFrame(material["results"][0]["locality"], columns = ["id"])
    ids_to_keep = df_material_ids["id"].unique() 
    df_region_material = df_region[df_region["id"].isin(ids_to_keep)]
    df_region_material = df_region_material[["id", "latitude", "longitude"]]
    df_region_material.to_csv(f"{path_str}mindat_data/{region}_{material_name}.csv")
    print(f"\nSuccesfully saved data to: {path_str}mindat_data/{region}_{material_name}.csv")
    return

def vector_rasterisation(output_path="../data/mines_raster.tif",
                         flo1k_path="../data/flo_IPB_2015.nc",
                         vector_path="../data/mine_polygons/74548_projected polygons.shp"):
    """Rasterises vector file to a reference raster and saves rasterised file
    
    Parameters
    ----------
    output_path : str, optional
        string of where output raster file should be output to, by default "../data/mines_raster.tif"
    flo1k_path : str, optional
        string of where reference raster (reference for the vector to raster transform) is located, by default "../data/flo_IPB_2015.nc"
    vector_path : str, optional
        string of where vector, to be rasterised, is located, by default "../data/mine_polygons/74548_projected polygons.shp"
    """
    ds = xr.open_dataset(flo1k_path)
    ref = ds["qav"].isel(time=0)

    lons = ds["lon"].values
    lats = ds["lat"].values
    res_x = float(lons[1] - lons[0])
    res_y = float(lats[1] - lats[0])

    transform = rasterio.transform.from_bounds(
        lons.min() - res_x/2, lats.min() - res_y/2,
        lons.max() + res_x/2, lats.max() + res_y/2,
        len(lons), len(lats)
    )

    gdf = gpd.read_file(vector_path)
    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS defined")
    gdf = gdf.to_crs("EPSG:4326")
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]

    raster_arr = rasterio.features.rasterize(
        shapes,
        out_shape=(len(lats), len(lons)),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    crs = rasterio.crs.CRS.from_epsg(4326)  # adjust if your data uses a different CRS

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=len(lats), width=len(lons),
        count=1, dtype="uint8",
        crs=crs, transform=transform
    ) as dst:
        dst.write(raster_arr, 1)

    print(f"Raster saved to {output_path}")

def flo1k_prep(flo1k_qav_path = "../data/FLO1K.ts.1960.2015.qav.nc", 
               flo1k_qma_path = "../data/FLO1K.ts.1960.2015.qma.nc", 
               flo1k_qmi_path = "../data/FLO1K.ts.1960.2015.qmi.nc", 
               basins_path = "../data/hybas_eu_lev01-04/hybas_eu_lev04_v1c.shp",
               basins_iloc = (45, 53),
               date = np.datetime64("2015-01-01"),
               aoi = None,
               length = 52,
               frequency = "W",
               time_first = "1960"):
    """Function to clip the full flo1k dataset to a specified area of interest and dates, 
    merges the qav, qmi and qma datasets and runs cubic hermite splining with max streamflow in winter,
    minimum streamflow in summer and average streamflow in spring and autumn (splining is done in splining() func)

    Parameters
    ----------
    flo1k_qav_path : str, optional
        string of where flo1k (qav: average) dataset is located, by default "../data/FLO1K.ts.1960.2015.qav.nc"
    flo1k_qma_path : str, optional
        string of where flo1k (qma: maximum) dataset is located, by default "../data/FLO1K.ts.1960.2015.qav.nc"
    flo1k_qmi_path : str, optional
        string of where flo1k (qmi: minimum) dataset is located, by default "../data/FLO1K.ts.1960.2015.qav.nc"
    basins_path : str, optional
        string of where HydroBASINC data is located, used for clipping of flo1k dataset to area of interest, by default "../data/hybas_eu_lev01-06_v1c/hybas_eu_lev04_v1c.shp"
    basins_iloc : tuple, optional
        tuple of which ilocs of the hydrobasins dataset should be used for clipping the flo1k dataset, by default (45, 53)
    date : np.datetime64 OR np.ndarray, optional
        np.datetime64 type object of what dates (1960-2015) the clipped flo1k dataset should save, if more than one should be used use np.ndarray, by default np.datetime64("2015-01-01") OR
        np.ndarray of np.datetime64 objects of what dates (1060-2015) the clipped flo1k dataset should save
    aoi : geopandas.GeoDataFrame, optional
        GeoDataFrame of area of interest to clip the flo1k dataset, if None the function uses basins_iloc, 
        the aoi can specify a more specific area of interest if needed
    length : int
        int of length of how many of the discrete timesteps in a year there are: e.g; 52 (weekly timesteps), 12 (monthly timesteps), (365) (daily timesteps)
    frequency : str
        str of what length type (weekly, monthly, daily, etc.) the discrete timestep is, see pandas.data_range() for options
    time_first : str
        string of the year where the dataset should start, by default "1960"

    Returns:
    ---------------------
    flo_da : xarray.DataSet
        dataset of streamflow (Q) splined to be seasonally distinct 
    """
    
    
    flo_one_qav = xr.open_dataset(flo1k_qav_path)
    flo_one_qma = xr.open_dataset(flo1k_qma_path)
    flo_one_qmi = xr.open_dataset(flo1k_qmi_path)
    basins = gpd.read_file(basins_path)
    if aoi is None:
        aoi = basins.iloc[basins_iloc[0]:basins_iloc[1]]
    aoi_lat = [float(aoi.total_bounds[1]), float(aoi.total_bounds[3])]
    aoi_lon = [float(aoi.total_bounds[0]), float(aoi.total_bounds[2])]

    if isinstance(date, np.datetime64):
        flo_qav = flo_one_qav["qav"].sel(time = date,
                                lon = slice(aoi_lon[0], aoi_lon[1]),
                                lat = slice(aoi_lat[0], aoi_lat[1]))
        flo_min = flo_one_qmi["qmi"].sel(time = date,
                            lon = slice(aoi_lon[0], aoi_lon[1]),
                            lat = slice(aoi_lat[0], aoi_lat[1]))
        flo_max = flo_one_qma["qma"].sel(time = date,
                            lon = slice(aoi_lon[0], aoi_lon[1]),
                            lat = slice(aoi_lat[0], aoi_lat[1]))
    elif isinstance(date, np.ndarray):
            flo_qav = flo_one_qav["qav"].sel(time = slice(date[0], date[-1]),
                            lon = slice(aoi_lon[0], aoi_lon[1]),
                            lat = slice(aoi_lat[0], aoi_lat[1]))
            flo_min = flo_one_qmi["qmi"].sel(time = slice(date[0], date[-1]),
                    lon = slice(aoi_lon[0], aoi_lon[1]),
                    lat = slice(aoi_lat[0], aoi_lat[1]))
            flo_max = flo_one_qma["qma"].sel(time = slice(date[0], date[-1]),
                    lon = slice(aoi_lon[0], aoi_lon[1]),
                    lat = slice(aoi_lat[0], aoi_lat[1]))

    flo_aoi_date = xr.merge([flo_qav, flo_min, flo_max])
    flo_aoi_date.rio.write_crs(aoi.crs, inplace = True)
    flo_aoi_date = flo_aoi_date.rio.clip(aoi.geometry, aoi.crs)

    # Drop lat/lon bands that are entirely NaN after clipping into the polygon AOI.
    # This removes the rectangular ocean padding before repeating and interpolation.
    flo_aoi_date = flo_aoi_date.dropna(dim="lon", how="all", subset=["qav", "qmi", "qma"])
    flo_aoi_date = flo_aoi_date.dropna(dim="lat", how="all", subset=["qav", "qmi", "qma"])

    if len(flo_aoi_date.lat) == 0 or len(flo_aoi_date.lon) == 0:
        raise ValueError("No valid land cells remain after dropping ocean-only rows/columns.")

    
    flo_da = splining(flo_aoi_date, time_first)
    return flo_da.to_dataset(name="Q")

def splining(flo, time_first):
    lats = flo.lat
    lons = flo.lon
    n_years = len(flo.time)
    years = pd.DatetimeIndex(flo.time.values)
    # create continuous time anchors spanning the series (4 seasonal anchors per year)
    time_anchors = []
    for i in range(n_years):
        offset = i * 52
        # Week 2 (Peak), Week 15 (Mean), Week 28 (Trough), Week 41 (Mean)
        time_anchors.extend([offset + 2, offset + 15, offset + 28, offset + 41])

    # append the final wrap-around anchor to close the timeline smoothly
    time_anchors.append(n_years * 52 + 2)
    time_anchors = np.array(time_anchors)

    # hemisphere-aware baseline mapping
    # If Lat >= 0 (North): winter max occurs first, summer min occurs in the middle
    # If Lat < 0 (South): summer min occurs first, winter max occurs in the middle
    y_0 = xr.where(flo["qmi"].lat >= 0, flo["qma"], flo["qmi"])
    y_2 = xr.where(flo["qmi"].lat >= 0, flo["qmi"], flo["qma"])
    y_1 = flo["qav"]
    y_3 = flo["qav"]

    # only interpolate land cells with valid data for all three variables over time.
    valid_mask = (
        np.isfinite(flo["qav"]) &
        np.isfinite(flo["qmi"]) &
        np.isfinite(flo["qma"])
    ).all(dim="time")
    if valid_mask.sum().item() == 0:
        raise ValueError("No valid land cells remain for interpolation.")

    valid_points = valid_mask.stack(point=("lat", "lon"))
    point_indices = np.where(valid_points.values)[0]

    # stack seasonal coordinates sequentially and filter to land points only.
    y_stacked = xr.concat([y_0, y_1, y_2, y_3], dim="season")
    y_stacked = y_stacked.transpose("time", "season", "lat", "lon")
    y_points = y_stacked.stack(point=("lat", "lon")).isel(point=point_indices)

    y_flat = y_points.transpose("time", "season", "point").values.reshape(n_years * 4, -1)

    # grab the final wrap-around value for the same valid land points.
    wrap_y = y_0.isel(time=-1).stack(point=("lat", "lon")).isel(point=point_indices).values[np.newaxis, :]
    y_final = np.concatenate([y_flat, wrap_y], axis=0)

    # calculate Hermite Slopes (dydx) across the multi-year boundary
    slopes = np.zeros_like(y_final)
    odd_indices = np.arange(1, len(time_anchors), 2)

    dy = y_final[odd_indices + 1] - y_final[odd_indices - 1]
    dx = time_anchors[odd_indices + 1] - time_anchors[odd_indices - 1]


    slopes[odd_indices] = dy / dx.reshape(-1, 1)

    # build the 52-week per year target prediction timeline
    total_weeks = n_years * 52
    target_weeks = np.arange(1, total_weeks + 1)


    spline = CubicHermiteSpline(x=time_anchors, y=y_final, dydx=slopes)
    predicted_weekly = spline(target_weeks)

    # clipping
    predicted_weekly = np.clip(predicted_weekly, 0, None)

    # back to rectangle raster
    output = np.full((total_weeks, len(lats), len(lons)), np.nan, dtype=predicted_weekly.dtype)
    lat_idx, lon_idx = np.where(valid_mask.values)
    output[:, lat_idx, lon_idx] = predicted_weekly

    # Generate a continuous weekly pandas date index starting at time_first
    time_index = pd.date_range(start=f"{time_first}-01-01", periods=total_weeks, freq="W-MON")

    flo_da = xr.DataArray(
        output,
        coords={
            'time': time_index,
            'lat': lats,
            'lon': lons
        },
        dims=['time', 'lat', 'lon'],
        name="Q"
    )

    print("Splining complete!")
    print(f"Output shape: {flo_da.shape} (Weeks: {total_weeks}, Lats: {len(lats)}, Lons: {len(lons)})")
    return flo_da

def hydir_IDs(ds, aoi):
    """Function to convert HydroSHEDS direction dataset to dataset containing unique cell IDs per cell and the ID of where the outflow from that cell goes to

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset of hydroSHEDS direction dataset to convert to unique cell IDs, and outflow IDs dataset
    aoi : geopandas.GeoDataFrame
        GeoDataFrame of area of interest that the input dataset should be clipped to

    Returns
    -------
    ds : xarray.Dataset
        Dataset of unique IDs per cell (ID) and where the outflow of each cell goes to (which ID the outflow from cells goes to) (outID)
    """
    # extract band of 2d shape
    if "band" in ds["hydir"].dims:
        hydir_2d = ds["hydir"].isel(band=0, drop=True)
    else:
        hydir_2d = ds["hydir"]  # already 2D

    nrows, ncols = hydir_2d.shape
    N = nrows * ncols
    print(f"Grid size: {nrows} rows, {ncols} cols")

    # cell IDs as x * y 
    cell_ids = np.arange(nrows * ncols).reshape((nrows, ncols))

    # outflow ID init with -1 (no outflow)
    outflow_ids = np.full((nrows, ncols), -1, dtype=np.int64)

    # HydroSHEDS encoding: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
    encoding_to_offset = {
        1: (0, 1),    # E
        2: (1, 1),    # SE
        4: (1, 0),    # S
        8: (1, -1),   # SW
        16: (0, -1),  # W
        32: (-1, -1), # NW
        64: (-1, 0),  # N
        128: (-1, 1), # NE
        0: (0, 0),    # inland depression (flows to itself)
        # 255 is handled separately (NoData)
    }

    for code, (dy, dx) in encoding_to_offset.items():
        if code == 255:
            continue

        mask = hydir_2d == code
        if not mask.any():
            continue

        # row/column incides
        rows, cols = np.where(mask)   

        # target cell locs
        target_rows = rows + dy
        target_cols = cols + dx

        # check if targets are valid
        valid = (target_rows >= 0) & (target_rows < nrows) & \
                (target_cols >= 0) & (target_cols < ncols)

        valid_rows = rows[valid]
        valid_cols = cols[valid]
        valid_target_rows = target_rows[valid]
        valid_target_cols = target_cols[valid]

        # outflow ID == ID of target
        outflow_ids[valid_rows, valid_cols] = cell_ids[valid_target_rows, valid_target_cols]

    # save to dataset on coordinates
    spatial_dims = hydir_2d.dims   

    ds["ID"] = xr.DataArray(
        cell_ids,
        dims=spatial_dims,
        attrs={"description": "Cell IDs", "units": "None"}
    )

    ds["outID"] = xr.DataArray(
        outflow_ids,
        dims=spatial_dims,
        attrs={"description": "Outflow cell IDs", "units": "None"}
    )

    # mark IDs that are outflow targets
    targets = np.zeros(N, dtype= bool)
    out_flat = ds["outID"].values.ravel()
    valid_out = out_flat[out_flat != -1]
    targets[valid_out] = True

    # filter out 255 hydir cells
    valid_mask = (ds["hydir"].values != 255).ravel()
    valid_indices = np.where(valid_mask)[0]

    # filters and masks for no inflow cells (network sources)
    no_inflow_indices = valid_indices[~targets[valid_indices]]

    rows, cols = np.unravel_index(no_inflow_indices, (nrows, ncols))

    no_inflow_mask = np.zeros((nrows, ncols), dtype= bool)
    no_inflow_mask[rows, cols] = True

    # save to dataset
    ds["source"] = xr.DataArray(
        no_inflow_mask,
        dims = ds["ID"].dims,
        attrs = {"description": "Bolean of True: no inflow, False: has inflow", "units": "None"})

    ds = ds.rio.clip(aoi.geometry, aoi.crs)
    return ds

def filter_mines_with_buffer(raster, points, crs, buffer_dist = 5000.0):
    """Function to filter a full raster dataset with mindat mineral location, mindat mineral location gets buffer: 
        if mine cell of raster falls within buffer the cell passes filter, otherwise cells gets dropped

    Parameters
    ----------
    raster : xarray.Dataset
        Dataset of mines cells
    points : geopandas.GeoDataFrame
        GeoDataFrame of mindat mineral locations to link with mines
    buffer_dist : float, optional
        float of m buffer in crs to buffer the points, by default 5000.0 (m)

    Returns
    -------
    raster_filtered : xarray.Dataset
        Dataset of mines that fall within buffer of points, thus likely containing the mineral of interest
    """
    points = points.set_geometry(col = 0)

    # set crs to metre based crs
    points.to_crs(crs, inplace = True)
    raster = raster.rio.reproject(crs)

    # create buffer in metres
    buffered_points = points.buffer(buffer_dist)

    # clip
    raster_filtered = raster.rio.clip(buffered_points, drop = True)
    raster_filtered = raster.where(raster_filtered["band_data"] == 1)
    raster_filtered = raster_filtered.rename_vars({"band_data": "mines"})

    # return to wg84 crs
    raster_filtered = raster_filtered.rio.reproject("EPSG:4326")
    return raster_filtered

def bool_to_int(ds, dtype="int16"):
    """Convert all boolean data variables to the specified integer type (not tested with other dtypes than int16), 
        used to save memory as size bool dataset > size int16 dataset

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset of bools
    dtype : str, optional
        string of dtype to convert to, by default "int16"

    Returns
    -------
    ds : xarray.Dataset
        Dataset of dtype
    """

    for var in ds.data_vars:
        if ds[var].dtype == bool:
            ds[var] = ds[var].astype(dtype)
    return ds

def cleanup_and_metadata(ds):
    """Function to cleanup dataset and add metadata

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to cleanup and add metadata to

    Returns
    -------
    ds : xarray.Dataset
        Cleaned dataset with metadata
    """
    ds = ds.drop_vars(["band", "hydir"])
    ds["mines"].attrs = {"units": "none", "description": "mapping of mines, all non nan are mines"}
    ds = ds.assign_coords(x=("x", ds.lon.values), y=("y", ds.lat.values))
    ds = ds.drop_vars(["lat", "lon"])        
    ds = ds.rename({"x": "lon", "y": "lat"})   
    ds["mines"] = ds["mines"].squeeze("band", drop=True)
    return ds


def estimate_ore(ds, F, ox_range = 27):
    """Function convert a dataset with only if mines are there (dataset["mines"], 1 = mine, 0 = no mine) to a dataset containing variable "ore" which has an estimate ore amount in square metre 

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing "mines" variable of int type with 1 = mine, 0 = no mine, dataset should be 1km*1km / 3 arc second resolution, otherwise the constant equation should be changed
    F : float
        fraction of reactive material per square metre of mine, used in the calculation: ox_range * 1000 * F
    ox_range : int, optional
        range of square metre of mineral exposed to real sqaure metre of mineral that can be oxidised, represents fractures and oxidation range, by default 27, but ranges between 27-161 are reported

    Returns
    -------
    ds : xarray.Dataset
        Dataset containing "ore" variable with estimated ore amount in square metre
    """

    constant = ox_range * 1000 * F
    ds["ore"] = xr.where(ds["mines"] == 1, constant, np.nan)
    ds["ore"].attrs = {"description": "Estimation of reactive ore per cell", "unit": "m2",}
    ds = ds.drop_vars(["mines"])
    return ds

def animation_plot(dataarray, aoi = None, size = (10, 8), cmap = "Reds", frame_skip = 1, dpi = 100):
    """Function to animate timeseries dataset of AMDFLOW

    Parameters
    ----------
    dataarray : xarray.Dataarray
        Dataarray of AMDFLOW output of one variable 
    aoi : geopandas.GeoDataFrame, optional
        area of interest to plot the boundaries of, usually a region/watershed, by default None
    size : tuple, optional
        size of plot in inches, by default (10, 8)
    cmap : str, optional
        colour map of plot, see matplotlib cmap documentatio, by default "Reds"
    frame_skip : int, optional
        show every Nth frame to reduce animation size, by default 1
    dpi : int, optional
        DPI for saved animation (lower = smaller file), by default 100
    """
    def _ani_update(t):
        im.set_data(plot_data[t].values)
        ax.set_title(f"{date_range[t][0]}/{date_range[t][1]}/{date_range[t][2]}")
        return [im]
    
    fig, ax = plt.subplots(figsize = size, dpi = dpi)
    
    # Sample frames to reduce file size
    plot_data = dataarray[::frame_skip]
    
    im = ax.imshow(plot_data[0].values, cmap = cmap, 
                    extent=[plot_data.lon.min(), plot_data.lon.max(), 
                            plot_data.lat.min(), plot_data.lat.max()],
                            origin='lower', aspect='auto')
    
    units = getattr(dataarray, 'units', '')
    cbar_label = f"{dataarray.name} ({units})" if units else str(dataarray.name)
    cbar = fig.colorbar(im, ax = ax, label = cbar_label)

    date_range = [(day, month, year) for day, month, year in zip(
        plot_data.time.dt.day.values,  
        plot_data.time.dt.month.values, 
        plot_data.time.dt.year.values)]
    
    title = ax.set_title(f"{date_range[0][0]}/{date_range[0][1]}/{date_range[0][2]}")

    if aoi is not None:
        aoi.boundary.plot(ax = ax, color = "k")
        
    ani = animation.FuncAnimation(fig, _ani_update, frames = len(date_range), 
                                  repeat = True, interval = 500, blit = True)
    plt.close()
    
    # Increase embed limit to allow larger animations
    plt.rcParams['animation.embed_limit'] = 100
    
    return HTML(ani.to_jshtml())

def vector_to_raster_sens(raster_file, vector_file):
    vectors = gpd.read_file(vector_file)
    raster = xr.load_dataset(raster_file)

    return

def sel_nearest_where(da, lat, lon, condition, max_distance_km=5, dist_km=None):
    """
    Select nearest grid cell where condition is met, within max distance.
    Optionally accepts a precomputed dist_km array to avoid redundant computation.
    Returns both the data and the distance.
    """
    lat = float(lat)
    lon = float(lon)

    # Create boolean mask for valid cells
    valid = condition(da)
    valid_mask = valid.any(dim="time").values if "time" in valid.dims else valid.values 

    if dist_km is None: 
        lats = da.lat.values.astype(float)
        lons = da.lon.values.astype(float)
        dlat_2d, dlon_2d = np.meshgrid(lats - lat, lons - lon, indexing="ij")
        dlat_km = dlat_2d * 111
        dlon_km = dlon_2d * 111 * np.cos(np.radians(lat))
        dist_km = np.sqrt(dlat_km**2 + dlon_km**2)

    dist_masked = dist_km.astype(float).copy()  
    dist_masked[~valid_mask | (dist_km > max_distance_km)] = np.nan  

    if np.all(np.isnan(dist_masked)):
        dist_fallback = dist_km.astype(float).copy()
        dist_fallback[dist_km > max_distance_km] = np.nan
        if np.all(np.isnan(dist_fallback)):
            print(f"Warning: No cells within {max_distance_km} km — using absolute nearest.")
            flat_idx = np.nanargmin(dist_km)
        else:
            flat_idx = np.nanargmin(dist_fallback)
    else:
        flat_idx = np.nanargmin(dist_masked)

    best_lat_idx, best_lon_idx = np.unravel_index(flat_idx, dist_km.shape)
    best_lat = float(da.lat.values[best_lat_idx])
    best_lon = float(da.lon.values[best_lon_idx])
    actual_distance = dist_km[best_lat_idx, best_lon_idx]

    return da.sel(lat=best_lat, lon=best_lon), actual_distance

def fill_flow_values(ds, hydir_filename, aoi):

    # Open with rioxarray
    flw_da = rioxarray.open_rasterio(hydir_filename)
    # Clip to AOI
    #flw_clip = flw_da.rio.clip(aoi.geometry, aoi.crs, drop=True)
    flw_clip = flw_da
    # Extract the numpy array, transform, and crs
    flw_data = flw_clip.values[0]  # assuming single band
    transform = flw_clip.rio.transform()
    crs = flw_clip.rio.crs

    ds = ds.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
    for var in ds.data_vars:
        if 'grid_mapping' in ds[var].attrs:
            del ds[var].attrs['grid_mapping']
    Q = ds["qav"]
    flw = pyflwdir.from_array(
        data=flw_data,
        transform=transform,
        ftype= "d8",
        latlon=True
    )

    for t in range(len(ds.time)):
        q_2d = Q.isel(time=t).values
        q_2d[q_2d == 0] = np.nan
        q_filled = flw.fillnodata(
            data = q_2d, 
            nodata = np.nan,
            direction="down",
            how = "sum")
        
        ds["qav"][t, :, :] = q_filled
    
    return ds
