# data_utils py file for AMDFLOW
# contains all helper functions to clip, transform and treat data before model run
from openmindat import LocalitiesRetriever, GeomaterialRetriever
import os
import json
import pandas as pd
import re
from osgeo import gdal
from osgeo import ogr
import xarray as xr
import geopandas as gpd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import matplotlib.pyplot as plt

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

def vector_rasterisation(output_path = "../data/mines_raster.tif", flo1k_path = "../data/flo_IPB_2015.nc", vector_path = "../data/mine_polygons/74548_projected polygons.shp"):
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
    # rasterise mining polygons
    # code sourced from: K. Haven 2019 https://medium.com/data-science/use-python-to-convert-polygons-to-raster-with-gdal-rasterizelayer-b0de1ec3267
    nc_path = flo1k_path

    raster = gdal.Open(f"NETCDF:{nc_path}:qav")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.Open(vector_path, 0)  # 0 = read-only
    if data_source is None:
        print("Could not open shapefile")
    else:
        print("Successfully opened shapefile")
        layer = data_source.GetLayer()
        print(f"Layer name: {layer.GetName()}")

    geo_transform = raster.GetGeoTransform()
    projection = raster.GetProjection()
    x_size = raster.RasterXSize
    y_size = raster.RasterYSize


    drv_tiff = gdal.GetDriverByName("GTiff")
    output_ds = drv_tiff.Create(
        output_path,
        x_size,
        y_size,
        1,  
        gdal.GDT_Float32  
    )

    output_ds.SetGeoTransform(geo_transform)
    output_ds.SetProjection(projection)


    options = [f"ATTRIBUTE=Shape_Area"]
    result = gdal.RasterizeLayer(
        output_ds,        # output dataset
        [1],              # band list
        layer,            # layer to rasterize
        options=options
    )
    output_ds.GetRasterBand(1).SetNoDataValue(0.0) 
    output_ds = None
    return

def flo1k_prep(flo1k_path = "../data/FLO1K.ts.1960.2015.qav.nc", 
               basins_path = "../data/hybas_eu_lev01-06_v1c/hybas_eu_lev04_v1c.shp",
               basins_iloc = (45, 53),
               date = np.datetime64("2015-01-01"),
               output_path = f"../data/flo_IPB_"):
    """Function to clip the full flo1k dataset to a specified area of interest and dates, saves clipped flo1k to new netcdf file

    Parameters
    ----------
    flo1k_path : str, optional
        string of where flo1k dataset is located, by default "../data/FLO1K.ts.1960.2015.qav.nc"
    basins_path : str, optional
        string of where HydroBASINC data is located, used for clipping of flo1k dataset to area of interest, by default "../data/hybas_eu_lev01-06_v1c/hybas_eu_lev04_v1c.shp"
    basins_iloc : tuple, optional
        tuple of which ilocs of the hydrobasins dataset should be used for clipping the flo1k dataset, by default (45, 53)
    date : np.datetime64 OR np.ndarray, optional
        np.datetime64 type object of what dates (1960-2015) the clipped flo1k dataset should save, if more than one should be used use np.ndarray, by default np.datetime64("2015-01-01") OR
        np.ndarray of np.datetime64 objects of what dates (1060-2015) the clipped flo1k dataset should save
    output_path : str, optional
        string of where the clipped flo1k dataset should be saved, date is used to append to this string to get full output path, by default f"../data/flo_IPB_"
    """
    
    
    flo_one_k = xr.open_dataset(flo1k_path)
    basins = gpd.read_file(basins_path)

    aoi = basins.iloc[basins_iloc[0]:basins_iloc[1]]
    aoi_lat = [float(aoi.total_bounds[1]), float(aoi.total_bounds[3])]
    aoi_lon = [float(aoi.total_bounds[0]), float(aoi.total_bounds[2])]

    if isinstance(date, np.datetime64):
        flo_aoi_date = flo_one_k["qav"].sel(time = date,
                                lon = slice(aoi_lon[0], aoi_lon[1]),
                                lat = slice(aoi_lat[0], aoi_lat[1]))
    elif isinstance(date, np.ndarray):
            flo_aoi_date = flo_one_k["qav"].sel(time = slice(date[0], date[-1]),
                            lon = slice(aoi_lon[0], aoi_lon[1]),
                            lat = slice(aoi_lat[0], aoi_lat[1]))

    flo_aoi_date.rio.write_crs(aoi.crs, inplace = True)
    flo_aoi_date = flo_aoi_date.rio.clip(aoi.geometry, aoi.crs)
    if isinstance(date, np.datetime64):
        output_path = f"{output_path}{pd.Timestamp(date).year}.nc"
    if isinstance(date, np.ndarray):
        output_path = f"{output_path}{pd.Timestamp(date[0]).year}-{pd.Timestamp(date[-1]).year}.nc"
    flo_aoi_date.to_netcdf(output_path)

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

def filter_mines_with_buffer(raster, points, buffer_deg = 0.045):
    """Function to filter a full raster dataset with mindat mineral location, mindat mineral location gets buffer: 
        if mine cell of raster falls within buffer the cell passes filter, otherwise cells gets dropped

    Parameters
    ----------
    raster : xarray.Dataset
        Dataset of mines cells
    points : geopandas.GeoDataFrame
        GeoDataFrame of mindat mineral locations to link with mines
    buffer_deg : float, optional
        !high uncertainty!, float of degree buffer in WG84 to buffer the points, by default 0.045

    Returns
    -------
    raster_filtered : xarray.Dataset
        Dataset of mines that fall within buffer of points, thus likely containing the mineral of interest
    """
    points.set_geometry(col = 0)

    # create buffer in degrees 
    points.to_crs(raster.rio.crs, inplace = True)
    buffered_points = points.buffer(buffer_deg)

    # clip
    raster_filtered = raster.rio.clip(buffered_points.geometry, drop = True)
    raster_filtered = raster_filtered.rename_vars({"band_data": "mines"})
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

def add_time(ds, length, frequency):
    """Function to add more discrete timestep to dataset with flo1k annual streamflow (Q) data, Q gets divided by by length to represent mean annual streamflow divided by discrete timestep,
        returns new dataset

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to add more discrete timestep to
    length : int
        int of length of how many of the discrete timesteps in a year there are: e.g; 52 (weekly timesteps), 12 (monthly timesteps), (365) (daily timesteps)
    frequency : str
        str of what length type (weekly, monthly, daily, etc.) the discrete timestep is, see pandas.data_range() for options

    Returns
    -------
    ds_new : xarray.Dataset
        Dataset with years divided into more discrete timesteps where Q (streamflow) is divided by length to represent mean annual streamflow divided by discrete timestep
    """
    start = ds.time.values[0]
    n_years = len(ds.time)
    date_range = pd.date_range(start, periods=n_years * length, freq=frequency)

    Q_repeated = np.repeat(ds["qav"].values, length, axis=0)

    Q_new = xr.DataArray(
        Q_repeated,
        dims=["time", "lat", "lon"],
        coords={
            "time": date_range,
            "lat": ds.lat,
            "lon": ds.lon,
        },
        attrs={
            "units": "m3/s",
            "description": f"Annual streamflow divided into {length} timesteps per year",
        }
    )
    Q_new = Q_new / length

    # start with Q, then add back every other data variable from the original
    ds_new = xr.Dataset({"Q": Q_new})
    for var in ds.data_vars:
        if var != "qav":
            # repeat static variables to match the new time dimension
            if "time" in ds[var].dims:
                var_repeated = np.repeat(ds[var].values, length, axis=0)
                ds_new[var] = xr.DataArray(
                    var_repeated,
                    dims=["time", "lat", "lon"],
                    coords={"time": date_range, "lat": ds.lat, "lon": ds.lon}
                )
            else:
                ds_new[var] = ds[var]

    # restore non-time scalar coordinates (
    for coord in ds.coords:
        if coord not in ("time", "lat", "lon") and coord not in ds_new.coords:
            ds_new[coord] = ds[coord]

    return ds_new

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
    ds["ore"] = xr.where(ds["mines"].notnull(), constant, np.nan)
    ds["ore"].attrs = {"description": "Estimation of reactive ore per cell", "unit": "m2",}
    ds = ds.drop_vars(["mines"])
    return ds

def animation_plot(dataarray, aoi = None, size = (10, 8), cmap = "Reds"):
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
    """
    def _ani_update(t):
        im.set_data(plot_data[t].values)
        ax.set_title(f"{date_range[t][0]}/{date_range[t][1]}/{date_range[t][2]}")
        return [im]
    
    fig, ax = plt.subplots(figsize = size)
    plot_data = dataarray
    im = ax. imshow(plot_data[0].values, cmap = cmap, 
                    extent=[plot_data.lon.min(), plot_data.lon.max(), 
                            plot_data.lat.min(), plot_data.lat.max()],
                            origin='lower', aspect='auto')
    
    cbar = fig.colorbar(im, ax = ax, label = f"{dataarray.name} ({dataarray.units})")

    date_range = [(day, month, year) for day, month, year in zip(
        dataarray.time.dt.day.values,  
        dataarray.time.dt.month.values, 
        dataarray.time.dt.year.values)]
    
    title = ax.set_title(f"{date_range[0][0]}/{date_range[0][1]}/{date_range[0][2]}")

    if aoi is not None:
        aoi.boundary.plot(ax = ax, color = "k")
        
    ani = animation.FuncAnimation(fig, _ani_update, frames = len(date_range), 
                                  repeat = True, interval = 500, blit = True)
    plt.close()
    return ani

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