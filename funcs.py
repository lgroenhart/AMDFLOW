# functions py file for AMDFLOW
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

def mindat_collector(region, material_id = 3314, mineral_strings = "(Fe|S)", material_name = "pyrite", path_str = "../data/", mindat_api_str = "mindat_API_key.txt"):

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

def vecor_rasterisation(output_path = "../data/mines_raster.tif", flo1k_path = "../data/flo_IPB_2015.nc", vector_path = "../data/mine_polygons/74548_projected polygons.shp"):
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
               date = pd.Timestamp(year = 2015, month = 1, day = 1),
               output_path = f"../data/flo_IPB_"):
    
    
    flo_one_k = xr.open_dataset(flo1k_path)
    basins = gpd.read_file(basins_path)

    aoi = basins.iloc[basins_iloc[0]:basins_iloc[1]]
    aoi_lat = [float(aoi.total_bounds[1]), float(aoi.total_bounds[3])]
    aoi_lon = [float(aoi.total_bounds[0]), float(aoi.total_bounds[2])]
    flo_aoi_date = flo_one_k["qav"].sel(time = date,
                                  lon = slice(aoi_lon[0], aoi_lon[1]),
                                  lat = slice(aoi_lat[0], aoi_lat[1]))
    flo_aoi_date.rio.write_crs(aoi.crs, inplace = True)
    flo_aoi_date = flo_aoi_date.rio.clip(aoi.geometry, aoi.crs)
    output_path = f"{output_path}{date.year}.nc"
    flo_aoi_date.to_netcdf(output_path)

def hydir_IDs(ds, aoi):
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
    points.set_geometry(col = 0)

    # create buffer in degrees 
    points.to_crs(raster.rio.crs, inplace = True)
    buffered_points = points.buffer(buffer_deg)

    # clip
    raster_filtered = raster.rio.clip(buffered_points.geometry, drop = True)
    raster_filtered = raster_filtered.rename_vars({"band_data": "mines"})
    return raster_filtered

def bool_to_int(ds, dtype="int16"):
    """Convert all boolean data variables to the specified integer type."""
    for var in ds.data_vars:
        if ds[var].dtype == bool:
            ds[var] = ds[var].astype(dtype)
    return ds

def cleanup_and_metadata(ds):
    ds = ds.drop_vars(["band", "hydir"])
    ds["mines"].attrs = {"units": "none", "description": "mapping of mines, all non nan are mines"}
    ds = ds.assign_coords(x=("x", ds.lon.values), y=("y", ds.lat.values))
    ds = ds.drop_vars(["lat", "lon"])        
    ds = ds.rename({"x": "lon", "y": "lat"})   
    ds["mines"] = ds["mines"].squeeze("band", drop=True)
    return ds

def add_time(ds, year, length, frequency):
    date_range = pd.date_range(f"{year}-01-01", periods = length, freq = frequency)
    # Rename before expanding to avoid issues
    ds = ds.rename({"qav": "Q"})
    # Expand Q with the time dimension
    ds["Q"] = ds["Q"].expand_dims(time=date_range)
    ds = ds.set_coords('time')
    
    # Broadcast all other variables to match Q's dimensions and coordinates
    for var in ds.data_vars:
        if var != "Q":
            ds[var] = ds[var].broadcast_like(ds["Q"])
    
    ds["Q"].attrs = {"units": "m3/s", "description": "Average annual streamflow divided by length of time within year (e.g.: 12 for months)"}
    ds["Q"] = ds["Q"] / length
    return ds

def estimate_ore(ds, H, F):
    # uses ds["mines"] values, might be useful when it is filled with area, but not right now
    #ds["ore"] = ds["mines"].where(ds["mines"].notnull(), 0) * (H * ((4 * 1000) / 4) * F * 27)

    constant = H * ((4 * 1000) / 4) * F * 27
    ds["ore"] = xr.where(ds["mines"].notnull(), constant, np.nan)
    ds["ore"].attrs = {"description": "Estimation of reactive ore per cell", "unit": "m2",}
    ds = ds.drop_vars(["mines"])
    return ds
