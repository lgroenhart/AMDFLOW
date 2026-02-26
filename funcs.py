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
    df_region = pd.json_normalize(region_json['results'])
    df_region = df_region[
    # check for mine/mining/quarry in description (case insensitive)
    df_region["description_short"].str.contains(r"\b(mine|mining|quarry)\b", 
                                        case=False, na=False, regex=True) &
    # check for Fe or S in elements string
    df_region["elements"].str.contains(f'{mineral_strings}', na=False, regex=True)
                            ]
    # filter region dataframe localities by only keeping the ids that are also in the material localities
    df_material_ids = pd.DataFrame(material["results"][0]["locality"], columns = ["id"])
    ids_to_keep = df_material_ids["id"].unique() 
    df_region_material = df_region[df_region["id"].isin(ids_to_keep)]
    df_region_material = df_region_material[["id", "latitude", "longitude"]]
    df_region_material.to_csv(f"{path_str}mindat_data/{region}_{material_name}.csv")
    print(f"\nSuccesfully saved data to: {path_str}mindat_data/{region}_{material_name}.csv")
    return

def vecor_rasterisation(output_path = "../data/mines_raster.tif", flo1k_path = "../data/flo_IPB_last_date.nc", vector_path = "../data/mine_polygons/74548_projected polygons.shp"):
    # rasterise mining polygons
    # code sourced from: K. Haven 2019 https://medium.com/data-science/use-python-to-convert-polygons-to-raster-with-gdal-rasterizelayer-b0de1ec3267
    nc_path = flo1k_path

    raster = gdal.Open(f"NETCDF:{nc_path}:qav")
    driver = ogr.GetDriverByName('ESRI Shapefile')
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


    options = [f'ATTRIBUTE=Shape_Area']
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
    output_path = f"{output_path}_{date.year}"
    flo_aoi_date.to_netcdf(output_path)
    
