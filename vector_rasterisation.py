# rasterise mining polygons
# code sourced from: K. Haven 2019 https://medium.com/data-science/use-python-to-convert-polygons-to-raster-with-gdal-rasterizelayer-b0de1ec3267
from osgeo import gdal
from osgeo import ogr
nc_path = "../data/flo_IPB_last_date.nc"

raster = gdal.Open(f"NETCDF:{nc_path}:qav")
driver = ogr.GetDriverByName('ESRI Shapefile')
data_source = driver.Open("../data/mine_polygons/74548_projected polygons.shp", 0)  # 0 = read-only
if data_source is None:
    print("Could not open shapefile")
else:
    print("Successfully opened shapefile")
    layer = data_source.GetLayer()
    print(f"Layer name: {layer.GetName()}")
#vector = gdal.Open("../data/mine_polygons/74548_projected polygons.shp")

geo_transform = raster.GetGeoTransform()
projection = raster.GetProjection()
x_size = raster.RasterXSize
y_size = raster.RasterYSize

#layer = vector.GetLayer()

drv_tiff = gdal.GetDriverByName("GTiff")
output_ds = drv_tiff.Create(
    "../data/mines_raster_IPB.tif",
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
