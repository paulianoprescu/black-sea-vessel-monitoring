import ee

def detect_ships(image, aoi, min_pixels=8):
    """Function for counting the ships detected in 1 image
    input : Sentinel-1 image, Area of Interest (aoi)
    output : FeatureCollection, one polygon per vessel
    """
    # Transform Image VH band raster cells for AoI region into a 2D Matrix
    vh = geemap.ee_to_numpy(image.select('VH'), region = aoi, scale = 10)
    vh = vh[:,:,0] # drop band dimension : 2D array

    # Defining dB scatter threshold
    threshold = - 15
    vh = image.select('VH').clip(aoi)
    # 1 where VH > threshold, masked (transparent) everywhere else
    bright = vh.gt(threshold).selfMask()

    # Object size : Adjacent Raster Cells Above Threshold Filter
    # each bright pixel gets the size of its 8-connected cluster
    # (maxSize only needs to exceed min_pixels; larger is slower)
    cluster_size = bright.connectedPixelCount(maxSize=100, eightConnected=True)

    # keep only clusters of at least min_pixels
    ships = bright.updateMask(cluster_size.gte(min_pixels))

    # Return polygon vector per remaining cluster
    return ships.reduceToVectors(
        geometry=aoi,
        scale=10,
        geometryType='polygon',
        eightConnected=True,
        tileScale=4,
    )
    #print(vector.size().getInfo())

    # Count Vessel Polygons detected 

import geemap
import geopandas as gpd 
from shapely.geometry import box


def load_water(shp_path, bbox, inward_buffer_m=10):
    """Clip the OSM water polygons to the study area and convert them for Earth Engine.

    input : path to the OSM water shapefile, bbox [xmin, ymin, xmax, ymax] in lon/lat,
            inward buffer in metres that keeps the bright quay edge out of the mask
    output: ee.Geometry of the water inside the bbox
    """
    # clip locally first, so only a small polygon is sent to Earth Engine
    water = gpd.read_file(shp_path).to_crs(4326).clip(box(*bbox))

    # buffer in metres (UTM), then merge everything into one polygon
    water = water.to_crs(water.estimate_utm_crs())
    water['geometry'] = water.buffer(-inward_buffer_m)
    water = water[~water.is_empty].dissolve().to_crs(4326)

    return geemap.geopandas_to_ee(water).geometry()