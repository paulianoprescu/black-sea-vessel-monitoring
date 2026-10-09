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


import pandas as pd

def s1_detect_ships(image, aoi, min_pixels=8, threshold=-15, vectors=False):
    """Detect ships in one Sentinel-1 image.
    input : Sentinel-1 image, area of interest (water geometry)
    output: ee.Feature with date, orbit and ship count (default),
            or ee.FeatureCollection with one polygon per vessel (vectors=True)
    """
    # 1 where VH > threshold, masked (transparent) everywhere else
    vh = image.select('VH').clip(aoi)
    bright = vh.gt(threshold).selfMask()

    # each bright pixel gets the size of its 8-connected cluster
    cluster_size = bright.connectedPixelCount(maxSize=100, eightConnected=True)

    # keep only clusters of at least min_pixels
    ships = bright.updateMask(cluster_size.gte(min_pixels))

    # polygon vector per remaining cluster
    ship_polygons = ships.reduceToVectors(
        geometry=aoi,
        scale=10,
        geometryType='polygon',
        eightConnected=True,
        tileScale=4,
    )

    if vectors:
        return ship_polygons

    # count vessel polygons detected
    return ee.Feature(None, {
        'date': image.date().format('YYYY-MM-dd HH:mm'),
        'orbit': image.get('relativeOrbitNumber_start'),
        'n_ships': ship_polygons.size(),
    })


def ship_counts(collection, aoi, **kwargs):
    """Run detect_ships on every image of a collection.
    output: pandas DataFrame indexed by acquisition date
    """
    fc = ee.FeatureCollection(collection.map(lambda img: s1_detect_ships(img, aoi, **kwargs)))
    info = fc.getInfo()
    df = pd.DataFrame([f['properties'] for f in info['features']])
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()