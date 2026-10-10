from pandas.core.computation.pytables import FilterBinOp
import ee
import geemap
import geopandas as gpd 
from shapely.geometry import box


def load_water(shp_path, bbox, buffer=10):
    """Load the OSM water polygons for the study area as an Earth Engine geometry.

    The polygons are clipped to the bbox and shrunk inward, so the bright edges of the docks are not mistaken for ships.
    Args:
        shp_path: Path to the OSM water shapefile
        bbox: Study area as bbox [xmin, ymin, xmax, ymax] in lon/lat
        inward_buffer_m: How many meters to shrink the water polygon inwards

    Returns:
        An ee.Geometry of the water inside the bbox
    """
    # Clip water polygon to our area of interest, so only a small polygon is sent to Earth Engine
    water = gpd.read_file(shp_path).to_crs(4326).clip(box(*bbox))

    # Buffer in metres (UTM), then merge everything into one polygon
    water = water.to_crs(water.estimate_utm_crs())
    water['geometry'] = water.buffer(-buffer)
    water = water[~water.is_empty].dissolve().to_crs(4326)

    return geemap.geopandas_to_ee(water).geometry()

def load_s1(aoi, start_date, end_date, orbit):
    """Load all Sentinel-1 scenes within a timeframe that fully cover the study area

    Args:
        aoi: ee.Geometry of the study area.
        start_date: First date as 'YYYY-MM-DD
        end_date: Last date as 'YYYY-MM-DD' (Not included)
        orbit: Relative orbit number as

    Returns:
        An ee.ImageCollection of Sentinel-1 scenes, sorted by date.
    """
    # https://tutorials.geemap.org/ImageCollection/image_collection_overview/
    # https://developers.google.com/earth-engine/apidocs/ee-filter-contains
    # https://claude.ai/share/e6ed9cd1-0001-40a2-b15a-c4957eef1492

    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterDate(start_date, end_date)
        .filterBounds(aoi)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('relativeOrbitNumber_start', orbit))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filter(ee.Filter.contains('.geo', aoi))
    )
    return collection
            
def cfar(image, guard=20, outer=25, k=4):
    """Applies a Constant False Alarm Rate Detection of bright targets in a a Sentinel-1 VH image

    Args:
        image: A GEE image
        guard: The maximum radius from the cell that get ignored for calculating the threshold in pixels (10m for Sentinel-1)
        outer: The maximum radius of cells that are used to calculate the threshold in pixels (10m for Sentinel-1)
        k: how many standard deviations above the background mean a pixel must be to count as a ship

    Returns:
        A GEE image with selected cells of ships, 1 where a ship is detected, masked everywhere else
    """
    # https://www.mathworks.com/help/phased/ug/constant-false-alarm-rate-cfar-detection.html 
    # https://developers.google.com/earth-engine/apidocs/ee-image-reduceneighborhood
    # https://developers.google.com/earth-engine/apidocs/ee-kernel-fixed
    # https://claude.ai/share/d0248a86-3e60-4c35-9cf8-85ab799be6ca
    size = outer * 2 + 1 

    # kernel looks like a square ring of 1s, with 0s in the middle as guard zone
    weights = []
    for i in range(size):
        row = []
        for j in range(size):
            di = i - outer
            dj = j - outer
            if (-guard <= di <= guard) and (-guard <= dj <= guard):
                row.append(0)
            else:
                row.append(1)
        weights.append(row)    
    kernel = ee.Kernel.fixed(size, size, weights, normalize=False)

    # For every pixel it computes the mean and std of the ring around it
    mean = image.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=kernel)
    std = image.reduceNeighborhood(reducer=ee.Reducer.stdDev(), kernel=kernel)

    # CFAR rule: ship if pixel > mean + k * std
    threshold = mean.add(std.multiply(k))
    ships = image.gt(threshold).selfMask()

    return ships

def detect_ships(image, water, method='cfar', guard=10, outer=15, threshold=-15, k=4, min_pixels=8):
    """Detect ships in one Sentinel-1 image

    Args:
        image: A Sentinel-1 ee.Image with a VH band
        water: ee.Geometry of the water mask
        method: 'cfar', local threshold or 'threshold', fixed value.
        threshold: Fixed theshold, used when method='threshold'
        k: Standard deviations above the background, used when method='cfar'
        min_pixels: Smallest group of bright pixels that count as a ship

    Returns:
        An ee.FeatureCollection with one polygon per detected ship
    """
    # https://developers.google.com/earth-engine/apidocs/ee-image-connectedpixelcount
    # https://developers.google.com/earth-engine/guides/reducers_reduce_to_vectors
    
    # Keep only the VH band of S1 image and only over the water mask
    vh = image.select('VH').clip(water)
    
    # Find bright pixels using cfar or static threshold
    if method == 'cfar':
        bright = cfar(vh, guard=guard, outer=outer, k=k)
    else:
        bright = vh.gt(threshold).selfMask()

    # Remove groups smaller than min_pixels
    cluster_size = bright.connectedPixelCount(maxSize=100, eightConnected=True)
    ships = bright.updateMask(cluster_size.gte(min_pixels))

    # Turn each group of pixels into a polygon
    vectors = ships.reduceToVectors(
        geometry=water,
        scale=10,
        geometryType='polygon',
        eightConnected=True,
        tileScale=4,
    )
    return vectors




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