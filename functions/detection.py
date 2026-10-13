import ee
import geemap
import geopandas as gpd 
from shapely.geometry import box
import pandas as pd

def load_water(shp_path, bbox, buffer=10):
    """Load the OSM water polygons for the study area as an Earth Engine geometry.

    The polygons are clipped to the bbox and shrunk inward, so the bright edges of the docks are not mistaken for ships.
    Args:
        shp_path: Path to the OSM water shapefile
        bbox: Study area as bbox [xmin, ymin, xmax, ymax] in lon/lat
        buffer: How many meters to shrink the water polygon inwards

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
        start_date: First date as 'YYYY-MM-DD'
        end_date: Last date as 'YYYY-MM-DD' (Not included)
        orbit: Relative orbit number

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
        .sort('system:time_start')
    )
    return collection
            
def cfar(image, guard=10, outer=15, k=4):
    """Applies a Constant False Alarm Rate Detection of bright targets in a Sentinel-1 VH image

    Args:
        image: A GEE image
        guard: The maximum radius from the cell that are ignored for calculating the threshold in pixels (10m for Sentinel-1)
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

def sea_threshold(image, water, k):
    """Calculates the threshold from the median VH of the water + k times the std (estimate at p84)
    
    Args:
        image: A Sentinel-1 ee.Image with a VH band
        water: ee.Geometry of the water mask
        k: Standard deviations above the median

    Returns:
        Threshold to use for detecting cells as ships.
    """
    # https://developers.google.com/earth-engine/apidocs/ee-reducer-percentile
    # https://claude.ai/share/45094391-4f12-42a3-a4d1-f95b4d5ccd00
    vh = image.select('VH')
    
    vh_percentiles = vh.reduceRegion(
        reducer=ee.Reducer.percentile([50, 84]),
        geometry=water,
        scale=30,
        maxPixels=1e9
    )

    median = ee.Number(vh_percentiles.get('VH_p50'))
    p84 = ee.Number(vh_percentiles.get('VH_p84'))

    one_std = p84.subtract(median)
    threshold = median.add(one_std.multiply(k))
    return threshold

def detect_ships(image, water, method='threshold', guard=10, outer=15, threshold=-15, k=4, min_pixels=8):
    """Detect ships in one Sentinel-1 image

    Args:
        image: A Sentinel-1 ee.Image with a VH band
        water: ee.Geometry of the water mask
        method: 'cfar', local threshold; 'threshold', fixed value; 'adaptive' uses k std
        guard: maximum radius of ignored pixels, used when method='cfar'
        outer: maximum radius of pixels used to calculate threshold, used when method='cfar'
        threshold: used when method='threshold'
        k: Standard deviations above the background, used when method='cfar' or method='adaptive'
        min_pixels: Smallest group of bright pixels that count as a ship

    Returns:
        An ee.FeatureCollection with one polygon per detected ship
    """
    # https://developers.google.com/earth-engine/apidocs/ee-image-connectedpixelcount
    # https://developers.google.com/earth-engine/guides/reducers_reduce_to_vectors
    
    # Keep only the VH band of S1 image and only over the water mask
    vh = image.select('VH').clip(water)
    
    # Find bright pixels using cfar, adaptive or static threshold
    if method == 'cfar':
        bright = cfar(vh, guard=guard, outer=outer, k=k)
    elif method == 'threshold':
        bright = vh.gt(threshold).selfMask()
    elif method == 'adaptive':
        adaptive_threshold = sea_threshold(image, water, k)
        bright = vh.gt(adaptive_threshold).selfMask()
        


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


def count_ships(image, water, **kwargs):
    """Count the ships in one Sentinel-1 image

    Args:
        image: A Sentinel-1 ee.Image with a VH band
        water: ee.Geometry of the water mask
        method: 'cfar', local threshold or 'threshold', fixed value.
        guard: maximum radius of ignored pixels
        outer: maximum radius of pixels used to calculate threshold
        threshold: Fixed theshold, used when method='threshold'
        k: Standard deviations above the background, used when method='cfar'
        min_pixels: Smallest group of bright pixels that count as a ship

    Returns: 
        An ee.Feature with the date and the number of ships
    """
    # https://developers.google.com/earth-engine/apidocs/ee-feature
    # https://claude.ai/share/edd311b1-6345-4eb1-a723-808e7efc20b0
    ships = detect_ships(image, water, **kwargs)

    row = ee.Feature(None, {
        'date': image.date().format('YYYY-MM-dd HH:mm'),
        'n_ships': ships.size(),
    })

    return row

def ship_counts(collection, water, **kwargs):
    """Count the ships in every image of a Sentinel-1 collection
    
    Args: 
        collection: ee.ImageCollection of Sentinel-1 images
        water: ee.Geometry of the water mask.
        **kwargs: Detection settings passed on to count_ships.
    
    Returns:
        A pandas DataFrame with the number of ships per date
    """
    # https://developers.google.com/earth-engine/guides/ic_mapping
    # https://claude.ai/share/02ac0efd-b5ef-4fd8-a20e-c1614b7cccd4
    # Count the ships in every image
    fc = ee.FeatureCollection(collection.map(lambda img: count_ships(img, water, **kwargs)))

    info = fc.getInfo()
    rows = []
    for feature in info['features']:
        rows.append(feature['properties'])
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])

    return df.set_index('date').sort_index()

import ee
import requests
from PIL import Image, ImageDraw
from functions.detection import detect_ships


def make_timelapse(scenes, water, aoi, out_gif, event_date, **kwargs):
    """Save a GIF of Sentinel-1 scenes with the detected ships in red

    The frames are made one at a time, because asking Earth Engine for all
    frames at once gives a 'Too many concurrent aggregations' error.

    Args:
        scenes: ee.ImageCollection of Sentinel-1 images
        water: ee.Geometry of the water mask
        aoi: ee.Geometry of the area shown in the GIF
        out_gif: Path of the output GIF
                event_date: Date of the event as 'YYYY-MM-DD'; frames are labelled pre- or post-event
        **kwargs: Detection settings passed on to detect_ships
    """
    # https://developers.google.com/earth-engine/guides/feature_collections_visualizing
    # https://developers.google.com/earth-engine/apidocs/ee-image-getthumburl
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif-saving
    frames = []

    # Loop over the scene IDs, so only one request is needed to list all scenes
    for scene_id in scenes.aggregate_array('system:index').getInfo():
        img = ee.Image('COPERNICUS/S1_GRD/' + scene_id)

        # Detect the ships and draw the polygons in red 
        ships = detect_ships(img, water, **kwargs).draw(color='D55E00')

        # Radar image in grey with the red ships on top
        frame = img.select('VH').visualize(min=-25, max=-5).blend(ships)

        # Let Earth Engine render the frame as a PNG and download it
        url = frame.getThumbURL({'region': aoi, 'dimensions': 600, 'format': 'png'})
        png = Image.open(requests.get(url, stream=True).raw).convert('RGB')

        # Write the date (from the scene ID, e.g. 20230601) as 2023-06-01, plus pre- or post-event
        d = scene_id.split('_')[4]
        date = f'{d[:4]}-{d[4:6]}-{d[6:8]}'
        label = 'pre-event' if date < event_date else 'post-event'
        ImageDraw.Draw(png).text((10, 10), f'{date}   {label}', fill='white', font_size=22)
        frames.append(png)

    # Combine all frames into one GIF: 400 ms per frame, looping forever
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=400, loop=0)