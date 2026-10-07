import geemap
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
    threshold = - 10
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