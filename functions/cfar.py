# https://www.mathworks.com/help/phased/ug/constant-false-alarm-rate-cfar-detection.html 
# https://developers.google.com/earth-engine/apidocs/ee-image-reduceneighborhood
# https://developers.google.com/earth-engine/apidocs/ee-kernel-fixed
# https://claude.ai/share/d0248a86-3e60-4c35-9cf8-85ab799be6ca
import ee
def cfar(image, guard=20, outer=25, k=4):
    """
    Applies a Constant False Alarm Rate Detection of bright targets in a a Sentinel-1 VH image

    Args:
        image: A GEE image
        guard: The maximum radius from the cell that get ignored for calculating the threshold in pixels (10m for Sentinel-1)
        outer: The maximum radius of cells that are used to calculate the threshold in pixels (10m for Sentinel-1)
        k: how many standard deviations above the background mean a pixel must be to count as a ship

    Returns:
        A GEE image with selected cells of ships, 1 where a ship is detected, masked everywhere else
    """

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