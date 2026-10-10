# Guard zone = 15 pixels
# Background ring = 5 pixels

import ee
def cfar(guard=20, outer=25):
    size = outer * 2 + 1 

    weights = []
    for i in range(size):
        row = []
        for j in range(size):
            di = i - outer
            dj = j - outer
            if (-20 < di < 20) or (-20 < dj < 20):
                row[j] = 0
            else:
                row[j] = 1
        weights.append(row)
                

    kernel = ee.Kernel.fixed(size, size, weights, normalize=False)