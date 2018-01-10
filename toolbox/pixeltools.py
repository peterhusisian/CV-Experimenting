import numpy as np

def get_white_pixels(img):
    pixels = []
    center = np.array([img.shape[0]/2, img.shape[1]/2])
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if(img[y, x] != 0):
                pixels.append(np.array([y, x]))
    return np.asarray(pixels)


def points_over_thresh(arr, thresh):
    extrema = np.where(arr > thresh)
    return np.dstack(extrema)[0,:,:]

def pixel_in_bounds(bounds, pixel):
    return pixel[0] >= 0 and pixel[0] < bounds[0] and pixel[1] >= 0 and pixel[1] < bounds[1]
