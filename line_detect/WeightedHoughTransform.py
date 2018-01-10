from line_detect.HoughTransform import HoughTransform
import toolbox.pixeltools
import numpy as np
from math import pi, cos, sin

class WeightedHoughTransform(HoughTransform):

    '''returns in the form:
    (hough transform, radius range).

    Radius is from the center of the image, with the origin at the bottom left.
    '''
    def hough_transform(self, img, n_radius_bins = 120, n_theta_bins = 120):
        '''
        hough = np.zeros((n_radius_bins, n_theta_bins), np.int)
        unbinned_projs, radius_range = self.get_prevote_projections(img, n_radius_bins, n_theta_bins)
        binned_projs = (n_radius_bins * (unbinned_projs - radius_range[0])/(radius_range[1]-radius_range[0])).astype(np.int)
        for theta_bin in range(0, hough.shape[1]):
            bin_counts = np.bincount(binned_projs[theta_bin], minlength = n_radius_bins)
            hough[:,theta_bin] += bin_counts
        return hough, radius_range
        '''
        return None, None
