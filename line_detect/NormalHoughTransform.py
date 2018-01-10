from line_detect.HoughTransform import HoughTransform
import toolbox.pixeltools
import numpy as np
from math import pi, cos, sin

class NormalHoughTransform(HoughTransform):

    '''returns in the form:
    (hough transform, radius range).

    Radius is from the center of the image, with the origin at the bottom left.
    '''
    def hough_transform(self, img, n_radius_bins = 120, n_theta_bins = 120):
        points, radius_range = self.get_points_and_radius_range(img)
        center = np.linalg.norm(np.array(img.shape) // 2)
        hough = np.zeros((n_radius_bins, n_theta_bins), np.int)
        THETA_STEP = 2.0*pi/n_theta_bins
        for theta_bin in range(0, hough.shape[1]):
            iter_unbinned_projs = self.get_prevote_projections(points, center, n_radius_bins, theta_bin * THETA_STEP)
            iter_binned_projs = (n_radius_bins * (iter_unbinned_projs - radius_range[0])/(radius_range[1]-radius_range[0])).astype(np.int)
            bin_counts = np.bincount(iter_binned_projs, minlength = n_radius_bins)
            hough[:,theta_bin] += bin_counts
        return hough, radius_range
