import cv2
import numpy as np
from math import pi, sin, cos
from scipy import signal
import toolbox.pixeltools

class HoughTransform:
    def get_points_and_radius_range(self, img):
        points = PixelTools.get_white_pixels(img)
        center = np.array([img.shape[0]//2, img.shape[1]//2])
        points -= center

        radiuses = np.linalg.norm(points, axis = 1)
        radius_max = radiuses.max()
        radius_min = -radius_max
        return points, (radius_min, radius_max)

    def get_prevote_projections(self, points, center, n_radius_bins, theta):
        angle_vec = np.array([-sin(theta), cos(theta)])
        projs = np.dot(points, angle_vec.T)
        return projs

    def remove_small_contours(self, img, min_contour_size):
        ret, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            if contours[i].shape[0] < min_contour_size:
                cv2.drawContours(img, contours, i, 0)
        return img

    def draw_houghs(self, image, hough, radius_bounds, hough_points, color = (0,255,0)):
        center = np.array([image.shape[0]//2, image.shape[1]//2])
        max_mag = np.linalg.norm(np.array(image.shape))
        t_bounds = (-max_mag, max_mag)
        for point in hough_points:
            r = (radius_bounds[1]-radius_bounds[0])*(point[0]/hough.shape[0])+radius_bounds[0]
            theta = 2.0 * pi * point[1]/hough.shape[1]
            base_x = r*cos(theta)+center[1]
            base_y = center[0] - r*sin(theta)

            p1 = (int(base_x + cos(theta + pi/2.0)*t_bounds[0]), int(base_y - sin(theta + pi/2.0)*t_bounds[0]))
            p2 = (int(base_x + cos(theta + pi/2.0)*t_bounds[1]), int(base_y - sin(theta + pi/2.0)*t_bounds[1]))
            image = cv2.line(image, p1, p2, color)
        return image
