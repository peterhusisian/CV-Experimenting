from line_detect.NormalHoughTransform import NormalHoughTransform
from line_detect.WeightedHoughTransform import WeightedHoughTransform
import numpy as np
import toolbox.pixeltools
import cv2
import toolbox.extrema.ExtremaFinder as ExtremaFinder
import scipy.signal as signal
from math import atan2, sin, cos, pi, acos
import graph.connected_components.conditional_components as conditional_components
SWT_DEFAULT_VALUE = 10000000

def SWT(canny, grad_angles, stroke_range):
    swt = np.full(canny.shape, SWT_DEFAULT_VALUE).astype(np.float64)
    edges = np.argwhere(canny != 0)
    for edge in edges:
        t1, path1 = SWT_path(canny, grad_angles, stroke_range, edge, grad_angles[edge[0], edge[1]])
        t2, path2 = SWT_path(canny, grad_angles, stroke_range, edge, pi + grad_angles[edge[0], edge[1]])
        stroke_path = []
        t = SWT_DEFAULT_VALUE
        if min(t1, t2) < SWT_DEFAULT_VALUE:
            stroke_path = path1 if t1 < t2 else path2
            t = t1 if t1 < t2 else t2
        for pixel in stroke_path:
            if t < swt[pixel[0], pixel[1]]:
                swt[pixel[0], pixel[1]] = t
    return swt


def SWT_path(canny, grad_angles, stroke_range, edge, edge_angle):
    sin_angle = sin(edge_angle)
    cos_angle = cos(edge_angle)
    path = []
    for t in range(0, stroke_range[1]):
        x = int(edge[1] + t*cos_angle)
        y = int(edge[0] + t*sin_angle)
        if in_bounds(canny.shape, x, y):
            path.append(np.array([y, x]))
            if not (x == edge[1] and y == edge[0]) and t >= stroke_range[0] and canny[y, x] != 0:
                if min(angle_between(edge_angle + pi, grad_angles[y,x]), angle_between(edge_angle, grad_angles[y,x])) < pi/3.0:
                    return t, path
                return SWT_DEFAULT_VALUE, []

    return SWT_DEFAULT_VALUE, []

def pixel_in_bounds(bounds, pixel):
    return pixel[0] >= 0 and pixel[0] < bounds[0] and pixel[1] >= 0 and pixel[1] < bounds[1]

def in_bounds(bounds, x, y):
    return y >= 0 and y < bounds[0] and x >= 0 and x < bounds[1]

def dot_angle_between(v1, v2):
    cos_angle = np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))
    if cos_angle > 1:
        cos_angle = 1
    if cos_angle < -1:
        cos_angle = -1
    return acos(cos_angle)

def angle_between(a1, a2):
    v1 = np.array([cos(a1), sin(a1)])
    v2 = np.array([cos(a2), sin(a2)])
    return dot_angle_between(v1, v2)

def SWT_ccomps_condition(img, p1, p2):
    ratio = img[p1[0], p1[1]]/img[p2[0], p2[1]]
    if ratio < 1: ratio = 1.0/ratio
    return ratio < 3

def remove_SWT_ccomps(swt, ccomps):
    i = 0
    ASPECT_RATIO_RANGE = (.1, 10)
    #further improvements: Rather than taking the vertical bounding box of each connected component, eliminate using the aspect ratio
    #of the minimum-area-bounding box of the connected component. will deal better with small sideways runway stripes, etc.
    #Also, the "diameter" limitation from the paper
    while i < len(ccomps):
        swt_subset = swt[ccomps[i][:,0], ccomps[i][:,1]]
        subset_mean = np.average(swt_subset)
        subset_variance = np.var(swt_subset)
        subset_bbox = cv2.boundingRect(ccomps[i])
        comp_aspect_ratio = float(subset_bbox[2])/float(subset_bbox[3])
        print("subset bbox: ", subset_bbox)

        if subset_variance > 0.5*subset_mean or not (comp_aspect_ratio >= ASPECT_RATIO_RANGE[0] and comp_aspect_ratio <= ASPECT_RATIO_RANGE[1]):
            del ccomps[i]
        else:
            i+=1




#PATH = "C:/Users/Peter/Desktop/Free Time CS Projects/Computer Vision Experimenting/images/letterR.png"
#PATH = "C:/Users/Peter/Desktop/Free Time CS Projects/Computer Vision Experimenting/images/300 crop 6480x4320.jpeg"
PATH = "C:/Users/Peter/Desktop/Free Time CS Projects/Computer Vision Experimenting/images/targets 300.JPG"
color_img = cv2.imread(PATH)
color_img = cv2.resize(color_img, (1024,768 ))
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
img = gray_img
img = cv2.GaussianBlur(gray_img, (5,5), 1, 1)
grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
grad_angles = np.arctan2(grad_y, grad_x)
canny = cv2.Canny(img, 30, 10)
#cv2.imshow("canny: ", canny)
#cv2.waitKey(0)

swt = SWT(canny, grad_angles, (3,35))
swt_draw = swt.copy()
swt_draw[swt_draw == SWT_DEFAULT_VALUE] = 0
swt_draw = np.uint8(255*swt_draw/swt_draw.max())
mask = 1-(swt == SWT_DEFAULT_VALUE)
ccomps = conditional_components.conditional_connected_components(swt, SWT_ccomps_condition, mask = mask)
remove_SWT_ccomps(swt, ccomps)
print("len ccomps; ", len(ccomps))

comp_draw = np.zeros(swt.shape)
for i in range(0, len(ccomps)):
    comp_draw[ccomps[i][:,0], ccomps[i][:,1]] = i+1

cv2.imshow("CCOMPS: ", np.uint8(255*comp_draw/comp_draw.max()))
cv2.waitKey(0)






















'''
PATH = "C:/Users/Peter/Desktop/Free Time CS Projects/Test Images/city.jpg"
NUM_BILAT_FILTERS = 5
NUM_HOUGH_SMOOTHS = 0
color_img = cv2.imread(PATH)
color_img = cv2.resize(color_img, (1920, 1080))

for i in range(0, NUM_BILAT_FILTERS):
    color_img = cv2.bilateralFilter(color_img, 5, 50,  50)

img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 1, 1)
img = cv2.Canny(img, 90, 50)


hough_maker = NormalHoughTransform()
img = hough_maker.remove_small_contours(img, 100)



hough, rad_bounds = hough_maker.hough_transform(img, n_radius_bins = 900, n_theta_bins = 900)

for i in range(0, NUM_HOUGH_SMOOTHS):
    hough = signal.convolve2d(hough, np.full((5,5), 1.0/25.0))




cv2.imshow("canny", img)

#hough_draw_points = PixelTools.points_over_thresh(hough, 500)
hough_draw_points =  ExtremaFinder.find_maxima(hough, min_val = 500)

color_hough = cv2.cvtColor(np.uint8(255*hough/hough.max()), cv2.COLOR_GRAY2RGB)
#color_hough[hough_draw_points[0], hough_draw_points[1]] = np.array([255,0,0])
radius = 2
for i in range(0, len(hough_draw_points)):
    cv2.rectangle(color_hough, (hough_draw_points[i][1]-radius, hough_draw_points[i][0]-radius), (hough_draw_points[i][1]+radius, hough_draw_points[i][0]+radius), (255,0,0))
    #color_hough[hough_draw_points[i][0] - radius : hough_draw_points[i][0] + radius+1, hough_draw_points[i][1] - radius : hough_draw_points[i][1] + radius+1] =  np.array([255,0,0])
cv2.imshow("hough", color_hough)

cv2.imshow('hough color: ', hough_maker.draw_houghs(color_img, hough, rad_bounds, hough_draw_points, color = (255,0,0)))#
cv2.waitKey(0)
'''
