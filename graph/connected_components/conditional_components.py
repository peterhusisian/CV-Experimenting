from graph.node import Node
from graph.queue import Queue
import toolbox.pixeltools as pixeltools
import numpy as np
import cv2

#takes an image and assumes all neighboring pixels as each pixel's children, and only considers those neighbors
#that satisfy the condition function that takes the mask and the two pixels as arguments
'''
def conditional_connected_components(img, condition, mask = None):
    #-1 means unexplored (white), otherwise pixels with the same number in ccomps are a member of the same connected component (and are black)
    if mask is None: mask = np.ones(img.shape)
    ccomps = np.full(img.shape, -1)
    current_ccomp = 1
    while len(np.argwhere((ccomps * mask) == -1)) > 0:
        queue = Queue()
        #picks a non-visited node to explore as a root
        root = np.argwhere((ccomps * mask) == -1)[0]
        ccomps[root[0], root[1]] = current_ccomp
        queue.push(root)
        while not queue.is_empty():
            v = queue.pop()
            ccomps[v[0], v[1]] = current_ccomp
            v_neighbors = neighbors(img, v, mask, condition = condition)
            for v_neighbor in v_neighbors:
                if ccomps[v_neighbor[0], v_neighbor[1]] == -1:
                    queue.push(v_neighbor)
                    ccomps[v_neighbor[0], v_neighbor[1]] = current_ccomp
        current_ccomp += 1
    print("total ccomps: ", current_ccomp)
    return ccomps
'''

def conditional_connected_components(img, condition, mask = None):
    if mask is None: mask = np.ones(img.shape)
    visited_pixels = np.full(img.shape, 0).astype(np.int)
    #sets pixels outside the mask as being visited
    visited_pixels[mask == 0] = 1

    ccomps = []
    num_visited = np.count_nonzero(visited_pixels)

    while num_visited < img.shape[0] * img.shape[1]:
        comp = []
        queue = Queue()
        root = np.argwhere(visited_pixels == 0)[0]
        num_visited += 1
        visited_pixels[root[0], root[1]] = 1
        queue.push(root)

        while not queue.is_empty():
            v = queue.pop()
            comp.append(v)
            v_neighbors = neighbors(img, v, mask, condition = condition)
            for v_neighbor in v_neighbors:
                if visited_pixels[v_neighbor[0], v_neighbor[1]] == 0:
                    queue.push(v_neighbor)
                    comp.append(v_neighbor)
                    visited_pixels[v_neighbor[0], v_neighbor[1]] = 1
                    num_visited += 1
        ccomps.append(np.asarray(comp).astype(np.int))
    return ccomps


#gives neighbors to the pixel in [y, x] (index i, j) form
def neighbors(img, pixel, mask, condition = None):
    neighbors = []
    for i in range(pixel[0]-1, pixel[0]+2):
        for j in range(pixel[1]-1, pixel[1]+2):
            neighbor = np.array([i, j], dtype = np.int)
            if not (pixel[0] == i and pixel[1] == j) and pixeltools.pixel_in_bounds(img.shape, neighbor) and mask[i,j] == 1:
                if condition is None:
                    neighbors.append(neighbor)
                else:
                    if condition(img, pixel, neighbor):
                        neighbors.append(neighbor)
    return np.asarray(neighbors)
