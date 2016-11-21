"""
Module for performing transforms on an image to determine its BOO.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from scipy.spatial import Delaunay

class DelaunayAnalyzer(object):
    """
    Class that performs Delaunay triangulation on a set of points.
    """
    def __init__(self, points):
        self.points = points
        self._triangles = Delaunay(self.points)

    def _find_neighbours(self, point_index):
        """
        Args:
            point_index: the index of the current point being analyzed
            triangles: the mesh of triangles computed in the Delaunay
            computation

        Returns:
            A list of neighbouring points for the point in consideration.
        """
        return self._triangles.vertex_neighbor_vertices[1][self._triangles.vertex_neighbor_vertices[0][point_index]:self._triangles.vertex_neighbor_vertices[0][point_index + 1]]

    def boo_graph(self):
        """
            Create Bond orientational graph of points
        """
        points = np.array([list(elem) for elem in self.points])
        triang = tri.Triangulation(x=points[:,0], y=points[:,1])
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.triplot(triang, 'bo-')
        plt.title('triplot of Delaunay triangulation')
        plt.show()
        return points


    def boo_psi(self, guess):
        """
        Args:
            guess: the guess of the number of nearest neigbours to try for
            the comuptation.
        """
        psi = []
        for i, point in enumerate(self.points):
            neighbours = self._find_neighbours(i)
            value = 0
            for _, index in enumerate(neighbours):
                vector = [self.points[index][0] - point[0], self.points[index][1] - point[1]]
                value += np.exp(guess * 1j * np.arccos(np.dot(vector, [0, 1]) / (np.linalg.norm(vector) * np.linalg.norm([0, 1]))))
                psi.append(np.linalg.norm(value / len(neighbours)))
        return psi

    def show(self, x_range):
        """
        Shows a response graph for various guesses at the number of neighbours
        for a given x value.

        Args
            x_range: the range over which to guess the number of neighbours
        """
        y_range = []
        for x_value in x_range:
            y_range.append(np.average(self.boo_psi(x_value)))
        plt.plot(x_range, y_range, 'o')
        plt.show()

class BlobDetector(object):
    """
    A thin wrapper over the blob detector in open cv.
    """
    def __init__(self, params=cv2.SimpleBlobDetector_Params()):
        self.params = params

    # TODO(ckitagawa): Automate the tuning of these parameters.
    def set_defaults(self):
        """
        Sets a default set of parameters that work well but should be tuned.
        """
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.minThreshold = 0.5
        self.params.maxThreshold = 100
        self.params.filterByArea = True
        self.params.minArea = 0.5
        self.params.filterByConvexity = True
        self.params.minConvexity = 0.4
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.01

    def detect_keypoints(self, img):
        """
        Performs blob detection on the image.

        Args
           img: an image to find blobs on.

        Returns a list of keypoints for the image.
        """
        # TODO(ckitagawa): This looks like an expensive operation consider not recreating it.
        detector = cv2.SimpleBlobDetector_create(self.params)
        keypoints = detector.detect(img)
        return keypoints

